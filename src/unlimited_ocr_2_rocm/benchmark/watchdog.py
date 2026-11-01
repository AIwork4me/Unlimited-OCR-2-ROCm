"""Runtime resource enforcement for supervised GPU attempts.

The P2-C stage gates CHECKED resource ceilings after completion; the L1
policy requires them ENFORCED during execution. A daemon thread samples an
injectable allocation callable (``torch.cuda.memory_allocated`` in the
worker), writes heartbeat receipts the supervisor can read live, and aborts
the process the moment the frozen hard limit is crossed. Allocation growth
between samples is bounded by the sampling interval: KV-cache grows
~60 KiB/token (~0.6 MiB/s at the 10.4 tok/s decode floor), so a 2 s interval
overshoots the 16 GiB limit by at most a few MiB — the gate stays honest
without killing a healthy run.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

WATCHDOG_EXIT_CODE = 86


class ResourceWatchdog:
    """Sample an allocation callable; abort the process above ``limit_bytes``.

    ``sampler`` returns currently-allocated bytes; ``abort`` defaults to
    writing a breach receipt next to the heartbeat and exiting the process
    with ``WATCHDOG_EXIT_CODE`` so the supervisor classifies the attempt as
    a failure (the outcome file carries the breach reason).
    """

    def __init__(
        self,
        *,
        sampler: Callable[[], int],
        limit_bytes: int,
        interval_seconds: float = 2.0,
        heartbeat_path: Path | None = None,
        heartbeat_seconds: float = 20.0,
        abort: Callable[[dict[str, Any]], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if limit_bytes <= 0:
            raise ValueError("limit_bytes must be positive")
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self._sampler = sampler
        self._limit_bytes = limit_bytes
        self._interval_seconds = interval_seconds
        self._heartbeat_path = heartbeat_path
        self._heartbeat_seconds = heartbeat_seconds
        self._abort = abort if abort is not None else self._default_abort
        self._clock = clock
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._samples = 0
        self._peak_bytes = 0
        self._breach: dict[str, Any] | None = None

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Watchdog already started")
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="resource-watchdog", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self._interval_seconds * 4, 10.0))
            self._thread = None
        return self.receipt()

    # -- sampling -----------------------------------------------------------

    def check_once(self) -> dict[str, Any]:
        """One sample; aborts when the frozen hard limit is crossed."""
        allocated = int(self._sampler())
        with self._lock:
            self._samples += 1
            self._peak_bytes = max(self._peak_bytes, allocated)
            over = allocated > self._limit_bytes
            if over and self._breach is None:
                self._breach = {
                    "kind": "GPU_ALLOCATED_LIMIT_BREACH",
                    "allocated_bytes": allocated,
                    "limit_bytes": self._limit_bytes,
                    "samples": self._samples,
                    "peak_bytes": self._peak_bytes,
                    "observed_monotonic": self._clock(),
                }
        if over and self._breach is not None:
            self._abort(self._breach)
        return {
            "allocated_bytes": allocated,
            "limit_bytes": self._limit_bytes,
            "over_limit": over,
        }

    def _loop(self) -> None:
        next_heartbeat = self._clock()
        next_sample = self._clock()
        while not self._stop.is_set():
            now = self._clock()
            if now >= next_sample:
                self.check_once()
                next_sample = now + self._interval_seconds
                if self._breach is not None:
                    return  # abort requested; the process is exiting
            if self._heartbeat_path is not None and now >= next_heartbeat:
                self._write_heartbeat()
                next_heartbeat = now + self._heartbeat_seconds
            self._stop.wait(timeout=min(self._interval_seconds / 4, 0.5))

    # -- receipts -----------------------------------------------------------

    def _write_heartbeat(self) -> None:
        if self._heartbeat_path is None:
            return
        try:
            self._heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "monotonic": round(self._clock(), 3),
                **self.receipt(),
            }
            self._heartbeat_path.write_text(
                json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
            )
        except OSError:
            pass  # heartbeat loss must never kill a healthy attempt

    def receipt(self) -> dict[str, Any]:
        with self._lock:
            return {
                "watchdog": {
                    "samples": self._samples,
                    "peak_allocated_bytes": self._peak_bytes,
                    "limit_bytes": self._limit_bytes,
                    "breach": self._breach,
                    "sink_errors": 0,
                }
            }

    # -- default abort --------------------------------------------------------

    def _default_abort(self, breach: dict[str, Any]) -> None:
        if self._heartbeat_path is not None:
            try:
                self._heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
                self._heartbeat_path.with_suffix(".breach.json").write_text(
                    json.dumps(breach, indent=1, sort_keys=True) + "\n", encoding="utf-8"
                )
            except OSError:
                pass
        os._exit(WATCHDOG_EXIT_CODE)
