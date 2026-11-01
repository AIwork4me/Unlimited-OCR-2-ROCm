"""Supervisor-side 1 Hz resource telemetry with a fail-closed staleness rule.

Rev-2 real-document admission policy (prerequisite P5): the supervised Windows
runner must sample available memory, worker allocator state and wall clock at
1 Hz, and treat >= 5 consecutive missed samples as *unknown* telemetry —
which stops the attempt instead of running unwatched. The worker side is the
PR #32 ``ResourceWatchdog`` (``watchdog.py``) publishing
``worker-heartbeat.json`` next to its attempt directory; this module reads it,
never trusts it (missing/stale heartbeats count as missed samples), and writes
the full series to ``telemetry.jsonl`` for the run ledger.

Fail-closed defaults follow the frozen policy envelope: admit-time and
in-flight floor of 48 GiB available, staleness threshold of 5 consecutive
missed 1 Hz samples, and a warm-up allowance before the worker's first
heartbeat is expected (the watchdog starts before model load, but process
creation and interpreter start are not instantaneous).
"""

from __future__ import annotations

import ctypes
import json
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

HEARTBEAT_NAME = "worker-heartbeat.json"
TELEMETRY_NAME = "telemetry.jsonl"

STOP_HOST_MEMORY_FLOOR = "HOST_AVAILABLE_MEMORY_BELOW_FLOOR"
STOP_TELEMETRY_STALE = "TELEMETRY_STALE_UNKNOWN"

#: Warm-up before a missing heartbeat starts counting as a missed sample.
DEFAULT_WARMUP_SECONDS = 60.0


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_uint64),
        ("ullAvailPhys", ctypes.c_uint64),
        ("ullTotalPageFile", ctypes.c_uint64),
        ("ullAvailPageFile", ctypes.c_uint64),
        ("ullTotalVirtual", ctypes.c_uint64),
        ("ullAvailVirtual", ctypes.c_uint64),
        ("ullAvailExtendedVirtual", ctypes.c_uint64),
    ]


def available_memory_bytes() -> int | None:
    """Host available RAM in bytes; ``None`` when it cannot be read (missed)."""
    try:
        if sys.platform == "win32":
            status = _MemoryStatus()
            status.dwLength = ctypes.sizeof(_MemoryStatus)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return None
            return int(status.ullAvailPhys)
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
        return None
    except Exception:  # noqa: BLE001 - a failed read is a missed sample
        return None


class ResourceTelemetryPolicy:
    """Frozen thresholds; all stops are process stops, never silent."""

    def __init__(
        self,
        *,
        min_available_bytes: int,
        interval_seconds: float = 1.0,
        staleness_samples: int = 5,
        warmup_seconds: float = DEFAULT_WARMUP_SECONDS,
    ) -> None:
        if min_available_bytes <= 0:
            raise ValueError("min_available_bytes must be positive")
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        if staleness_samples < 1:
            raise ValueError("staleness_samples must be at least 1")
        self.min_available_bytes = int(min_available_bytes)
        self.interval_seconds = float(interval_seconds)
        self.staleness_samples = int(staleness_samples)
        self.warmup_seconds = float(warmup_seconds)

    def as_dict(self) -> dict[str, Any]:
        return {
            "min_available_bytes": self.min_available_bytes,
            "interval_seconds": self.interval_seconds,
            "staleness_samples": self.staleness_samples,
            "warmup_seconds": self.warmup_seconds,
        }


class TelemetryMonitor:
    """Sample host + worker heartbeat at 1 Hz; decide policy stops.

    The monitor never kills anything itself — the supervisor owns the Job.
    ``stop_reason`` turns non-None the moment a frozen stop condition holds;
    the supervisor must then terminate the attempt and record the reason.
    """

    def __init__(
        self,
        attempt_directory: Path,
        policy: ResourceTelemetryPolicy,
        *,
        sampler: Callable[[], int | None] = available_memory_bytes,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._directory = attempt_directory
        self._policy = policy
        self._sampler = sampler
        self._clock = clock
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._samples = 0
        self._missed_streak = 0
        self._max_missed_streak = 0
        self._min_available: int | None = None
        self._stop_reason: str | None = None
        self._started: float | None = None
        self._last_heartbeat: dict[str, Any] | None = None
        self._telemetry_path = attempt_directory / TELEMETRY_NAME

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Telemetry monitor already started")
        self._stop.clear()
        self._started = self._clock()
        self._thread = threading.Thread(target=self._loop, name="resource-telemetry", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self._policy.interval_seconds * 4, 10.0))
            self._thread = None
        return self.receipt()

    # -- sampling -----------------------------------------------------------

    def check_once(self) -> None:
        """One telemetry tick; records the sample and evaluates both stops."""
        now = self._clock()
        available = self._sampler()
        heartbeat = self._read_heartbeat()
        elapsed = 0.0 if self._started is None else now - self._started
        warmup = elapsed < self._policy.warmup_seconds
        missed = available is None
        fresh = self._heartbeat_is_fresh(heartbeat, now)
        if not fresh and not warmup:
            missed = True
        with self._lock:
            self._samples += 1
            self._missed_streak = self._missed_streak + 1 if missed else 0
            self._max_missed_streak = max(self._max_missed_streak, self._missed_streak)
            if available is not None:
                previous = self._min_available
                self._min_available = available if previous is None else min(previous, available)
            self._last_heartbeat = heartbeat
            stop = self._stop_reason
            streak = self._missed_streak
        if stop is None:
            if available is not None and available < self._policy.min_available_bytes:
                self._record_stop(STOP_HOST_MEMORY_FLOOR)
            elif streak >= self._policy.staleness_samples:
                self._record_stop(STOP_TELEMETRY_STALE)
        self._append_sample(now, available, heartbeat, missed)

    def _heartbeat_is_fresh(self, heartbeat: dict[str, Any] | None, now: float) -> bool:
        if heartbeat is None:
            return False
        age = now - float(heartbeat.get("monotonic", 0.0))
        # Fresh means written within twice the worker's own heartbeat cadence;
        # ResourceWatchdog heartbeats use 20 s, so 40 s bounds the check even
        # when the pilot tightens the sampling interval to 1 Hz.
        return age <= 40.0

    def _read_heartbeat(self) -> dict[str, Any] | None:
        path = self._directory / HEARTBEAT_NAME
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def _append_sample(
        self,
        now: float,
        available: int | None,
        heartbeat: dict[str, Any] | None,
        missed: bool,
    ) -> None:
        watchdog = (heartbeat or {}).get("watchdog") or {}
        record = {
            "monotonic": round(now, 3),
            "available_bytes": available,
            "missed": missed,
            "heartbeat_present": heartbeat is not None,
            "worker_allocated_bytes": watchdog.get("peak_allocated_bytes"),
            "worker_limit_bytes": watchdog.get("limit_bytes"),
        }
        try:
            with self._telemetry_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        except OSError:
            pass  # series loss must never stop supervision; receipt still holds

    def _record_stop(self, reason: str) -> None:
        with self._lock:
            if self._stop_reason is None:
                self._stop_reason = reason

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.check_once()
            self._stop.wait(timeout=self._policy.interval_seconds)

    # -- evidence -----------------------------------------------------------

    @property
    def stop_reason(self) -> str | None:
        with self._lock:
            return self._stop_reason

    def receipt(self) -> dict[str, Any]:
        with self._lock:
            return {
                "telemetry": {
                    "policy": self._policy.as_dict(),
                    "samples": self._samples,
                    "max_missed_streak": self._max_missed_streak,
                    "min_available_bytes": self._min_available,
                    "stop_reason": self._stop_reason,
                    "heartbeat": self._last_heartbeat,
                }
            }


def attach_worker_watchdog(
    heartbeat_path: Path,
    *,
    limit_bytes: int,
    interval_seconds: float = 1.0,
    heartbeat_seconds: float = 1.0,
) -> Any:
    """Start the in-worker ResourceWatchdog over CUDA **reserved** memory.

    Fail-closed: if torch/CUDA is not importable the worker must not run
    unwatched, so the ImportError propagates to the caller. The pilot worker
    calls this *before* model load so the supervisor's staleness rule sees a
    heartbeat from the first seconds of the attempt.
    """
    import torch.cuda  # noqa: PLC0415 - deliberate lazy, worker-process-only import

    from .watchdog import ResourceWatchdog

    watchdog = ResourceWatchdog(
        sampler=lambda: int(torch.cuda.memory_reserved()),
        limit_bytes=limit_bytes,
        interval_seconds=interval_seconds,
        heartbeat_path=heartbeat_path,
        heartbeat_seconds=heartbeat_seconds,
    )
    watchdog.start()
    return watchdog
