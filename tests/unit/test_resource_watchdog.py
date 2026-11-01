"""ResourceWatchdog: runtime enforcement of the frozen resource ceilings."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from unlimited_ocr_2_rocm.benchmark.watchdog import WATCHDOG_EXIT_CODE, ResourceWatchdog


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_watchdog_aborts_once_on_limit_breach() -> None:
    allocations = iter([1_000, 2_000, 10_000])
    aborts: list[dict] = []
    watchdog = ResourceWatchdog(
        sampler=lambda: next(allocations),
        limit_bytes=5_000,
        abort=aborts.append,
    )
    assert watchdog.check_once() == {
        "allocated_bytes": 1_000,
        "limit_bytes": 5_000,
        "over_limit": False,
    }
    assert watchdog.check_once()["over_limit"] is False
    assert watchdog.check_once()["over_limit"] is True
    assert len(aborts) == 1
    assert aborts[0]["kind"] == "GPU_ALLOCATED_LIMIT_BREACH"
    assert aborts[0]["allocated_bytes"] == 10_000
    assert watchdog.receipt()["watchdog"]["peak_allocated_bytes"] == 10_000


def test_watchdog_background_loop_enforces_and_stops(tmp_path: Path) -> None:
    clock = _FakeClock()
    current = {"allocated": 1_000}
    aborts: list[dict] = []

    def fake_wait(timeout: float | None = None) -> bool:
        clock.advance(0.25)
        return False

    watchdog = ResourceWatchdog(
        sampler=lambda: current["allocated"],
        limit_bytes=5_000,
        interval_seconds=1.0,
        abort=aborts.append,
        clock=clock,
    )
    watchdog._stop.wait = fake_wait  # type: ignore[method-assign]
    watchdog.start()
    clock.advance(2.5)  # several samples under the limit
    current["allocated"] = 9_000
    clock.advance(2.0)  # the breach sample runs
    deadline = time.monotonic() + 5
    while not aborts and time.monotonic() < deadline:
        time.sleep(0.01)
    watchdog.stop()
    assert aborts and aborts[0]["allocated_bytes"] == 9_000
    assert watchdog.receipt()["watchdog"]["breach"] is not None


def test_watchdog_heartbeat_receipt_written(tmp_path: Path) -> None:
    clock = _FakeClock()
    heartbeat = tmp_path / "attempt.pending" / "heartbeat.json"
    watchdog = ResourceWatchdog(
        sampler=lambda: 1_234,
        limit_bytes=5_000,
        heartbeat_path=heartbeat,
        heartbeat_seconds=10.0,
        abort=lambda _breach: None,
        clock=clock,
    )
    watchdog.check_once()  # heartbeat carries the latest known sample
    watchdog._write_heartbeat()
    clock.advance(10.0)
    watchdog._write_heartbeat()
    payload = json.loads(heartbeat.read_text(encoding="utf-8"))
    assert payload["watchdog"]["peak_allocated_bytes"] == 1_234
    assert payload["watchdog"]["limit_bytes"] == 5_000
    assert heartbeat.with_suffix(".breach.json").exists() is False


def test_watchdog_default_abort_writes_breach_receipt(tmp_path: Path) -> None:
    heartbeat = tmp_path / "hb.json"
    watchdog = ResourceWatchdog(sampler=lambda: 0, limit_bytes=1, heartbeat_path=heartbeat)
    breach = {"kind": "GPU_ALLOCATED_LIMIT_BREACH", "allocated_bytes": 2, "limit_bytes": 1}
    exit_code: list[int] = []
    original_exit = __import__("os")._exit

    def fake_exit(code: int) -> None:  # type: ignore[no-untyped-def]
        exit_code.append(code)
        raise SystemExit(code)

    import os

    os._exit = fake_exit  # type: ignore[assignment]
    try:
        with pytest.raises(SystemExit):
            watchdog._default_abort(breach)
    finally:
        os._exit = original_exit  # type: ignore[assignment]
    assert exit_code == [WATCHDOG_EXIT_CODE]
    written = json.loads(heartbeat.with_suffix(".breach.json").read_text(encoding="utf-8"))
    assert written["allocated_bytes"] == 2


def test_watchdog_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError):
        ResourceWatchdog(sampler=lambda: 0, limit_bytes=0)
    with pytest.raises(ValueError):
        ResourceWatchdog(sampler=lambda: 0, limit_bytes=10, interval_seconds=0)


def test_watchdog_stop_without_start_is_safe() -> None:
    watchdog = ResourceWatchdog(sampler=lambda: 0, limit_bytes=1)
    assert watchdog.receipt()["watchdog"]["samples"] == 0
    assert "breach" in watchdog.receipt()["watchdog"]
    assert threading.current_thread().name != "resource-watchdog"
