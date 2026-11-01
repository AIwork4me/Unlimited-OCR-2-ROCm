"""P5: 1 Hz resource telemetry with staleness rule, wired into supervision."""

import json
import sys
from pathlib import Path

import pytest

from unlimited_ocr_2_rocm.benchmark.runner import supervise_worker, worker_outcome_to_finish
from unlimited_ocr_2_rocm.benchmark.telemetry import (
    HEARTBEAT_NAME,
    STOP_HOST_MEMORY_FLOOR,
    STOP_TELEMETRY_STALE,
    ResourceTelemetryPolicy,
    TelemetryMonitor,
    available_memory_bytes,
)

GIB = 1024**3


def write_heartbeat(directory: Path, monotonic: float, reserved: int = 0) -> None:
    payload = {
        "monotonic": monotonic,
        "watchdog": {
            "samples": 1,
            "peak_allocated_bytes": reserved,
            "limit_bytes": 16 * GIB,
            "breach": None,
            "sink_errors": 0,
        },
    }
    (directory / HEARTBEAT_NAME).write_text(json.dumps(payload), encoding="utf-8")


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_monitor_records_series_and_no_stop_when_healthy(tmp_path: Path) -> None:
    clock = FakeClock()
    policy = ResourceTelemetryPolicy(min_available_bytes=48 * GIB)
    monitor = TelemetryMonitor(tmp_path, policy, sampler=lambda: 64 * GIB, clock=clock)
    for tick in range(4):
        clock.now += 1.0
        write_heartbeat(tmp_path, clock.now, reserved=2 * GIB * (tick + 1))
        monitor.check_once()
    receipt = monitor.stop()
    telemetry = receipt["telemetry"]
    assert telemetry["stop_reason"] is None
    assert telemetry["samples"] == 4
    assert telemetry["max_missed_streak"] == 0
    assert telemetry["min_available_bytes"] == 64 * GIB
    series = [
        json.loads(line)
        for line in (tmp_path / "telemetry.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(series) == 4
    assert [row["worker_allocated_bytes"] for row in series] == [
        2 * GIB,
        4 * GIB,
        6 * GIB,
        8 * GIB,
    ]


def test_monitor_stops_after_five_consecutive_missed_samples(tmp_path: Path) -> None:
    clock = FakeClock()
    policy = ResourceTelemetryPolicy(min_available_bytes=48 * GIB, warmup_seconds=0.0)
    monitor = TelemetryMonitor(tmp_path, policy, sampler=lambda: 64 * GIB, clock=clock)
    monitor.start()
    for _ in range(5):
        clock.now += 1.0
        monitor.check_once()  # no heartbeat ever appears
        if monitor.stop_reason is None:
            continue
        break
    assert monitor.stop_reason == STOP_TELEMETRY_STALE
    receipt = monitor.stop()
    assert receipt["telemetry"]["max_missed_streak"] >= 5


def test_monitor_warmup_tolerates_missing_heartbeat(tmp_path: Path) -> None:
    clock = FakeClock()
    policy = ResourceTelemetryPolicy(min_available_bytes=48 * GIB)
    monitor = TelemetryMonitor(tmp_path, policy, sampler=lambda: 64 * GIB, clock=clock)
    monitor.start()
    for _ in range(20):
        clock.now += 1.0
        monitor.check_once()  # within 60 s warm-up: recorded, not missed
    assert monitor.stop_reason is None
    receipt = monitor.stop()
    assert receipt["telemetry"]["max_missed_streak"] == 0


def test_monitor_stops_below_available_memory_floor(tmp_path: Path) -> None:
    clock = FakeClock()
    policy = ResourceTelemetryPolicy(min_available_bytes=48 * GIB)
    monitor = TelemetryMonitor(tmp_path, policy, sampler=lambda: 40 * GIB, clock=clock)
    monitor.start()
    write_heartbeat(tmp_path, clock.now)
    clock.now += 1.0
    monitor.check_once()
    assert monitor.stop_reason == STOP_HOST_MEMORY_FLOOR


def test_monitor_sampler_failure_counts_as_missed(tmp_path: Path) -> None:
    clock = FakeClock()
    policy = ResourceTelemetryPolicy(min_available_bytes=1, warmup_seconds=0.0)
    monitor = TelemetryMonitor(tmp_path, policy, sampler=lambda: None, clock=clock)
    monitor.start()
    for _ in range(5):
        clock.now += 1.0
        write_heartbeat(tmp_path, clock.now)
        monitor.check_once()
        if monitor.stop_reason:
            break
    assert monitor.stop_reason == STOP_TELEMETRY_STALE


def test_stale_heartbeat_counts_as_missed(tmp_path: Path) -> None:
    clock = FakeClock()
    policy = ResourceTelemetryPolicy(min_available_bytes=1, warmup_seconds=0.0)
    monitor = TelemetryMonitor(tmp_path, policy, sampler=lambda: 64 * GIB, clock=clock)
    monitor.start()
    write_heartbeat(tmp_path, clock.now)  # fresh once, then frozen
    for _ in range(50):
        clock.now += 1.0
        monitor.check_once()
        if monitor.stop_reason:
            break
    assert monitor.stop_reason == STOP_TELEMETRY_STALE


def test_available_memory_sampler_reads_host() -> None:
    value = available_memory_bytes()
    if sys.platform in {"win32", "linux"}:
        assert isinstance(value, int) and value > 4 * GIB
    else:
        assert value is None or isinstance(value, int)


def test_resource_limit_breach_maps_to_oom_state() -> None:
    state, reason, metadata = worker_outcome_to_finish(
        None,
        "WORKER_RESOURCE_LIMIT_BREACH",
        attempt="000001",
        max_new_tokens=8192,
        started=0.0,
    )
    assert state == "OOM"
    assert reason == "WORKER_RESOURCE_LIMIT_BREACH"
    assert metadata["error"] == "WORKER_RESOURCE_LIMIT_BREACH"


def test_resource_stop_maps_to_fail_state() -> None:
    state, reason, _ = worker_outcome_to_finish(
        None,
        f"WORKER_RESOURCE_STOP_{STOP_TELEMETRY_STALE}",
        attempt="000001",
        max_new_tokens=8192,
        started=0.0,
    )
    assert state == "FAIL"
    assert reason == f"WORKER_RESOURCE_STOP_{STOP_TELEMETRY_STALE}"


# supervise_worker appends "--result <path>" to the command, so the worker
# sees: argv[1]=heartbeat_dir argv[2]=mode argv[3]="--result" argv[4]=result
WORKER_SCRIPT = """
import json, sys, time
from pathlib import Path
heartbeat_dir = Path(sys.argv[1])
mode = sys.argv[2]
result = Path(sys.argv[4])
if mode == "healthy":
    tick = 0
    end = time.monotonic() + 3
    while time.monotonic() < end:
        tick += 1
        payload = {"monotonic": time.monotonic(), "watchdog": {
            "samples": tick, "peak_allocated_bytes": 1024,
            "limit_bytes": 17179869184, "breach": None, "sink_errors": 0}}
        (heartbeat_dir / "worker-heartbeat.json").write_text(
            json.dumps(payload), encoding="utf-8")
        time.sleep(0.5)
    result.write_text(
        json.dumps({"kind": "selftest", "raw": "ok", "report": {}}), encoding="utf-8"
    )
    raise SystemExit(0)
if mode == "silent":
    time.sleep(30)
if mode == "breach":
    raise SystemExit(86)
"""


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows Job supervision")
@pytest.mark.parametrize(
    ("mode", "expected_failure"),
    [
        ("healthy", ""),
        ("silent", "WORKER_RESOURCE_STOP_TELEMETRY_STALE_UNKNOWN"),
        ("breach", "WORKER_RESOURCE_LIMIT_BREACH"),
    ],
)
def test_supervision_enforces_telemetry_policy(
    tmp_path: Path, mode: str, expected_failure: str
) -> None:
    script = tmp_path / "telemetry_worker.py"
    script.write_text(WORKER_SCRIPT, encoding="utf-8")
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    policy = ResourceTelemetryPolicy(
        min_available_bytes=1,
        interval_seconds=0.2,
        staleness_samples=5,
        warmup_seconds=1.0,
    )
    outcome, failure = supervise_worker(
        [sys.executable, str(script), str(attempt), mode],
        attempt_directory=attempt,
        timeout_seconds=20,
        resource_policy=policy,
    )
    assert failure == expected_failure
    supervision = json.loads((attempt / "supervision.json").read_text(encoding="utf-8"))
    assert supervision["telemetry"]["samples"] >= 1
    if mode == "healthy":
        assert supervision["telemetry"]["stop_reason"] is None
        assert outcome is not None and outcome["kind"] == "selftest"
    if mode == "silent":
        assert supervision["telemetry"]["stop_reason"] == STOP_TELEMETRY_STALE
        assert supervision["cleanup"]["active_processes"] == 0
