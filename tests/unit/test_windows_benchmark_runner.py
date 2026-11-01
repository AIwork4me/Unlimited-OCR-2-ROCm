"""Real torch-free Windows workers and non-destructive process ownership checks."""

import ctypes
import json
import subprocess
import sys
import time

import pytest

from unlimited_ocr_2_rocm.benchmark import runner

WINDOWS = pytest.mark.skipif(sys.platform != "win32", reason="native Windows Job Objects")


def process_alive(pid):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        assert ctypes.get_last_error() == 87, "failed to inspect process"
        return False
    try:
        code = ctypes.c_ulong()
        assert kernel.GetExitCodeProcess(handle, ctypes.byref(code))
        return code.value == 259
    finally:
        kernel.CloseHandle(handle)


def cleanup_pid(pid):
    """Only called after assertions; never used to establish successful cleanup."""
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(1, False, pid)
    if handle:
        try:
            kernel.TerminateProcess(handle, 1)
        finally:
            kernel.CloseHandle(handle)


WORKER = """
import json, os, pathlib, subprocess, sys, time
mode, pidfile = sys.argv[1:3]
result = pathlib.Path(sys.argv[sys.argv.index('--result') + 1])
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'],
    creationflags=0x8 | 0x200, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
pathlib.Path(pidfile).write_text(json.dumps([os.getpid(), child.pid]))
print('worker evidence retained', flush=True)
if mode == 'hang': time.sleep(300)
if mode == 'crash': os._exit(17)
if mode == 'malformed': result.write_text('{broken')
elif mode == 'list': result.write_text('[]')
elif mode == 'unknown': result.write_text(json.dumps({'kind':'surprise'}))
elif mode == 'kind-list': result.write_text(json.dumps({'kind':[]}))
elif mode == 'kind-dict': result.write_text(json.dumps({'kind':{}}))
else: result.write_text(json.dumps({'kind':'ok', 'report': {'status':'PASS',
    'finish_reason':'stop','output_token_count':1,'max_new_tokens':100}}))
if mode == 'nonzero': sys.exit(7)
"""


def worker(tmp_path, mode):
    pidfile = tmp_path / "pids.json"
    return [sys.executable, "-c", WORKER, mode, str(pidfile)], pidfile


def assert_dead(pidfile):
    pids = json.loads(pidfile.read_text())
    try:
        deadline = time.monotonic() + 5
        while any(process_alive(pid) for pid in pids) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not any(process_alive(pid) for pid in pids), "owned descendant survived"
    finally:
        for pid in pids:
            cleanup_pid(pid)


@WINDOWS
@pytest.mark.parametrize("mode", ["normal", "crash", "hang"])
def test_supervision_drains_detached_children_after_every_worker_exit(tmp_path, mode):
    command, pidfile = worker(tmp_path, mode)
    outcome, failure = runner.supervise_worker(
        command, attempt_directory=tmp_path, timeout_seconds=1 if mode == "hang" else 20
    )
    assert_dead(pidfile)
    accounting = json.loads((tmp_path / "supervision.json").read_text())
    assert accounting["cleanup"]["active_processes"] == 0
    assert accounting["cleanup"]["status"] == "TERMINATED"
    assert accounting["cleanup"]["total_processes"] >= 2
    assert (tmp_path / "worker.log").read_text().strip() == "worker evidence retained"
    if mode == "normal":
        assert not failure and outcome["kind"] == "ok"
        assert (tmp_path / "worker-outcome.json").is_file()
    elif mode == "hang":
        assert outcome is None and failure == "WORKER_TIMEOUT"
        assert accounting["timed_out"] is True
        assert isinstance(accounting["exit_code"], int)
    else:
        assert outcome is None and failure == "WORKER_CRASH_EXIT_17"


@WINDOWS
def test_keyboard_interrupt_drains_job_and_retains_accounting(tmp_path, monkeypatch):
    command, pidfile = worker(tmp_path, "hang")
    original_wait = subprocess.Popen.wait
    interrupted = False

    def interrupt_once(process, timeout=None):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            deadline = time.monotonic() + 10
            while not pidfile.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            raise KeyboardInterrupt
        return original_wait(process, timeout=timeout)

    monkeypatch.setattr(subprocess.Popen, "wait", interrupt_once)
    with pytest.raises(KeyboardInterrupt):
        runner.supervise_worker(command, attempt_directory=tmp_path, timeout_seconds=20)
    assert_dead(pidfile)
    accounting = json.loads((tmp_path / "supervision.json").read_text())
    assert accounting["cleanup"]["active_processes"] == 0
    assert accounting["exception"] == "KeyboardInterrupt"
    assert isinstance(accounting["exit_code"], int)
    assert accounting["timed_out"] is False


@WINDOWS
def test_supervisor_death_closes_job_and_kills_workers(tmp_path):
    command, pidfile = worker(tmp_path, "hang")
    parent_code = """
import os, pathlib, sys, threading, time
from unlimited_ocr_2_rocm.benchmark.runner import supervise_worker
def exit_after_spawn():
    path = pathlib.Path(sys.argv[2])
    while True:
        try:
            if len(__import__('json').loads(path.read_text())) == 2: break
        except (OSError, ValueError): pass
        time.sleep(0.01)
    os._exit(23)
threading.Thread(target=exit_after_spawn, daemon=True).start()
supervise_worker(__import__('json').loads(sys.argv[1]),
    attempt_directory=pathlib.Path(sys.argv[3]), timeout_seconds=30)
"""
    parent = subprocess.Popen(
        [sys.executable, "-c", parent_code, json.dumps(command), str(pidfile), str(tmp_path)]
    )
    try:
        assert parent.wait(timeout=20) == 23
        assert_dead(pidfile)
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=10)


@WINDOWS
@pytest.mark.parametrize("mode", ["malformed", "list", "unknown", "nonzero"])
def test_invalid_outcome_or_exit_never_succeeds_and_preserves_bytes(tmp_path, mode):
    command, pidfile = worker(tmp_path, mode)
    outcome, failure = runner.supervise_worker(
        command, attempt_directory=tmp_path, timeout_seconds=20
    )
    assert_dead(pidfile)
    assert failure
    assert (tmp_path / "worker-outcome.json").is_file()
    state, _, _ = runner.worker_outcome_to_finish(
        outcome, failure, attempt="000001", max_new_tokens=100, started=time.time()
    )
    assert state == "FAIL"


@WINDOWS
@pytest.mark.parametrize("mode", ["kind-list", "kind-dict"])
def test_nonstring_outcome_kind_fails_with_drained_job(tmp_path, mode):
    command, pidfile = worker(tmp_path, mode)
    try:
        outcome, failure = runner.supervise_worker(
            command, attempt_directory=tmp_path, timeout_seconds=20
        )
    finally:
        assert_dead(pidfile)
    assert outcome is None and failure == "WORKER_INVALID_OUTCOME"
    assert (tmp_path / "worker-outcome.json").is_file()
    accounting = json.loads((tmp_path / "supervision.json").read_text())
    assert accounting["cleanup"]["active_processes"] == 0


@WINDOWS
def test_interrupt_at_created_child_handoff_kills_unassigned_child(tmp_path, monkeypatch):
    from unlimited_ocr_2_rocm.benchmark import windows_job

    created = []
    original_popen = subprocess.Popen

    def create_child(*args, **kwargs):
        child = original_popen(*args, **kwargs)
        created.append(child)
        return child

    def interrupt_at_handoff(frame, event, arg):
        if (
            frame.f_code is windows_job.NativeJob.spawn.__code__
            and event == "line"
            and frame.f_locals.get("child") is not None
        ):
            raise KeyboardInterrupt
        return interrupt_at_handoff

    monkeypatch.setattr(windows_job.subprocess, "Popen", create_child)
    job = object.__new__(windows_job.NativeJob)
    original_trace = sys.gettrace()
    marker = tmp_path / "must-not-execute.txt"
    try:
        sys.settrace(interrupt_at_handoff)
        with pytest.raises(KeyboardInterrupt):
            job.spawn(
                [
                    sys.executable,
                    "-c",
                    "from pathlib import Path; import sys; Path(sys.argv[1]).touch()",
                    str(marker),
                ]
            )
    finally:
        sys.settrace(original_trace)
    try:
        assert len(created) == 1
        assert created[0].returncode is not None, "unassigned suspended child was not reaped"
        assert not process_alive(created[0].pid)
        assert not marker.exists()
    finally:
        for child in created:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=10)


def test_nonzero_exit_with_good_report_cannot_pass():
    state, _, _ = runner.worker_outcome_to_finish(
        {"kind": "ok", "report": {"status": "PASS", "finish_reason": "stop"}},
        "WORKER_EXIT_7",
        attempt="000001",
        max_new_tokens=100,
        started=time.time(),
    )
    assert state == "FAIL"


@pytest.mark.parametrize("value", ["c:/users/person/input", "C:\\uSeRs\\person\\input"])
def test_private_path_refusal_does_not_echo_path(value):
    with pytest.raises(ValueError) as caught:
        runner.sanitize_tree({"error": value})
    assert value not in str(caught.value)


def test_environment_retains_windows_triton_identity():
    result = runner.sanitized_environment(
        {"native_dependencies": {"triton-windows": "3.7.1.post27"}}
    )
    assert result["packages"]["triton-windows"] == "3.7.1.post27"


@WINDOWS
def test_cleanup_failure_raises_instead_of_allowing_next_document(tmp_path, monkeypatch):
    from unlimited_ocr_2_rocm.benchmark.windows_job import NativeJob

    command, pidfile = worker(tmp_path, "normal")
    terminate = NativeJob.terminate

    def fail_after_drain(job, **kwargs):
        terminate(job, **kwargs)
        raise OSError("injected accounting failure")

    monkeypatch.setattr(NativeJob, "terminate", fail_after_drain)
    with pytest.raises(runner.WorkerCleanupError, match="campaign stopped"):
        runner.supervise_worker(command, attempt_directory=tmp_path, timeout_seconds=20)
    assert_dead(pidfile)
    accounting = json.loads((tmp_path / "supervision.json").read_text())
    assert accounting["cleanup"]["status"] == "UNVERIFIED"
    assert accounting["cleanup"]["error"] == "OSError"
    assert "injected accounting failure" in accounting["cleanup"]["detail"]


@WINDOWS
def test_job_initialization_failure_retains_evidence_without_starting_worker(tmp_path, monkeypatch):
    from unlimited_ocr_2_rocm.benchmark.windows_job import NativeJob

    def refuse_job(job):
        raise OSError("injected Job initialization failure")

    monkeypatch.setattr(NativeJob, "__init__", refuse_job)
    command, pidfile = worker(tmp_path, "normal")
    with pytest.raises(OSError, match="initialization failure"):
        runner.supervise_worker(command, attempt_directory=tmp_path, timeout_seconds=20)
    assert not pidfile.exists()
    accounting = json.loads((tmp_path / "supervision.json").read_text())
    assert accounting["exception"] == "OSError"
    assert accounting["cleanup"]["status"] == "NOT_STARTED"
