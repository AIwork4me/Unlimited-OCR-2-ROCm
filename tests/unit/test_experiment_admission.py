"""CPU workers prove refusal means no process execution, not just an error."""

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from unlimited_ocr_2_rocm.benchmark.admission import Gate, Rejected
from unlimited_ocr_2_rocm.benchmark.manifest import digest, file_hash


class Worker:
    def __init__(self, active=0):
        self.active = active
        self.starts = 0
        self.child = None

    def accounting(self):
        return {"active_processes": self.active}

    def spawn(self, command, **kwargs):
        self.starts += 1
        self.child = subprocess.Popen(command, **kwargs)
        self.active = 1
        return self.child

    def terminate(self):
        if self.child:
            if self.child.poll() is None:
                self.child.kill()
            self.child.wait(timeout=5)
        self.active = 0


def setup(tmp_path):
    groups = {}
    for group in ("sources", "wheels", "fixtures", "policy", "artifacts"):
        p = tmp_path / group
        p.write_text(group, encoding="utf-8")
        groups[group] = {str(p): file_hash(p)}
    identity = dict(groups, target="windows/gfx1151")
    marker = tmp_path / "worker-started"
    script = tmp_path / "fake_worker.py"
    script.write_text("from pathlib import Path\nimport sys\nPath(sys.argv[1]).touch()\n")
    identity["sources"][str(script)] = file_hash(script)
    command = [sys.executable, str(script), str(marker)]
    identity["sources"][sys.executable] = file_hash(Path(sys.executable))
    policy = Path(next(iter(identity["policy"])))
    policy.write_text(
        json.dumps(
            {
                name: {
                    "argv": command,
                    "cwd": str(Path.cwd()),
                    "env": dict(os.environ),
                    "dependency": None if name == "correctness" else "correctness",
                }
                for name in ("correctness", "performance", "first", "second")
            }
        ),
        encoding="utf-8",
    )
    identity["policy"][str(policy)] = file_hash(policy)
    gate = Gate(tmp_path / "gate.sqlite")
    return gate, identity, command, marker


def qualify(gate, identity, command, tmp_path, *, status="PASS", complete=True):
    gate.prepare("correctness", identity, command, initial_qualification=True)
    job = Worker()
    gate.dispatch("correctness", None, identity, job, timeout=5)
    receipt = tmp_path / "qualification.json"
    receipt.write_text(
        json.dumps(
            {
                "identity": digest(identity),
                "execution": gate.execution("correctness"),
                "status": status,
                "complete": complete,
                "strict_correctness": status == "PASS",
                "artifacts": identity["artifacts"],
            }
        ),
        encoding="utf-8",
    )
    gate.evaluate("correctness", receipt, file_hash(receipt))
    return receipt


@pytest.mark.parametrize(
    "fault",
    [
        "FAIL",
        "PARTIAL",
        "missing",
        "incomplete",
        "sources",
        "wheels",
        "fixtures",
        "policy",
        "artifacts",
        "receipt",
        "target",
        "active",
        "missing_fixture",
    ],
)
def test_rejection_never_starts_worker(tmp_path, fault):
    gate, identity, command, marker = setup(tmp_path)
    gate.prepare("performance", identity, command)
    if fault != "missing":
        receipt = qualify(
            gate,
            identity,
            command,
            tmp_path,
            status=fault if fault in ("FAIL", "PARTIAL") else "PASS",
            complete=fault != "incomplete",
        )
        marker.unlink()
        if fault in ("sources", "wheels", "fixtures", "policy", "artifacts"):
            Path(next(iter(identity[fault]))).write_text("changed", encoding="utf-8")
        if fault == "receipt":
            receipt.write_text("{}", encoding="utf-8")
    if fault == "target":
        identity = dict(identity, target="linux/gfx1100")
    if fault == "missing_fixture":
        identity = dict(identity, fixtures={})
    job = Worker(active=int(fault == "active"))
    with pytest.raises(Rejected):
        gate.dispatch("performance", "correctness", identity, job, timeout=5)
    assert job.starts == 0
    assert not marker.exists()
    assert job.child is None


def test_pass_starts_exactly_one_worker_and_waits_for_evaluation(tmp_path):
    gate, identity, command, marker = setup(tmp_path)
    qualify(gate, identity, command, tmp_path)
    marker.unlink()
    gate.prepare("performance", identity, command)
    job = Worker()
    gate.dispatch("performance", "correctness", identity, job, timeout=5)
    assert marker.exists() and job.starts == 1
    assert job.active == 0 and job.child.poll() == 0
    assert gate.state("performance") == "EVALUATING"


def test_restart_does_not_promote_incomplete_evaluation(tmp_path):
    gate, identity, command, marker = setup(tmp_path)
    gate.prepare("correctness", identity, command, initial_qualification=True)
    gate.dispatch("correctness", None, identity, Worker(), timeout=5)
    marker.unlink()
    restarted = Gate(gate.path)
    restarted.prepare("performance", identity, command)
    job = Worker()
    with pytest.raises(Rejected):
        restarted.dispatch("performance", "correctness", identity, job, timeout=5)
    assert restarted.state("correctness") == "EVALUATING"
    assert job.starts == 0 and not marker.exists()


def test_second_check_rejects_changed_receipt_before_spawn(tmp_path):
    gate, identity, command, marker = setup(tmp_path)
    receipt = qualify(gate, identity, command, tmp_path)
    marker.unlink()
    gate.prepare("performance", identity, command)

    class Mutator(Worker):
        def accounting(self):
            receipt.write_text("{}", encoding="utf-8")
            return super().accounting()

    job = Mutator()
    with pytest.raises(Rejected):
        gate.dispatch("performance", "correctness", identity, job, timeout=5)
    assert job.starts == 0 and not marker.exists()


def test_simultaneous_launches_cannot_bypass_gate(tmp_path):
    import threading

    gate, identity, command, marker = setup(tmp_path)
    qualify(gate, identity, command, tmp_path)
    marker.unlink()
    for name in ("first", "second"):
        gate.prepare(name, identity, command)
    # Initialization may legitimately reject both zero-timeout contenders.
    # Pin this race to the actual admission/spawn boundary instead.
    gates = [Gate(gate.path), Gate(gate.path)]
    owner_at_spawn, contender_done = threading.Event(), threading.Event()

    class PausedOwner(Worker):
        def spawn(self, command, **kwargs):
            owner_at_spawn.set()
            assert contender_done.wait(5)
            return super().spawn(command, **kwargs)

    jobs = [PausedOwner(), Worker()]

    def attempt(i):
        if i == 1:
            assert owner_at_spawn.wait(5)
        try:
            gates[i].dispatch(("first", "second")[i], "correctness", identity, jobs[i], timeout=5)
            return True
        except Rejected:
            return False
        finally:
            if i == 1:
                contender_done.set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, range(2)))
    assert sum(results) == sum(j.starts for j in jobs) == 1
    assert all(j.active == 0 for j in jobs)
    assert marker.exists() and gate.state("first") == "EVALUATING"


def test_busy_constructor_refuses_without_launch(tmp_path):
    import sqlite3

    gate, identity, command, marker = setup(tmp_path)
    job = Worker()
    with sqlite3.connect(gate.path, isolation_level=None) as db:
        db.execute("BEGIN IMMEDIATE")
        with pytest.raises(Rejected, match="Busy or invalid admission ledger"):
            Gate(gate.path).dispatch("correctness", None, identity, job, timeout=5)
        db.rollback()
    assert job.starts == 0 and not marker.exists()


def test_failure_after_reservation_is_not_admitted_on_restart(tmp_path):
    gate, identity, command, marker = setup(tmp_path)
    gate.prepare("correctness", identity, command, initial_qualification=True)

    class Broken(Worker):
        def spawn(self, command, **kwargs):
            raise KeyboardInterrupt("interrupted before spawn")

    with pytest.raises(KeyboardInterrupt):
        gate.dispatch("correctness", None, identity, Broken(), timeout=5)
    assert not marker.exists()
    assert Gate(gate.path).state("correctness") != "PASS"


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_timeout_refuses_worker(tmp_path, timeout):
    gate, identity, command, marker = setup(tmp_path)
    gate.prepare("correctness", identity, command, initial_qualification=True)
    job = Worker()
    with pytest.raises(Rejected):
        gate.dispatch("correctness", None, identity, job, timeout=timeout)
    assert job.starts == 0 and not marker.exists()


def test_omitted_dependency_cannot_authorize_performance(tmp_path):
    gate, identity, command, marker = setup(tmp_path)
    gate.prepare("performance", identity, command)
    job = Worker()
    with pytest.raises(Rejected):
        gate.dispatch("performance", None, identity, job, timeout=5)
    assert job.starts == 0 and not marker.exists()


@pytest.mark.skipif(os.name != "nt", reason="native Windows Job ownership")
def test_native_job_rejection_owns_no_workers(tmp_path):
    from unlimited_ocr_2_rocm.benchmark.windows_job import NativeJob

    gate, identity, command, marker = setup(tmp_path)
    gate.prepare("performance", identity, command)
    job = NativeJob()
    try:
        with pytest.raises(Rejected):
            gate.dispatch("performance", "absent", identity, job, timeout=5)
        assert job.accounting()["total_processes"] == 0
        assert job.accounting()["active_processes"] == 0
        assert not marker.exists()
    finally:
        job.close()


@pytest.mark.parametrize("change", ["entry", "argv", "cwd", "env", "shell"])
def test_unqualified_execution_contract_never_starts_worker(tmp_path, change):
    gate, identity, command, marker = setup(tmp_path)
    qualify(gate, identity, command, tmp_path)
    marker.unlink()
    kwargs = {}
    if change == "entry":
        entry = tmp_path / "unqualified.py"
        entry.write_text("raise RuntimeError('unqualified entry')", encoding="utf-8")
        command = [sys.executable, str(entry)]
    elif change == "argv":
        command = [*command, "--backend=AUTO"]
    elif change == "cwd":
        kwargs["cwd"] = tmp_path
    elif change == "env":
        kwargs["env"] = dict(os.environ, UNQUALIFIED_BACKEND="AUTO")
    else:
        kwargs["shell"] = True
    job = Worker()
    with pytest.raises(Rejected):
        gate.prepare("performance", identity, command)
        gate.dispatch("performance", "correctness", identity, job, timeout=5, **kwargs)
    assert job.starts == 0 and not marker.exists()


def test_receipt_cannot_be_reused_for_a_different_execution(tmp_path):
    gate, identity, command, marker = setup(tmp_path)
    receipt = qualify(gate, identity, command, tmp_path)
    marker.unlink()
    gate.prepare("performance", identity, command)
    gate.dispatch("performance", "correctness", identity, Worker(), timeout=5)
    gate.evaluate("performance", receipt, file_hash(receipt))
    assert gate.state("performance") == "FAIL"


def test_nonzero_worker_is_durably_failed_and_drained(tmp_path):
    gate, identity, command, marker = setup(tmp_path)
    script = Path(command[1])
    script.write_text(script.read_text() + "\nraise SystemExit(1)\n", encoding="utf-8")
    identity["sources"][str(script)] = file_hash(script)
    gate.prepare("correctness", identity, command, initial_qualification=True)
    job = Worker()
    with pytest.raises(Rejected):
        gate.dispatch("correctness", None, identity, job, timeout=5)
    assert marker.exists() and job.starts == 1 and job.active == 0
    assert Gate(gate.path).state("correctness") == "FAIL"


def test_separate_controllers_cannot_bypass_device_ledger(tmp_path):
    gate, identity, command, marker = setup(tmp_path)
    qualify(gate, identity, command, tmp_path)
    marker.unlink()
    for name in ("first", "second"):
        gate.prepare(name, identity, command)
    manifest = tmp_path / "identity.json"
    manifest.write_text(json.dumps(identity), encoding="utf-8")
    code = """
import json, sys, time
from pathlib import Path
from test_experiment_admission import Worker
from unlimited_ocr_2_rocm.benchmark.admission import Gate, Rejected
folder = Path(sys.argv[3]).parent
def wait(name):
    deadline = time.monotonic() + 5
    while not (folder / name).exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(name)
        time.sleep(0.01)
class PausedOwner(Worker):
    def spawn(self, command, **kwargs):
        (folder / 'owner-at-spawn').touch()
        wait('contender-done')
        return super().spawn(command, **kwargs)
job = PausedOwner() if sys.argv[2] == 'first' else Worker()
identity = json.loads(Path(sys.argv[3]).read_text())
environment = json.loads(Path(next(iter(identity['policy']))).read_text())['first']['env']
gate = Gate(Path(sys.argv[1]))
(folder / ('ready-' + sys.argv[2])).touch()
wait('ready-second' if sys.argv[2] == 'first' else 'owner-at-spawn')
try:
    gate.dispatch(sys.argv[2], 'correctness',
        identity, job, timeout=5, env=environment)
except Rejected:
    pass
finally:
    if sys.argv[2] == 'second':
        (folder / 'contender-done').touch()
print(job.starts, job.active)
"""
    root = Path(__file__).resolve().parents[2]
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(
            [
                str(root / "src"),
                str(root / "tests/unit"),
            ]
        ),
    )
    import time

    children = []
    for name in ("first", "second"):
        children.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    code,
                    str(gate.path),
                    name,
                    str(manifest),
                ],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        )
        deadline = time.monotonic() + 5
        while not (tmp_path / ("ready-" + name)).exists():
            assert children[-1].poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.01)
    outputs = [child.communicate(timeout=15) for child in children]
    assert [child.returncode for child in children] == [0, 0], outputs
    assert sorted(out.strip() for out, err in outputs) == [b"0 0", b"1 0"]
    assert marker.exists()
    assert gate.state("first") == "EVALUATING"


@pytest.mark.skipif(os.name != "nt", reason="native Windows Job ownership")
@pytest.mark.parametrize("timeout", [False, True])
def test_native_job_drains_worker_and_descendant(tmp_path, timeout):
    from unlimited_ocr_2_rocm.benchmark.windows_job import NativeJob

    gate, identity, command, marker = setup(tmp_path)
    script = Path(command[1])
    script.write_text(
        script.read_text()
        + """
import subprocess, time
subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
if '--sleep' in sys.argv:
    time.sleep(30)
""",
        encoding="utf-8",
    )
    identity["sources"][str(script)] = file_hash(script)
    if timeout:
        command = [*command, "--sleep"]
    policy = Path(next(iter(identity["policy"])))
    data = json.loads(policy.read_text())
    data["correctness"]["argv"] = command
    policy.write_text(json.dumps(data), encoding="utf-8")
    identity["policy"][str(policy)] = file_hash(policy)
    gate.prepare("correctness", identity, command, initial_qualification=True)
    job = NativeJob()
    try:
        if timeout:
            with pytest.raises(subprocess.TimeoutExpired):
                gate.dispatch("correctness", None, identity, job, timeout=1)
            assert gate.state("correctness") == "FAIL"
        else:
            gate.dispatch("correctness", None, identity, job, timeout=5)
            assert gate.state("correctness") == "EVALUATING"
        assert marker.exists()
        assert job.accounting()["total_processes"] >= 2
        assert job.accounting()["active_processes"] == 0
    finally:
        job.terminate()
        job.close()


def test_missing_policy_dependency_is_not_an_initial_qualification(tmp_path):
    gate, identity, command, marker = setup(tmp_path)
    path = Path(next(iter(identity["policy"])))
    policy = json.loads(path.read_text())
    del policy["correctness"]["dependency"]
    path.write_text(json.dumps(policy), encoding="utf-8")
    identity["policy"][str(path)] = file_hash(path)
    job = Worker()
    with pytest.raises(Rejected):
        gate.prepare("correctness", identity, command, initial_qualification=True)
        gate.dispatch("correctness", None, identity, job, timeout=5)
    assert job.starts == 0 and not marker.exists()


def test_reserved_owner_waits_for_short_handoff_contention(tmp_path, monkeypatch):
    import sqlite3
    import threading
    import time
    from contextlib import contextmanager

    gate, identity, command, marker = setup(tmp_path)
    qualify(gate, identity, command, tmp_path)
    marker.unlink()
    gate.prepare("performance", identity, command)
    reserved, locked, second = threading.Event(), threading.Event(), threading.Event()
    original = gate._transaction
    count = 0

    @contextmanager
    def instrumented(*args, **kwargs):
        nonlocal count
        count += 1
        number = count
        if number == 2:
            second.set()
        with original(*args, **kwargs) as db:
            yield db
        if number == 1:
            reserved.set()
            assert locked.wait(3)

    def contender():
        assert reserved.wait(3)
        with sqlite3.connect(gate.path, isolation_level=None) as db:
            db.execute("BEGIN IMMEDIATE")
            locked.set()
            assert second.wait(3)
            time.sleep(0.2)
            db.commit()

    monkeypatch.setattr(gate, "_transaction", instrumented)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(contender)
        job = Worker()
        gate.dispatch("performance", "correctness", identity, job, timeout=5)
        future.result(timeout=5)
    assert job.starts == 1 and marker.exists() and job.active == 0


@pytest.fixture
def cli_output():
    from tempfile import TemporaryDirectory

    # The CLI permits evidence only below cwd; hosted pytest's tmp_path is
    # normally outside the checkout. Keep this output within the owned root.
    parent = Path.cwd() / ".b" / "admission-cli-tests"
    parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=parent) as directory:
        yield Path(directory) / "output"


@pytest.mark.skipif(os.name != "nt", reason="native prospective Windows dispatcher")
@pytest.mark.parametrize("inherited_flags", [None, "0", "1"])
def test_prospective_cli_refuses_before_worker_execution(
    tmp_path, monkeypatch, inherited_flags, cli_output
):
    # Freeze the CLI's normalized worker environment, independently of the
    # controller's inherited flags. Rejection must reach the missing predecessor.
    monkeypatch.setenv("PYTHONUTF8", "1")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    gate, identity, command, marker = setup(tmp_path)
    gate.prepare("performance", identity, command)
    manifest = tmp_path / "identity.json"
    manifest.write_text(json.dumps(identity), encoding="utf-8")
    root = Path(__file__).resolve().parents[2]
    cli_env = dict(os.environ)
    for name in ("PYTHONUTF8", "PYTHONDONTWRITEBYTECODE"):
        if inherited_flags is None:
            cli_env.pop(name)
        else:
            cli_env[name] = inherited_flags
    result = subprocess.run(
        [
            sys.executable,
            str(root / "experiments/admission/run.py"),
            "--ledger",
            str(gate.path),
            "--stage",
            "performance",
            "--qualification",
            "correctness",
            "--identity",
            str(manifest),
            "--output",
            str(cli_output),
            "--timeout",
            "5",
            "--execute-gpu",
        ],
        capture_output=True,
        timeout=10,
        env=cli_env,
    )
    assert result.returncode != 0
    assert (
        b"Missing qualification/stage" in (cli_output / "stderr.log").read_bytes()
        or b"Missing qualification/stage" in result.stderr
    )
    assert not marker.exists()
    assert gate.state("performance") == "PREPARED"
