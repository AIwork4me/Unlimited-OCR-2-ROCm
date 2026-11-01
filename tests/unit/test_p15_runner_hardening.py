"""P1.5 runner hardening: realistic CPU failure injection through real subprocesses.

Every scenario drives the true supervision path: the parent supervises an
actual worker process and records the outcome in the append-only ledger.
"""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "scripts/run_mpdoc_sample.py"
REPO_ROOT = Path(__file__).parents[2]


def dataset(tmp_path: Path) -> Path:
    docs = []
    for name, count in (("z", 2), ("a", 2)):
        pages = []
        for index in range(1, count + 1):
            path = f"images/{name}/page_{index}.png"
            target = tmp_path / path
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = f"{name}:{index}".encode()
            target.write_bytes(payload)
            pages.append(path)
        docs.append({"page_info": {"image_path": f"{name}.pdf", "images_list": pages}})
    gt = tmp_path / "gt.json"
    gt.write_text(json.dumps(docs), encoding="utf-8")
    return gt


def make_run(tmp_path: Path):
    from unlimited_ocr_2_rocm.benchmark.ledger import Run
    from unlimited_ocr_2_rocm.benchmark.manifest import inventory

    gt = dataset(tmp_path)
    manifest = inventory(gt, tmp_path, "synthetic", "a" * 40, 2, 4)
    # The worker re-checks every page against a frozen index before inference;
    # tests provide an equivalent index over the synthetic bytes.
    pages = [
        {
            "path": page["path"],
            "sha256": page["sha256"],
            "bytes": 4,
        }
        for doc in manifest["documents"]
        for page in doc["pages"]
    ]
    index_dir = tmp_path / "eval/benchmarks/g05/datasets"
    index_dir.mkdir(parents=True)
    (index_dir / "MPDocBench-page-index.json").write_text(
        json.dumps({"pages": pages}), encoding="utf-8"
    )
    run = Run.create(
        tmp_path / "runs",
        manifest,
        {"protocol": "frozen", "timeout_seconds_per_document": 30},
        {"python": "test"},
    )
    return run, gt


def worker_command(mode: str, repo_root: Path) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT),
        "worker",
        "--repo-root",
        str(repo_root),
        *(["--self-test-failure", mode] if mode else []),
    ]


def execute(run, selection, mode, timeout=30.0):
    from unlimited_ocr_2_rocm.benchmark import runner as benchmark_runner

    return benchmark_runner.execute_selection(
        run,
        selection,
        worker_command=worker_command(mode, run.path.parents[2]),
        data_root=run.path.parents[2],
        max_new_tokens=100,
        timeout_seconds=timeout,
    )


SELECTION = {"a": {"reason": "synthetic"}, "z": {"reason": "synthetic"}}


def state_of(summary, doc_id):
    row = next(row for row in summary["documents"] if row["doc_id"] == doc_id)
    return row


def test_good_document_passes_and_resume_skips(tmp_path):
    run, _ = make_run(tmp_path)
    result = execute(run, SELECTION, "good")
    assert result["counts"] == {
        "PASS": 2,
        "FAIL": 0,
        "TIMEOUT": 0,
        "OOM": 0,
        "TRUNCATED": 0,
        "NOT_RUN": 0,
    }
    again = execute(run, SELECTION, "good")
    assert again["executed"] == []
    assert len(run.attempts("a")) == 1


def test_native_length_becomes_truncated_with_preserved_raw(tmp_path):
    run, _ = make_run(tmp_path)
    execute(run, SELECTION, "length")
    row = state_of(run.summary(), "a")
    assert row["state"] == "TRUNCATED"
    assert row["reason"] == "TOKEN_LIMIT_NATIVE_LENGTH"
    raw = (run.path / "documents/a/000001/raw.txt").read_text(encoding="utf-8")
    assert raw.endswith("truncat")
    strict = tmp_path / "strict-export"
    run.export(strict, include=set(SELECTION))
    assert (strict / "a.md").read_bytes() == b""
    assert json.loads((strict / "accounting.json").read_text("utf-8"))["export"]["mode"] == "strict"
    diagnostic = tmp_path / "diagnostic-export"
    run.export(diagnostic, include=set(SELECTION), include_truncated=True)
    assert (diagnostic / "a.md").read_text(encoding="utf-8").strip()
    accounting = json.loads((diagnostic / "accounting.json").read_text("utf-8"))
    assert accounting["export"]["mode"] == "truncation_content_diagnostic"
    assert accounting["export"]["truncated_documents_as_predictions"] == ["a", "z"]


def test_worker_oom_is_classified_oom(tmp_path):
    run, _ = make_run(tmp_path)
    execute(run, {"a": SELECTION["a"]}, "oom")
    row = state_of(run.summary(), "a")
    assert row["state"] == "OOM"
    assert row["reason"] == "WORKER_OUT_OF_MEMORY"


def test_hanging_worker_times_out_and_is_killed(tmp_path):
    run, _ = make_run(tmp_path)
    started = time.time()
    result = execute(run, {"a": SELECTION["a"]}, "hang", timeout=3.0)
    assert time.time() - started < 30
    row = state_of(run.summary(), "a")
    assert row["state"] == "TIMEOUT"
    assert row["reason"] == "WORKER_TIMEOUT"
    assert (run.path / "documents/a/000001/worker.log").is_file()
    assert result["counts"]["TIMEOUT"] == 1


def test_killed_worker_crash_is_fail_then_retry_passes(tmp_path):
    run, _ = make_run(tmp_path)
    execute(run, {"a": SELECTION["a"]}, "crash")
    row = state_of(run.summary(), "a")
    assert row["state"] == "FAIL"
    assert row["reason"].startswith("WORKER_CRASH_EXIT")
    execute(run, {"a": SELECTION["a"]}, "good")
    row = state_of(run.summary(), "a")
    assert row["state"] == "PASS"
    assert len(run.attempts("a")) == 2


def test_stop_without_page_structure_is_refused(tmp_path):
    run, _ = make_run(tmp_path)
    execute(run, {"a": SELECTION["a"]}, "malformed")
    row = state_of(run.summary(), "a")
    assert row["state"] == "FAIL"
    assert row["reason"] == "EMPTY_MALFORMED_OR_PAGE_MISMATCH"


def test_abandoned_attempt_is_interrupted_failure_then_recovered(tmp_path):
    run, _ = make_run(tmp_path)
    run.begin("a")
    row = state_of(run.summary(), "a")
    assert row["state"] == "FAIL"
    assert row["reason"] == "INTERRUPTED"
    execute(run, {"a": SELECTION["a"]}, "good")
    assert state_of(run.summary(), "a")["state"] == "PASS"
    assert len(run.attempts("a")) == 2


def test_unselected_documents_are_not_run(tmp_path):
    run, _ = make_run(tmp_path)
    execute(run, {"a": SELECTION["a"]}, "good")
    counts = run.summary()["counts"]
    assert counts["PASS"] == 1
    assert counts["NOT_RUN"] == 1


def test_selection_outside_manifest_is_rejected(tmp_path):
    from unlimited_ocr_2_rocm.benchmark import runner as benchmark_runner

    run, _ = make_run(tmp_path)
    try:
        benchmark_runner.execute_selection(
            run,
            {"ghost": {"reason": "x"}},
            worker_command=worker_command("good", tmp_path),
            data_root=tmp_path,
            max_new_tokens=10,
            timeout_seconds=5,
        )
    except ValueError as error:
        assert "outside frozen manifest" in str(error)
    else:
        raise AssertionError("selection scope must be enforced")


def test_gt_slice_is_limited_to_selection_with_original_bytes(tmp_path):
    from unlimited_ocr_2_rocm.benchmark import runner as benchmark_runner

    run, gt = make_run(tmp_path)
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(SELECTION), encoding="utf-8")
    out = tmp_path / "slice.json"
    evidence = benchmark_runner.write_gt_slice(run, gt, selection_path, SELECTION, out)
    rows = json.loads(out.read_text(encoding="utf-8"))
    assert sorted(row["page_info"]["image_path"] for row in rows) == ["a.pdf", "z.pdf"]
    assert evidence["selection_sha256"] == hashlib.sha256(selection_path.read_bytes()).hexdigest()


def test_receipt_has_no_private_paths_or_host_ids(tmp_path):
    from unlimited_ocr_2_rocm.benchmark import runner as benchmark_runner

    run, _ = make_run(tmp_path)
    execute(run, SELECTION, "good")
    receipt = benchmark_runner.sanitized_receipt(
        run, selection=SELECTION, selection_sha256="0" * 64
    )
    text = json.dumps(receipt)
    assert "/home/" not in text and "node" not in text and "executable" not in text
    assert receipt["counts"]["PASS"] == 2
    assert receipt["label"] == "REAL_DATA_INTEGRATION_ONLY"


def test_sanitize_tree_refuses_local_paths_and_drops_ids():
    from unlimited_ocr_2_rocm.benchmark import runner as benchmark_runner

    cleaned = benchmark_runner.sanitize_tree(
        {
            "executable": "/x/python",
            "nested": {"machine_id_sha256": "abc", "keep": 1},
            "gpu": {"uuid": "u", "name": "AMD Radeon 8060S"},
        }
    )
    assert cleaned == {"nested": {"keep": 1}, "gpu": {"name": "AMD Radeon 8060S"}}
    try:
        benchmark_runner.sanitize_tree({"path": "/home/amd/secret"})
    except ValueError as error:
        assert "Private local path" in str(error)
    else:
        raise AssertionError("local paths must be refused")


def _process_alive(pid: int) -> bool:
    """Portable existence probe used only by the termination tests."""
    if sys.platform == "win32":
        # Returncode 0 means the process existed (the probe force-terminated
        # it); 128 means not found. The same user owns both processes.
        probe = subprocess.run(
            ["taskkill", "/PID", str(pid), "/F"], check=False, capture_output=True
        )
        return probe.returncode == 0
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def test_termination_kills_worker_descendants_not_only_the_worker(tmp_path):
    """A supervision timeout must not leave orphaned descendants behind."""
    from unlimited_ocr_2_rocm.benchmark import runner as benchmark_runner

    log = tmp_path / "tree.log"
    script = (
        "import subprocess, sys, time\n"
        "grandchild = subprocess.Popen("
        "[sys.executable, '-c', 'import time; time.sleep(300)'])\n"
        "print(grandchild.pid, flush=True)\n"
        "time.sleep(300)\n"
    )
    with log.open("xb") as sink:
        worker = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=sink,
            stderr=subprocess.STDOUT,
            start_new_session=os.name != "nt",
        )
        try:
            grandchild_pid = None
            for _ in range(100):  # up to ~10 s for the descendant announcement
                if log.stat().st_size:
                    announced = log.read_text(encoding="utf-8").splitlines()[0]
                    if announced.strip().isdigit():
                        grandchild_pid = int(announced)
                        break
                time.sleep(0.1)
            assert grandchild_pid is not None, "worker never announced its descendant"
            assert benchmark_runner.terminate_process_tree(worker) == "TERMINATED"
            deadline = time.time() + 30
            while time.time() < deadline and _process_alive(grandchild_pid):
                time.sleep(0.2)
        finally:
            if worker.poll() is None:
                benchmark_runner.terminate_process_tree(worker)
    assert not _process_alive(grandchild_pid), "descendant outlived supervised termination"


def test_termination_tolerates_worker_that_exited_before_the_kill():
    """The kill races a worker that already exited; reaping must still confirm."""
    from unlimited_ocr_2_rocm.benchmark import runner as benchmark_runner

    worker = subprocess.Popen(
        [sys.executable, "-c", "pass"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=os.name != "nt",
    )
    worker.wait()  # exited but not yet reaped: the signal targets a dead group
    assert benchmark_runner.terminate_process_tree(worker) == "TERMINATED"
    assert worker.returncode == 0
