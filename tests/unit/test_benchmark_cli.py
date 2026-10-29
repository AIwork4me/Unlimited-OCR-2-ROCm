import hashlib
import io
import json
import os
import subprocess
import sys
import time
import urllib.error

import pytest
from test_benchmark_ledger import complete, make_run

from unlimited_ocr_2_rocm.benchmark.cli import fetch_scorer, main, run_process


def test_cli_reports_all_documents_and_artifact_hashes(tmp_path):
    run = make_run(tmp_path)
    complete(run, "a")
    output = tmp_path / "report"
    assert main(["report", "--run", str(run.path), "--output", str(output)]) == 0
    report = json.loads((output / "summary.json").read_text())
    assert report["counts"]["NOT_RUN"] == 1
    assert report["expected_documents"] == 2
    assert len((output / "documents.csv").read_text().splitlines()) == 3
    assert (output / "SHA256SUMS.json").is_file()


def test_cli_scorer_missing_source_preserves_blocked_report(tmp_path):
    output = tmp_path / "scorer"
    lock = tmp_path / "lock.json"
    lock.write_text('{"qwen":{"files":{"absent.py":"abc"}}}', encoding="utf-8")
    assert (
        main(["score", "--source", str(tmp_path), "--lock", str(lock), "--output", str(output)])
        == 2
    )
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "BLOCKED"
    assert report["overall"] is None


def test_source_download_retries_transport_and_verifies_bytes(tmp_path, monkeypatch):
    data = b"# inert original fixture\n"
    count = 0

    def response(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 1:
            raise urllib.error.URLError("transient fixture transport failure")
        return io.BytesIO(data)

    monkeypatch.setattr("urllib.request.urlopen", response)
    source = tmp_path / "source"
    lock = {
        "qwen": {"revision": "a" * 40, "files": {"fixture.py": hashlib.sha256(data).hexdigest()}}
    }
    fetch_scorer(source, lock, "qwen")
    assert (source / "fixture.py").read_bytes() == data


def test_timeout_preserves_logs_and_stops_descendant_process(tmp_path):
    marker = tmp_path / "leaked-child"
    child = tmp_path / "child.py"
    child.write_text(
        "import time,pathlib\ntime.sleep(2)\npathlib.Path("
        + repr(str(marker))
        + ").write_text('leaked')\ntime.sleep(20)\n"
    )
    parent = (
        "import subprocess,sys,time; subprocess.Popen([sys.executable, sys.argv[1]]); "
        "print('started',flush=True); time.sleep(20)"
    )
    stdout_path = tmp_path / "stdout"
    with stdout_path.open("wb") as stdout, (tmp_path / "stderr").open("wb") as stderr:
        with pytest.raises(subprocess.TimeoutExpired):
            run_process(
                [sys.executable, "-c", parent, str(child)], stdout, stderr, 1, dict(os.environ)
            )
    time.sleep(2)
    assert b"started" in stdout_path.read_bytes()
    assert not marker.exists()
