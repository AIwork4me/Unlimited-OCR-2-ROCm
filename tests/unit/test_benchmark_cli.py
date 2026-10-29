import hashlib
import io
import json
import urllib.error

from test_benchmark_ledger import complete, make_run

from unlimited_ocr_2_rocm.benchmark.cli import fetch_scorer, main


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
