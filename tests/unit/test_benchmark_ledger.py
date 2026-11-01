import json

import pytest
from test_benchmark_manifest import dataset

from unlimited_ocr_2_rocm.benchmark.ledger import Run
from unlimited_ocr_2_rocm.benchmark.manifest import inventory


def make_run(tmp_path):
    gt = dataset(tmp_path)
    manifest = inventory(gt, tmp_path, "synthetic", "a" * 40, 2, 12)
    return Run.create(tmp_path / "runs", manifest, {"protocol": "frozen"}, {"python": "test"})


def complete(run, doc_id, state="PASS"):
    count = 10 if doc_id == "a" else 2
    raw = "<PAGE><BLOCK>text[0,0,1,1]<FORK>hello" * count
    attempt = run.begin(doc_id)
    run.finish(
        doc_id,
        attempt,
        raw,
        state,
        {"finish_reason": "stop", "output_tokens": 7, "max_tokens": 100, "memory": {"rss": 1}},
    )
    return attempt


def test_failed_and_missing_documents_stay_in_denominator_and_resume(tmp_path):
    run = make_run(tmp_path)
    complete(run, "z", "OOM")
    summary = run.summary()
    assert summary["expected_documents"] == 2
    assert summary["expected_pages"] == 12
    assert summary["counts"]["OOM"] == 1
    assert summary["counts"]["NOT_RUN"] == 1
    assert run.todo() == ["a", "z"]
    complete(run, "z")
    assert run.todo() == ["a"]
    assert len(run.attempts("z")) == 2
    assert run.summary()["status"] == "PARTIAL"


@pytest.mark.parametrize("state", ["FAIL", "TIMEOUT", "OOM", "TRUNCATED"])
def test_explicit_failures_preserve_raw_and_never_become_pass(tmp_path, state):
    run = make_run(tmp_path)
    complete(run, "a", state)
    assert run.summary()["counts"][state] == 1
    assert "a" in run.todo()
    assert (run.path / "documents/a/000001/raw.txt").read_bytes().startswith(b"<PAGE>")


def test_pending_attempt_is_interrupted_failure_and_retry_appends(tmp_path):
    run = make_run(tmp_path)
    first = run.begin("a")
    assert run.summary()["documents"][0]["reason"] == "INTERRUPTED"
    second = complete(run, "a")
    assert second != first
    assert (run.path / "documents/a/000001.pending/started.json").is_file()


def test_tampered_pass_cannot_be_skipped_on_resume(tmp_path):
    run = make_run(tmp_path)
    complete(run, "a")
    (run.path / "documents/a/000001/raw.txt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        run.todo()


def test_missing_completion_metadata_preserves_raw_as_explicit_failure(tmp_path):
    run = make_run(tmp_path)
    attempt = run.begin("a")
    run.finish("a", attempt, "partial raw", "PASS", {})
    assert run.summary()["documents"][0]["state"] == "FAIL"
    assert (run.path / "documents/a/000001/raw.txt").read_bytes() == b"partial raw"


def test_unbounded_heading_is_retained_without_expanding_markdown(tmp_path):
    run = make_run(tmp_path)
    attempt = run.begin("z")
    raw = "<PAGE><BLOCK>title<LEVEL17>[0,0,1,1]<FORK>x" * 2
    run.finish(
        "z",
        attempt,
        raw,
        "PASS",
        {
            "finish_reason": "stop",
            "output_tokens": 3,
            "max_tokens": 100,
            "memory": {},
        },
    )
    assert run.summary()["documents"][1]["state"] == "FAIL"


@pytest.mark.parametrize("kind", ["empty", "length", "pages", "malformed"])
def test_pass_requires_valid_completion_and_page_structure(tmp_path, kind):
    run = make_run(tmp_path)
    attempt = run.begin("z")
    raw = "<PAGE><BLOCK>text[0,0,1,1]<FORK>hello" * 2
    meta = {"finish_reason": "stop", "output_tokens": 2, "max_tokens": 3, "memory": {}}
    if kind == "empty":
        raw = ""
    elif kind == "length":
        meta["finish_reason"] = "length"
    elif kind == "pages":
        raw += "<PAGE><BLOCK>text[0,0,1,1]<FORK>extra"
    else:
        raw = "<PAGE>not a block" * 2
    run.finish("z", attempt, raw, "PASS", meta)
    row = run.summary()["documents"][1]
    assert row["state"] == ("TRUNCATED" if kind == "length" else "FAIL")
    assert run.todo() == ["a", "z"]


def test_run_identity_binds_manifest_protocol_and_environment(tmp_path):
    run = make_run(tmp_path)
    data = json.loads((run.path / "run.json").read_text())
    same = Run.create(tmp_path / "runs", data["manifest"], data["config"], data["environment"])
    assert same.path == run.path
    different = Run.create(
        tmp_path / "runs", data["manifest"], {"protocol": "changed"}, {"python": "test"}
    )
    assert different.path != run.path
    data["manifest"]["dataset"] = "other"
    other = Run.create(tmp_path / "runs", data["manifest"], data["config"], data["environment"])
    assert other.path.parent != run.path.parent


def test_export_keeps_failed_documents_as_empty_predictions(tmp_path):
    run = make_run(tmp_path)
    complete(run, "a")
    complete(run, "z", "TIMEOUT")
    target = tmp_path / "predictions"
    run.export(target)
    assert sorted(p.name for p in target.glob("*.md")) == ["a.md", "z.md"]
    assert (target / "z.md").read_bytes() == b""
    assert "hello" in (target / "a.md").read_text()
    assert (target / "accounting.json").is_file()
    with pytest.raises(FileExistsError):
        run.export(target)
