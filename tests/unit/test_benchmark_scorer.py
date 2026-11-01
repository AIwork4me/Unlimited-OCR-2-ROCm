import json

import pytest

from unlimited_ocr_2_rocm.benchmark.manifest import file_hash
from unlimited_ocr_2_rocm.benchmark.scorer import (
    aggregate,
    check_dependencies,
    check_sources,
    metric_values,
    sample_coverage,
)


def official_result():
    result = {}
    for task in ("text_block", "merged_text_block", "reading_order"):
        result[task] = {"all": {"Edit_dist": {"ALL_page_avg": 0.25, "edit_whole": 0.9}}}
    for task, metric in (
        ("table", "TEDS"),
        ("merged_table", "TEDS"),
        ("head", "HeadTEDS"),
        ("figure", "FigureF1"),
        ("display_formula", "CDM"),
    ):
        result[task] = {"all": {metric: {"all": 0.5}}}
    return result


def test_aggregation_uses_official_fields_and_inverts_only_edit_distances():
    values = metric_values(official_result())
    assert values["TextEdit"] == 75
    assert values["T-TextEdit"] == 75
    assert values["ReadOrderEdit"] == 75
    assert values["FormulaCDM"] == 50
    assert aggregate(values) == pytest.approx(59.375)


@pytest.mark.parametrize("bad", [None, "NaN", float("nan"), float("inf"), -0.1, 1.1])
def test_missing_or_invalid_metric_blocks_overall(bad):
    result = official_result()
    result["display_formula"]["all"]["CDM"]["all"] = bad
    values = metric_values(result)
    assert values["FormulaCDM"] is None
    assert aggregate(values) is None


def test_missing_dependency_is_not_scored_as_zero():
    result = official_result()
    del result["display_formula"]
    assert aggregate(metric_values(result)) is None


def test_scorer_coverage_keeps_missing_documents_and_per_metric_denominators():
    result = sample_coverage(["a.pdf", "b.pdf"], {"text_block": [{"img_id": "a.pdf"}]})
    assert result["expected_documents"] == 2
    assert result["tasks"]["text_block"]["documents_with_samples"] == 1
    assert result["tasks"]["text_block"]["documents_without_samples"] == ["b.pdf"]


def test_source_hash_gate_rejects_changed_and_extra_code_before_import(tmp_path):
    source = tmp_path / "scorer"
    source.mkdir()
    script = source / "original.py"
    script.write_text("raise RuntimeError('must not execute')", encoding="utf-8")
    lock = {"files": {"original.py": file_hash(script)}}
    check_sources(source, lock)
    script.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        check_sources(source, lock)
    lock["files"]["original.py"] = file_hash(script)
    (source / "injected.py").write_text("bad", encoding="utf-8")
    with pytest.raises(ValueError, match="unlocked"):
        check_sources(source, lock)


def test_dependency_gate_rejects_drift_and_gpu_environment():
    check_dependencies({"numpy": "1.26.4"}, ["numpy==1.26.4"])
    with pytest.raises(ValueError, match="dependency"):
        check_dependencies({"numpy": "2.0"}, ["numpy==1.26.4"])
    with pytest.raises(ValueError, match="torch-free"):
        check_dependencies({"numpy": "1.26.4", "torch": "2.12.0"}, ["numpy==1.26.4"])


@pytest.mark.parametrize(
    "relative", ["__pycache__/original.cpython-310.pyc", "original.pyd", "original.so"]
)
def test_source_gate_rejects_unverified_import_artifacts(tmp_path, relative):
    source = tmp_path / "source"
    source.mkdir()
    script = source / "original.py"
    script.write_text("VALUE = 'verified'", encoding="utf-8")
    payload = source / relative
    payload.parent.mkdir(parents=True, exist_ok=True)
    payload.write_bytes(b"unreviewed executable artifact")
    with pytest.raises(ValueError, match="unlocked"):
        check_sources(source, {"files": {"original.py": file_hash(script)}})


def test_committed_dataset_scopes_reconcile_full_denominators():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "eval/benchmarks/manifests"
    for name, docs, pages in (("MPDocBench", 420, 3135), ("MarathonDocBench", 100, 18272)):
        rows = [json.loads(line) for line in (root / (name + ".jsonl")).read_text().splitlines()]
        assert len(rows) == docs
        assert len({r["id"] for r in rows}) == docs
        assert sum(r["pages"] for r in rows) == pages
