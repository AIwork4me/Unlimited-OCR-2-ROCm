"""Guard against treating a swallowed renderer exception as a valid CDM score."""

import json
import os
import signal
import subprocess
import sys

import pytest

from scripts.g05_scorer_closure import (
    classify_cdm,
    cleanup_process_group,
    pipeline_passed,
    prepare_pipeline,
    render_artifacts,
)


def test_exact_requires_real_renderer_artifacts():
    assert classify_cdm("exact", 1.0, True, False) == "PASS"
    assert classify_cdm("exact", 1.0, False, False) == "FAIL"
    assert classify_cdm("exact", 0.0, False, False) == "FAIL"


def test_mismatch_requires_rendering_and_a_lower_score():
    assert classify_cdm("mismatch", 0.4, True, False) == "PASS"
    assert classify_cdm("mismatch", 0.8, True, False) == "FAIL"
    assert classify_cdm("mismatch", 1.0, True, False) == "FAIL"
    assert classify_cdm("mismatch", 0.0, False, False) == "FAIL"


def test_failure_cases_remain_distinct_from_quality_scores():
    assert classify_cdm("error", 0.0, False, False, error_observed=True) == "EXPECTED_RENDER_ERROR"
    assert classify_cdm("error", 0.0, False, False, error_observed=False) == "FAIL"
    assert classify_cdm("timeout", None, False, True, True) == "EXPECTED_TIMEOUT"
    assert classify_cdm("timeout", None, False, True, False) == "FAIL"
    assert classify_cdm("exact", None, False, True) == "FAIL"


def test_pipeline_includes_formula_and_cross_page_relations(tmp_path):
    gt_path, predictions = prepare_pipeline(tmp_path)
    gt = json.loads(gt_path.read_text(encoding="utf-8"))[0]
    assert gt["layout_dets"][-1]["category_type"] == "equation_isolated"
    assert gt["layout_dets"][-1]["latex"] == "x+y=z"
    assert len(gt["extra"]["relation"]) == 3
    assert "$$x+y=z$$" in (predictions / "synthetic.md").read_text(encoding="utf-8")


def test_render_proof_rejects_empty_boxes_and_invalid_png(tmp_path):
    from PIL import Image

    (tmp_path / "gt/bbox").mkdir(parents=True)
    (tmp_path / "gt/vis").mkdir()
    bbox = tmp_path / "gt/bbox/exact.jsonl"
    image = tmp_path / "gt/vis/exact_base.png"
    bbox.write_text('{"bbox": [0, 0, 1, 1], "token": "x"}\n', encoding="utf-8")
    image.write_bytes(b"invalid PNG")
    assert not render_artifacts(tmp_path, "exact", "gt")["rendered"]
    Image.new("RGB", (2, 2), "black").save(image)
    assert render_artifacts(tmp_path, "exact", "gt")["rendered"]
    bbox.write_text('{"bbox": [], "token": "x"}\n', encoding="utf-8")
    assert not render_artifacts(tmp_path, "exact", "gt")["rendered"]


def test_pipeline_cannot_pass_with_an_absent_formula_population():
    from unlimited_ocr_2_rocm.benchmark.scorer import METRICS

    scores = {"metrics": dict.fromkeys(METRICS, 100.0), "overall": 100.0}
    coverage = {"tasks": {"display_formula": {"sample_count": 1}}}
    assert pipeline_passed(scores, coverage)
    coverage["tasks"]["display_formula"]["sample_count"] = 0
    assert not pipeline_passed(scores, coverage)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process-group control")
def test_cleanup_kills_descendant_after_leader_has_exited():
    command = (
        "import subprocess,sys; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        "print(p.pid,flush=True)"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", command], stdout=subprocess.PIPE, start_new_session=True
    )
    try:
        assert proc.stdout is not None
        assert int(proc.stdout.readline()) > 0
        proc.wait(timeout=10)
        assert cleanup_process_group(proc) == []
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
