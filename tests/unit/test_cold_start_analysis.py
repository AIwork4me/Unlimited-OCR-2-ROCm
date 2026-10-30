"""CPU-only checks for the new diagnostic's strict comparison semantics."""

import importlib.util
import sys
from pathlib import Path

import pytest

EXPERIMENT = Path(__file__).resolve().parents[2] / "experiments/cold_start"
sys.path.insert(0, str(EXPERIMENT))
spec = importlib.util.spec_from_file_location("cold_analysis_test", EXPERIMENT / "analyze.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)
report_spec = importlib.util.spec_from_file_location("cold_report_test", EXPERIMENT / "report.py")
report = importlib.util.module_from_spec(report_spec)
report_spec.loader.exec_module(report)
reliability_spec = importlib.util.spec_from_file_location(
    "cold_reliability_test", EXPERIMENT / "reliability.py"
)
reliability = importlib.util.module_from_spec(reliability_spec)
reliability_spec.loader.exec_module(reliability)


def native(text, tokens=None):
    return dict(text=text, token_ids=tokens or [1, 2], prompt_token_ids=[3])


def test_body_coordinate_like_text_is_not_masked():
    a = native("<PAGE><BLOCK>text[1,2,30,40]<FORK>array [1,2,3,4]")
    b = native("<PAGE><BLOCK>text[1,2,30,40]<FORK>array [1,2,3,5]")
    assert analysis.compare(a, b)["exact_nonbbox_bytes"] is False


def test_body_fork_does_not_turn_array_into_bbox():
    a = native("<PAGE><BLOCK>text[1,2,30,40]<FORK>array [1,2,3,4]<FORK>end")
    b = native("<PAGE><BLOCK>text[1,2,30,40]<FORK>array [1,2,3,5]<FORK>end")
    assert analysis.compare(a, b)["exact_nonbbox_bytes"] is False


@pytest.mark.parametrize("header", ["text[1,2,30,40,999]<FORK>x", "text[1,2,30,40]x"])
def test_lossy_or_missing_header_syntax_rejects(header):
    with pytest.raises(ValueError):
        analysis.boxes("<PAGE><BLOCK>" + header)


def test_prompt_mismatch_cannot_pass_exact_trial():
    a = native("<PAGE><BLOCK>text[1,2,30,40]<FORK>x")
    b = dict(a, prompt_token_ids=[4])
    assert analysis.compare(a, b)["acceptance"] == "EXACT_FAIL"


def test_signed_deltas_do_not_relax_exact_acceptance():
    a = native("<PAGE><BLOCK>text[1,2,30,40]<FORK>x", [1, 2])
    b = native("<PAGE><BLOCK>text[2,1,30,40]<FORK>x", [1, 5])
    result = analysis.compare(a, b)
    assert result["boxes"][0]["signed_deltas"] == [1, -1, 0, 0]
    assert result["acceptance"] == "EXACT_FAIL"
    assert result["first_differing_token"] == 1


def test_invalid_boxes_have_no_iou_and_missing_boxes_reject():
    assert analysis.iou((2, 0, 1, 4), (0, 0, 4, 4)) is None
    assert analysis.iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1
    with pytest.raises(ValueError):
        analysis.boxes("<PAGE><BLOCK>text<FORK>missing")


def test_publication_requires_success_completion_and_cleanup(tmp_path):
    import json

    worker = tmp_path / "worker"
    (worker / "request-0").mkdir(parents=True)
    (worker / "request-0/summary.json").write_text("{}")
    assert report.process_gate(tmp_path) is False

    (worker / "complete.json").write_text('{"requests": 1}')
    finish = tmp_path / "finished.json"
    finish.write_text(
        json.dumps(dict(exit_code=1, accounting_after_cleanup=dict(active_processes=0)))
    )
    assert report.process_gate(tmp_path) is False
    finish.write_text(
        json.dumps(dict(exit_code=0, accounting_after_cleanup=dict(active_processes=1)))
    )
    assert report.process_gate(tmp_path) is False
    finish.write_text(
        json.dumps(dict(exit_code=0, accounting_after_cleanup=dict(active_processes=0)))
    )
    assert report.process_gate(tmp_path) is True
    (worker / "failure.json").write_text("{}")
    assert report.process_gate(tmp_path) is False


def test_reliability_categories_do_not_relabel_process_failure_as_parity():
    assert reliability.failure_reason(True, True, False) == "PROCESS_COMPLETION_CLEANUP_FAIL"
    assert reliability.failure_reason(False, True, True) == "EXACT_GOLDEN_PARITY_FAIL"
    assert reliability.failure_reason(True, False, True) == "FUNCTIONAL_RECOVERY_FAIL"
    assert reliability.failure_reason(True, True, True) == ""


def test_external_termination_requires_live_clean_checkpoint_and_no_return():
    receipt = dict(
        exit_code=1,
        fault_checkpoint=dict(completed_decoder_forwards=3, synchronized=True, hip_zero=True),
        native_return_before_kill=False,
        accounting_after_cleanup=dict(active_processes=0),
    )
    assert reliability.fault_contract(receipt) is True
    assert reliability.fault_contract(dict(receipt, exit_code=0)) is False
    assert reliability.fault_contract(dict(receipt, native_return_before_kill=True)) is False
    assert reliability.fault_contract(dict(receipt, fault_checkpoint={})) is False
