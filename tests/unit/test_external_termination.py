import importlib.util
from pathlib import Path

import pytest


def diagnostic():
    path = Path(__file__).parents[2] / "experiments/windows_reliability/physical.py"
    assert path.exists(), "External-termination diagnostic not implemented"
    spec = importlib.util.spec_from_file_location("external_termination", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_completed_or_unobserved_fault_is_never_qualified():
    module = diagnostic()
    for event, exit_code, active in (
        (None, 1, 0),
        ({"completed_decoder_forwards": 3, "hip_zero": True}, 0, 0),
        ({"completed_decoder_forwards": 2, "hip_zero": True}, 1, 0),
        ({"completed_decoder_forwards": 3, "hip_zero": False}, 1, 0),
        ({"completed_decoder_forwards": 3, "hip_zero": True}, 1, 1),
    ):
        assert module.qualify_fault(event, exit_code, active) is False


@pytest.mark.parametrize("code", [1, -1, 137])
def test_observed_live_fault_and_empty_job_can_be_qualified(code):
    assert (
        diagnostic().qualify_fault({"completed_decoder_forwards": 3, "hip_zero": True}, code, 0)
        is True
    )


def test_late_native_return_is_detected_before_qualifying_a_fault(tmp_path):
    module = diagnostic()
    assert hasattr(module, "has_native_return")
    assert module.has_native_return(tmp_path) is False
    path = tmp_path / "request-0/diagnostics/native-001.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}", encoding="utf-8")
    assert module.has_native_return(tmp_path) is True


def test_ledger_downgrade_or_missing_failure_history_cannot_be_pass():
    import copy

    module = diagnostic()
    assert hasattr(module, "ledger_pass")
    good = dict(
        status="COMPLETE",
        expected_documents=2,
        expected_pages=4,
        counts={"PASS": 2},
        documents=[
            dict(doc_id=doc, attempts=[dict(state=state), dict(state="PASS")])
            for doc, state in (("external_kill", "FAIL"), ("timeout", "TIMEOUT"))
        ],
    )
    assert module.ledger_pass(good) is True
    downgraded = copy.deepcopy(good)
    downgraded["documents"][0]["attempts"][-1]["state"] = "FAIL"
    assert module.ledger_pass(downgraded) is False
    downgraded["status"] = "PARTIAL"
    downgraded["counts"]["PASS"] = 1
    assert module.ledger_pass(downgraded) is False
    assert module.ledger_pass({}) is False
