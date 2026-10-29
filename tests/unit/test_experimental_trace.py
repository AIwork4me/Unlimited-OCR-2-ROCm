import pytest

from experiments.native_ocr.validate import fused_trace_summary


@pytest.mark.parametrize("backend", ["MATH", "UNOBSERVED"])
def test_fused_gate_rejects_quality_pass_without_fusion(backend):
    assert (
        fused_trace_summary([{"status": "PASS", "observed_backend": backend}])["status"] == "FAIL"
    )


def test_empty_or_incomplete_trace_cannot_pass():
    assert fused_trace_summary([])["status"] == "FAIL"
    assert fused_trace_summary([{"status": "ATTEMPTED"}])["status"] == "FAIL"


def test_successful_observed_fusion_is_distinct_from_quality_status():
    result = fused_trace_summary(
        [
            {"status": "PASS", "observed_backend": "FLASH_ATTENTION"},
            {"status": "PASS", "observed_backend": "EFFICIENT_ATTENTION"},
        ]
    )
    assert result["status"] == "PASS"
    assert result["calls"] == 2
