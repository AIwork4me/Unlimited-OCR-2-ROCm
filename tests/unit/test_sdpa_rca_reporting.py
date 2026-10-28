"""Evidence classifications must not promote unsupported, nonfinite or unobserved results."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/sdpa_rca"))


def reporting():
    path = ROOT / "experiments/sdpa_rca/reporting.py"
    assert path.exists(), "RCA reporting implementation missing"
    spec = importlib.util.spec_from_file_location("rca_reporting", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bad_numerics_and_pending_hip_error_cannot_pass():
    m = reporting()
    assert m.classify({"failing_elements": 0, "nonfinite_actual": 0}, 0) == "PASS"
    assert m.classify({"failing_elements": 1, "nonfinite_actual": 0}, 0) == "FAIL"
    assert m.classify({"failing_elements": 0, "nonfinite_actual": 1}, 0) == "FAIL"
    assert m.classify(None, 1) == "FAIL"
    assert m.classify(None, 0) == "INCOMPLETE"


def test_requested_backend_is_not_dispatch_evidence():
    m = reporting()
    assert m.observed_backend([]) == "UNOBSERVED"
    assert m.observed_backend(["aten._scaled_dot_product_attention_math.default"]) == "MATH"
    assert (
        m.observed_backend(["aten._scaled_dot_product_flash_attention.default"])
        == "FLASH_ATTENTION"
    )
    assert (
        m.observed_backend(["aten._scaled_dot_product_efficient_attention.default"])
        == "EFFICIENT_ATTENTION"
    )
    assert (
        m.observed_backend(["aten._scaled_dot_product_cudnn_attention.default"])
        == "CUDNN_ATTENTION"
    )


def test_private_paths_are_redacted_without_losing_error_details():
    m = reporting()
    value = r"C:\Users\someone\venv\torch.dll: hipErrorInvalidValue (1)"
    assert (
        m.sanitize(value, r"C:\Users\someone")
        == r"<USERPROFILE>\venv\torch.dll: hipErrorInvalidValue (1)"
    )
    assert m.sanitize("C:/Users/someone/test", r"C:\Users\someone") == "<USERPROFILE>/test"


def test_explicit_reference_mask_and_scale_semantics():
    torch = pytest.importorskip("torch")
    m = reporting()
    q = torch.zeros(1, 1, 2, 1)
    v = torch.tensor([[[[2.0], [6.0]]]])
    assert m.reference_attention(q, q, v, None, False, None).flatten().tolist() == [4, 4]
    allowed = torch.tensor([[True, False], [False, True]])
    assert m.reference_attention(q, q, v, allowed, False, None).flatten().tolist() == [2, 6]
    assert m.reference_attention(q, q, v, None, True, None).flatten().tolist() == [2, 4]
    bias = torch.tensor([[0.0, -float("inf")], [-float("inf"), 0.0]])
    assert m.reference_attention(q, q, v, bias, False, None).flatten().tolist() == [2, 6]
    q1 = torch.ones(1, 1, 1, 1)
    k = torch.tensor([[[[0.0], [1.0]]]])
    out = m.reference_attention(q1, k, v, None, False, 0.0)
    assert out.item() == 4


def test_nonfinite_statistics_and_near_zero_relative_error():
    torch = pytest.importorskip("torch")
    m = reporting()
    stats = m.numerical_stats(
        torch.tensor([0.0, float("nan"), float("inf"), 0.1]), torch.zeros(4), 0.03, 0.03
    )
    assert stats["failing_elements"] == 3
    assert stats["nonfinite_actual"] == 2
    assert stats["total_elements"] == 4
    assert stats["failing_percent"] == 75


def test_fixture_preserves_exact_values_across_stride_variants():
    pytest.importorskip("torch")
    path = ROOT / "experiments/sdpa_rca/repro_sdpa.py"
    assert path.exists(), "Standalone reproducer missing"
    spec = importlib.util.spec_from_file_location("rca_repro", path)
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    a, ha = module.make_fixture([1, 2, 3, 4], "bfloat16", "bias", 7992, False)
    b, hb = module.make_fixture([1, 2, 3, 4], "bfloat16", "bias", 7992, True)
    assert ha == hb
    assert not b[0].is_contiguous()
    for left, right in zip(a, b, strict=True):
        assert left.equal(right)


def test_runner_retains_real_failed_process_and_stderr(tmp_path):
    import sys

    path = ROOT / "experiments/sdpa_rca/compare_backends.py"
    assert path.exists(), "Fresh process runner missing"
    spec = importlib.util.spec_from_file_location("rca_runner", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    result = module.run_case(
        [
            sys.executable,
            "-c",
            "import sys; print('real stdout'); print('real error', file=sys.stderr); sys.exit(2)",
        ],
        tmp_path / "failed",
        ROOT,
    )
    assert result["exit_code"] == 2
    assert (tmp_path / "failed.stdout.log").read_text() == "real stdout\n"
    assert (tmp_path / "failed.stderr.log").read_text() == "real error\n"
    assert len(result["stdout_sha256"]) == 64


def test_provenance_parses_runtime_commit_independently_of_image_signature():
    path = ROOT / "experiments/sdpa_rca/collect_provenance.py"
    assert path.exists(), "Provenance collector missing"
    spec = importlib.util.spec_from_file_location("rca_provenance", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    text = (
        "#define AOTRITON_VERSION_MINOR 12u\n"
        "#define AOTRITON_GIT_SHA1 05f26c2d8fbdcf8dd6d95b9544d04f7a39fc9920\n"
    )
    result = module.parse_header(text)
    assert result["AOTRITON_VERSION_MINOR"] == "12u"
    assert result["AOTRITON_GIT_SHA1"] == "05f26c2d8fbdcf8dd6d95b9544d04f7a39fc9920"


def test_evidence_export_retains_original_digest_and_sanitizes_nested_json(tmp_path):
    path = ROOT / "experiments/sdpa_rca/export_evidence.py"
    assert path.exists(), "Evidence exporter missing"
    spec = importlib.util.spec_from_file_location("rca_export", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "private.json"
    source.write_bytes(
        b'{"traceback":"C:\\\\Users\\\\someone\\\\probe.py: hipErrorInvalidValue",'
        b'"run_id":"original-run"}\r\n'
    )
    target = tmp_path / "public.json"
    record = module.export_file(source, target, r"C:\Users\someone")
    import hashlib
    import json

    assert record["original_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert record["published_sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert json.loads(target.read_text())["run_id"] == "original-run"
    assert "someone" not in target.read_text()
    assert "hipErrorInvalidValue" in target.read_text()


def test_runner_timeout_keeps_partial_streams(tmp_path):
    import sys

    path = ROOT / "experiments/sdpa_rca/compare_backends.py"
    spec = importlib.util.spec_from_file_location("rca_runner_timeout", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.run_case(
        [sys.executable, "-u", "-c", "import time; print('started'); time.sleep(2)"],
        tmp_path / "timeout",
        ROOT,
        timeout=0.5,
    )
    assert result["exit_code"] == 124
    assert result["timed_out"] is True
    assert (tmp_path / "timeout.stdout.log").read_text() == "started\n"


def test_ab_audit_rejects_a_changed_dll_or_package_even_when_images_change():
    import copy

    m = reporting()
    assert hasattr(m, "audit_ab_provenance"), "Controlled A/B audit missing"
    a = {
        "packages": {"torch": "exact"},
        "architecture": "gfx1151",
        "system": {"driver": "same"},
        "runtime_header": {"sha256": "header"},
        "loaded_modules": [{"path": r"C:\env\runtime.dll", "sha256": "runtime"}],
        "image_files": [{"path": r"C:\env\aotriton.images\old.zip", "sha256": "old"}],
    }
    b = copy.deepcopy(a)
    b["image_files"] = [{"path": r"C:\env\aotriton.images\new.aks2", "sha256": "new"}]
    assert m.audit_ab_provenance(a, b)["only_image_payload_changed"] is True
    b["loaded_modules"][0]["sha256"] = "other-runtime"
    with pytest.raises(ValueError, match="loaded_modules"):
        m.audit_ab_provenance(a, b)
    b = copy.deepcopy(a)
    b["packages"]["torch"] = "newer"
    with pytest.raises(ValueError, match="packages"):
        m.audit_ab_provenance(a, b)
