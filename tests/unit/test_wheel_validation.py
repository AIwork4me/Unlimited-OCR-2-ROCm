"""CPU-only evidence gates: no torch or model imports."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "experiments/pytorch212_wheel/compare_results.py"
SPEC = importlib.util.spec_from_file_location("wheel_comparison", SCRIPT)
assert SPEC and SPEC.loader
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


def clean_report():
    return {
        "acceptance": True,
        "status": "PASS",
        "exit_code": 0,
        "backend_requested": "FLASH_ATTENTION",
        "backend_observed": "FLASH_ATTENTION",
        "fixture_sha256": "a" * 64,
        "arguments": {"dtype": "bfloat16", "operation": "copy"},
        "events": [
            {"stage": s, "hip_before_sync": 0, "hip_after_sync": 0, "synchronize": "OK"}
            for s in ("before_sdpa", "after_sdpa", "after_copy")
        ],
        "numerical": {
            "atol": 0.03,
            "rtol": 0.03,
            "total_elements": 10,
            "failing_elements": 0,
            "nonfinite_actual": 0,
            "nonfinite_reference": 0,
        },
    }


def test_clean_fused():
    assert comparison.classify_case(clean_report(), 0) == "FUSED_PASS"


def test_default_math_is_not_fused():
    r = clean_report()
    r.update(backend_requested="DEFAULT", backend_observed="MATH")
    assert comparison.classify_case(r, 0) == "MATH_PASS"


@pytest.mark.parametrize("field", ["UNOBSERVED", "MATH", "EFFICIENT_ATTENTION"])
def test_forced_backend_must_be_observed(field):
    r = clean_report()
    r["backend_observed"] = field
    assert comparison.classify_case(r, 0) == "INCONCLUSIVE"


@pytest.mark.parametrize("stage", ["before_sdpa", "after_sdpa", "after_copy"])
def test_any_sticky_error_fails(stage):
    r = clean_report()
    next(e for e in r["events"] if e["stage"] == stage)["hip_before_sync"] = 1
    assert comparison.classify_case(r, 0) == "FAIL"


@pytest.mark.parametrize("field", ["failing_elements", "nonfinite_actual", "nonfinite_reference"])
def test_bad_numerics_fail(field):
    r = clean_report()
    r["numerical"][field] = 1
    assert comparison.classify_case(r, 0) == "FAIL"


@pytest.mark.parametrize("change", ["clear", "no_numeric", "threshold", "no_events", "no_copy"])
def test_missing_or_altered_acceptance_is_inconclusive(change):
    r = clean_report()
    if change == "clear":
        r["acceptance"] = False
    elif change == "no_numeric":
        r.pop("numerical")
    elif change == "threshold":
        r["numerical"]["atol"] = 0.1
    elif change == "no_events":
        r["events"] = []
    else:
        r["events"].pop()
    assert comparison.classify_case(r, 0) == "INCONCLUSIVE"


def test_process_failure_dominates_report():
    assert comparison.classify_case(clean_report(), 1) == "FAIL"


def test_unsupported_is_not_pass():
    r = clean_report()
    r.update(status="UNSUPPORTED", exit_code=2)
    assert comparison.classify_case(r, 2) == "UNSUPPORTED"


def test_unsupported_with_dirty_hip_is_failure():
    r = clean_report()
    r.update(status="UNSUPPORTED", exit_code=2)
    r["events"][-1]["hip_after_sync"] = 1
    assert comparison.classify_case(r, 2) == "FAIL"


def test_only_named_optional_unsupported_control_is_excluded():
    statuses = comparison.acceptance_statuses(
        [
            {"case": "vision_flash_unsupported", "status": "UNSUPPORTED"},
            {"case": "small_flash_1", "status": "UNSUPPORTED"},
            {"case": "small_efficient_1", "status": "FUSED_PASS"},
        ]
    )
    assert statuses == ["UNSUPPORTED", "FUSED_PASS"]
    assert comparison.decide(True, True, True, statuses, matrix_complete=True) == "INCONCLUSIVE"
    assert comparison.acceptance_statuses(
        [
            {"case": "vision_flash_unsupported", "status": "FAIL"},
        ]
    ) == ["FAIL"]


def test_report_parser_rejects_non_object_and_nonfinite(tmp_path):
    p = tmp_path / "report.json"
    for text in ("[]", '{"x": NaN}', '{"x": Infinity}'):
        p.write_text(text)
        with pytest.raises(ValueError):
            comparison.read_report(p)
    p.write_text(json.dumps(clean_report()))
    assert comparison.read_report(p) == clean_report()


def test_provenance_requires_matching_hash_and_paths():
    p = {
        "built_wheel_sha256": "b" * 64,
        "installed_wheel_sha256": "b" * 64,
        "source_sha": "c" * 40,
        "installed_git_version": "c" * 40,
        "architecture": "gfx1151",
        "python_abi": "cp310",
        "platform": "win_amd64",
        "import_within_venv": True,
        "loaded_binaries_within_venv": True,
        "runtime_images_verified": True,
        "pip_check_exit": 0,
        "source_overrides": {},
    }
    assert comparison.valid_provenance(p)
    for field, bad in (
        ("installed_wheel_sha256", "d" * 64),
        ("installed_git_version", "d" * 40),
        ("architecture", "gfx1100"),
        ("python_abi", "cp313"),
        ("platform", "linux_x86_64"),
        ("loaded_binaries_within_venv", False),
        ("runtime_images_verified", False),
        ("pip_check_exit", 1),
        ("source_overrides", {"PYTORCH_AOTRITON_COMMIT": "custom"}),
    ):
        changed = copy.deepcopy(p)
        changed[field] = bad
        assert not comparison.valid_provenance(changed), field
    assert not comparison.valid_provenance({})


def test_final_decision_never_infers_candidate_from_baseline():
    assert comparison.decide(False, False, False, []) == "BUILD_BLOCKED"
    assert comparison.decide(True, False, False, []) == "INSTALL_BLOCKED"
    assert comparison.decide(True, True, False, ["FUSED_PASS"]) == "INCONCLUSIVE"
    assert comparison.decide(True, True, True, ["FAIL"]) == "REGRESSION_PERSISTS"
    assert comparison.decide(True, True, True, ["FUSED_PASS"]) == "INCONCLUSIVE"
    assert comparison.decide(True, True, True, [], matrix_complete=True) == "INCONCLUSIVE"
    assert (
        comparison.decide(True, True, True, ["FUSED_PASS", "MATH_PASS"], matrix_complete=True)
        == "FIX_VERIFIED"
    )
    assert (
        comparison.decide(True, True, True, ["FUSED_PASS", "UNSUPPORTED"], matrix_complete=True)
        == "INCONCLUSIVE"
    )


@pytest.mark.parametrize("dtype,tol", [("bfloat16", 0.03), ("float16", 0.005), ("float32", 0.0001)])
def test_frozen_thresholds(dtype, tol):
    assert comparison.TOLERANCES[dtype] == (tol, tol)


def test_sanitization_handles_case_slashes_and_json_escaping():
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("wheel_export", SCRIPT.with_name("export_evidence.py"))
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    home = "C:/Users/Example"
    for text in ("c:/users/example/x", "C:\\Users\\Example\\x", r"C:\\Users\\Example\\x"):
        result = module.sanitize_text(text, home)
        assert "example" not in result.lower()
        assert "<USERPROFILE>" in result
    payload = json.dumps({"path": "C:\\Users\\Example\\x"})
    assert json.loads(module.sanitize_text(payload, home))["path"] == "<USERPROFILE>\\x"


@pytest.mark.parametrize(
    "name,digest",
    [
        ("repro_sdpa.py", "9673a18c38e48b1a5969d04b50ef691c13a8a6099092150ae5eda2d0380baf13"),
        ("reporting.py", "eb1ed2c5ca146e78c973495825ee52272aee46f442f71df2ef3fa0332b7ed79c"),
        (
            "collect_provenance.py",
            "4aaaf1c469eb31ae382affa1aff7e51eac9ba5106c862811455d80e18cf287a9",
        ),
    ],
)
def test_immutable_rca_producers(name, digest):
    assert hashlib.sha256(SCRIPT.with_name(name).read_bytes()).hexdigest() == digest
