"""CPU tests for passive physical-regression evidence tooling."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def harness():
    path = Path(__file__).resolve().parents[2] / "experiments/native_regression/runner.py"
    assert path.is_file(), "G1 harness implementation is missing"
    spec = importlib.util.spec_from_file_location("g1_harness_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_observer_preserves_exact_return_and_captures_before_caller_discards(harness, tmp_path):
    result = {
        "text": "<PAGE>中文\n",
        "token_ids": [128827, 42],
        "prompt_token_ids": [128827],
        "finish_reason": "length",
        "num_pages_out": 1,
        "expected_pages": 1,
        "time": 1.2,
        "prefill_time": 0.2,
    }
    model = SimpleNamespace(infer=lambda *args, **kwargs: result)
    original = model.infer
    with harness.NativeObserver(model, SimpleNamespace(), tmp_path) as observer:
        observed = model.infer("tokenizer", image_file="fixture.png", return_dict=True)
        assert observed is result
        result["token_ids"].append(100)
    assert model.infer is original
    assert observer.records[0]["native"]["token_ids"] == [128827, 42]
    assert (tmp_path / "native-001.raw.txt").read_bytes() == "<PAGE>中文\n".encode()
    assert (
        json.loads((tmp_path / "native-001.json").read_text(encoding="utf-8"))["finish_reason"]
        == "length"
    )


@pytest.mark.parametrize(
    ("state", "expected", "cap", "branch"),
    [
        ({"marker_cum": 3, "page_len": 1}, 2, 20, "force_eos_excess_pages"),
        ({"marker_cum": 1, "page_len": 8}, 2, 8, "force_page_at_page_cap"),
        ({"marker_cum": 2, "page_len": 8}, 2, 8, "force_eos_at_page_cap"),
        ({"marker_cum": 1, "page_len": 1}, 2, 20, "suppress_eos_missing_pages"),
        ({"marker_cum": 2, "page_len": 1}, 2, 20, "unmodified"),
        ({"marker_cum": 0, "page_len": 8}, 0, 8, "disabled_expected_pages_zero"),
    ],
)
def test_guard_branch_uses_official_post_state(harness, state, expected, cap, branch):
    assert harness.guard_branch(state, expected, cap) == branch


def test_guard_observer_calls_original_once_and_returns_same_logits(harness, tmp_path):
    logits = object()
    state = {"marker_cum": 0, "page_len": 0, "scanned": 0}

    def official(value, ids, expected, page_id, eos_id, cap, st):
        assert value is logits
        st.update(marker_cum=2, page_len=8, scanned=12)
        return value

    module = SimpleNamespace(_uocr_page_count_guard=official)
    model = SimpleNamespace(infer=lambda: None)
    with harness.NativeObserver(model, module, tmp_path) as observer:
        actual = module._uocr_page_count_guard(logits, [1] * 12, 2, 10, 11, 8, state)
        assert actual is logits
    assert module._uocr_page_count_guard is official
    assert observer.guard_events[0]["branch"] == "force_eos_at_page_cap"
    assert observer.guard_events[0]["state_before"]["marker_cum"] == 0
    assert observer.guard_events[0]["state_after"]["marker_cum"] == 2


def test_normal_cases_match_smoke_coverage_and_per_page_budget(harness):
    cases = harness.normal_cases()
    assert [case["name"] for case in cases] == [
        "english",
        "chinese",
        "mixed",
        "table",
        "math",
        "repeat_unicode_recovery",
        "pages_2_run_1",
        "pages_2_run_2",
        "pages_3_run_1",
        "pages_3_run_2",
        "pages_5_run_1",
        "pages_5_run_2",
        "table_2_pages",
    ]
    assert [case["max_new_tokens"] for case in cases] == [
        1024,
        1024,
        1024,
        1024,
        1024,
        1024,
        2048,
        2048,
        3072,
        3072,
        5120,
        5120,
        2048,
    ]
    assert cases[-1]["truth_keys"] == ["table_page_1", "table_page_2"]


def test_alias_loader_keeps_package_relative_imports(harness, tmp_path):
    source = tmp_path / "before.py"
    source.write_text("from .config import file_sha256\ndef run_native(): return file_sha256\n")
    module = harness.load_before(source)
    from unlimited_ocr_2_rocm.config import file_sha256

    assert module.run_native() is file_sha256
    assert module.__name__ == "unlimited_ocr_2_rocm._g1_before"


def test_comparison_ignores_timing_but_rejects_native_or_kernel_changes(harness):
    left = {
        "native": {
            "text": "A",
            "token_ids": [1],
            "prompt_token_ids": [2],
            "finish_reason": "stop",
            "num_pages_out": 1,
            "expected_pages": 1,
            "time": 1.0,
            "prefill_time": 0.1,
        },
        "raw_sha256": "abc",
        "kernels": [{"query_shape": [1, 2]}],
        "status": "PASS",
        "correctness": {"status": "PASS"},
        "runtime_state_restored": True,
        "physical_contract": "PASS",
    }
    right = json.loads(json.dumps(left))
    right["native"]["time"] = 99
    assert harness.compare_case(left, right)["status"] == "PASS"
    right["native"]["finish_reason"] = "length"
    right["kernels"][0]["query_shape"] = [9, 2]
    comparison = harness.compare_case(left, right)
    assert comparison["status"] == "FAIL"
    assert set(comparison["differences"]) == {"native.finish_reason", "kernels"}


def test_fresh_output_refuses_reuse_without_erasing(harness, tmp_path):
    evidence = tmp_path / "raw.txt"
    evidence.write_bytes(b"retained")
    with pytest.raises(ValueError, match="fresh"):
        harness.fresh_output(tmp_path)
    assert evidence.read_bytes() == b"retained"


def test_comparison_rejects_missing_native_evidence_even_if_both_arms_match(harness):
    case = {"status": "PASS", "correctness": {"status": "PASS"}}
    assert harness.compare_case(case, case)["status"] == "FAIL"


def test_comparison_rejects_wrapper_failure_even_if_native_matches(harness):
    case = {
        "native": {
            "text": "A",
            "token_ids": [1],
            "prompt_token_ids": [2],
            "finish_reason": "stop",
            "num_pages_out": 1,
            "expected_pages": 1,
        },
        "raw_sha256": "abc",
        "kernels": [],
        "status": "FAIL",
        "correctness": {"status": "PASS"},
    }
    assert "wrapper_status" in harness.compare_case(case, case)["differences"]


def test_interrupt_occurs_after_third_real_forward_and_restores(harness):
    events = []
    decoder = SimpleNamespace(forward=lambda: events.append("forward") or object())
    original = decoder.forward
    with harness.interrupt_after_forward(decoder, lambda: events.append("sync"), 3) as state:
        decoder.forward()
        decoder.forward()
        with pytest.raises(KeyboardInterrupt):
            decoder.forward()
        assert state["completed_forward_calls"] == 3
        assert state["injected"] is True
    assert decoder.forward is original
    assert events == ["forward", "forward", "forward", "sync"]


def test_helper_proof_rejects_changed_helper(harness, tmp_path):
    before, after = tmp_path / "before", tmp_path / "after"
    before.mkdir()
    after.mkdir()
    for name in ("model.py", "output.py", "compat.py", "config.py", "doctor.py"):
        (before / name).write_bytes(b"original")
        (after / name).write_bytes(b"original")
    assert harness.helper_proof(before, after)["model.py"]["byte_identical"] is True
    (after / "model.py").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="Historical helper differs"):
        harness.helper_proof(before, after)


def test_after_suite_cannot_pass_without_observed_interruption_and_identical_retry(harness):
    summaries = {
        case["name"]: {
            "status": "PASS",
            "correctness": {"status": "PASS"},
            "runtime_state_restored": True,
        }
        for case in harness.normal_cases()
    }
    summaries["length_2_pages_8_tokens"] = {
        "status": "FAIL",
        "native": {"finish_reason": "length", "token_ids": list(range(8))},
        "raw_sha256": "abc",
    }
    suite = harness.suite_contract("after", summaries, None)
    assert suite["status"] == "FAIL"
    assert suite["interruption_failure_contract"] == "FAIL"
    assert suite["retry_identity_contract"] == "FAIL"


def test_case_reader_detects_raw_artifact_tampering(harness, tmp_path):
    diagnostics = tmp_path / "diagnostics"
    diagnostics.mkdir()
    raw = diagnostics / "native-001.raw.txt"
    raw.write_bytes(b"original")
    (diagnostics / "native-001.json").write_text('{"text":"original"}', encoding="utf-8")
    harness.write_json(
        tmp_path / "summary.json",
        {"native": {"text": "original"}, "raw_sha256": harness.sha256(raw)},
    )
    assert harness.read_case(tmp_path)["native"]["text"] == "original"
    raw.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="raw artifact"):
        harness.read_case(tmp_path)


def test_hip_probe_preserves_nonzero_peek_as_failure_without_clearing(harness):
    def runtime(pointer):
        pointer._obj.value = 71400000
        return 0

    def driver(pointer):
        pointer._obj.value = 70000000
        return 0

    api = SimpleNamespace(
        hipRuntimeGetVersion=runtime, hipDriverGetVersion=driver, hipPeekAtLastError=lambda: 7
    )
    result = harness.hip_probe(api)
    assert result["status"] == "FAIL"
    assert result["hipPeekAtLastError"] == 7
    assert result["hipRuntimeGetVersion"]["version"] == 71400000


def test_baseline_proof_rejects_wrong_original_caller(harness, tmp_path):
    before, after = tmp_path / "before", tmp_path / "after"
    before.mkdir()
    after.mkdir()
    hashes = {}
    for name in ("model", "output", "compat", "config", "doctor"):
        for directory in (before, after):
            (directory / f"{name}.py").write_bytes(b"original")
        hashes[name] = harness.sha256(before / f"{name}.py")
    source = tmp_path / "inference.py"
    source.write_bytes(b"historical")
    hashes["inference"] = harness.sha256(source)
    receipt = tmp_path / "hashes.json"
    harness.write_json(receipt, hashes)
    assert harness.baseline_proof(source, before, after, receipt)["caller_verified"] is True
    source.write_bytes(b"wrong caller")
    with pytest.raises(RuntimeError, match="Historical caller hash"):
        harness.baseline_proof(source, before, after, receipt)


def test_control_contract_rejects_missing_controls(harness):
    assert harness.control_contracts({}) == {
        "max_length_control": "FAIL",
        "page_cap_control": "FAIL",
    }


def test_comparison_arm_gate_rejects_reversed_arms_and_failed_suite(harness, tmp_path):
    before, after = tmp_path / "before", tmp_path / "after"
    before.mkdir()
    after.mkdir()
    for directory in (before, after):
        harness.write_json(directory / "run.json", {"arm": "after", "harness_sha256": "same"})
        harness.write_json(directory / "suite.json", {"status": "FAIL"})
    assert harness.comparison_arm_gate(before, after)["status"] == "FAIL"


def test_json_write_rejects_invalid_values_without_partial_or_overwritten_artifact(
    harness, tmp_path
):
    path = tmp_path / "evidence.json"
    with pytest.raises(ValueError):
        harness.write_json(path, {"bad": float("nan")})
    assert not path.exists()
    harness.write_json(path, {"value": "retained"})
    with pytest.raises(FileExistsError):
        harness.write_json(path, {"value": "replacement"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"value": "retained"}


def test_expected_length_failure_cannot_hide_telemetry_failure(harness):
    summaries = {
        "length_2_pages_8_tokens": {
            "status": "FAIL",
            "native": {"finish_reason": "length", "token_ids": list(range(8))},
            "raw_sha256": "abc",
            "runtime_state_restored": True,
            "physical_contract": "PASS",
            "telemetry_error": "synchronize failed",
        }
    }
    assert harness.suite_contract("before", summaries, None)["length_failure_contract"] == "FAIL"


def test_case_preserves_failure_when_initial_synchronize_raises(harness, tmp_path, monkeypatch):
    import sys

    import unlimited_ocr_2_rocm.doctor as doctor

    def failed_sync():
        raise RuntimeError("initial sync failure")

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(reset_peak_memory_stats=lambda: None, synchronize=failed_sync)
        ),
    )
    monkeypatch.setattr(doctor, "memory_snapshot", lambda: {})
    monkeypatch.setattr(harness, "runtime_state", lambda *args: {})
    image = tmp_path / "image.png"
    image.write_bytes(b"fixture")
    truth = {"english": {"image": "image.png", "sha256": harness.sha256(image)}}

    def infer(*args, **kwargs):
        return None

    model = SimpleNamespace(infer=infer, infer_multi=infer, _uocr_generate=infer)
    native = SimpleNamespace(model=model, kernels=[], vision=[])
    caller = SimpleNamespace(run_native=infer)
    case = {"name": "english", "truth_keys": ["english"], "max_new_tokens": 1024}
    summary = harness.run_case(
        native, SimpleNamespace(), caller, case, truth, tmp_path, tmp_path / "failure"
    )
    assert summary["status"] == "FAIL"
    assert "initial sync failure" in summary["traceback"]
    assert (tmp_path / "failure/summary.json").is_file()
    assert summary["physical_contract"] == "FAIL"
