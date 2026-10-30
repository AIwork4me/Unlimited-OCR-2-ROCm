"""CPU-only scientific acceptance gates reject incomplete physical controls."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[2] / "experiments/te_mitigation/analyze.py"
spec = importlib.util.spec_from_file_location("te_qualification_test", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def valid_request():
    summary = dict(
        status="PASS",
        correctness={"status": "PASS"},
        validated_metadata={"finish_reason": "stop"},
        runtime_state_restored=True,
        hip_before={"status": "PASS"},
        hip_after={"status": "PASS"},
        kernels=[dict(dtype="torch.bfloat16", device="cuda:0", synchronized=True)] * 12,
        guard_branch_counts={"unmodified": 1},
        physical_contract="PASS",
    )
    observation = dict(
        status="PASS",
        mode="MATH",
        dispatch={
            name: {"aten::_scaled_dot_product_attention_math": 1}
            for name in ("sam_model", "vision_model")
        },
    )
    events = [
        dict(
            layer=layer,
            call=1,
            graph="aten::sigmoid",
            profile=1,
            dtype="torch.bfloat16",
            device="cuda:0",
            shape=[1, 257, 4096],
            stride=[1052672, 4096, 1],
            graph_optimize=True,
            te_enabled=False,
        )
        for layer in range(24)
    ]
    profiles = [
        dict(serial=0, operators=[]),
        dict(serial=1, operators=[dict(name="aten::sigmoid", count=24)]),
    ]
    return summary, observation, events, profiles


def gates(parts):
    return module.request_gates(*parts, "te-off", "MATH", 1)


def test_complete_control_qualifies():
    assert all(gates(valid_request()).values())


@pytest.mark.parametrize(
    "fault",
    [
        "hip",
        "fallback",
        "missing_layer",
        "duplicate_call",
        "fused",
        "no_dispatch",
        "no_profile",
        "wrong_dtype",
        "global_graph_off",
        "length",
        "dash",
        "unrestored",
    ],
)
def test_incomplete_controls_never_qualify(fault):
    parts = copy.deepcopy(valid_request())
    summary, observation, events, profiles = parts
    if fault == "hip":
        summary["hip_after"]["status"] = "FAIL"
    elif fault == "fallback":
        observation["dispatch"]["vision_model"] = {"aten::_scaled_dot_product_flash_attention": 1}
    elif fault == "missing_layer":
        events.pop()
    elif fault == "duplicate_call":
        events[0]["call"] = 2
    elif fault == "fused":
        events[0]["graph"] = "prim::TensorExprGroup"
    elif fault == "no_dispatch":
        profiles[1]["operators"] = []
    elif fault == "no_profile":
        events[0]["profile"] = 77
    elif fault == "wrong_dtype":
        events[0]["dtype"] = "torch.float32"
    elif fault == "global_graph_off":
        events[0]["graph_optimize"] = False
    elif fault == "length":
        summary["validated_metadata"]["finish_reason"] = "length"
    elif fault == "dash":
        summary["kernels"].pop()
    elif fault == "unrestored":
        summary["runtime_state_restored"] = False
    assert not all(gates(parts).values())


def test_empty_or_same_process_repeats_cannot_qualify():
    schedule = [dict(option="te-off", modes=None)] * 2
    one = dict(option="te-off", modes=None, pid=1, created=1)
    two = dict(one, pid=2, created=2)
    assert module.independent_coverage([one, two], schedule)
    assert not module.independent_coverage([], [])
    assert not module.independent_coverage([one], schedule[:1])
    assert not module.independent_coverage([one, one], schedule)


def test_auto_silent_math_fallback_is_rejected():
    parts = valid_request()
    parts[1]["mode"] = "AUTO"
    assert not module.request_gates(*parts, "te-off", "AUTO", 1)["attention"]


def test_single_image_crop_contract_is_exact_not_a_shape_tolerance():
    summary, observation, global_events, profiles = valid_request()
    local_events = copy.deepcopy(global_events)
    for event in local_events:
        event.update(shape=[6, 101, 4096], stride=[413696, 4096, 1])
    for event in global_events:
        event.update(call=2, profile=3)
    profiles += [
        dict(serial=2, operators=[]),
        dict(serial=3, operators=[dict(name="aten::sigmoid", count=24)]),
    ]
    parts = summary, observation, local_events + global_events, profiles
    shapes = [[6, 101, 4096], [1, 257, 4096]]
    assert not all(module.request_gates(*parts, "te-off", "MATH", 1).values())
    assert all(module.request_gates(*parts, "te-off", "MATH", 1, shapes).values())
    local_events[0]["shape"] = [5, 101, 4096]
    assert not all(module.request_gates(*parts, "te-off", "MATH", 1, shapes).values())


publication_spec = importlib.util.spec_from_file_location(
    "te_publication_test", path.with_name("publish.py")
)
publication = importlib.util.module_from_spec(publication_spec)
publication_spec.loader.exec_module(publication)


@pytest.mark.parametrize(
    "payload",
    [
        {"file": "C:/Users/private/file"},
        {"file": r"C:\Users\private\file"},
        {"file": "/home/private/file"},
        {"text": "synthetic payload"},
        {"prompt": "synthetic payload"},
        {"token_ids": [1, 2]},
    ],
)
def test_publication_rejects_payloads_and_both_windows_path_styles(payload):
    with pytest.raises(ValueError):
        publication.validate_public_text(json.dumps(payload))


def test_comparison_export_drops_unexpected_fields_and_token_context():
    result = publication.comparison(
        dict(
            raw_exact=True,
            boxes=[],
            invalid_boxes=[],
            text="PRIVATE_CANARY",
            token_context=[123],
            raw_output="PRIVATE_CANARY",
        )
    )
    assert result == dict(raw_exact=True, boxes=[], invalid_boxes=[])
    publication.validate_public_text(json.dumps(result))
