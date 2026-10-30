"""CPU CI validates scientific comparison contracts without torch or models."""

import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[2] / "experiments/clip_mlp/analyze.py"
spec = importlib.util.spec_from_file_location("clip_mlp_contract_test", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def identity():
    return dict(
        shape=[1, 257, 4096],
        stride=[1052672, 4096, 1],
        dtype="torch.bfloat16",
        device="cuda:0",
        storage_offset=0,
        sha256="a" * 64,
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("shape", [1052672]),
        ("stride", [1052672, 1, 257]),
        ("dtype", "torch.float32"),
        ("device", "cpu"),
        ("storage_offset", 1),
        ("sha256", "b" * 64),
    ],
)
def test_comparison_rejects_any_input_identity_change(key, value):
    with pytest.raises(ValueError):
        module.require_same_inputs(identity(), dict(identity(), **{key: value}))


def test_comparison_requires_complete_original_gpu_bf16_identity():
    module.require_same_inputs(identity(), identity())
    with pytest.raises(ValueError):
        module.require_same_inputs({})
    with pytest.raises(ValueError):
        module.require_same_inputs(dict(identity(), device="cpu"))


def test_graph_sanitization_preserves_constants_and_types():
    graph = (
        "%1 : float = prim::Constant[value=1.702]() # private/source.py:237\n"
        "%2 : Tensor = aten::mul(%x, %1)\n"
    )
    sanitized = module.graph_without_locations(graph)
    assert "private/source" not in sanitized
    assert "Constant[value=1.702]" in sanitized
    assert "aten::mul(%x, %1)" in sanitized


@pytest.mark.parametrize(
    "key,value",
    [
        ("hip_before", {"status": "FAIL"}),
        ("hip_after", {"status": "FAIL"}),
        ("fallback_allowed", True),
        ("cpu_original_exact", False),
    ],
)
def test_reduction_with_bad_hip_fallback_or_reference_cannot_publish(key, value):
    row = dict(
        hip_before={"status": "PASS"},
        hip_after={"status": "PASS"},
        fallback_allowed=False,
        cpu_original_exact=True,
        cpu_emitted_coefficient_exact=True,
    )
    module.require_reduction(row)
    with pytest.raises(ValueError):
        module.require_reduction(dict(row, **{key: value}))
