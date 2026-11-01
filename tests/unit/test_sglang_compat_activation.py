"""CPU tests for the B1 activation fallbacks: exactness vs the torch formulas
and the in-place / shape contracts the sgl_kernel call sites rely on."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from unlimited_ocr_2_rocm.sglang_compat.activation import (
    gelu_and_mul,
    gelu_quick,
    gelu_tanh_and_mul,
    silu_and_mul,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(16, 64), (2, 3, 7, 128)])
def test_silu_and_mul_matches_the_reference_formula(dtype, shape):
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(shape, generator=gen, dtype=torch.float32).to(dtype)
    expected = F.silu(x[..., : x.shape[-1] // 2]) * x[..., x.shape[-1] // 2 :]
    assert torch.equal(silu_and_mul(x), expected)
    out = torch.empty(*shape[:-1], shape[-1] // 2, dtype=dtype)
    result = silu_and_mul(x, out)
    assert result is out and torch.equal(out, expected)


def test_gelu_variants_match_their_formulas():
    gen = torch.Generator().manual_seed(1)
    x = torch.randn((8, 32), generator=gen, dtype=torch.float32)
    gate, up = x[..., :16], x[..., 16:]
    assert torch.allclose(gelu_and_mul(x), F.gelu(gate) * up, rtol=1e-6)
    assert torch.allclose(gelu_tanh_and_mul(x), F.gelu(gate, approximate="tanh") * up, rtol=1e-6)
    # gelu_quick is ELEMENTWISE at the frozen AOT reference (act_only_kernel,
    # out same shape as input) — NOT the gated and-mul form.
    assert torch.allclose(gelu_quick(x), x * torch.sigmoid(1.702 * x), rtol=1e-6)
    out_q = torch.empty_like(x)
    assert gelu_quick(x, out_q) is out_q
    assert torch.allclose(out_q, x * torch.sigmoid(1.702 * x), rtol=1e-6)


def test_routed_variant_refuses_instead_of_silently_misfiltering():
    x = torch.randn((8, 32))
    with pytest.raises(ValueError, match="expert_ids"):
        silu_and_mul(x, expert_ids=torch.zeros((2,), dtype=torch.int32))


def test_odd_last_dim_refused():
    with pytest.raises(ValueError):
        silu_and_mul(torch.randn((4, 7)))
