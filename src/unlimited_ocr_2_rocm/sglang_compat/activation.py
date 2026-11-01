"""Exact torch fallbacks for the sgl_kernel elementwise activations.

Second CUDA-only dependency ring on the B1 HIP serving path (verified from
server-v2.log after the moe_align fallback): the HIP branches of
``sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe`` and
``sglang.srt.layers.activation`` bind ``silu_and_mul`` / ``gelu_and_mul`` /
``gelu_tanh_and_mul`` / ``gelu_quick`` directly from the CUDA-only
``sgl_kernel`` wheel. Each has an exact torch equivalent:

- silu_and_mul:  out = silu(x[..., :H]) * x[..., H:]
- gelu_and_mul:  out = gelu_erf(x[..., :H]) * x[..., H:]  (sgl-kernel default)
- gelu_tanh_and_mul: out = gelu_tanh(x[..., :H]) * x[..., H:]
- gelu_quick:    out = (x1 * sigmoid(1.702 * x1)) * x2

The routed/filtered variants (expert_ids) are refused fail-loud: the HIP
call sites pass the plain two-argument form, and a silent wrong filter would
corrupt MoE outputs. bf16 numerics: torch elementwise ops on HIP follow the
same rounding as the Triton DASH path already validated on gfx1151; any
divergence shows up in the bounded smoke's golden-page comparison, not
silently.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

GELU_QUICK_COEFFICIENT = 1.702


def _split(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if input.shape[-1] % 2 != 0:
        raise ValueError(f"activation input last dim must be even, got {input.shape[-1]}")
    half = input.shape[-1] // 2
    return input[..., :half], input[..., half:]


def _finish(result: torch.Tensor, out: torch.Tensor | None) -> torch.Tensor:
    if out is None:
        return result
    out.copy_(result)
    return out


def silu_and_mul(
    input: torch.Tensor,
    out: torch.Tensor | None = None,
    expert_ids: torch.Tensor | None = None,
    expert_step: int = 1,
) -> torch.Tensor:
    if expert_ids is not None:
        raise ValueError(
            "torch silu_and_mul fallback does not implement the routed "
            "expert_ids variant; HIP call sites use the plain two-argument form"
        )
    gate, up = _split(input)
    return _finish(F.silu(gate) * up, out)


def gelu_and_mul(input: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    gate, up = _split(input)
    return _finish(F.gelu(gate) * up, out)  # exact erf, sgl-kernel default


def gelu_tanh_and_mul(input: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    gate, up = _split(input)
    return _finish(F.gelu(gate, approximate="tanh") * up, out)


def gelu_quick(input: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Elementwise quick-GELU per the frozen AOT reference.

    sglang v0.5.20's AOT activation kernel implements gelu_quick with the
    ELEMENTWISE act_only_kernel (out same shape as input, d = input.size(-1);
    csrc/elementwise/activation.cu@v0.5.20 gelu_quick, lines 152-170), and the
    wheel's only call site (QuickGELU.forward_hip) passes out with x.shape —
    NOT the gated and-mul form used by some other ecosystems. An earlier
    revision of this module used the gated form; corrected to the reference
    contract (independent review finding, delivery session 2).
    """
    result = input * torch.sigmoid(GELU_QUICK_COEFFICIENT * input)
    if out is None:
        return result
    out.copy_(result)
    return out
