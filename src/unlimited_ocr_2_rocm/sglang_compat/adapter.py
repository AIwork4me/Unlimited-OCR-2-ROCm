"""B1 adapter: route sglang's CUDA-only sgl_kernel ops to torch fallbacks.

The frozen wheel resolves ``moe.moe_align_block_size`` through
``KernelBackend.AOT`` to the ``sgl_kernel`` wheel (CUDA-only), and the HIP
branches of the fused-MoE and activation layers bind the elementwise
activations from the same wheel. This adapter rebinds those names to the
CPU-tested torch fallbacks in :mod:`unlimited_ocr_2_rocm.sglang_compat`:

- ``sglang.kernels.ops.moe.moe_align_block_size`` (the single import point
  the HIP serving path uses for the align op), and
- the ``sgl_kernel`` module attributes for the elementwise activations
  (the single choke point every consumer binds through).

It must be installed BEFORE the consuming sglang modules are imported (the
B1 launcher does this at interpreter startup via sitecustomize, spawn-safe).
Nothing inside the sglang tree is modified; unpatched sgl_kernel attributes
keep raising the env's loud-fail stub. The receipt records which fallbacks
are installed and how often each served, so every smoke receipt can state
the kernel identity.
"""

from __future__ import annotations

from typing import Any

_STATE: dict[str, Any] = {"installed": False, "call_counts": {}}


def _counting(name: str, impl):  # type: ignore[no-untyped-def]
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        _STATE["call_counts"][name] = _STATE["call_counts"].get(name, 0) + 1
        return impl(*args, **kwargs)

    return wrapper


def install() -> None:
    """Rebind the CUDA-only ops to the torch fallbacks (idempotent)."""
    if _STATE["installed"]:
        return

    import sgl_kernel

    from .activation import gelu_and_mul, gelu_quick, gelu_tanh_and_mul, silu_and_mul
    from .moe_align import torch_moe_align_block_size

    for name, impl in (
        ("silu_and_mul", silu_and_mul),
        ("gelu_and_mul", gelu_and_mul),
        ("gelu_tanh_and_mul", gelu_tanh_and_mul),
        ("gelu_quick", gelu_quick),
        ("moe_align_block_size", torch_moe_align_block_size),
    ):
        setattr(sgl_kernel, name, _counting(name, impl))

    # The align op's HIP import point binds the ops-wrapper function, not the
    # sgl_kernel attribute; rebind it too (idempotent, same fallback).
    import sglang.kernels.ops.moe as ops_moe

    ops_moe.moe_align_block_size = _counting("moe_align_block_size", torch_moe_align_block_size)
    _STATE["installed"] = True


def receipt() -> dict[str, Any]:
    """Kernel identity for the smoke receipts."""
    return {
        "fallbacks": {
            "moe_align_block_size": "sglang_compat.moe_align.torch_moe_align_block_size "
            "(upstream ref: sglang v0.5.20 aot/csrc/moe/moe_align_kernel.cu)",
            "silu_and_mul": "sglang_compat.activation (exact torch)",
            "gelu_and_mul": "sglang_compat.activation (exact torch, erf)",
            "gelu_tanh_and_mul": "sglang_compat.activation (exact torch, tanh)",
            "gelu_quick": "sglang_compat.activation (exact torch, 1.702)",
        },
        "installed": _STATE["installed"],
        "call_counts": dict(_STATE["call_counts"]),
    }
