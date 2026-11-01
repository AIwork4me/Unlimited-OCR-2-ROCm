"""B1 adapter: route sglang's ``moe.moe_align_block_size`` to the torch fallback.

The frozen wheel resolves the op through ``KernelBackend.AOT`` to the
``sgl_kernel`` wheel (CUDA-only). This adapter rebinds the single import
point the HIP serving path uses —
``sglang.kernels.ops.moe.moe_align_block_size`` — to the CPU-tested torch
fallback in :mod:`unlimited_ocr_2_rocm.sglang_compat.moe_align`. It must be
installed BEFORE ``sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size``
is imported (the consumer binds the name at import time); the B1 launch
wrapper imports this module first.

Isolation: nothing inside the sglang tree is modified; the adapter is a
runtime rebinding performed by the launch wrapper, recorded in the kernel
identity receipt (``call_count``) so every smoke receipt can state which
implementation served the call.
"""

from __future__ import annotations

from typing import Any

_STATE: dict[str, Any] = {"installed": False, "call_count": 0}


def _counting_wrapper(torch_impl):  # type: ignore[no-untyped-def]
    def wrapper(*args: Any, **kwargs: Any) -> None:
        _STATE["call_count"] += 1
        return torch_impl(*args, **kwargs)

    return wrapper


def install() -> None:
    """Rebind ``sglang.kernels.ops.moe.moe_align_block_size`` (idempotent)."""
    if _STATE["installed"]:
        return
    import sglang.kernels.ops.moe as ops_moe

    from .moe_align import torch_moe_align_block_size

    ops_moe.moe_align_block_size = _counting_wrapper(torch_moe_align_block_size)
    _STATE["installed"] = True


def receipt() -> dict[str, Any]:
    """Kernel identity for the smoke receipts."""
    return {
        "moe_align_block_size": "unlimited_ocr_2_rocm.sglang_compat.moe_align."
        "torch_moe_align_block_size (B1 HIP fallback, upstream ref "
        "sglang v0.5.20 aot/csrc/moe/moe_align_kernel.cu)",
        "installed": _STATE["installed"],
        "call_count": _STATE["call_count"],
    }
