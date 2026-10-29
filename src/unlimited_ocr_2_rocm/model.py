"""Reviewed official model loading and non-algorithmic kernel instrumentation."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .compat import wrap_vision_forward
from .config import download_snapshot, file_sha256, read_lock, verify_files
from .doctor import collect_environment, memory_snapshot, require_stack


@dataclass
class NativeModel:
    tokenizer: Any
    model: Any
    evidence: dict[str, Any]
    kernels: list[dict[str, Any]] = field(default_factory=list)
    vision: list[dict[str, Any]] = field(default_factory=list)


def require_model_dependencies(versions: dict[str, str]) -> None:
    expected = {
        "transformers": "4.52.1",
        "triton-windows": "3.7.1.post27",
        "huggingface-hub": "0.36.0",
        "tokenizers": "0.21.4",
    }
    for name, version in expected.items():
        if versions.get(name) != version:
            raise RuntimeError(f"Native model requires {name}=={version}")


def require_native_environment(environment: dict[str, str]) -> None:
    allowed = {
        "DASH_STAGE0": "head_cls",
        "DASH_SPARSITY_SCHED": "0",
        "WARMUP_MAIN_LOCAL_WINDOW": "1",
        "PAGE_SCOPE_PREFILL": "0",
        "TRITON_INTERPRET": "0",
    }
    for name, value in environment.items():
        if name in {"HSA_OVERRIDE_GFX_VERSION", "TRITON_OVERRIDE_DIR"} and value:
            raise RuntimeError(f"Native validation refuses override {name}")
        if name in allowed and value != allowed[name]:
            raise RuntimeError(f"Native checkpoint semantics conflict with {name}")
        if name.startswith("DASH_") and not name.startswith("DASH_LOG_"):
            if name not in allowed and name != "DASH_DEBUG_ANOMALY":
                raise RuntimeError(f"Unreviewed DASH environment override: {name}")


def load_model(
    cache_dir: Path | None = None, offline: bool = False, *, vision_sdpa: str = "math"
) -> NativeModel:
    if vision_sdpa not in {"math", "auto"}:
        raise ValueError("Vision SDPA must be math or auto")
    import torch

    require_native_environment(dict(os.environ))
    environment = collect_environment()
    require_stack(environment)
    names = ("transformers", "triton-windows", "huggingface-hub", "tokenizers")
    versions = {name: importlib.metadata.version(name) for name in names}
    require_model_dependencies(versions)
    if "transformers" in sys.modules:
        raise RuntimeError("Load in a clean process so reviewed code uses a fresh module cache")
    lock = read_lock()
    snapshot = download_snapshot(cache_dir, offline)
    hashes = verify_files(snapshot, {**lock["files"], **lock["weights"]})
    # A fresh process-local module cache prevents reuse of an unchecked code copy.
    os.environ["HF_MODULES_CACHE"] = tempfile.mkdtemp(prefix="uocr2-reviewed-modules-")
    from transformers import AutoModel, AutoTokenizer

    before = memory_snapshot()
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot),
        trust_remote_code=True,
        local_files_only=True,
    )
    model = (
        AutoModel.from_pretrained(
            str(snapshot),
            trust_remote_code=True,
            local_files_only=True,
            use_safetensors=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        .eval()
        .cuda()
    )
    torch.cuda.synchronize()
    for name, parameter in model.named_parameters():
        if parameter.device.type != "cuda" or parameter.dtype != torch.bfloat16:
            raise RuntimeError(f"Model parameter device/dtype mismatch: {name}")
    evidence = {
        "status": "PASS",
        "model_id": lock["model_id"],
        "model_revision": lock["model_revision"],
        "code_revision": lock["code_revision"],
        "source_and_weight_hashes": hashes,
        "source_patch": None,
        "environment": environment,
        "native_dependencies": versions,
        "wrapper_source_hashes": {
            str(path.relative_to(Path(__file__).parent)): file_sha256(path)
            for path in sorted(Path(__file__).parent.rglob("*.py"))
        },
        "cold_load_seconds": time.perf_counter() - start,
        "parameter_device": str(next(model.parameters()).device),
        "parameter_dtype": str(next(model.parameters()).dtype),
        "memory_before_load": before,
        "memory_after_load": memory_snapshot(),
        "attention_dispatch": "mha_eager with official DASH enabled",
        "runtime_compatibility": {
            "vision_sdpa": vision_sdpa,
            "scope": "SAM and CLIP forward only",
            "reason": (
                "historical default SDPA fails; MATH is the accepted default"
                if vision_sdpa == "math"
                else "experimental AUTO; actual fused execution requires separate evidence"
            ),
            "decoder_DASH_and_Triton": "unchanged",
            "global_attention_settings": "restored after each vision call",
        },
    }
    native = NativeModel(tokenizer, model, evidence)
    for name in ("sam_model", "vision_model"):
        encoder = getattr(model.model, name)
        encoder.forward = wrap_vision_forward(
            encoder.forward, name, native.vision, backend=vision_sdpa
        )
    prefix = model.__class__.__module__.rsplit(".", 1)[0]
    kernel_module: Any = importlib.import_module(prefix + ".sparse_gather_e2e_triton")
    original = kernel_module.triton_gather_attention_e2e

    def observed_kernel(*args: Any, **kwargs: Any) -> Any:
        require_native_environment(dict(os.environ))
        query = args[0] if args else kwargs["query_states"]
        if query.device.type != "cuda" or query.dtype != torch.bfloat16:
            raise RuntimeError("DASH query is not BF16 on GPU")
        output = original(*args, **kwargs)
        if output is None or output.device.type != "cuda":
            raise RuntimeError("Native DASH prefill declined the input; fallback refused")
        torch.cuda.synchronize()
        native.kernels.append(
            {
                "name": "triton_gather_attention_e2e",
                "query_shape": list(query.shape),
                "device": str(query.device),
                "dtype": str(query.dtype),
                "synchronized": True,
            }
        )
        return output

    kernel_module.triton_gather_attention_e2e = observed_kernel
    routers = [module for module in model.modules() if module.__class__.__name__ == "DashAttention"]
    if len(routers) != 12 or any(not router.use_triton for router in routers):
        raise RuntimeError("Official 12-layer DASH Triton configuration is not active")

    def refuse_prefill_fallback(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Dense prefill fallback refused; native Triton execution is required")

    for router in routers:
        router._forward_dense = refuse_prefill_fallback
    return native
