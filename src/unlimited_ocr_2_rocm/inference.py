"""Native infer/infer_multi calls with raw output and evidence preservation."""

from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Any

from .config import file_sha256
from .doctor import memory_snapshot, write_json
from .model import NativeModel
from .output import assess, normalize


def validate_inputs(images: list[Path], multi: bool) -> list[Path]:
    if not images or (not multi and len(images) != 1):
        raise ValueError("single requires exactly one image; multi requires ordered pages")
    if len(images) > 5:
        raise ValueError("Phase 1 supports at most five fixture pages per request")
    resolved = [path.resolve() for path in images]
    for path in resolved:
        if not path.is_file():
            raise FileNotFoundError(path)
    return resolved


def run_native(
    native: NativeModel,
    images: list[Path],
    output: Path,
    *,
    multi: bool = False,
    truth: list[dict[str, Any]] | None = None,
    max_new_tokens: int = 1024,
) -> dict[str, Any]:
    import torch
    from PIL import Image

    images = validate_inputs(images, multi)
    if not 1 <= max_new_tokens <= 8192:
        raise ValueError("max_new_tokens must be between 1 and 8192")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh output directory to preserve earlier evidence")
    output.mkdir(parents=True, exist_ok=True)
    for path in images:
        with Image.open(path) as image:
            if image.width * image.height > 16_000_000:
                raise ValueError("Phase 1 fixture limit: 16 million pixels per page")
            image.verify()
    report: dict[str, Any] = {
        "native_entrypoint": "infer_multi" if multi else "infer",
        "inputs": [{"path": str(path), "sha256": file_sha256(path)} for path in images],
        "memory_before": memory_snapshot(),
        "max_new_tokens": max_new_tokens,
        "model": native.evidence,
    }
    previous_kernels = len(native.kernels)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        with torch.inference_mode():
            if multi:
                raw, tokens = native.model.infer_multi(
                    native.tokenizer,
                    image_files=[str(path) for path in images],
                    save_results=False,
                    max_new_tokens=max_new_tokens,
                    eval_mode=True,
                )
                finish = "length" if tokens >= max_new_tokens else "upstream_tuple_no_finish_field"
            else:
                result = native.model.infer(
                    native.tokenizer,
                    image_file=str(images[0]),
                    save_results=False,
                    eval_mode=True,
                    return_dict=True,
                    max_new_tokens=max_new_tokens,
                )
                raw, tokens, finish = (
                    result["text"],
                    len(result["token_ids"]),
                    result["finish_reason"],
                )
                report["upstream_timing"] = {
                    "generation_seconds": result["time"],
                    "prefill_seconds": result["prefill_time"],
                }
        torch.cuda.synchronize()
        (output / "raw.txt").write_text(raw, encoding="utf-8")
        (output / "normalized.txt").write_text(normalize(raw), encoding="utf-8")
        kernels = native.kernels[previous_kernels:]
        report.update(
            output_token_count=tokens,
            finish_reason=finish,
            raw_sha256=file_sha256(output / "raw.txt"),
            raw_output=raw,
            native_kernels=kernels,
            native_kernel_count=len(kernels),
        )
        if len(kernels) < 12:
            raise RuntimeError("No successful native DASH kernel recorded for every decoder layer")
        if not raw.strip() or finish == "length":
            raise RuntimeError("Empty or token-budget-truncated OCR output")
        if truth is not None:
            report["correctness"] = assess(raw, truth)
            report["status"] = report["correctness"]["status"]
        else:
            report["status"] = "INFERENCE_ONLY"
    except Exception:
        report["status"] = "FAIL"
        report["traceback"] = traceback.format_exc()
    finally:
        report["synchronized_wall_seconds"] = time.perf_counter() - start
        report["memory_after"] = memory_snapshot()
        write_json(output / "manifest.json", report)
    return report
