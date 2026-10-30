"""Native infer/infer_multi calls with raw output and evidence preservation."""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any

from .completion import (
    completion_metadata,
    diagnostic_evidence,
    infer_document,
    require_complete_pages,
)
from .config import file_sha256
from .doctor import memory_snapshot, write_json
from .model import NativeModel
from .output import assess, convert_readable, normalize, preserve_raw

PRODUCTION_MAX_NEW_TOKENS = 8192


def validate_budget(max_new_tokens: int, policy_path: Path | None = None) -> None:
    """Production cap is 8192; larger budgets require the frozen P2-A policy."""
    if 1 <= max_new_tokens <= PRODUCTION_MAX_NEW_TOKENS:
        return
    if policy_path is None:
        raise ValueError(
            "max_new_tokens beyond the production cap requires the frozen "
            "experimental budget policy (production defaults are unchanged)"
        )
    try:
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Unreadable experimental budget policy: {error}") from error
    budgets = policy.get("budgets_new_tokens")
    if policy.get("schema") != 1 or not isinstance(budgets, list) or max_new_tokens not in budgets:
        raise ValueError(
            f"Requested budget {max_new_tokens} is not authorized by the frozen policy"
        )


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
    budget_policy: Path | None = None,
) -> dict[str, Any]:
    import torch
    from PIL import Image

    images = validate_inputs(images, multi)
    validate_budget(max_new_tokens, budget_policy)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh output directory to preserve earlier evidence")
    output.mkdir(parents=True, exist_ok=True)
    for path in images:
        with Image.open(path) as image:
            if image.width * image.height > 16_000_000:
                raise ValueError("Phase 1 fixture limit: 16 million pixels per page")
            image.verify()
    report: dict[str, Any] = {
        "native_entrypoint": "infer",
        "native_mode": "infer_multi_equivalent_base" if multi else "single_auto_crop",
        "completion_interpretation": (
            "Native EOS/length only; stop may be forced by the official page guard. "
            "Valid page structure does not prove semantic content completeness."
        ),
        "inputs": [{"path": str(path), "sha256": file_sha256(path)} for path in images],
        "memory_before": memory_snapshot(),
        "max_new_tokens": max_new_tokens,
        "model": native.evidence,
    }
    if budget_policy is not None:
        report["experimental_budget_policy_sha256"] = file_sha256(budget_policy)
    previous_kernels = len(native.kernels)
    previous_vision = len(getattr(native, "vision", []))
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        with torch.inference_mode():
            result = infer_document(
                native.model,
                native.tokenizer,
                [str(path) for path in images],
                multi=multi,
                max_new_tokens=max_new_tokens,
            )
        # Persist all available native evidence before metadata or Markdown
        # validation. Failure must not erase returned model text/finish fields.
        report["native_generation"], malformed_evidence = diagnostic_evidence(result)
        if malformed_evidence:
            report["native_generation_diagnostic"] = True
        if isinstance(result, dict):
            report["finish_reason"] = diagnostic_evidence(result.get("finish_reason"))[0]
            raw = result.get("text")
            if isinstance(raw, str):
                preserve_raw(output / "raw.txt", raw)
                report["raw_output"] = raw
                report["raw_sha256"] = file_sha256(output / "raw.txt")
        torch.cuda.synchronize()
        # Convert only after retaining raw; metadata validation below remains
        # independent of any best-effort readable representation.
        if isinstance(result, dict) and isinstance(result.get("text"), str):
            raw = result["text"]
            preserve_raw(output / "readable.md", convert_readable(raw))
        metadata = completion_metadata(
            result, expected_pages=len(images), max_new_tokens=max_new_tokens
        )
        raw, tokens, finish = result["text"], metadata["output_tokens"], metadata["finish_reason"]
        report["upstream_timing"] = {
            "generation_seconds": result["time"],
            "prefill_seconds": result["prefill_time"],
        }
        (output / "normalized.txt").write_text(normalize(raw), encoding="utf-8")
        kernels = native.kernels[previous_kernels:]
        report.update(
            output_token_count=tokens,
            finish_reason=finish,
            raw_sha256=file_sha256(output / "raw.txt"),
            raw_output=raw,
        )
        if len(kernels) < 12:
            raise RuntimeError("No successful native DASH kernel recorded for every decoder layer")
        if not raw.strip() or finish == "length":
            raise RuntimeError("Empty or native length-truncated OCR output")
        require_complete_pages(raw, len(images), max_chars=1_000_000)
        if truth is not None:
            report["correctness"] = assess(raw, truth)
            report["status"] = report["correctness"]["status"]
        else:
            report["status"] = "INFERENCE_ONLY"
    except KeyboardInterrupt:
        report["status"] = "FAIL"
        report["error"] = "INTERRUPTED"
        report["traceback"] = traceback.format_exc()
        raise
    except Exception:
        report["status"] = "FAIL"
        report["traceback"] = traceback.format_exc()
    finally:
        kernels = native.kernels[previous_kernels:]
        report["native_kernels"] = kernels
        report["native_kernel_count"] = len(kernels)
        report["vision_inputs"] = getattr(native, "vision", [])[previous_vision:]
        report["synchronized_wall_seconds"] = time.perf_counter() - start
        try:
            report["memory_after"] = memory_snapshot()
        except Exception:
            report["memory_after_error"] = traceback.format_exc()
        # Malformed native/diagnostic values must never break the FAIL manifest.
        write_json(output / "manifest.json", diagnostic_evidence(report)[0])
    return report
