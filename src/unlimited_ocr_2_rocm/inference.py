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
#: Pinned digest of the frozen P2-B policy (eval/experiments/p2b-ngram-suppression/
#: policy.json). The gate compares the presented file against this digest, so a
#: locally forged schema-1 policy cannot authorize suppression
#: (docs/reviews/POLICY_IDENTITY_AUDIT.md hardening requirement).
EXPECTED_NGRAM_POLICY_SHA256 = "0095b0c47da9a693d6bef70305a4385b986ea79aac129d628a500dac0ef8e1f5"


def validate_ngram(ngram_size: int, ngram_window: int, policy_path: Path | None) -> None:
    """No-repeat n-gram suppression is opt-in via the frozen P2-B policy only.

    The window must be passed explicitly and positively: the frozen model
    silently coerces a non-positive window to 100 (modeling_unlimitedocr.py
    infer()), which is NOT the official 1024 (docs/rca/NGRAM35_SOURCE_AUDIT.md).
    """
    if ngram_size == 0 and ngram_window == 0:
        return
    if policy_path is None:
        raise ValueError(
            "no_repeat_ngram suppression requires the frozen experimental "
            "n-gram policy (production default is suppression off)"
        )
    if ngram_size <= 0 or ngram_window <= 0:
        raise ValueError(
            "Repetition suppression requires an explicit positive ngram_size and "
            "ngram_window from the frozen policy (a non-positive window would be "
            "silently coerced to 100 by the frozen model, not the official 1024)"
        )
    try:
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Unreadable experimental n-gram policy: {error}") from error
    if file_sha256(policy_path) != EXPECTED_NGRAM_POLICY_SHA256:
        raise ValueError(
            "Experimental n-gram policy digest does not match the pinned frozen identity"
        )
    sizes = policy.get("ngram_sizes") if isinstance(policy, dict) else None
    if (
        policy.get("schema") != 1
        or policy.get("experiment") != "P2B_NGRAM_SUPPRESSION_RCA"
        or not isinstance(sizes, list)
        or ngram_size not in sizes
        or policy.get("ngram_window") != ngram_window
    ):
        raise ValueError(
            f"Requested n-gram configuration {ngram_size}/{ngram_window} is not "
            "authorized by the frozen policy"
        )


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
    no_repeat_ngram_size: int = 0,
    ngram_window: int = 0,
    ngram_policy: Path | None = None,
) -> dict[str, Any]:
    import torch
    from PIL import Image

    images = validate_inputs(images, multi)
    validate_budget(max_new_tokens, budget_policy)
    validate_ngram(no_repeat_ngram_size, ngram_window, ngram_policy)
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
    if no_repeat_ngram_size > 0 and ngram_policy is not None:
        # validate_ngram already refused suppression without the policy.
        report["no_repeat_ngram_size"] = no_repeat_ngram_size
        report["ngram_window"] = ngram_window
        report["experimental_ngram_policy_sha256"] = file_sha256(ngram_policy)
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
                no_repeat_ngram_size=no_repeat_ngram_size,
                ngram_window=ngram_window,
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
