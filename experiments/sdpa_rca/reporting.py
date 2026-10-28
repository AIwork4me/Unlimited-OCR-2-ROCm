"""Shared evidence helpers; torch is imported only for explicit reference arithmetic."""

import hashlib
import json
import math
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize(text, home):
    for prefix in (
        home.replace("\\", "\\\\"),
        home,
        home.replace("\\", "/"),
        home.replace("/", "\\"),
    ):
        text = text.replace(prefix, "<USERPROFILE>")
    return text


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def classify(stats, hip_error):
    if hip_error or (stats and (stats["failing_elements"] or stats["nonfinite_actual"])):
        return "FAIL"
    return "PASS" if stats else "INCOMPLETE"


def hip_failed(events):
    """Inspect the entire recorded sequence, including downstream checkpoints."""
    return any(
        event.get("hip_before_sync") or event.get("hip_after_sync") or event.get("traceback")
        for event in events
    )


def audit_ab_provenance(left, right):
    """Reject a purported images-only A/B if a recorded core component changed."""
    checked = ["packages", "architecture", "system"]
    for field in checked:
        if left[field] != right[field]:
            raise ValueError(f"A/B changed {field}")
    if left["runtime_header"]["sha256"] != right["runtime_header"]["sha256"]:
        raise ValueError("A/B changed runtime_header")

    def modules(report):
        return {
            p["path"].replace("\\", "/").split("/")[-1].lower(): p["sha256"]
            for p in report["loaded_modules"]
        }

    if modules(left) != modules(right):
        raise ValueError("A/B changed loaded_modules")
    return {
        "unchanged_fields": [*checked, "runtime_header", "loaded_modules"],
        "unchanged_module_sha256": modules(left),
        "only_image_payload_changed": left["image_files"] != right["image_files"],
    }


def observed_backend(operators):
    mapping = {
        "_scaled_dot_product_attention_math": "MATH",
        "_scaled_dot_product_flash_attention": "FLASH_ATTENTION",
        "_scaled_dot_product_efficient_attention": "EFFICIENT_ATTENTION",
        "_scaled_dot_product_cudnn_attention": "CUDNN_ATTENTION",
    }
    found = {value for name, value in mapping.items() if any(name in op for op in operators)}
    return next(iter(found)) if len(found) == 1 else "UNOBSERVED"


def reference_attention(q, k, v, mask, causal, scale):
    """CPU FP32 reference on exactly the quantized fixture sent to the GPU."""
    import torch

    if any(t.device.type != "cpu" for t in (q, k, v)):
        raise ValueError("Independent reference requires CPU tensors")
    scores = q.float() @ k.float().transpose(-1, -2)
    scores *= (1 / math.sqrt(q.shape[-1])) if scale is None else scale
    if causal:
        allowed = torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool).tril()
        scores.masked_fill_(~allowed, -float("inf"))
    if mask is not None:
        if mask.dtype == torch.bool:
            scores.masked_fill_(~mask, -float("inf"))
        else:
            scores += mask.float()
    # SDPA defines an entirely masked row as zero, rather than NaN.
    probabilities = torch.softmax(scores, dim=-1).nan_to_num(0.0)
    return probabilities @ v.float()


def numerical_stats(actual, reference, atol, rtol):
    import torch

    actual, reference = actual.float().cpu(), reference.float().cpu()
    if actual.shape != reference.shape:
        raise ValueError("Comparison shape mismatch")
    finite = torch.isfinite(actual) & torch.isfinite(reference)
    error = (actual - reference).abs()
    failures = ~finite | (error > atol + rtol * reference.abs())
    finite_error = error[finite]
    relative = error / reference.abs().clamp_min(1e-6)
    return {
        "atol": atol,
        "rtol": rtol,
        "relative_denominator_floor": 1e-6,
        "max_abs_error_finite": finite_error.max().item() if finite_error.numel() else None,
        "mean_abs_error_finite": finite_error.mean().item() if finite_error.numel() else None,
        "max_relative_error_finite": relative[finite].max().item()
        if finite_error.numel()
        else None,
        "failing_elements": failures.sum().item(),
        "total_elements": actual.numel(),
        "failing_percent": 100 * failures.sum().item() / actual.numel(),
        "nonfinite_actual": (~torch.isfinite(actual)).sum().item(),
        "nan_actual": torch.isnan(actual).sum().item(),
        "inf_actual": torch.isinf(actual).sum().item(),
        "nonfinite_reference": (~torch.isfinite(reference)).sum().item(),
    }
