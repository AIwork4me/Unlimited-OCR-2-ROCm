"""Numerical summaries and strict input contracts; imports no torch at module load."""

from __future__ import annotations

import hashlib

INPUT_KEYS = ("shape", "stride", "dtype", "device", "storage_offset", "sha256")


def require_same_inputs(*rows):
    if not rows:
        raise ValueError("No input evidence")
    try:
        expected = {k: rows[0][k] for k in INPUT_KEYS}
        if expected["dtype"] != "torch.bfloat16" or expected["device"] != "cuda:0":
            raise ValueError("Original GPU BF16 required")
        if any({k: row[k] for k in INPUT_KEYS} != expected for row in rows[1:]):
            raise ValueError("Input bytes/layout/dtype/device differ")
    except KeyError as error:
        raise ValueError("Incomplete input identity") from error


def graph_without_locations(graph):
    return "\n".join(line.split(" #", 1)[0] for line in graph.splitlines()) + "\n"


def require_reduction(result):
    if (
        result.get("hip_before", {}).get("status") != "PASS"
        or result.get("hip_after", {}).get("status") != "PASS"
        or result.get("fallback_allowed") is not False
        or result.get("cpu_original_exact") is not True
        or result.get("cpu_emitted_coefficient_exact") is not True
    ):
        raise ValueError("Reduction HIP/reference/fallback qualification failed")


def tensor_hash(value):
    import torch

    return hashlib.sha256(
        value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def difference(left, right):
    import torch

    if left.shape != right.shape or left.dtype != right.dtype or left.dtype != torch.bfloat16:
        raise ValueError("Comparison shape/dtype differs")
    a, b = left.detach().cpu().contiguous(), right.detach().cpu().contiguous()
    changed = (a.view(torch.int16) != b.view(torch.int16)).flatten().nonzero().flatten()
    delta = b.float() - a.float()
    finite = torch.isfinite(delta)
    values = delta[finite]
    first = int(changed[0]) if len(changed) else None
    coordinates = []
    if first is not None:
        remainder = first
        for dimension in reversed(a.shape):
            coordinates.append(remainder % dimension)
            remainder //= dimension
        coordinates.reverse()
    return dict(
        elements=a.numel(),
        different_elements=changed.numel(),
        left_sha256=tensor_hash(a),
        right_sha256=tensor_hash(b),
        max_abs=float(values.abs().max()) if len(values) else None,
        mean_abs=float(values.abs().mean()) if len(values) else None,
        min_signed=float(values.min()) if len(values) else None,
        max_signed=float(values.max()) if len(values) else None,
        first_differing_flat_index=first,
        first_differing_coordinate=coordinates,
        first_signed_delta=float(delta.flatten()[first])
        if first is not None and finite.flatten()[first]
        else None,
        nonfinite_left=int((~torch.isfinite(a)).sum()),
        nonfinite_right=int((~torch.isfinite(b)).sum()),
    )
