"""Independent attention equations, not a backend emulator or OCR acceptance gate."""

import math

import numpy as np


def attention_reference(
    q, k, v, *, mask=None, scale=None, causal=False, dropout=0.0, dtype=np.float64, chunk_heads=8
):
    """Compute softmax(Q K^T * scale + mask) V on CPU in FP32 or FP64.

    Boolean masks use True=attend. Fully masked rows return zero, matching
    safe-softmax semantics. Dropout and grouped-query expansion are outside
    this frozen experiment; unsupported configurations fail closed.
    """
    if dropout != 0 or (causal and mask is not None):
        raise ValueError("Only zero dropout and unambiguous masks supported")
    if dtype not in (np.float32, np.float64) or chunk_heads < 1:
        raise ValueError("FP32/FP64 and positive chunk size required")
    q, k, v = (np.asarray(t, dtype=dtype) for t in (q, k, v))
    if (
        q.ndim != 4
        or k.ndim != 4
        or v.ndim != 4
        or q.shape[:2] != k.shape[:2]
        or k.shape[:3] != v.shape[:3]
        or q.shape[-1] != k.shape[-1]
    ):
        raise ValueError("Expected matched [batch, heads, sequence, dim] tensors")
    if not all(np.isfinite(t).all() for t in (q, k, v)):
        raise ValueError("Nonfinite inputs need a separately specified contract")
    b, h, length, dim = q.shape
    source = k.shape[-2]
    factor = dtype(1 / math.sqrt(dim) if scale is None else scale)
    bias = np.zeros((length, source), dtype=dtype)
    if causal:
        bias = np.where(np.tril(np.ones((length, source), dtype=bool)), bias, -np.inf)
    if mask is not None:
        mask = np.asarray(mask)
        bias = np.where(mask, dtype(0), -np.inf) if mask.dtype == bool else mask.astype(dtype)
    bias = np.broadcast_to(np.asarray(bias, dtype=dtype), (b, h, length, source)).reshape(
        b * h, length, source
    )
    qs, ks, vs = (t.reshape(b * h, *t.shape[-2:]) for t in (q, k, v))
    result = np.empty((b * h, length, v.shape[-1]), dtype=dtype)
    for start in range(0, b * h, chunk_heads):
        stop = start + chunk_heads
        scores = (qs[start:stop] @ ks[start:stop].swapaxes(-1, -2)) * factor
        scores = scores + bias[start:stop]
        maxima = scores.max(axis=-1, keepdims=True)
        safe_maxima = np.where(np.isneginf(maxima), dtype(0), maxima)
        weights = np.exp(scores - safe_maxima)
        sums = weights.sum(axis=-1, keepdims=True, dtype=dtype)
        probabilities = np.divide(weights, sums, out=np.zeros_like(weights), where=sums != 0)
        result[start:stop] = probabilities @ vs[start:stop]
    return result.reshape(b, h, length, v.shape[-1])


def error_summary(observed, reference):
    """Descriptive error only. No observed difference becomes a tolerance."""
    a, r = np.asarray(observed, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if a.shape != r.shape:
        raise ValueError("Shape mismatch")
    delta = a - r
    finite = np.isfinite(a) & np.isfinite(r)
    absolute = np.abs(delta[finite])
    nonzero = finite & (r != 0)
    relative = np.abs(delta[nonzero] / r[nonzero])
    changed = np.argwhere(a != r)
    norm = float(np.linalg.norm(r[finite]))
    return dict(
        elements=a.size,
        nonfinite_observed=int((~np.isfinite(a)).sum()),
        nonfinite_reference=int((~np.isfinite(r)).sum()),
        max_absolute=float(absolute.max()) if absolute.size else None,
        mean_absolute=float(absolute.mean()) if absolute.size else None,
        relative_l2=float(np.linalg.norm(delta[finite]) / norm) if norm else None,
        max_relative_nonzero=float(relative.max()) if relative.size else None,
        mean_relative_nonzero=float(relative.mean()) if relative.size else None,
        absolute_quantiles={
            str(p): float(np.quantile(absolute, p)) for p in (0, 0.5, 0.9, 0.99, 0.999, 1)
        }
        if absolute.size
        else {},
        first_differing_coordinate=changed[0].tolist() if changed.size else None,
        exact_differing_elements=int((a != r).sum()),
    )
