"""Device-agnostic torch fallback for sglang's ``moe_align_block_size``.

SGLang B1 root cause (delivery-20261101/P2_SGLANG_B1_RCA.md): the frozen
Baidu wheel (sglang 0.5.20.dev10+g8d62f5fb9c) resolves
``moe.moe_align_block_size`` through the AOT backend to the ``sgl_kernel``
wheel, which ships CUDA/C++ binaries only (no ROCm build) — the first
failing call of the B1 serving path, and a pure index-bookkeeping op with
no HIP-specific semantics.

This module reimplements the exact reference semantics of the frozen AOT
CUDA kernel (python/sglang/kernels/aot/csrc/moe/moe_align_kernel.cu at
sglang v0.5.20, the public reference for the 0.4.7-class binary), as
called by the triton_utils caller with ``pad_sorted_token_ids=True``:

- "+1 offset" convention: bucket = expert_id + 1; bucket 0 collects
  EP-filtered/invalid ids and its blocks carry expert_ids = -1 (skipped by
  fused_moe's filter_expert);
- every bucket is padded up to a ``block_size`` multiple, offsets in bucket
  order;
- ``sorted_token_ids`` holds flat (token, slot) indices; pad slots hold
  ``numel`` (the whole buffer when ``pad_sorted_token_ids``, otherwise only
  the published region [0, total));
- ``num_tokens_post_pad`` = total padded length; ``expert_ids`` is written
  only below ``total // block_size`` — beyond that the caller's allocation
  stays untouched (consumers read nothing above the published total);
- ``cumsum_buffer`` ends as per-bucket INCLUSIVE ends with
  ``cumsum[num_experts] = total`` (the CUDA count_and_sort pass advances
  the prefix cursors in place).

Intentional deviations, invisible to the fused_moe consumer:
- intra-bucket order is STABLE in flat index (the reference's atomicAdd
  order is scheduling-dependent; consumers place rows per block);
- out-of-range expert ids raise ValueError instead of the reference's
  undefined behavior;
- with pad_sorted_token_ids=False the module fills [0, total) with numel
  while the reference leaves pad slots unwritten (the frozen caller always
  passes True); the small-batch regime the reference short-circuits
  (numel < 1024 and experts <= 64, where its cumsum_buffer is never
  written) is not special-cased here — the frozen caller's buffers make
  the difference unobservable and B1 (257 buckets) never enters it.

Performance note: this is a CORRECTNESS fallback for bounded B1 smoke
validation, not a benchmark identity. It adds one host synchronization
(the range check) per call; a HIP AOT/Triton kernel remains the perf path.
"""

from __future__ import annotations

import torch


def torch_moe_align_block_size(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    cumsum_buffer: torch.Tensor,
    pad_sorted_token_ids: bool = False,
    ignore_invalid_expert: bool = False,
) -> None:
    """In-place implementation of the frozen AOT contract (all writes land in
    the caller-provided output buffers; nothing is returned)."""
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    flat = topk_ids.reshape(-1).to(torch.long)
    numel = flat.numel()
    if ignore_invalid_expert:
        ids = flat[flat >= 0] + 1
    else:
        ids = flat + 1
    device = topk_ids.device
    if bool((ids < 0).any().item()) or bool((ids >= num_experts).any().item()):
        raise ValueError(
            "topk_ids contains expert ids outside [-1, num_experts-2] after the "
            f"+1 offset mapping (num_experts={num_experts}, "
            "ignore_invalid_expert="
            f"{ignore_invalid_expert}); the CUDA reference is undefined here "
            "and this fallback refuses silently-wrong routing"
        )

    counts = torch.bincount(ids, minlength=num_experts)
    padded = (counts + (block_size - 1)) // block_size * block_size
    prefix = torch.zeros(num_experts + 1, dtype=torch.long, device=device)
    torch.cumsum(padded, 0, out=prefix[1:])
    total = int(prefix[-1].item())

    # Stable intra-bucket placement: sort (bucket, flat index) pairs. `order`
    # indexes the FILTERED id array, so map it back through the surviving flat
    # positions to place original (token, slot) indices.
    order = torch.argsort(ids, stable=True)
    flat_index = (
        torch.nonzero(flat >= 0, as_tuple=False).reshape(-1)
        if ignore_invalid_expert
        else torch.arange(numel, device=device)
    )
    sorted_buckets = ids[order]
    group_start = torch.cumsum(counts, 0) - counts  # first sorted-index per bucket
    ranks = torch.arange(sorted_buckets.numel(), device=device) - group_start[sorted_buckets]
    positions = prefix[:-1][sorted_buckets] + ranks

    if pad_sorted_token_ids:
        sorted_token_ids.fill_(numel)
    elif total > 0:
        sorted_token_ids[:total].fill_(numel)
    sorted_token_ids[positions] = flat_index[order].to(torch.int32)

    num_blocks = total // block_size
    if num_blocks > 0:
        block_index = torch.arange(num_blocks, device=device)
        bucket_of_block = torch.searchsorted(prefix, block_index * block_size, right=True) - 1
        expert_ids[:num_blocks] = (bucket_of_block - 1).to(torch.int32)

    num_tokens_post_pad.fill_(total)
    # The CUDA count_and_sort pass leaves cumsum_buffer as inclusive ends.
    cumsum_buffer[:num_experts] = (prefix[:-1] + counts).to(torch.int32)
    cumsum_buffer[num_experts] = total
