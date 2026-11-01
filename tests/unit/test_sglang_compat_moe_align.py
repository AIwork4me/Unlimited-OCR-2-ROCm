"""CPU differential tests for the sglang B1 moe_align_block_size torch fallback.

The reference semantics are the frozen AOT CUDA kernel at sglang v0.5.20
(python/sglang/kernels/aot/csrc/moe/moe_align_kernel.cu) as called by
sglang 0.5.20.dev10's triton_utils caller with pad_sorted_token_ids=True:
- "+1 offset" convention: bucket = expert_id + 1 (bucket 0 = EP-filtered/invalid,
  its blocks carry expert_ids = -1 which fused_moe's filter_expert skips);
- every bucket padded up to a block_size multiple, offsets in bucket order;
- sorted_token_ids holds flat (token, slot) indices; pad slots (both inside
  [0, total) and the whole buffer when pad_sorted_token_ids) hold `numel`;
- num_tokens_post_pad = total padded length; expert_ids written only below
  total // block_size (beyond that the reference leaves the caller's
  allocation untouched);
- cumsum_buffer ends as per-bucket INCLUSIVE ends with cumsum[num_experts] =
  total (the count_and_sort pass advances the prefix cursors in place).

Known intentional deviations of the torch fallback (documented in the module):
- intra-bucket order is stable in flat index (the CUDA reference's atomicAdd
  order is scheduling-dependent; consumers place rows by block, not order);
- out-of-range expert ids raise instead of the reference's undefined behavior.
"""

from __future__ import annotations

from collections import Counter

import pytest
import torch

from unlimited_ocr_2_rocm.sglang_compat.moe_align import torch_moe_align_block_size


def reference(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    max_num_tokens_padded: int,
    max_num_m_blocks: int,
    pad_sorted_token_ids: bool,
    ignore_invalid_expert: bool,
):
    """Straightforward loop reference written independently of the module."""
    flat = [int(v) for v in topk_ids.reshape(-1).tolist()]
    numel = len(flat)
    buckets = [v + 1 for v in flat if not (ignore_invalid_expert and v < 0)]
    assert all(0 <= b < num_experts for b in buckets)
    counts = Counter(buckets)
    padded = [((counts[b] + block_size - 1) // block_size) * block_size for b in range(num_experts)]
    prefix = [0] * (num_experts + 1)
    for b in range(num_experts):
        prefix[b + 1] = prefix[b] + padded[b]
    total = prefix[num_experts]

    sorted_ids = [numel] * (max_num_tokens_padded if pad_sorted_token_ids else total)
    cumsum = prefix[:-1] + [prefix[num_experts]]  # cursors start at prefix
    placed: dict[int, list[int]] = {b: [] for b in range(num_experts)}
    for i, v in enumerate(flat):
        if ignore_invalid_expert and v < 0:
            continue
        b = v + 1
        sorted_ids[cumsum[b]] = i
        cumsum[b] += 1
        placed[b].append(i)
    # cursor advanced = inclusive ends
    cumsum_final = [prefix[b] + counts.get(b, 0) for b in range(num_experts)] + [total]

    expert_ids = [None] * max_num_m_blocks
    num_blocks = total // block_size
    for i in range(num_blocks):
        block_start = i * block_size
        left = 0
        right = num_experts
        while left < right:
            mid = (left + right) // 2
            if prefix[mid] <= block_start:
                left = mid + 1
            else:
                right = mid
        expert_ids[i] = left - 2
    return sorted_ids, expert_ids, total, cumsum_final


def run_module(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    max_num_tokens_padded: int,
    max_num_m_blocks: int,
    pad_sorted_token_ids: bool,
    ignore_invalid_expert: bool,
):
    sorted_ids = torch.full((max_num_tokens_padded,), -123, dtype=torch.int32)
    expert_ids = torch.full((max_num_m_blocks,), -123, dtype=torch.int32)
    num_tokens_post_pad = torch.full((1,), -123, dtype=torch.int32)
    cumsum = torch.full((num_experts + 1,), -123, dtype=torch.int32)
    torch_moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        cumsum,
        pad_sorted_token_ids=pad_sorted_token_ids,
        ignore_invalid_expert=ignore_invalid_expert,
    )
    return sorted_ids, expert_ids, int(num_tokens_post_pad.item()), cumsum


@pytest.mark.parametrize("block_size", [1, 4, 16, 64])
@pytest.mark.parametrize("num_experts", [9, 65, 257])  # caller passes E+1 buckets
@pytest.mark.parametrize("ignore_invalid", [False, True])
def test_differential_against_loop_reference(block_size, num_experts, ignore_invalid):
    gen = torch.Generator().manual_seed(hash((block_size, num_experts, ignore_invalid)) % 2**31)
    tokens, topk = 13, 6
    ids = torch.randint(-1, num_experts - 1, (tokens, topk), generator=gen, dtype=torch.int64)
    numel = ids.numel()
    cap = numel + num_experts * (block_size - 1)
    m_blocks = (cap + block_size - 1) // block_size

    exp_sorted, exp_expert, exp_total, exp_cumsum = reference(
        ids, num_experts, block_size, cap, m_blocks, True, ignore_invalid
    )
    got_sorted, got_expert, got_total, got_cumsum = run_module(
        ids, num_experts, block_size, cap, m_blocks, True, ignore_invalid
    )
    assert got_total == exp_total
    assert got_sorted.tolist() == exp_sorted
    # expert_ids below num_blocks must match; ABOVE it the reference leaves the
    # caller's allocation untouched — the module must not write there either.
    nb = exp_total // block_size
    assert got_expert.tolist()[:nb] == exp_expert[:nb]
    assert all(v == -123 for v in got_expert.tolist()[nb:])
    assert got_cumsum.tolist() == exp_cumsum
    # intra-bucket content matches as multisets (order deviation is documented)
    for b in range(num_experts):
        lo, hi = exp_cumsum[b - 1] if b else 0, exp_cumsum[b]
        assert sorted(got_sorted.tolist()[lo:hi]) == sorted(exp_sorted[lo:hi])


def test_pad_false_fills_only_the_published_region():
    ids = torch.tensor([[0, 2], [1, 2], [0, -1]], dtype=torch.int64)
    numel, ne, bs = ids.numel(), 4, 4
    cap = numel + ne * (bs - 1)
    got_sorted, _, total, _ = run_module(ids, ne, bs, cap, cap // bs, False, False)
    # buckets 0..3 carry counts 1,2,1,2 -> four padded blocks of 4
    assert total == 4 * bs
    # published region holds no poison; beyond it the caller's buffer is untouched
    published = got_sorted.tolist()[:total]
    assert -123 not in published
    assert all(v == -123 for v in got_sorted.tolist()[total:])


def test_invalid_bucket_zero_carries_expert_id_minus_one():
    ids = torch.tensor([[-1, -1], [-1, 0]], dtype=torch.int64)
    ne, bs = 3, 4
    cap = ids.numel() + ne * (bs - 1)
    got_sorted, got_expert, total, cumsum = run_module(ids, ne, bs, cap, cap // bs, True, False)
    # bucket 0 (the two -1 ids) owns the first block -> expert_ids[0] == -1
    assert got_expert.tolist()[0] == -1
    assert total == 2 * bs
    # inclusive end of bucket 0 = prefix 0 + 3 placed tokens (NOT the padded 4 —
    # the CUDA count_and_sort pass advances the cursor once per placed token)
    assert cumsum.tolist()[0] == 3
    # bucket 0's block: tokens 0,1,2 then pad slot holding numel=4;
    # bucket 1's single token (flat idx 3) sits at position 4
    assert got_sorted.tolist()[:5] == [0, 1, 2, 4, 3]


def test_ignore_invalid_expert_drops_invalid_entries():
    ids = torch.tensor([[-1, 0], [1, -1]], dtype=torch.int64)
    ne, bs = 3, 4
    cap = ids.numel() + ne * (bs - 1)
    _, _, total, cumsum = run_module(ids, ne, bs, cap, cap // bs, True, True)
    # buckets 1 and 2 hold one token each -> two padded blocks
    assert total == 2 * bs
    assert cumsum.tolist()[0] == 0  # bucket 0 unused when invalid ids are dropped


def test_out_of_range_expert_ids_raise_instead_of_undefined_behavior():
    ids = torch.tensor([[5]], dtype=torch.int64)  # bucket 6 out of range for ne=4
    sorted_ids = torch.empty((16,), dtype=torch.int32)
    expert_ids = torch.empty((4,), dtype=torch.int32)
    ntp = torch.empty((1,), dtype=torch.int32)
    cumsum = torch.empty((5,), dtype=torch.int32)
    with pytest.raises(ValueError):
        torch_moe_align_block_size(
            ids, 4, 4, sorted_ids, expert_ids, ntp, cumsum, pad_sorted_token_ids=True
        )


def test_caller_capacity_contract_shapes():
    """The sglang triton_utils caller allocates exactly these shapes."""
    for tokens, topk, ne, bs in ((1, 8, 257, 32), (33, 6, 129, 16), (1, 1, 17, 1)):
        ids = torch.randint(0, ne - 1, (tokens, topk), dtype=torch.int64)
        numel = ids.numel()
        if numel < ne:
            cap = numel * bs
        else:
            cap = numel + ne * (bs - 1)
        m_blocks = (cap + bs - 1) // bs
        got_sorted, got_expert, total, cumsum = run_module(ids, ne, bs, cap, m_blocks, True, False)
        assert got_sorted.numel() == cap and got_expert.numel() == m_blocks
        assert cumsum.numel() == ne + 1
        assert total % bs == 0 and total <= cap
