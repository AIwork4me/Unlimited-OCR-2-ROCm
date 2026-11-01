"""CPU tests for the PDL support decision (B1 ring #3)."""

from __future__ import annotations

import pytest

from unlimited_ocr_2_rocm.sglang_compat.pdl import pdl_supported


@pytest.mark.parametrize(
    "major,hip,expected",
    [
        (11, True, False),  # gfx1151 reports major 11 via torch.cuda — no PDL on ROCm
        (9, False, True),  # NVIDIA Hopper: upstream rule
        (8, False, False),  # pre-Hopper: upstream rule
        (9, True, False),  # HIP never gets the kwarg
    ],
)
def test_pdl_decision(major, hip, expected):
    assert pdl_supported(major, hip) is expected
