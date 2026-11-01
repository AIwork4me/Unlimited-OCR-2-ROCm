"""PDL support decision for the DASH Triton kernels on ROCm.

Root cause #3 of the B1 HIP cascade: the frozen wheel's
``unlimited_dash_ops.common._compute_pdl_support()`` treats "CUDA compute
capability major >= 9" as "launch_pdl supported". On gfx1151,
``torch.cuda.get_device_capability()`` reports major 11, so the wheel passes
``launch_pdl=True`` to Triton — which ROCm Triton rejects outright
(``KeyError: 'Keyword argument launch_pdl was specified but unrecognised'``).
PDL (programmatic dependent launch) is an NVIDIA SM90+ feature; on HIP the
``USE_PDL=False`` path is the ordinary non-Hopper path, fully supported
upstream.

This module holds the decision as a pure function so it is testable on CPU:
HIP → False; NVIDIA → the upstream SM90+ rule.
"""

from __future__ import annotations


def pdl_supported(compute_capability_major: int, is_hip: bool) -> bool:
    """Upstream-faithful PDL gate with the HIP correction."""
    if is_hip:
        return False
    return compute_capability_major >= 9
