# Next upstream action

No upstream PR or issue is created by this validation mission.

The current source pin alone does not establish that the packaging defect is
fixed. Complete the Windows source-wheel build and fresh-install provenance
checks, then run the same frozen 27-process matrix on this physical gfx1151 GPU.
Preserve all failed build attempts and use the existing isolated environments.

If the artifact earns **FIX_VERIFIED**, the recommended next repository is
**ROCm/TheRock**: add a packaged-wheel SDPA regression gate that checks both actual
fused routes, runtime/image content, CPU-reference tolerances, non-clearing HIP
error state and downstream operations. Keep CPU hosted CI separate from physical
Windows GPU acceptance. A source-built pass does not validate a published AMD wheel.

If a correctly installed source-built artifact reproduces the failure, determine
the first fault and its loaded binary/image contract before proposing a minimal
fix in ROCm/pytorch, ROCm/TheRock packaging, or ROCm/aotriton as the evidence directs.
Do not infer ownership from a changed commit alone, and do not deploy image swaps.

While the current artifact is unvalidated, retain the production scoped GPU MATH
adapter. Its production implementation is untouched by this PR. SAM/model/PDF
cross-checks are deferred until the torch-only fused acceptance passes.
