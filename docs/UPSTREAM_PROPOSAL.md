# Prepared upstream compatibility report

This proposal is prepared locally; no upstream issue/comment/PR was posted.

## PyTorch / TheRock vision SDPA regression

Native Windows 11 build26200, Ryzen AI MAX+395, Radeon8060S/gfx1151,
driver32.0.22032.6002, CPython3.10.20. Exact required packages:
torch2.12.0+rocm7.14.1, torchvision0.27.0+rocm7.14.1,
torchaudio2.11.0+rocm7.14.1; torch.version.hip7.14.60850.

The official Unlimited OCR2 weights load correctly. Its SAM encoder fails
at a subsequent permute/contiguous after default SDPA. Synchronizing first
does not eliminate the error. A plain copy loop with the same layout passes.
HIP peek tracing isolates error1 immediately after default SDPA. Its result
also fails an independent FP32 attention reference.

Reproduce the reduced SAM/SDPA case without model weights:

```powershell
python scripts/probe_vision_sdpa.py --source "C:\reviewed pinned model source" --mode default --output "default-failure.json"
python scripts/probe_vision_sdpa.py --source "C:\reviewed pinned model source" --mode math --output "math-reference.json"
```

The probe verifies the downloaded official source hashes before importing
deepencoder.py. It uses three forwards of one random-weight SAM block,
B6/H40/W40/C768, 12heads, window14, then a BF16 attention reference with
shape[54,12,196,64]. MATH passes and restores prior attention settings.
Neither test silently clears HIP errors.

Related history: [TheRock#7992](https://github.com/ROCm/TheRock/issues/7992)
reports the same wheel-family symptom on another architecture. Our evidence
adds physical gfx1151 data; it does not establish support on that other GPU.
Investigate/backport the relevant SDPA/kernel-loading correction while keeping
the requested public package identity/provenance explicit. This project did
not downgrade, replace wheels or patch DLLs.

## Baidu compatibility documentation / optional backend selection

Suggest documenting a model-local vision SDPA backend option for affected
Windows ROCm stacks. Default behavior on other platforms can remain unchanged.
Our workaround wraps only SAM and CLIP forward in the public GPU MATH SDPA
context. It preserves learned attention, BF16 tensors, official weights and
the decoder's native DASH/Triton path; no global persistent backend change.
The compatibility manifest names this adaptation explicitly.

The native generation guard can produce one empty terminal PAGE after all
correct content. Documentation should distinguish raw delimiters from
content pages and explain that the V2 block parser skips the final empty
segment. We preserve raw output and do not change the generation algorithm.

Use the official utils/process_unlimitedOCR2.py for V2 block output. The HF
legacy drawing helper expects older ref/det markers and uses eval; avoid
calling it on untrusted generated coordinates. Any security patch to that
helper should use bounded non-executing parsing with regression coverage.
