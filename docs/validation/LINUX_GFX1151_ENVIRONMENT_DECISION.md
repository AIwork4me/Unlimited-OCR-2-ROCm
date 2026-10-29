# Linux gfx1151 compatibility and environment decision

Date: 2026-10-29 (Asia/Shanghai). Decides the native Linux ROCm environment for
issue #6 physical validation on the newly available
**Ryzen AI MAX+ PRO 395 / Radeon 8060S (gfx1151)** Linux host, which becomes
the **primary Linux target**. Radeon PRO W7900 / gfx1100 is retained as a
*future optional independent hardware-compatibility target* only (see
ADAPTATION_MATRIX.md); no gfx1100 claim derives from this host.

## Decision

| Question | Decision | Rationale |
| --- | --- | --- |
| Use the reviewed Windows candidate wheel on Linux? | **No** | `torch-2.12.0+rocm7.14.1-cp310-cp310-win_amd64.whl` is a Windows-only, locally source-built artifact. It must not be installed on Linux and no Linux evidence may cite it. |
| Upgrade the system ROCm 7.2.1 userland / kernel / drivers? | **No** | Read-only-host constraint; not needed — the officially published Linux `+rocm7.14` wheel bundles its ROCm/HIP runtime and needs only the inbox amdgpu kernel driver, which the matrix supports for this kernel. |
| PyTorch source | Officially published **`torch-2.12.0+rocm7.2`** cp310 manylinux wheel from `https://download.pytorch.org/whl/rocm7.2` | Same torch 2.12.0 line as the Windows candidate (comparable semantics, different binary), first-listed PyTorch in the ROCm 7.14.1 matrix, bundles its own ROCm/HIP runtime so the untouched system stack suffices. |
| Triton source | **`triton-rocm==3.7.0`** (cp310, module `triton`) from the same official index | Official Linux Triton required by the torch wheel; same 3.7 Triton line as Windows' `triton-windows==3.7.1.post27`. |
| Rejected alternative | `torch-2.14.1+rocm7.14` (downloaded, SHA-verified, then discarded) | Its `rocm==7.14.*` + `triton-rocm~=3.8.0` dependencies are published only on AMD's own pip index, unreachable from this host; not installable offline-from-PyPI. |
| Python | CPython **3.10** (venv) | Repository-supported interpreter (`requires-python >=3.10,<3.11`); matches the Windows Phase-1 evidence's interpreter so fixture/CLI behavior is comparable. Deviation from the ROCm 7.14.1 matrix's example list (3.11–3.14) recorded below. |
| Environment location | `../envs/linux-gfx1151-rocm` venv outside the repository; model cache `../hf-cache` | No global packages touched; no repo-tree pollution; retained short paths per the runtime-cache convention. |
| Attention/precision policy | **MATH default** for vision SDPA (scoped wrapper), BF16 parameters, eager decoder attention with original DASH Triton enabled | Unchanged reviewed semantics; no AUTO switch; zero silent fallback. |
| torchvision / torchaudio | Not installed on the Linux native profile | Not in the native model dependency set (`requirements/native.txt`); avoids unverified extra GPU components. |

## Pinned identities

| Artifact | Identity |
| --- | --- |
| `torch-2.12.0+rocm7.2-cp310-cp310-manylinux_2_28_x86_64.whl` (6,176,896,129 B, runtime bundled) | SHA-256 `099e265de73094cb6a2112ce9d7940c59bae0c884e9516f16719e392a5471df6` |
| `triton_rocm-3.7.0-cp310-cp310-linux_x86_64.whl` (315,212,279 B) | SHA-256 `b7a9e90ceae67668202958f7b180d51421df441dc73ae100b89cf0aa19e0b71e` |
| model snapshot | `baidu/Unlimited-OCR-2` @ `9cf077c4b206e0e2a31879df937b0b0bcba3661c`, all file/weight SHA-256 per `src/unlimited_ocr_2_rocm/upstream-lock.json` |
| native deps | `requirements/native.txt` unchanged (transformers 4.52.1, huggingface-hub 0.36.0, tokenizers 0.21.4, …) |
| full environment freeze | `requirements/linux-cp310.lock.txt` (generated from the installed venv, recorded with the run evidence) |

Runtime identities reported by the installed wheel (recorded by `doctor`):
`torch.version.hip`, `torch.version.rocm`, arch list including gfx1151 — see
the doctor manifest linked in the validation report. Acceptance is *physical*:
the doctor profile, BF16 probe, DASH/Triton kernel checks and page-structure
contract must all pass on this host; matrix listing alone proves nothing.

## Known deviations and residual risks (explicit)

1. **Wheel-line choice within the matrix.** AMD's docs-7.14.1 compatibility
   matrix lists PyTorch 2.12.0/2.11.0/2.10.0; we pin the officially published
   torch 2.12.0+rocm7.2 cp310 wheel (first-listed matrix version, and the same
   2.12.0 line as the reviewed Windows candidate). The published 7.14-line
   wheels (2.14.x) are newer than the matrix list and depend on AMD-index-only
   packages; they were downloaded, hash-verified, and set aside. Wheel bytes
   are the identity, not the version string; everything is re-validated
   physically regardless.
2. **System ROCm userland (7.2.1) vs wheel runtime (7.2).** The wheel bundles
   its own ROCm/HIP 7.2 libraries; system `rocminfo` (7.2.1) is used solely
   for host enumeration and matches the same 7.2 generation. Any future global
   upgrade is out of scope.
3. **Triton artifacts differ across platforms.** Linux uses triton-rocm 3.7.0;
   Windows uses triton-windows 3.7.1.post27 (same 3.7 line, separately built
   and pinned). DASH kernel identity is
   pinned by the frozen upstream source hashes (`dash_attention.py`,
   `sparse_gather_e2e_triton.py` per the upstream lock), which are identical on
   both platforms — kernel *source* parity holds; kernel *compiler/runtime*
   differs by design and is validated per platform.
4. **APU shared memory.** All reported GPU memory figures are unified-memory
   accounting (32 GiB carve-out + GTT up to 100 GiB), not dedicated VRAM.
5. **Network egress constraint.** `huggingface.co` unreachable directly; the
   frozen-revision snapshot is fetched via `HF_ENDPOINT=https://hf-mirror.com`
   with post-download SHA-256 verification against the upstream lock. The
   mirror is a transport only; identity is hash-verified.

## What this decision does NOT authorize

- No claim that Windows evidence (Phase 1, PR #5 draft, wheel #4) transfers to
  Linux, or vice versa.
- No gfx1100/W7900 support claim of any kind.
- No cap removal, no SGLang substitution, no production AUTO switch, no
  release or upstream contact.
