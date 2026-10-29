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
| PyTorch source | Officially published **`torch-2.14.1+rocm7.14`** cp310 manylinux wheel from `https://download.pytorch.org/whl/rocm7.14` | Independently pinned AMD/PyTorch Linux artifact; wheel SHA-256 pinned below; distinct binary identity from any Windows wheel. |
| Triton source | **`pytorch-triton-rocm==3.5.0`** (cp310) from the same official index | Official Linux Triton for the rocm7.14 wheel stack; `triton-windows==3.7.1.post27` remains the separate Windows pin. |
| Python | CPython **3.10** (venv) | Repository-supported interpreter (`requires-python >=3.10,<3.11`); matches the Windows Phase-1 evidence's interpreter so fixture/CLI behavior is comparable. Deviation from the ROCm 7.14.1 matrix's example list (3.11–3.14) recorded below. |
| Environment location | `../envs/linux-gfx1151-rocm` venv outside the repository; model cache `../hf-cache` | No global packages touched; no repo-tree pollution; retained short paths per the runtime-cache convention. |
| Attention/precision policy | **MATH default** for vision SDPA (scoped wrapper), BF16 parameters, eager decoder attention with original DASH Triton enabled | Unchanged reviewed semantics; no AUTO switch; zero silent fallback. |
| torchvision / torchaudio | Not installed on the Linux native profile | Not in the native model dependency set (`requirements/native.txt`); avoids unverified extra GPU components. |

## Pinned identities

| Artifact | Identity |
| --- | --- |
| `torch-2.14.1+rocm7.14-cp310-cp310-manylinux_2_28_x86_64.whl` | SHA-256 `7e803db5a702e3321ae43d03ea0439c7c422642507df048663a1acc0ec1a7daf` |
| `pytorch_triton_rocm-3.5.0-cp310-cp310-linux_x86_64.whl` | SHA-256 `f2ea8cb272afcdb8b08a85eb947ed01d0921a56b26aee8648c6d1de1aaf4db08` |
| model snapshot | `baidu/Unlimited-OCR-2` @ `9cf077c4b206e0e2a31879df937b0b0bcba3661c`, all file/weight SHA-256 per `src/unlimited_ocr_2_rocm/upstream-lock.json` |
| native deps | `requirements/native.txt` unchanged (transformers 4.52.1, huggingface-hub 0.36.0, tokenizers 0.21.4, …) |
| full environment freeze | `requirements/linux-cp310.lock.txt` (generated from the installed venv, recorded with the run evidence) |

Runtime identities reported by the installed wheel (recorded by `doctor`):
`torch.version.hip`, `torch.version.rocm`, arch list including gfx1151 — see
the doctor manifest linked in the validation report. Acceptance is *physical*:
the doctor profile, BF16 probe, DASH/Triton kernel checks and page-structure
contract must all pass on this host; matrix listing alone proves nothing.

## Known deviations and residual risks (explicit)

1. **ROCm matrix example-list deviation.** AMD's docs-7.14.1 compatibility
   matrix lists PyTorch 2.12.0/2.11.0/2.10.0 (Python 3.11–3.14) for the
   framework stack; the officially published Linux `rocm7.14` wheel index
   offers 2.14.x (cp310–cp314) and 2.15.0 (cp311/312). We pin the officially
   published 2.14.1 cp310 wheel and re-validate everything physically instead
   of assuming matrix equivalence. Wheel bytes are the identity, not the
   version string.
2. **System ROCm userland (7.2.1) ≠ wheel runtime (7.14).** Only the wheel's
   bundled runtime is exercised by the venv; system `rocminfo` (7.2.1) is used
   solely for host enumeration. Any future global upgrade is out of scope.
3. **Triton line differs across platforms.** Linux uses pytorch-triton-rocm
   3.5.0; Windows uses triton-windows 3.7.1.post27. DASH kernel identity is
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
