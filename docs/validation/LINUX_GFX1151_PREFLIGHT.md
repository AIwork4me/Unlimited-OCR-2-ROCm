# Linux gfx1151 host preflight — read-only

Date: 2026-10-29 (Asia/Shanghai). Task: issue #6 P1 Linux native profile, with
the primary Linux target changed to the physically available
**AMD Ryzen AI MAX+ PRO 395 / Radeon 8060S (gfx1151)** native Linux host
(issue #6 previously deferred Linux work to a future W7900/gfx1100 device).
This is a read-only survey: no kernel, driver or global package was changed.

## Host identity

| Item | Observed value |
| --- | --- |
| Product | HP ZBook Ultra G1a 14 inch Mobile Workstation PC (board 8D01) |
| CPU (first `/proc/cpuinfo` model name) | `AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S` |
| GPU (rocminfo agent 1) | `gfx1151`, marketing `AMD Radeon Graphics`, 40 CUs, max 2900 MHz; ISA `amdgcn-amd-amdhsa--gfx11-generic` also reported |
| NPU (rocminfo agent 2) | `aie2p`, `RyzenAI-npu5` (present, **not used** by the native Transformers stack) |
| OS | Ubuntu 24.04.4 LTS (Noble) |
| Kernel | `6.17.0-1033-oem` (Ubuntu OEM kernel, `#33-Ubuntu SMP PREEMPT_DYNAMIC Mon Aug 24 05:45:54 UTC 2026`) |
| amdgpu driver | in-tree kernel module `/lib/modules/6.17.0-1033-oem/.../amdgpu.ko.zst`; no `amdgpu-dkms` package installed |

## ROCm / HIP userland (system, global — not modified)

| Package | Version |
| --- | --- |
| rocm-core | `7.2.1.70201-81~24.04` (`/opt/rocm/.version` = `7.2.1`) |
| hip-dev / hip-runtime-amd | `7.2.53211.70201-81~24.04` |
| rccl / rccl-dev | `2.27.7.70201-81~24.04` |
| rocm-smi-lib | `7.8.0.70201-81~24.04` |
| rocminfo | `1.0.0.70201-81~24.04` |
| hipcc | `1.1.1.70201-81~24.04` |

The system userland is **ROCm 7.2.1**, not 7.14.1. It is retained untouched;
the pinned native environment uses the officially published
`torch-2.14.1+rocm7.14` manylinux wheel, which bundles its own ROCm/HIP
runtime libraries and requires only the kernel-mode driver above.

## Device nodes and permissions

| Node | Permissions | Access |
| --- | --- | --- |
| `/dev/kfd` | `crw-rw---- root render (511,0)` | user `amd` in `render` → read/write |
| `/dev/dri/renderD128` | `crw-rw---- root video (226,128)` | user in `video` → read/write |
| `/dev/dri/card1` | `crw-rw----+ root video` | user in `video` |

User groups include `render`, `video`, `docker`, `ollama`; `sudo` requires a
password (no unattended global changes possible, by design).

## Memory / APU configuration

| Quantity | Value |
| --- | --- |
| System RAM | 94 GiB visible (96 GiB nominal), 8 GiB swap |
| Dedicated VRAM carve-out (BIOS UMA) | 32.0 GiB (`mem_info_vram_total` 34,359,738,368 B) |
| GTT (shared, resizable) | 100.0 GiB (`mem_info_gtt_total`) |
| Idle VRAM in use at preflight | ~1.5 GiB (desktop session + services) |

This is APU **unified/shared memory**: `torch` reported budgets are not
dedicated VRAM and must be labeled as shared system memory in all evidence
(matches the Windows evidence labeling rule).

## Python / ML software state (global)

| Item | Value |
| --- | --- |
| System Python | 3.12.3 (Ubuntu) |
| Global torch / triton / transformers | **not installed** (no `torch`, no `triton`, no `transformers` in system or user site-packages) |
| User-level packages | `amdsmi 26.2.2`, `pillow 10.2.0` |
| `uv` | 0.12.3 |
| GPU services active | `ollama` (active), `docker` (active) — co-tenancy: baseline ~1.5 GiB VRAM used before any run; no model loaded at preflight |
| Existing project envs | none (`~/.conda`, conda, other venvs absent); new env created separately at `../envs/linux-gfx1151-rocm` (outside the repository) |
| HF cache | `~/.cache/huggingface` 420 GiB; `baidu/Unlimited-OCR-2` snapshot **absent** → downloaded fresh to the frozen revision |
| Free disk | ~155 GiB on `/` (1.9 TiB NVMe) |

## Network

`github.com`, `pypi.org`, `download.pytorch.org` reachable directly;
`huggingface.co` is **not reachable directly** (connection fails). Downloads
use the `HF_ENDPOINT=https://hf-mirror.com` mirror endpoint with unchanged
revision pinning and SHA-256 verification against `upstream-lock.json`; the
mirror serves the same git-LFS oids (verified against the lock before use).

## Comparison with the AMD ROCm 7.14.1 official matrix

Source: AMD ROCm documentation, version selector `docs-7.14.1`
(`rocm.docs.amd.com` compatibility matrix, Radeon/Ryzen APU table):

| Matrix requirement (Ryzen AI MAX+ 395 family, gfx1151) | This host | Match |
| --- | --- | --- |
| Ubuntu 24.04.4 with **OEM kernel 6.17** (or Ubuntu 26.04 GA) | Ubuntu 24.04.4 + `6.17.0-1033-oem` | ✅ exact |
| Kernel driver: inbox driver of the supported Ubuntu version | in-tree `amdgpu` of 6.17 OEM | ✅ |
| Windows 11 25H2 (Windows rows) | n/a (Linux runs are Linux-only evidence) | — |
| PyTorch listed for 7.14.1: 2.12.0 / 2.11.0 / 2.10.0, Python 3.11–3.14 | pinned wheel `torch-2.14.1+rocm7.14` on CPython 3.10 | ⚠️ recorded deviation, see environment decision |

Findings:

1. The host exactly matches the officially supported Linux configuration
   (Ubuntu 24.04.4 + OEM kernel 6.17, inbox amdgpu) for the Ryzen AI MAX+ 395
   family. No driver/kernel change is required or was made.
2. The installed global ROCm userland (7.2.1) is older than the 7.14.x
   matrix generation. It is not upgraded: the native environment instead uses
   the officially published `+rocm7.14` wheel stack with its bundled runtime,
   keeping the global system untouched. `rocminfo` from 7.2.1 already enumerates
   gfx1151 correctly, which is how the hardware was confirmed.
3. The ROCm 7.14.1 matrix lists PyTorch 2.12.0/2.11.0/2.10.0 on Python
   3.11–3.14; the officially published Linux rocm7.14 wheel index itself
   provides torch 2.14.x (cp310–cp314) and 2.15.0 (cp311/cp312). The chosen
   cp310 2.14.1 wheel is an officially published AMD/PyTorch Linux artifact,
   byte-identity pinned by SHA-256; the deviation from the matrix's example
   list is recorded explicitly in the environment decision document and is
   re-validated physically on this host (doctor/BF16/Triton/DASH) rather than
   assumed from the matrix.
4. gfx1151 (Radeon 8060S) here is RDNA 3.5 and shares the family support row
   with gfx1150/1152/1153; no per-chip difference applies. NPU (aie2p) is out
   of scope for the native Transformers stack.

## Standing constraints honored

- Read-only preflight: no kernel, driver, or global package changes; no sudo.
- The Windows candidate wheel (`torch-2.12.0+rocm7.14.1-cp310-win_amd64.whl`,
  SHA-256 `6a67d15861538ee217afdccd1ccc09ff318d7f44bff29a10f74c0d51730e6e1e`)
  is **not installed** on this host and no Linux result may cite it.
- No gfx1100 (W7900) claim is made from this host; gfx1100 remains a separate
  future physical target (see ADAPTATION_MATRIX.md).
- Windows and Linux evidence remain fully separate records.
