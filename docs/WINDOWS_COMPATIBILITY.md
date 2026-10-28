# Windows compatibility

This project's support target is native Windows 11, Radeon 8060S / gfx1151,
Ryzen AI MAX+ 395, CPython 3.10 x64. The current acceptance state is in
[VALIDATION.md](VALIDATION.md); this matrix distinguishes upstream references
from actual local execution.

| Component | Baidu native reference | Phase 1 target / observation |
|---|---|---|
| OS | Linux implied by reference workflow | Windows 11 Home Chinese, build 26200 |
| CPU/GPU | NVIDIA A800 80GB | Ryzen AI MAX+ 395, Radeon 8060S, gfx1151 |
| Memory | Discrete accelerator reference | Shared APU RAM, OS usable 132766306304 bytes |
| Driver | Not pinned in model card | AMD 32.0.22032.6002 |
| Python | 3.10 | CPython 3.10.20 x64; exact AMD wheels available |
| torch | 2.13.0 | Required 2.12.0+rocm7.14.1 |
| torchvision | Unpinned | Required 0.27.0+rocm7.14.1 |
| torchaudio | Not specified | Required 2.11.0+rocm7.14.1; import checked |
| ROCm | Not part of CUDA reference | Package release 7.14.1, HIP build 7.14.60850 |
| Transformers | 4.52.1 | 4.52.1 pinned; no 5.x migration |
| Hub / tokenizers | 0.36.0 / 0.21.4 | Same exact versions |
| Triton | Custom kernels present | triton-windows 3.7.1.post27, HIP backend |
| Attention dispatch | Custom native implementation | mha_eager with DASH/Triton enabled |
| Serving client | Customized SGLang | Excluded from Phase 1 |

The [pinned official model card](https://huggingface.co/baidu/Unlimited-OCR-2/blob/9cf077c4b206e0e2a31879df937b0b0bcba3661c/README.md)
states the Baidu reference versions. The [Triton post27 release](https://github.com/triton-lang/triton-windows/releases/tag/v3.7.1-windows.post27)
reports RDNA3.5 testing and offers CPython 3.10 Windows wheels. Our actual
vector-add and DASH-kernel manifests provide local evidence separately.

## Windows behavior

- DLLs/runtime are supplied by the exact AMD multi-architecture packages;
  do not mix CUDA wheels or install NVIDIA CUDA. torch.cuda is the supported
  API for ROCm devices.
- Visual Studio Build Tools 2022 17.14 is already installed at the discovery
  machine. Triton automatically found its bundled AMD compiler and the
  existing Python headers/libraries; no developer-shell or global PATH change
  was needed by the successful kernel probes.
- Native vector-add compilation emitted three Python 3.10 header warnings
  about struct timeval visibility. Execution still passed. The official
  sparse kernel emitted a make_block_ptr deprecation warning; no rewrite
  was made merely to suppress it.
- Use pathlib and quoted PowerShell argument arrays. Raw text and manifests
  use UTF-8; stdout display depends on the terminal encoding.
- All GPU scripts have main guards. This implementation launches no model
  multiprocessing workers and installs no persistent CI runner.
- APU reported GPU budget is 83954728960 bytes. It is shared system memory,
  not an independent 80GB physical VRAM pool. Record system available RAM,
  RSS and GPU allocation together.
- No Defender, Smart App Control, registry, long-path policy, driver,
  system GPU budget or permanent environment changes were made.
- Long-path and antivirus interference are not inferred from successful
  short paths. If one blocks a future run, preserve the error and investigate
  the smallest reproduction before requesting a security-setting decision.

An actual clean install exposed the legacy path boundary in an AMD
hipBLASLt wheel: the same Tensile file exists at total path lengths251/257,
but extraction failed at260 under the longer venv name. Setup now detects
that known entry before downloading/installing. Use a short dedicated
VenvPath; no registry or security change was made. The failed installation
was confined to its new isolated environment and is retained as a diagnostic.

No Linux inference, WSL2, Docker, virtual machine, W7900/gfx1100, gfx1200,
or other physical GPU validation is claimed. Linux CPU CI checks only the
package's hardware-independent contracts.
