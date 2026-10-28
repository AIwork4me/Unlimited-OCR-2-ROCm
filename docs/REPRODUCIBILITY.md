# Reproducing Phase 1

Use the same physical Windows gfx1151 machine, 64-bit CPython 3.10 and
sufficient free RAM/disk. CPU CI does not perform these tests.

```powershell
# Select an installed CPython 3.10 interpreter, for example:
$Python310 = (uv python find 3.10)
.\scripts\reproduce_windows.ps1 -Python $Python310 -VenvPath ".venv-a" -OutputPath "eval/results/run-a"
.\scripts\reproduce_windows.ps1 -Python $Python310 -VenvPath ".venv-b" -OutputPath "eval/results/run-b" -BaselinePath "eval/results/run-a/reproduction.json"
```

uv is optional: replace $Python310 with the full path of an existing CPython
3.10 executable. An existing global/base Python is only used to create a venv.
Scripts refuse an existing non-venv directory and never install into global
Python. Clean reproduction refuses an existing environment; choose a new
VenvPath. Setup alone can be repeated on an existing dedicated venv.

The first run establishes A–D. Gate E requires a linked baseline pair:
different producing environment IDs, matching machine/artifacts/locked packages,
all required cases, and an environment-creation witness tied to the workflow
run ID. Re-summarizing under another interpreter cannot manufacture that proof.
A fresh flag contradicting the recorded workflow is rejected.
Choose a short venv path: an AMD wheel entry fails at260 total path characters.
Setup detects that boundary; do not change the registry.

The mandatory installation inside that environment is:

```powershell
python -m pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ "torch[device-gfx1151]==2.12.0+rocm7.14.1" "torchvision[device-gfx1151]==0.27.0+rocm7.14.1" "torchaudio==2.11.0+rocm7.14.1"
```

Then setup installs the hash-pinned Triton candidate and exact dependency lock,
checks pip and the GPU, and installs the wrapper without dependency resolution.
It records install reports and freeze under the venv's phase1-install directory.
The package deliberately has no automatic torch dependency.

The orchestrator checks isolated Python, pip, strict identity and BF16, then
Triton vector-add, pinned model data/checksums, the official DASH kernel,
English/Chinese/mixed/table OCR, repeated Unicode-path recovery and native
2/3/5-page repeated inference. It stops on failure. Output includes individual
manifests, raw/normalized OCR, command lists, logs and reproduction.json.

Once installed and cached, an offline model rerun is:

```powershell
.\scripts\reproduce_windows.ps1 -VenvPath ".venv-b" -ReuseEnvironment -Offline
```

This rerun does not invoke pip installation. A fresh offline package install
also requires a separately prepared wheelhouse; cached model data alone does
not supply Python dependencies.

Model/code pin: 9cf077c4b206e0e2a31879df937b0b0bcba3661c.
GitHub pin: 4d0bebab8714e0f3b0c811cd44bf422ddb49daae.
The installed package's upstream-lock.json includes every reviewed Python,
config, tokenizer and weight checksum. --cache-dir selects the HF cache.
--offline refuses network model lookup. Fresh module cache copies prevent
reuse of an unchecked dynamic-code copy; use a clean process for loading.

Run CPU checks in a separate environment:

```powershell
python -m pip install -r requirements/dev.txt
python -m pip install --no-deps -e .
python -m ruff check src scripts tests
python -m ruff format --check src scripts tests
python -m mypy src
python -m pytest
python -m build
python -m pip check
```

Explicit physical tests require both the opt-in and the real model cache:

```powershell
$env:UOCR_RUN_GPU = "1"
$env:UOCR_MODEL_CACHE = "C:\model-cache"
python -m pytest tests/gpu -v
```

Only a successful second clean environment on the same machine establishes
Gate E. It is not second-machine or cross-platform reproduction. Read the
actual evidence in VALIDATION.md before assigning any acceptance gate.
Sanitize logs before publishing; installation logs can contain local paths.
