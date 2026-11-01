# G05 Phase A scorer closure implementation plan

Goal: execute the locked Qwen and Marathon CPU scorers with real FormulaCDM,
including negative cases, without changing the official source or the frozen lock.

Architecture: use WSL Ubuntu 22.04 only for CPU scoring. Keep its new Python
environment, downloaded packages, extracted tools, fonts, caches and raw evidence
under `.b/g05-scorer`. Invoke every command from the primary repository root.
Existing system tools may be read; missing packages are extracted locally.

## Planned steps

- [x] Inventory and hash the frozen source/requirements and existing Linux tools.
- [x] Create a separate CPython 3.10 environment, install the frozen scorer lock,
  and record package inventory and the absence of torch.
- [x] Supply XeLaTeX, xeCJK, the exact Source Han Sans SC font, Node,
  ImageMagick and Ghostscript through process-local environment settings.
- [x] Run all eight original metric known answers and converter/matcher fixture
  for both source profiles, retaining stdout/stderr and official JSON outputs.
- [x] Run real CDM exact and intentional mismatch cases; exercise invalid LaTeX
  and an outer process timeout, retaining original zero/error behavior explicitly.
- [x] Publish sanitized evidence, source/tool hashes, failure catalog and exact
  reproduction commands under `eval/benchmarks/g05/scorer`.

## Constraints and review focus

No model or torch downloads, scorer substitutions, source-byte changes, global
PATH/security/driver/SDK changes, or edits to historical evidence. Linux scoring
does not validate Linux inference hardware. Renderer failures must remain visible
even when the original scorer converts exceptions into a zero score. Known-answer
aggregation is a fixture result and is not a model benchmark score.

## Results

PASS for the authored CPU scorer controls, for both locked profiles. This is
scorer readiness evidence; it is not a model-quality or GPU benchmark result.
See [the receipt](../eval/benchmarks/g05/scorer/summary.json) and
[recorded commands](../eval/benchmarks/g05/scorer/commands.json).

| Check | Qwen | Marathon |
| --- | --- | --- |
| Original eight metric known answers | 8/8 PASS | 8/8 PASS |
| Original converter/matcher relation fixture | PASS | PASS |
| Full relation + formula pipeline, all eight metrics | 100 each | 100 each |
| Real exact CDM, five rendered tokens per side | 1.0 | 1.0 |
| Real mismatch CDM, two shared symbols out of five | 0.4 | 0.4 |
| Undefined-control renderer error | EXPECTED_RENDER_ERROR | EXPECTED_RENDER_ERROR |
| Forced outer deadline during actual prediction XeTeX | EXPECTED_TIMEOUT | EXPECTED_TIMEOUT |
| Active renderer processes after each control | 0 | 0 |

The error check requires valid GT boxes and a decoded, nonwhite GT PNG, failed
prediction rendering, and an original TeX diagnostic. A swallowed CDM exception
returning zero is never accepted as a valid quality score. Exact/mismatch checks
require nonempty positive-area boxes and verified, decoded PNGs on both sides.
The comparison tolerance is `1e-7`.

The timeout check observes the actual prediction `/usr/bin/xetex` process in the
child's process group, then applies a deliberate 0.05-second outer deadline.
Startup has its own 120-second cap. Cleanup runs in `finally`, kills the process
group, reaps its leader and checks that no active descendants remain. A separate
Linux regression also checks an orphaned descendant after its leader exits.

**NOT EXERCISED:** upstream's hardcoded 30-second native `Timer` path. Two
additional candidate expressions returned naturally; their complete diagnostic
receipts are retained. They do not prove native timeout handling. Full dataset
scoring, model inference and all GPU benchmarks are NOT RUN in this phase.

## Frozen inputs and runtime

The benchmark lock is byte-identical to PR #7 `d842ce6`, SHA256
`39f3dc09a9f9aa917f3ecf5f0deea690a7d1ed91ffb4e8e70803aac9e8dd7e08`.
The scorer dependency lock remains
`12e19dfd496976e804c2f89a50038ea03875fe086fcde66f6e4a788d26fe2c40`.
Every official file is individually verified; neither official source bytes nor
the existing worker were changed for this closure. Historical `__pycache__`
directories were preserved, and only locked files were copied into clean local
snapshots before import.

The torch-free environment uses CPython 3.10.12, Node 16.13.1, XeTeX 0.999993
(TeX Live 2022/dev/Debian), xeCJK 3.8.8, Source Han Sans SC 2.004R,
ImageMagick 7.1.2-26 and Ghostscript 9.55.0. `pip check` passes. Exact packages,
download URLs, hashes, underlying XeTeX binary hash, wrapper hash, system package
versions and process environment overrides are in
[runtime/inventory.json](../eval/benchmarks/g05/scorer/runtime/inventory.json).
The font is bound to Adobe commit `a8b073bbf80f7226af03abeeb31e27017d5e3f67`.

All new retained tools, downloads, venv files, caches and raw run outputs are
inside `.b/g05-scorer`. The final environment uses explicit Linux tool directories
and process-local `XDG_*`, `TEXMF*`, `FONTCONFIG_*`, `HF_*`, `MPLCONFIGDIR` and
temporary-directory settings. It does not change global PATH or HOME. The local
TeX tree needs `mktexlsr`, `TEXMFHOME=!!<local tree>` and the recorded `TEXMFDBS`
to avoid repeated filesystem traversal. Generic missing libGL loader libraries
were extracted locally for the unchanged OpenCV package; no driver was installed.

## Reproduce in the retained environment

Run every command from the primary repository root. Preserve the original runs;
use fresh output names for each replay. Tool binary hashes must match the receipt.
A new host needs separately provisioned and verified base TeX, ImageMagick and
Ghostscript; this receipt does not certify another host.

Save this small launcher as `.b/g05-scorer/replay.py`. The published inventory
contains the complete environment recipe, and the published config and wrapper
are sanitized copies of the actual files. Retain the recorded local tool prefix,
font, TeX format and frozen venv.

```python
import json
import os
import subprocess
import sys
from pathlib import Path

root = Path.cwd()
base = root / ".b/g05-scorer"
receipt = root / "eval/benchmarks/g05/scorer/runtime"
inventory = json.loads((receipt / "inventory.json").read_text())
expand = lambda value: value.replace("<repo>", str(root))
controls = {key: expand(value) for key, value in inventory["environment_overrides"].items()}
(base / "fontconfig.xml").write_text(expand((receipt / "fontconfig.xml").read_text()))
wrapper = base / "bin/xelatex"
wrapper.write_text(expand((receipt / "xelatex").read_text()))
wrapper.chmod(0o755)
command = [str(base / "venv/bin/python"), "scripts/g05_scorer_closure.py", *sys.argv[1:]]
raise SystemExit(subprocess.call(command, cwd=root, env=os.environ | controls))
```

Invoke the launcher through WSL from PowerShell, keeping its working directory
at this repository root. The formula pipeline also runs the original known
answers and relation fixture. Repeat both commands with `marathon` in the three
profile-specific arguments.

```powershell
$linuxRoot = "/mnt/c" + (Get-Location).Path.Substring(2).Replace("\", "/")
wsl -d Ubuntu2204 --cd $linuxRoot --exec python3 .b/g05-scorer/replay.py --source .b/g05-scorer/source/qwen --profile qwen --output .b/g05-scorer/replay-qwen-pipeline --pipeline
wsl -d Ubuntu2204 --cd $linuxRoot --exec python3 .b/g05-scorer/replay.py --source .b/g05-scorer/source/qwen --profile qwen --output .b/g05-scorer/replay-qwen-cases
```

The source snapshots and venv are intentionally local. Source reconstruction
copies only paths listed in the unchanged lock and verifies their SHA256 before
use. Package reconstruction uses `python3 -m venv` and the unchanged
`requirements/benchmark-scorer.lock.txt`, with `TMPDIR` and pip cache set below
`.b/g05-scorer` before creating the venv. Ubuntu packages are downloaded from
the recorded URLs, hash checked, and extracted with `dpkg-deb -x` into the local
prefix. Node's archive is checked against the retained official SHA256 list.
No global installation is required for these overlays.

## Evidence and failures

[artifact-manifest.json](../eval/benchmarks/g05/scorer/artifact-manifest.json)
binds original raw bytes to sanitized text copies. PNG/PDF/TeX-format binaries
are retained locally and represented by hashes, sizes and image validation
metadata. Source and tool receipts retain their original hashes. Sanitization
replaces host paths, including derived, line-wrapped TeX filenames; it does not
change raw local evidence. Public paths longer than 100 repository-relative
characters use short `artifacts/<id>` names to fit Windows build extraction;
the manifest preserves their logical paths and original hashes. Peak RSS uses
Linux `RUSAGE_CHILDREN.ru_maxrss` in KiB,
the maximum child process resident set, not summed process memory, dedicated
VRAM, APU GPU allocation or total system memory consumption.

[failure-catalog.json](../eval/benchmarks/g05/scorer/failure-catalog.json)
records the initial libGL import failure, invalid negative-fixture assumptions,
the interrupted-process cleanup review and retained warnings. Earlier outputs
are preserved under `.b/g05-scorer/runs` and public `preliminary/`; final outputs
use separate directories. No metric threshold was relaxed to obtain a pass.
