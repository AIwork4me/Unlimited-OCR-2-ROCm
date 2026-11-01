# Native Windows clean-environment baseline

Code under physical qualification: `3556e3867a086437a43a645fa907689f4dd25260`
in a fresh local clone, `.worktrees/wi`. CPU integration follow-up at
`5436f80233477cf53f0e5eda4406c27a9e09c035` changes only the PowerShell cache
entry and its test; it does not change model/inference code. The final docs and
packaging commit is separate. Do not attribute historical GPU results to either.

Two fresh CPython 3.10.20 environments completed offline installation and pip
check. The complete wheelhouse has 43 artifacts. All 13,343 non-RECORD members
from the original nine AMD runtime wheels match both installations byte for
byte, with no missing members. Nine installer-regenerated RECORD files are
explicit exclusions. The original torch archive is
`415d2dbd4b6ab002cff1c54795fdd66322abd9a46bf982cb64eebb4493ac2e7c`.
Installed-member receipt SHA256:
`d1d184069891e8f5e6a07a17d62ea4b437b42a3a533fb6ca991c080aceb706e9`.

## Preserved first qualification failure

The first reviewed identity, `golden-original-v1`, failed during operator
environment collection, before BF16/Triton probes, model loading or original
OCR inference. The explicit environment omitted username variables and did not
set TORCHINDUCTOR_CACHE_DIR. Importing torchvision reached PyTorch's default
Inductor cache path, where getpass.getuser attempted to import the POSIX-only
pwd module. This is an environment-policy failure, not an OCR correctness score.

The durable ledger records w1 FAIL and w2 PREPARED with no execution. Total
persisted original infer entries: **0**. The failed identity/commands, full
exception, memory telemetry and ledger remain private and immutable; no stage
was reset or automatically retried. An independent post-exit process inventory
found no remaining Python worker. The original Job termination routine verifies
zero active owned processes before returning.

Identity-file SHA256:
`509f2a01fcb61653286bacd33fce0c648398c526b4823bfda43b2df9c455bc18`;
policy-file SHA256:
`fe99b356d18240a1f10479ccf1487bf49dca37fd73a44798e16c01dab45517c5`.

## Existing regression contract

Only the unchanged 13 normal cases are in scope: English, Chinese, mixed text,
table, simple mathematics, Unicode path with recoverable missing-input guard,
two repetitions each of 2/3/5 pages, and a two-page table. Each suite has 28 input
page uses, including repeated synthetic fixtures. These are not benchmark-dataset
documents. The maximum remains 26 original infer entries across two environments.

Product admission independently recomputes the existing fixture content contract
at CER <=0.02, important numbers/table requirements, ordered complete PAGE/BLOCK
structure, native stop/metadata, original default options, restored runtime state,
actual synchronized BF16 CUDA DASH dispatches, both GPU MATH vision forwards,
clean HIP and complete memory evidence. torch.cuda is the ROCm API here.

The existing completion contract allows one terminal empty PAGE marker. Preserve
raw/native marker counts separately from ordered nonempty content pages; this
does not permit missing or extra content pages. This interpretation was checked
against all 13 historical cases before new GPU execution and changed no original
threshold or model behavior.

Exact output-token, raw-bbox and cold/warm parity are separate classifications.
Product PASS cannot convert a historical strict parity difference into PASS.
No AUTO/fused attention or experimental scoped QuickGELU is used.

## Resource and supervision boundaries

The reviewed policy requires >=64 GiB host available memory before each suite;
sampled stops are <48 GiB host available or >16 GiB GPU allocator reserved.
Unknown/malformed readings or a capture older than five seconds stop the worker.
Each case is bounded by 120 seconds and each suite by 900 seconds, with no
automatic retry. Owned suspended-create/assign/resume Jobs are drained on exit.
The hard-owner-death gap before Job assignment remains documented.

APU allocation/reservation is shared system memory, not dedicated VRAM, process
RSS or total host memory. One-Hz observations are not an unsampled peak guarantee.
Actual allocator peak counters and process peak RSS are recorded separately.

## Current execution result

The separately reviewed corrected identity completed both suites. All26 original
infer entries passed independent worker, controller and root raw-artifact rereads.
W1 and W2 durable states are PASS. The first failure stays FAIL; no old ledger or
source was changed. V2 binds its complete sealed failure chain. Post-exit native
process inventory found zero matching workers.

V2 identity-file SHA256: `253f7ba17ec29671973e8489ecae3f373fe2f19f3f62cd7ca9aa04e1952db649`;
policy-file SHA256: `ce6db725b90b52b289c25b5113c9dfc71b10f7282b69a3ca7b49e13b3846ebb9`.
Public allowlisted receipt SHA256: `8d2fda03ad3e979ecf0f33e15b15a7f48ccc1cf1cf94f360b833769f52be4410`.

| Classification | This fixed sample and scope |
| --- | --- |
| CONTENT_CORRECTNESS | PASS26/26, every fixture page CER0 at predeclared <=0.02 |
| STRUCTURAL_CORRECTNESS | PASS26/26, ordered content pages, complete blocks, native stop and consistent metadata |
| EXACT_TOKEN_PARITY | PASS13/13 corresponding cross-environment pairs |
| EXACT_BBOX_PARITY | PASS13/13 corresponding cross-environment pairs; integer block coordinates compared without tolerance |
| COLD_WARM_PARITY | PASS6/6 within-suite2/3/5-page strict pairs |
| Physical contract | PASS26/26,312 synchronized BF16 DASH dispatches, both actual GPU MATH vision modules, clean HIP, restored original state |

Historical strict differences remain immutable. These new paired observations
provide no proof that prior numerical differences are causally fixed, no general
determinism guarantee and no AUTO/fused/scoped-QuickGELU admission.
Here cold/warm means the existing ordered pages_N_run_1/run_2 pairs after the
single-page cases; it does not mean a cold boot or a globally empty driver cache.

Actual OS/device: Windows11 build26200, Ryzen AI MAX+395 / Radeon8060S gfx1151,
driver32.0.22032.6002, CPython3.10.20, ROCm7.14.1 /HIP7.14.60850. Both installations
report the pinned torch/vision/audio trio, triton-windows3.7.1.post27 and
Transformers4.52.1. BF16 compute, HIP Triton compilation/execution and immutable
offline official model loading passed independently in each environment.

| Observation | w1 | w2 |
| --- | ---: | ---: |
| Original infer entries / synthetic input page uses | 13 /28 | 13 /28 |
| Generated tokens | 1698 | 1698 |
| Sum of synchronized case wall seconds | 166.618 | 163.990 |
| GPU peak allocated GiB | 8.589 | 8.589 |
| GPU peak reserved GiB | 10.510 | 10.510 |
| Process peak working set GiB | 11.211 | 11.212 |
| Sampled minimum system available GiB | 76.632 | 76.430 |
| One-Hz validated samples | 201 | 196 |

The wall sums exclude operator checks/model load/identity hashing; they are not
full-corpus performance. Allocator peaks, process working set and host available
memory are distinct UMA observations. No OOM, truncation, timeout or HIP error
occurred in these26 original calls. One pre-infer environment failure is counted
separately and retained. Full MPDocBench/Marathon documents/pages attempted remain
0, and all eight metrics/Overall/deltas remain null. See
[readiness](WINDOWS_NATIVE_READINESS.md) and
[campaign status](../../eval/windows/benchmark-status.json).
