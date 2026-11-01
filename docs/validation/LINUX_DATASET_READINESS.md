# Linux dataset readiness — D0 full local verification

Date: 2026-10-30 (Asia/Shanghai). Task: issue #6 mandatory phase **D0** —
verify the locally provided benchmark datasets on the Linux gfx1151 host
before any GPU dataset benchmarking. Read-only with one explicitly recorded
exception (a single authorized page repair, receipt chain preserved below).

Host: HP ZBook Ultra G1a, Ryzen AI MAX+ PRO 395 / Radeon 8060S (gfx1151),
Ubuntu 24.04.4, kernel `6.17.0-1033-oem`. Repo baseline: PR #11
(`3ad8809e452f616f1e353f0e632982fe0fc3c6b1`), branch
`feat/linux-gfx1151-native-validation`.

## Verdict

| Dataset | Documents | Pages | Status |
| --- | --- | --- | --- |
| MPDocBench | 420/420 | 3135/3135 | **PASS** |
| MarathonDocBench | 100/100 | 18272/18272 | **PASS** (after one authorized page repair) |
| SlideVQA source shards | 12/12 | — | **PASS** (frozen SHA256 match) |

**D0 exit gate: satisfied.** GPU dataset benchmarking (P1) is unblocked from
the dataset side.

## Data roots (absolute, on this host)

| Item | Path |
| --- | --- |
| Data root | `/home/amd/Desktop/UnlimitedOCR/Data` |
| MPDocBench images | `/home/amd/Desktop/UnlimitedOCR/Data/MPDocBench/images/<doc>/page_n.jpg` |
| MarathonDocBench images | `/home/amd/Desktop/UnlimitedOCR/Data/MarathonDocBench/images/<doc>/page_n.png` |
| SlideVQA parquet shards | `/home/amd/Desktop/UnlimitedOCR/Data/SlideVQA/data/test-*-of-00012.parquet` |

No dataset file is inside Git. Directory audit: 0 symlinks, 0 unreadable
files, no case-colliding names, no extra files beyond the frozen indexes
(3135 jpg + 18272 png + 12 parquet), 115 GiB free on the hosting filesystem.

## Frozen identities used

| Identity | Value |
| --- | --- |
| Benchmark lock | `eval/benchmarks/benchmark-lock.json`, SHA256 `39f3dc09a9f9aa91ffb9e8e70803aac9e8dd7e08` |
| MPDocBench scope (manifest jsonl) | SHA256 `785f18884dc586aa836f096dcba7c96c1fd0cc50a6dae9f62aabb0152d4bdcc8` |
| MarathonDocBench scope (manifest jsonl) | SHA256 `2515b579828e787c6cc928421abd6c693086d64084ba26a459f82ccf21a9fe76` |
| MPDocBench GT (not present locally) | SHA256 `0a579dca6cf875a41fdab0975efee6d4e0fea999e0fa6c3ca95ad08b948f40df` (34,593,225 B) |
| MarathonDocBench GT (not present locally) | SHA256 `7db0e659c76661ce87d4e1b3d47e010edbd48adfb446e3e2ee37260c60adf51c` (114,122,727 B) |
| MPDocBench image archive (Windows acquisition) | revision `48428421ee985717fbda3a6521165aaedfc42077`, 600,152,009 B |
| MarathonDocBench revision | `fa05fccbe82ee6ec628006a1d58bb153954c3ee3` (HF LFS) |
| Verifier | `scripts/verify_linux_datasets.py`, SHA256 at run time `903989b97f8b5683bba8310136a2eace6ed64eef7dd0054c22119121c15934db` (attempt-1 ran byte-identical logic pre-formatting, SHA256 `0f2b60170bb701d5…` recorded inside its own receipt) |

Both manifest JSONL files were re-hashed on this host and match the lock.
**The original GT JSON files are not present on this Linux host** (they live
only in the Windows `.b/` audit tree). Document-level integrity is therefore
re-derived from the scope-bound manifests: every document's
`ordered_paths_sha256` and per-page-hash digest (`archive_page_hashes_sha256`
for the 409 archive documents / `published_page_hashes_sha256` for Marathon)
was recomputed from the frozen page index and matched. Annotation-level
re-verification (`annotation_sha256`) requires the GT files and remains
deferred to the scorer phase; it is covered transitively by the scope hash.
The 11 SlideVQA decks (84 pages, `official_slidevqa_jpeg_serialization`)
were verified against the page index, which is their authoritative frozen
serialization; their archive digest is intentionally not comparable.

## Per-page verification results

Streaming SHA256 over every byte of every page, byte-count check, PIL decode
check, document/page reconciliation against the frozen manifests, extra-file
scan:

| Metric | MPDocBench | MarathonDocBench |
| --- | --- | --- |
| Expected documents / pages | 420 / 3135 | 100 / 18272 |
| Verified documents (digests) | 420 | 100 |
| Verified pages | 3135 | 18272 |
| Missing | 0 | 0 |
| Mismatched | 0 | 0 |
| Extra files | 0 | 0 |
| Verified bytes | 688,382,455 | 11,805,490,064 |
| Decode failures | 0 | 0 (1 before repair) |
| Page-index digest (verified pages) | `24d8f234e25ac020…` | `e48a2b213f0ccdbe…` |
| Wall time (8 decode workers) | 1.6 s | 35.4 s |

SlideVQA source: all 12 pinned test parquet shards match their frozen
SHA256 (1,393,939,599 B total). The extracted JPEG outputs are the
`pptSlides_eng_*` directories inside MPDocBench and are covered by the
page verification above.

## The 569,985-byte Marathon discrepancy — reconciled

The Linux preflight had recorded 11,804,920,079 local image bytes against
the frozen 11,805,490,064. A size-only scan isolated the entire gap to
**one page**: `images/mp3lan_0165/page_43.png` — expected 1,225,345 B,
actual 655,360 B (**exactly 640 KiB**: a transfer truncation), SHA256
`4ae261c4…` ≠ frozen `580555a9…`, PIL decode failed
("image file is truncated"). All 18,271 other pages were byte-exact.
Cause: an incomplete download stream during the dataset copy; not a
dataset-revision or measurement difference.

Per the D0 rules the failed state was first recorded immutably
(`dataset-verification-attempt-1-blocked.json`), then a single-page
authorized repair was performed: re-download of the LFS object via
hf-mirror (huggingface.co unreachable from this host), gated on
size == 1,225,345, SHA256 == the frozen page hash, and a successful PIL
decode. The truncated original is preserved outside Git at
`/home/amd/Desktop/UnlimitedOCR/d0-evidence/mp3lan_0165_page_43.png.truncated-655360`.
A full re-verification of both datasets followed (attempt-2 receipt,
PASS/PASS). Repair receipt:
`eval/results/Linux gfx1151 verified/datasets/page-repair-mp3lan_0165-43.json`.

## Receipt chain (all committed)

| Receipt | Content |
| --- | --- |
| `eval/results/Linux gfx1151 verified/datasets/dataset-verification-attempt-1-blocked.json` | Full verification, Marathon BLOCKED on 1 page (immutable failure evidence) |
| `eval/results/Linux gfx1151 verified/datasets/page-repair-mp3lan_0165-43.json` | Authorized repair record with gates and backup location |
| `eval/results/Linux gfx1151 verified/datasets/dataset-verification-attempt-2-repaired-pass.json` | Full re-verification after repair: PASS / PASS |
| `eval/results/Linux gfx1151 verified/datasets/dataset-verification.json` | Latest run (identical to attempt-2) |

## Verifier and tests

`scripts/verify_linux_datasets.py` (read-only; accepts `--data-root`,
`--benchmark-lock`, `--output`, `--dataset`, `--decode/--no-decode`,
`--workers`): streaming SHA256, size check, decode check, symlink/path-escape
and case-collision rejection, duplicate-path rejection, scope-hash binding,
document digest reconciliation, extra-file scan, atomic receipt (written only
on completion — an interrupted run leaves no receipt). 16 CPU unit tests in
`tests/unit/test_verify_linux_datasets.py` cover: valid inventory, missing
page, corrupted SHA256, size mismatch, truncated-image decode failure, wrong
page order, wrong annotation row, duplicate paths, case collisions, path
escape, symlinked page, extra files, scope-hash mismatch, page-count
mismatch, interrupted run, end-to-end CLI receipt. Full CPU suite after the
additions: **270 passed, 5 skipped** (torch-dependent tests skip on the
system interpreter), no regressions.

## Limitations / next actions

- GT JSON files are not on this host; scorer-stage work must bring them (or
  re-verify against the Windows audit tree) before official scoring.
- The verifier trusts the frozen page index as the per-page authority (it is
  itself scope-bound and was cross-checked against published LFS hashes at
  Windows acquisition time).
- Dataset readiness now PASS for both benchmarks; P0 integration is the next
  gate before any P1 GPU dataset run.
