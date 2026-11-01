# Windows GPU Inference → Linux Official Scorer Bridge Design (Phase 4.5, Option B)

Design only. No GT or prediction transfer is authorized by this document;
any actual transfer requires separate explicit authorization with frozen
input/output digests (per the standing real-document admission rules).

## Principle

The official scorer stays **unchanged on Linux**, where it is already
qualified twice (WSL PR #9; native Linux gfx1151 PR #13). Only the artifact
that is portable by nature — per-document Markdown predictions — crosses the
boundary. Windows keeps inference evidence; Linux keeps scoring evidence;
each side re-derives identity from hashes rather than trusting the other.

## Contract

| Element | Specification |
| --- | --- |
| Scorer revision | `Tongyi-Zhiwen/Qwen-Doc` @ `4e5aee32280febb7b0844237ba087f837e3c8905` (85 files, per-file SHA-256 from `eval/benchmarks/benchmark-lock.json`); Marathon patches @ `fa05fccbe82ee6ec628006a1d58bb153954c3ee3` (3 files) |
| Dependency closure | `requirements/benchmark-scorer.lock.txt` (75 pins, torch-free; lock SHA bound by the benchmark lock) in a **separate torch-free CPU venv** |
| Host tools (CDM) | Node, ImageMagick+Ghostscript, XeLaTeX/TeX Live with `xeCJK`, Source Han Sans SC, `kpsewhich`; `latexmlc` pinned and self-tested before any run exercising the LaTeX-table branch |
| Ground truth | **Not transferred.** The Linux side re-uses its verified local GT; admission requires the GT SHA-256 on both sides to equal the frozen lock value (`0a579dca…` MPDocBench / `7db0e659…` Marathon) |
| Prediction schema | flat directory, exactly one `<doc_id>.md` per GT document; failures = explicit empty files; inventory equality enforced by the worker before scoring |
| Transfer artifact | archive of the prediction directory + `predictions-manifest.json` (per-file SHA-256, doc count, campaign id, producer receipts) — model outputs only, never GT; sanitized by the standard publication scanner rules before leaving the Windows host |
| Provenance carried | Windows run receipt digests (source commit, wheel hash, model revision, policy digests, ledger seal) recorded alongside scores |
| Metric coverage | all eight official metrics; Overall null if any missing (see OFFICIAL_METRIC_COVERAGE_GATE.md) |
| Failure semantics | Windows-side TRUNCATED/TIMEOUT/OOM/INTERRUPTED documents are scored as their exported artifacts (empty or partial per strict mode) and reported in the failure catalog; never silently excluded |
| Platform responsibility | Windows: inference, export, sanitization, manifest. Linux: scorer integrity, tool chain, scoring, coverage report |
| Reproduction | Windows: install guide + campaign runner; Linux: scorer closure procedure (`scripts/g05_scorer_closure.py` / `scripts/check_official_scorer.py` lineage), re-run from the manifest digests |

## Boundary rules

1. Predictions are model outputs; treat them as private until the campaign is
   explicitly published. GT never crosses in either direction.
2. The Linux scorer result is **Linux scorer evidence**; it is combined with
   Windows inference evidence only through the shared hash chain (predictions
   manifest ↔ scored input digest), never by relabeling platform scope.
3. A Linux scorer PASS never becomes "Windows scorer PASS"; product status
   keeps `Native Windows eight-metric scorer: BLOCKED` and records the bridge
   route explicitly.
4. Empty-prediction documents stay in the denominator; the campaign report
   must show documents attempted / passed / failed per state alongside metrics.
5. Until a complete campaign: all metrics, Overall and deltas remain null.

## Why not Option A (native Windows scorer)

Faithfulness risk: hosting TeX/Ghostscript/Node on Windows and touching
fork/SIGALRM paths inside the official scorer changes the measured thing.
Cost: large new dependency surface to pin and maintain. Benefit over the
bridge: none for correctness. BLOCKED / NOT_RECOMMENDED.
