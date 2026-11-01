# Windows Integration Closure — Execution Ledger (2026-11-01)

Mission: finish Windows product integration (PR #33 preparation), approve and
implement the official Linux scorer bridge (Option B), and prepare exactly one
safe real-document pilot for final execution authorization. States:
PASS / FAIL / PARTIAL / BLOCKED / NOT_RUN / UNVERIFIED.

Evidence roots referenced below are private and hash-sealed
(`.b/integration-20261101/SHA256SUMS.json`); this ledger publishes digests and
conclusions only. The prior mission ledger
(`docs/windows-delivery-closure/` at local commit `03d8cd8`) was verified
locally before this mission began: all 23 sealed documents match their
recorded SHA-256 (that commit is local-only by design; this ledger reconciles
it against the real GitHub integration tree).

## D1 — PR #33 integration preparation — PASS

- Actual combined tree constructed as a two-parent merge of origin/main
  `a2b91aa` (PR #32 merged) and PR #33 head `a987dca`: conflict-free, tree
  `7eb5c1cd` byte-identical to the prior simulation; merge commit `028e0d8`
  on `codex/windows-integration-20261101`.
- Hardening coexistence verified from both parents: PR #32 strict
  JSON/duplicate-key/Windows-path detection + `ResourceWatchdog`; PR #33
  selected-root scan fix + hash-pinned WHITELIST.
- Combined publication gate + full CPU CI: ruff, format, pytest **521 passed
  / 10 skipped / 0 failed** (test imports pinned to the combined tree; the 10
  unpinned-run failures were the pre-documented environment-binding artifact),
  mypy win32+linux, build, pip check, repo-root publication scan — PASS.
- Hosted CI on PR #33 head after the documentation update (`c80a93c`): all 8
  jobs PASS (cpu ubuntu/windows, scorer qwen/marathon, two run groups).
- Windows GPU regression within the existing approved synthetic scope:
  executed as the fresh golden rerun at the frozen execution commit (see P6).
- Stale documents updated on the delivery branch (`c80a93c`, pushed; PR #33
  body updated): PRODUCT_STATUS, WINDOWS_BENCHMARK_PROTOCOL,
  REAL_DOCUMENT_ADMISSION (full reconciliation, see D3), WINDOWS_HANDOFF.
- Independent review of the combined tree: **APPROVED** — 0 P0 / 0 P1 / 6 P2
  (pre-existing test fragility ×2, vendored-supervisor drain-grace
  documentation note 30 s→10 s fail-closed-earlier, deviation-only CLI shape
  validation, non-strict export semantics, pre-existing POSIX exit-86
  asymmetry). No blockers to merge readiness.
- PR #33 remains **Draft**; no merge, no release.

## D2 — Official scorer bridge, Option B — PASS (implementation + synthetic tests)

- Implemented `benchmark/bridge.py` + `scripts/scorer_bridge.py` on the
  execution tree (`59e22c1`, `1788a5a`): strict-mode prediction export (failed
  documents = explicit empty files, kept in denominators and the failure
  catalog; publication-detector sanitization refuses a dirty export before
  anything is written), hash-bound manifest, Linux-side verification (per-file
  hashes + inventory equality against LOCAL ground truth; GT never crosses),
  eight-metric + Overall comparison report with explicit scope labels
  (`linux-cpu-official`; Overall null while any metric missing; never a
  Windows-native scorer claim).
- Real eight-metric positive control: the frozen original scorer's verbatim
  synthetic selftest `metric_result.json` (license-safe numeric output,
  frozen as `tests/fixtures/scorer_selftest_metric_result.json`) passes
  through unchanged — seven real values, FormulaCDM null without the Linux
  CDM renderer, Overall null by design until scored on the qualified Linux
  host. 9/9 tests + CLI round-trip; reviewer confirmed no silent-exclusion or
  GT-leak path in strict mode.
- No GT or prediction transfer executed; nothing published.

## D3 — Real-document pilot preparation — COMPLETE; EXECUTION NOT AUTHORIZED

- Sealed ledger re-verified locally (23/23 hashes at `03d8cd8`); the
  one-character typo in the policy's pinned PR #28 head hash was confirmed
  and corrected in the packet (binding identity: the vendored file hash
  `9df35b9b…`, branch head `39584ac…7d319…`).
- **P1 CLOSED**: PR #32 merged (`a2b91aa`); the execution tree carries the
  ngram/budget gates — proved by the combined tree's own content and the
  vendoring diff.
- **P2 CLOSED**: admission contract vendored byte-identical (`b6ae19d`);
  real-gate dry-run re-executed against the REAL Gate with the production
  NativeJob: **18 cases / 14 verified refusals / 0 reason mismatches,
  DRYRUN_PASS** (receipt hash `4aae6a16…`), including policy-swap and
  receipt-swap TOCTOU, concurrent ledger instances, ordered-pages swap,
  crash-FAIL terminal state, resume-after-failure refusal, and the positive
  golden→dependent-arm dependency shape.
- **P5 CLOSED**: 1 Hz telemetry with staleness rule + watchdog wired into the
  supervised runner (`b9fc306`): memory-floor and telemetry-stale process
  stops (`WORKER_RESOURCE_STOP_*`), watchdog breach → OOM classification,
  12-case CPU suite; reviewer found no correctness bugs.
- **P6 CLOSED**: fresh synthetic golden rerun at the frozen execution commit
  `6d8e481` (GOLDEN_PROTOCOL_V3; same wheels/environments/model/13-case
  contract, prior v2 PASS verified immutable): **w1 PASS + w2 PASS = 26/26
  equivalents, exact cross-environment parity (13/13), allocator envelope
  identical to v2 (10.510 GiB peak reserved / 8.589 GiB peak allocated),
  522 s**; receipt `95858192…`. This is simultaneously D1's approved-scope
  GPU regression. 26 original infer entries; zero real-document requests.
- **P3/P4 remain open by design** (owner authorization; receipt minted at
  authorization time).
- Envelope-vs-policy reconciliation published in
  `docs/windows/REAL_DOCUMENT_ADMISSION.md` (six axes, all resolved to
  revision 2 with deviations D1–D4 recorded).
- **Pilot packet prepared and returned for owner GO/NO-GO** (private; exact
  identities, both arms at 8192 tokens, resource limits, failure semantics,
  evidence destinations, reviewer table). Policy re-review verdict recorded
  in the packet; **no GPU real-document request executed**.

## Documentation and publication — DONE (review PR pending merge decision)

- Delivery-branch docs updated and pushed (`c80a93c`); PR #33 body updated;
  PR #33 Draft preserved.
- This ledger publishes only independently reviewed, license-safe content;
  the review PR is `codex/windows-closure-docs-20261101` → main. No automatic
  merge or release.

## Invariants maintained

No force-push/rewrite of any branch; PRs #28/#32 untouched; 42 dirty files
and the sealed prior ledger preserved byte-exact; no wheel/model
substitution; no tolerance relaxation; no production default change; the only
GPU requests were the 26 approved-scope synthetic golden entries at the
frozen execution commit; no real-document inference, no 15-page/39321 arm,
no GT or prediction transfer, no private data in Git; unrun official metrics
remain null.
