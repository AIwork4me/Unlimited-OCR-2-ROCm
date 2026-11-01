# Windows Bounded Real-Document Admission Policy (Phase 5, revision 2)

Design + CPU dry-run only. **GPU execution is NOT authorized by this
document.** Revision 2 incorporates all six required changes from
`WINDOWS_REAL_DOCUMENT_INDEPENDENT_REVIEW.md` (verdict REVISION_REQUIRED);
the revised protocol has not been re-reviewed — re-review precedes any
authorization decision.

This protocol concretizes the envelope committed on the delivery branch
(`docs/windows/REAL_DOCUMENT_ADMISSION.md`) **with recorded deviations**
(see Deviation register).

## Research question (as narrowed by the review)

Retained Linux evidence for this exact document shows: at 8192 new tokens it
finishes `length`/TRUNCATED (551.7 s ngram0 arm; 572.3 s ngram35 arm), and
reaches native `stop` only at ~10021 output tokens (~731–740 s at ≥12288
budgets). The pilot therefore does **not** ask "does it stop natively at
8192" — the open, decision-relevant questions on Windows are:

- Q1 Memory/wall envelope: does a real dense 5-page document stay within the
  frozen resource envelope on Windows (evidence so far: synthetic-only,
  10.51 GiB reserved peak)?
- Q2 Failure semantics: are TRUNCATED/`length` outcomes classified and
  preserved exactly (never accepted as PASS) with complete accounting?
- Q3 Parity: do outputs and envelope metrics match the retained Linux
  evidence for the same document under the same policy?

**Predeclared expected outcome: both arms TRUNCATED-at-cap.** A native stop
would be a surprise worth investigating; it is not the success criterion.
Demonstrating native stop is explicitly deferred to a future, separately
authorized ≥16384-token arm with a ≥750 s ceiling.

## 5.2 Exact execution contract

| Element | Frozen value (revision 2) |
| --- | --- |
| Host | Windows 11 (build 26200), Ryzen AI MAX+ PRO 395, Radeon 8060S gfx1151, driver 32.0.22032.6002, ROCm 7.14.1 / HIP 7.14.60850 |
| Torch wheel | original AMD `torch-2.12.0+rocm7.14.1-cp310`, full SHA-256 `415d2dbd4b6ab002cff1c54795fdd66322abd9a46bf982cb64eebb4493ac2e7c` (candidate `6a67d158…` forbidden; full-hash compare, no prefix) |
| Model | revision `9cf077c4b206e0e2a31879df937b0b0bcba3661c`, all 18 file hashes verified pre-load |
| Source | the frozen execution commit, defined at authorization time as: delivered `src/` tree + merged PR #32 ngram/budget gates + vendored PR #28 admission contract (exact commit SHA recorded in the admission receipt; no claim of byte-identity to `3556e38`) |
| Document | `handwritten_chn_1ae487be`, 5 ordered pages, identities per `WINDOWS_REAL_DOCUMENT_SELECTION.json` (incl. `ordered_paths_sha256`) |
| Arms | exactly 2, sequential: **arm 1** = 8192 tokens, ngram 0 (production default); **arm 2** = 8192 tokens, ngram 35 / window 1024 (pinned policy). No 16384 arm, no 39321 arm (deviation register D1) |
| Sequencing | arm 1 then arm 2. **TRUNCATED completes the arm's diagnostic record and does NOT cancel arm 2** (TRUNCATED is never PASS). A process-level failure (FAIL/TIMEOUT/OOM/INTERRUPTED) in arm 1 cancels arm 2 (stop-after-failure) and blocks the ledger pending review |
| Inference | native Transformers BF16, scoped MATH vision SDPA, native DASH/Triton/SR-SWA, original weights, sequential pages 1→5 |
| Memory stops | admit only ≥64 GiB system available at start; hard stop at <48 GiB available or >16 GiB allocator reserved (1 Hz samples); **no process working-set cap below 16 GiB** — Windows UMA golden evidence peaked 11.2 GiB working set / 10.51 GiB reserved; the envelope's 4 GiB host-RSS rule is dropped as undefined-and-contradicted (deviation register D3) |
| Telemetry | 1 Hz sampler (available memory, allocator reserved/peak, wall clock, token count); **staleness rule: ≥5 consecutive missed samples ⇒ treat as unknown ⇒ stop**; enforcing component: PR #32 `ResourceWatchdog` wired into the supervised Windows runner (prerequisite P5) |
| Timeout | per-arm ceiling **1200 s** (retained Linux 8192-token wall 551.7–572.3 s; ~2× margin for unmeasured Windows real-doc speed); total wall clock **45 min** including model load and receipts; per-arm ceiling governs its arm; total-wall expiry kills the in-flight arm as INTERRUPTED; TIMEOUT never masquerades as length |
| Failure rules | native authoritative stop or `length` recorded verbatim; `length` ⇒ TRUNCATED (recorded, never PASS); FAIL/TIMEOUT/OOM/INTERRUPTED preserve partial outputs + accounting; stdout/stderr captured per attempt |
| Cancellation | host-side cancel = Job terminate, 10 s grace; partial attempt retained as INTERRUPTED |
| Ownership | Windows Job Object: suspended create → assign → resume; KILL_ON_JOB_CLOSE; owned-descendant drain; supervisor pre-assignment death gap documented |
| Artifacts | new private run directory under the approved local evidence root (never the Git tree); raw outputs, telemetry, receipts private; strict-mode export (empty file on failure) produced at run time OR deterministically re-derivable from retained raw output under the same identity |
| Scorer route | `SCORE_NOT_RUN` declared for this pilot (gate asserts the decision); any future scoring uses the Linux bridge with separately authorized transfer |
| Completion metadata | per attempt: finish reason, token count, DASH dispatch count, allocator peaks, telemetry series, page-bytes re-verification, stdout/stderr, attempt accounting, ledger seal |

## 5.3 Gate design (fail closed — refuses execution when)

1. Source identity differs from the frozen execution commit.
2. Model identity differs (any of the 18 hashes).
3. Wheel identity differs (full SHA-256 compare).
4. Any dataset identity differs (per-page SHA-256 + `annotation_sha256` +
   `ordered_paths_sha256` + GT row digest).
5. Scorer route decision not declared (`SCORE_NOT_RUN`) / bridge readiness
   receipt absent for any future scoring arm.
6. Resource or timeout policy missing or not a valid object (non-empty dict
   with the frozen thresholds; positive numeric timeout).
7. Admission receipt absent, unreadable, or hash-mismatched (unreadable ⇒
   refuse, never crash).
8. Correctness prerequisite failed: **a fresh synthetic golden rerun executed
   on Windows GPU at the frozen execution commit** (binding: 26/26-equivalent
   receipt for that exact tree; the `3556e38` 26/26 receipt is historical
   context only and cannot substitute — it predates the #32/#28 code the
   execution tree must carry).
9. Another owned GPU experiment is active (exclusive device ledger refuses
   any non-IDLE state — RUNNING **and** blocked-after-failure).
10. Output would be written outside the approved private directory.

**Admission mechanism:** the independently reviewed fail-closed design of
PR #28 (`codex/experiment-admission-safety` @ head
`39584aca3a1723d319b925a54d880bf2caa23dce`,
`src/unlimited_ocr_2_rocm/benchmark/admission.py`): identity-bound `_verify`
(sources/wheels/fixtures/policy/artifacts/target groups, absolute paths,
per-file hashes, single-read policy), SQLite ledger with exclusive device
ownership, `Rejected` ⇒ no worker starts, crashed RUNNING blocks future work,
recovery never invents PASS, trusted-controller assumption (the controller
owns the ledger and immutable files). PR #28 is not in main and not installed
by the delivery branch: at implementation time the contract must be
**vendored or integrated under a separate reviewed commit** binding the exact
source; relying on the unmerged side branch is forbidden. The dry-run receipt
in this ledger validates the policy's gate logic only (a mock
re-implementation); the real vendored gate must be dry-run again before
authorization, including TOCTOU (receipt/policy swap), concurrent gate
instances, and `ordered_paths_sha256` cases.

## Prerequisites (all open — honest blockers)

- P1: PR #32 (or successor) merged so the ngram arm exists on the execution tree.
- P2: PR #28 admission contract vendored/integrated by a reviewed commit;
  dry-run re-executed against the real gate.
- P3: explicit GPU execution authorization (see
  `WINDOWS_GPU_AUTHORIZATION_REQUEST.md`).
- P4: admission receipt minted (policy + selection + contract hashed; scope
  digest recorded).
- P5: resource telemetry (1 Hz sampler with staleness rule) and PR #32
  `ResourceWatchdog` wired into the supervised Windows runner by a reviewed
  commit (the delivered runner has neither).
- P6: fresh synthetic golden rerun at the frozen execution commit on Windows
  GPU (also supplies the first Windows GPU evidence for the #32 code paths).

## Deviation register (vs the committed envelope — requires sign-off with the authorization request)

| ID | Deviation | Rationale |
| --- | --- | --- |
| D1 | No 16384-token arm (envelope allows one opt-in 16384 arm) | smallest first run; retained evidence already characterizes 12288/16384 on Linux; native-stop demonstration explicitly deferred |
| D2 | Baseline arm uses ngram 0 (envelope: both arms ngram35/window1024, only budget differs) | arm-1 measures the production default; the experiment's axis is ngram on/off at fixed budget — inverted from the envelope, recorded here |
| D3 | Envelope's 4 GiB host-RSS soft limit dropped | contradicted by Windows golden working-set peak 11.2 GiB (UMA shared memory); no Windows host-only RSS measurement exists; replaced by available-memory/reserved stops + watchdog |
| D4 | Timeouts raised: 1200 s/arm, 45 min total (envelope: ≤600 s/arm, ≤20 min) | retained same-document Linux wall 551.7–572.3 s at 8192 leaves <5% margin at 600 s; alternate candidate hit 600.9 s; two arms cannot fit 20 min |

## Success criteria (attainable, review-aligned)

The pilot succeeds as a *diagnostic* if both arms complete within the frozen
envelope (or fail with exactly-preserved semantics), telemetry is complete,
memory stays within stops, and the Windows results are recorded against the
retained Linux evidence for parity review. It produces no official metric,
score, or comparability claim.

Evidence that would justify expanding to 15 pages: parity with retained Linux
evidence for this document (predicted TRUNCATED-at-cap reproduced or
explained), memory and wall within the frozen envelope with measured margin,
marker/page-guard behavior clean (on the #32-merged tree), complete failure
semantics, and a reviewed decision on the ≥16384-token native-stop question
(which requires its own arm and ≥750 s ceiling, separately authorized).

Evidence that would justify a full dataset: official protocol-equivalence
resolution (windowing/KV/ngram semantics), the Linux scorer bridge executed
end-to-end on a bounded document set with full coverage, multi-document
memory/wall diversity evidence, and campaign authorization.

## Revision record

- rev 1 (2026-11-01): initial policy; CPU dry-run found and fixed three
  fail-closed defects (null-valued resource policy admitting; unreadable
  receipt crashing instead of refusing; post-failure re-admission because only
  the RUNNING ledger state was refused). Dry-run receipt:
  `WINDOWS_REAL_DOCUMENT_ADMISSION_DRYRUN.json` (15 cases, 12 refusals
  verified, DRYRUN_PASS).
- rev 2 (2026-11-01): independent review REVISION_REQUIRED — applied all six
  required changes: (1) evidence-based timeouts 1200 s/45 min + predeclared
  TRUNCATED expectation + precedence rules; (2) gate 8 re-bound to a fresh
  golden rerun at the execution commit, Source row fixed; (3) two-arm
  sequencing specified (TRUNCATED does not cancel the sibling arm);
  (4) 4 GiB RSS rule dropped with evidence; (5) P5 telemetry/watchdog
  prerequisite with staleness rule; (6) deviation register added, request
  shape corrected. Also adopted notes: full wheel hash in gate 3,
  strict-export condition, recast expansion criteria. **Not yet re-reviewed.**
