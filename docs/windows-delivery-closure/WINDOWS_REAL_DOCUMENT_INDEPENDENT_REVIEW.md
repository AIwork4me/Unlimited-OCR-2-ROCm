# Independent Review — Windows Bounded Real-Document Admission Protocol (Phase 5.5)

Reviewer: separate fresh-context agent (no stake in the work), 2026-11-01.
Scope: `WINDOWS_REAL_DOCUMENT_POLICY.md`, `WINDOWS_REAL_DOCUMENT_SELECTION.json`,
`WINDOWS_REAL_DOCUMENT_ADMISSION_DRYRUN.json`, scorer readiness/bridge
designs, the committed envelope (`REAL_DOCUMENT_ADMISSION.md`), benchmark
protocol, PR #28 admission source, evidence matrix. The reviewer independently
re-hashed the dry-run seal, cross-checked all five page byte-sizes/SHA-256
prefixes against the public page-index receipt, re-verified annotation counts
against retained GT evidence, extracted retained Linux timings for the exact
selected document, and confirmed PR #28's head.

## Verdict

**REVISION_REQUIRED**

Approval of this review is design-stage only and does **NOT authorize GPU
execution**.

## Summary of findings (full answers retained in the review transcript)

1. Real unresolved product question — PARTIAL: Windows memory/wall envelope on
   a real dense document, failure-semantics preservation, cross-platform
   parity are open; BUT retained Linux evidence already answers the stated
   headline question negatively at 8192 tokens: this exact document finished
   `length`/TRUNCATED at 8192 (551.7 s ngram0; 572.3 s ngram35) and reaches
   native `stop` only at ~10021 output tokens (~740 s at ≥12288 budgets). The
   policy must disclose this prediction.
2. Sample representativeness — YES with caveats (deterministic rule verified;
   hardest qualifying candidate; n=1 supports no corpus claim; max-stress
   results will not generalize to typical documents).
3. Scorer executable for this pilot — NO, and the policy handles it correctly
   (SCORE_NOT_RUN declared; native Windows scoring proven BLOCKED in source).
4. Scorability later without changing official semantics — YES (deterministic
   converter, hash-manifested per-doc Markdown, unchanged Linux scorer);
   condition recorded: strict-mode export at run time or deterministically
   re-derivable.
5. Timeouts/resources — PARTIAL: memory stops evidence-grounded (golden peak
   10.51 GiB reserved; Linux real-doc 8.78 GiB at full 8192); **4 GiB host-RSS
   soft limit unsafe** (Windows golden working set peaked 11.2 GiB — UMA
   shared memory counts); **timeouts unsafe as frozen** (<5% margin at 600 s;
   both-arms arithmetic exceeds 20 min; likeliest outcome TIMEOUT-instead-of-
   length, destroying the diagnostic).
6. Fail-closed admission — PARTIAL: 10 gates + 12 dry-run refusals verified
   real; three found-then-fixed defects confirmed in the harness; still
   missing: gate 8 unsatisfiable as written (golden receipt binds `3556e38`,
   execution tree must carry #32+#28 — can never match); dry-run was a mock,
   not PR #28's code; no TOCTOU/concurrency/ordered-paths coverage; in-run
   1 Hz sampler/staleness criterion unbound; two-arm ledger sequencing
   ambiguous; wheel gate used an 8-hex prefix.
7. Failed run explainable — YES (states, partial outputs, accounting, seal);
   specify stdout/stderr capture and arm ordering.
8. Artifacts private and reproducible — YES (hashes/prefixes only public;
   full digests in private receipt; identity chain complete).
9. 15-page expansion evidence — PARTIAL: right kinds of criteria but "native
   stop at 8192" bar is predicted-unattainable; recast in attainable terms;
   any native-stop demonstration needs ≥16384 tokens and ≥750 s under separate
   authorization.
10. Full-dataset evidence — YES as necessary conditions (protocol-equivalence
    resolution, bridge executed end-to-end on a bounded set, campaign
    authorization) plus multi-document diversity evidence.

Envelope deviations found (must be recorded for sign-off): (a) no 16384 arm
(envelope allows opt-in 16384); (b) baseline arm ngram0 vs envelope's
"both arms ngram35/window1024, only budget differs"; the "same request shape"
claim was false. Referenced `WINDOWS_GPU_AUTHORIZATION_REQUEST.md` did not
exist at review time.

## Required changes (from the reviewer, verbatim)

1. Reconcile time limits with retained evidence (cite 551.7/572.3/600.9/~740 s
   timings; raise ceilings with rationale or predeclare expected
   TRUNCATED-at-cap; define TIMEOUT-vs-length precedence and total-wall kill
   semantics).
2. Make gate 8 satisfiable (fresh synthetic golden rerun at the frozen
   execution commit as prerequisite; fix the self-contradictory Source row).
3. Specify two-arm sequencing under the ledger (TRUNCATED must not cancel the
   sibling arm; process-failure must).
4. Define or drop the 4 GiB host-RSS limit (contradicted by 11.2 GiB Windows
   working-set peak; no Windows host-only measurement exists).
5. Add the in-run watchdog/1 Hz telemetry sampler as an explicit prerequisite
   with a staleness criterion and a named enforcing component.
6. Record the envelope deviations for sign-off and correct the request-shape
   claim.

Recommended notes: retain records of the three dry-run defects; re-run dry-run
against the real vendored PR #28 gate; add TOCTOU/concurrency/ordered-paths
cases; use full wheel SHA-256; create the authorization request; recast
15-page criteria; confirm strict-mode export condition.

## Disposition

All six required changes and the adoptable notes were applied to
`WINDOWS_REAL_DOCUMENT_POLICY.md` (revision 2, same day) — see its
"Deviation register" and "Revision record" sections. The revised protocol has
**not** been re-reviewed; re-review is a precondition of any authorization
decision. GPU execution remains BLOCKED.
