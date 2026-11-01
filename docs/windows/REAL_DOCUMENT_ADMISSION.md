# Bounded real-document qualification — reconciled admission state

Status: **PREPARATION COMPLETE; GPU EXECUTION STILL NOT AUTHORIZED**. The
earlier envelope on this page was a scope proposal with deferred identities.
It has been superseded by the reviewed revision-2 policy
(`WINDOWS_REAL_DOCUMENT_POLICY.md` in the local delivery-closure ledger) and
is now reconciled with it below. This page records the reconciled envelope
and prerequisite state; it is not a run authorization.

## Reconciliation — this branch's earlier envelope vs revision 2

The committed envelope and the revision-2 policy disagreed on six axes. In
every case the revision-2 value governs; each deviation from the original
envelope is individually recorded in the policy's deviation register (D1–D4)
and requires sign-off with the authorization request.

| Axis | Earlier envelope (superseded) | Revision 2 (governs) | Why |
| --- | --- | --- | --- |
| Requests / token budgets | Two calls: 8192 baseline **and opt-in 16384**; no 39321 | Exactly two arms, **both 8192**: arm 1 ngram 0 (production default), arm 2 ngram 35/window 1024; no 16384 arm, no 39321 (D1, D2) | Retained Linux evidence for this exact document reaches native stop only near 10021 output tokens; an 8192/16384 pair would blur the diagnostic. Native-stop demonstration is deferred to a separately authorized >=16384 arm with a >=750 s ceiling |
| Timeouts / duration | <=600 s per arm, <=20 min total | **1200 s per arm, 45 min total** including model load and receipts (D4) | Retained same-document Linux wall is 551.7–572.3 s at 8192 — under 5% margin at 600 s; two arms cannot fit 20 min |
| Memory limits / monitoring | >=64 GiB admit; stops <48 GiB available, >16 GiB reserved; 1 Hz + allocator peaks; 4 GiB host-RSS soft limit | Same thresholds; **4 GiB RSS rule dropped** (D3, contradicted by 11.2 GiB UMA working-set peak); **staleness rule >=5 consecutive missed samples => unknown => stop**; `ResourceWatchdog` enforcement wired into the supervised runner (prerequisite P5, implemented) | No Windows host-only RSS measurement exists; UMA working sets legitimately exceed 4 GiB |
| Correctness acceptance | "No truncation/repetition acceptance; length is TRUNCATED" (implied native-stop success framing) | **Predeclared expected outcome: both arms TRUNCATED-at-cap**; TRUNCATED completes the arm's diagnostic record and is never PASS; the pilot's satisfiable objective is the memory/wall envelope (Q1), failure semantics (Q2) and Linux parity (Q3) | A pilot whose success criterion contradicts retained evidence is unsatisfiable by construction |
| Execution / stop sequence | Stop-after-failure (single clause) | Arm 1 -> arm 2 sequential; **TRUNCATED does not cancel arm 2**; a process-level failure (FAIL/TIMEOUT/OOM/INTERRUPTED) in arm 1 cancels arm 2 and blocks the ledger; total-wall expiry kills the in-flight arm as INTERRUPTED | Exact two-arm sequencing with preserved partial evidence |
| Frozen identities | Deferred ("the proposer must freeze…") | Frozen now: document `handwritten_chn_1ae487be` (5 ordered pages, per-page SHA-256 + `ordered_paths_sha256` in the selection record), original AMD wheel `415d2dbd…`, model revision `9cf077c4…` with all 18 file hashes, frozen execution commit, scorer route `SCORE_NOT_RUN` | The admission receipt (prerequisite P4) is minted at authorization time over these identities |

Stale reference corrected: the envelope cited "PR #24 OPEN" as the source
dependency. The ngram/budget/guard/watchdog path is now **merged into main
via PR #32** (origin/main `a2b91aa`), which closes prerequisite P1.

## Prerequisite state

| ID | Requirement | State |
| --- | --- | --- |
| P1 | PR #32 (or successor) merged so the ngram arm exists on the execution tree | **CLOSED** — main `a2b91aa` merged #32; the combined execution tree carries the ngram/budget gates and `ResourceWatchdog` |
| P2 | PR #28 admission contract vendored/integrated by a reviewed commit; dry-run re-executed against the real gate | **CLOSED** — vendored byte-identical at the execution tree (admission.py sha `9df35b9b…`, branch head `39584ac`); real-gate dry-run 18 cases / 14 verified refusals / DRYRUN_PASS (receipt in the integration ledger) |
| P3 | Explicit GPU execution authorization | **OPEN by design** — see the authorization request; the exact executable pilot packet is prepared for the owner's decision |
| P4 | Admission receipt minted (policy + selection + contract hashed) | **OPEN by design** — minted at authorization time |
| P5 | 1 Hz telemetry with staleness rule + watchdog wired into the supervised runner | **CLOSED** — `benchmark/telemetry.py` + runner wiring on the execution tree; 12-case CPU suite |
| P6 | Fresh synthetic golden rerun at the frozen execution commit on Windows GPU | See the integration-closure ledger and `PRODUCT_STATUS.md` for the recorded outcome at commit `6d8e481…` |

Known correction carried into the packet: the policy text's pinned PR #28
branch-head hash contains a one-character typo (`…a1723d319…` for the actual
`…a1727d319…`); the vendored file hash `9df35b9b…` is the binding identity.

One exact executable pilot packet (frozen Git/wheel/model/document/policy
identities, exact commands, resource limits, failure handling, scorer route,
evidence destinations, reviewer GO/NO-GO) has been prepared and returned to
the owner. **No GPU real-document request has been executed.**
