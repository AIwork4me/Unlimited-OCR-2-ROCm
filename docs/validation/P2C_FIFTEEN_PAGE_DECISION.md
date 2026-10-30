# P2-C Phase 3 decision — 15-page Native admission

**Date: 2026-10-30. Inputs: `P2C_COMBINED_RCA.md` (ledger runs bccb2252,
140f8c20), `P2C_GUARD_COLLISION_SAFETY.md`, the P2-B decision record and the
immutable ngram=0 controls. Newspapers evaluated first, per the admission
order.**

## Gate evaluation

### Newspapers (the gate carriers)

| Gate | newspaper_chn_eb87830d | newspaper_eng_ed12985c |
| --- | --- | --- |
| Completes all expected pages | **FAIL** — 3/5 at 16384, page yield ~0 from 12288→16384 | PARTIAL — 4/4 pages emitted at 16384, finish_reason=length cuts the last page's content |
| Long/short repetition controlled | PASS — no cycle at 16384 (was 86.7 % loop at ngram=0) | PASS — no cycle at 16384 (was 93.1 % loop at ngram=0) |
| Text/layout quality acceptable | conditional — every landed page scores, diagnostic ReadOrder 79.17 at stage level | conditional — same |
| No forced-PAGE inconsistency | PASS — 0 mismatches, 0 overshoot, all 8 attempts | PASS |
| Memory/time admissible | PASS — 8.55/8.77 GiB peak, ≤ 2.51 GiB RSS, ≤ 1 540 s/doc | PASS |

**Newspaper page-completion gate: FAILED at the maximum tested budget
(16384).** newspaper_chn needs more than 16 384 tokens to reach page 4 of 5;
newspaper_eng reaches all pages but not a natural stop. Neither is a
suppression failure — repetition is fully controlled and the guard is
consistent — it is a budget-class fact.

### Controls (side-effect check)

- handwritten and paper complete naturally at 12288 with **byte-identical
  stop points at 16384** (10 043 / 8 202 tokens); paper's official edit
  distance is flat at 0.041 across all stages — suppression changed nothing
  that the budget did not pay for.
- Official scorer: no metric regressed at stage level versus P2-B@8192;
  diagnostic improvements across the board (TextEdit 59.85→66.63,
  ReadOrder 59.17→79.17). No unwanted suppression side effects detected.

## Verdict

**The 15-page Native ladder remains BLOCKED** at the P2-C budgets: the
newspaper page-completion gate is unmet at 12288 and 16384, so a 15-page
one-request identity at these budgets would inherit the same failure.

What changed versus the P2-B decision: every failure that remains is a
budget-class failure. Suppression (gate 1) now passes with margin — no
repetition cycles at 16 384 tokens, ban engagement ≤ 30 steps, zero
guard↔ban inconsistencies; controls complete naturally with byte-stable
stop points; resources are flat. The official 39 321-class budget is the
only untested lever, and it is also the budget where the Phase-1
guard-collision corner becomes reachable (page cap 20 000 < 39 321), which
the L1 spec must treat as a live risk with the replay detector armed.

**Admissible next step: prepare — but do not execute — the L1 15-page
specification** at the reviewed official budget class with suppression
enabled as one supervised identity, per `L1_FIFTEEN_PAGE_SPEC.md`.
No 30/50/100-page escalation is sought or justified by this evidence.
