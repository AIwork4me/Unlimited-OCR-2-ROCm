# P2-C combined experiment — official suppression × raised budgets on the four frozen documents

**Date: 2026-10-30. Ledger runs bccb2252 (budget 12288) and 140f8c20
(budget 16384). Policy digest 2068efad…, budget source of truth the frozen
P2-A policy (jointly required at the gate). Controls: the immutable ngram=0
P1.5/P2-A evidence and P2-B run 6e1ae22e — untouched, never re-run.**

## Identity (all gates green before any GPU minute)

- Suppression: ngram=35, trailing window 1024, **generated ids only**
  (Phase-1 revalidation `docs/validation/P2C_GUARD_COLLISION_SAFETY.md`:
  frozen decoder functions hash-pinned, 7/7 guard-collision fixtures, the
  forced-`<PAGE>`-while-banned corner structurally unreachable below the
  20 000-token per-page cap — both stage budgets qualify).
- Unchanged against controls: image bytes and page hashes (frozen index
  gate), model revision 9cf077c4, official prompt, `crop_mode=False`
  (infer_multi), eager + DASH Triton decoder, vision SDPA MATH, greedy
  temperature 0, 5-page limit, native page guard, supervised isolated
  workers (7200 s timeout), atomic ledger, per-attempt token-id
  persistence for replay.
- Stage admission was enforced, not assumed: budget 16384 ran only after
  the 12288 stage passed the frozen gates (4/4 documents completed, 0
  supervision failures, 0 forced-marker mismatches, GPU peak 8.55 GiB ≤ 10,
  RSS 2.51 GiB ≤ 4). Both stage gate receipts: `stage-*-gates.json`, both
  PASS with empty failure lists.

## Termination and page coverage (the P2-C question)

| Document | 8192 (P2-B) | 12288 | 16384 |
| --- | --- | --- | --- |
| handwritten_chn_1ae487be | TRUNCATED 4/5 | **stop 5/5 @ 10 043 tok** | stop 5/5 @ 10 043 (byte-identical) |
| newspaper_chn_eb87830d | TRUNCATED 2/5 | TRUNCATED 3/5 | TRUNCATED 3/5 |
| newspaper_eng_ed12985c | TRUNCATED 3/4 | TRUNCATED 3/4 | TRUNCATED **4/4** pages, finish=length |
| paper_chn_3567e55d | TRUNCATED 5/5 | **stop 5/5 @ 8 202 tok** | stop 5/5 @ 8 202 (byte-identical) |

- The two control documents complete **naturally** at 12288 with identical
  stop points at 16384 — budget independence below the binding constraint,
  and zero suppression side effects on their bytes.
- newspaper_eng emits every expected page at 16384; the cap still cuts the
  final page's content (finish_reason=length, never relabeled).
- newspaper_chn stalls at 3/5 between 12288 and 16384: 4 096 extra tokens
  bought zero new pages — the remaining pages of this document are denser
  than ~2 k tokens/page under suppression.

## Repetition, ban-set and guard statistics (replayed from persisted ids)

- **No sustained repetition cycle remains at either stage**: repeated-tail
  analysis finds only nascent 2–4-token cycles at the 12288 cap edge, and
  **nothing at 16384** (`{}` for both newspapers). The ngram=0 controls had
  86.7–93.1 % of output inside a hard loop; the P2-B@8192 run broke the
  loops; P2-C holds them broken across 12 288 and 16 384 tokens.
- Ban engagement stays rare and small: first bans at the replayed steps
  (711 / 6324 / 5421 / 6434 — deterministic prefixes), 19–30 banning steps
  per 12–16 k tokens, max ban size 2, mean ≈ 1.
- **0 forced-marker mismatches and 0 marker overshoots in all 8 attempts**
  (guard↔ban consistency verified per step by the replay detector).

## Resource envelope

Identical to every previous stage of this stack: GPU peak 8.546 GiB
allocated / 8.77 GiB reserved at both budgets (vision-prefill dominated —
the output budget moves nothing), host RSS ≤ 2.51 GiB, prefill ≈ 4.1–4.7 s,
decode 12.6–13.9 tok/s, wall 745–1 540 s per document. No TIMEOUT, no OOM,
no crash, one attempt per document per stage.

## Official scorer (same 4-doc GT slice, identical per-metric populations;
Overall null by rule: T-TableTEDS has no sample)

| Metric (diagnostic) | ngram=0 @8192 | P2-B @8192 | **P2-C @12288** | **P2-C @16384** |
| --- | --- | --- | --- | --- |
| TextEdit | 53.94 | 59.85 | **66.63** | 65.76 |
| T-TextEdit | 45.30 | 50.92 | **55.10** | **58.19** |
| FigureF1 | 58.33 | 67.60 | 81.28 | **82.00** |
| ReadOrder | 51.38 | 59.17 | 71.10 | **79.17** |
| FormulaCDM | 90.64 | 91.35 | 92.71 | 92.71 |
| HeadingTEDS | 29.07 | 31.65 | 34.00 | **37.20** |
| TableTEDS | 46.97 | 46.97 | 46.97 | 46.97 (deterministic tables) |

Strict mode (TRUNCATED ⇒ empty predictions, 2 newspapers) at both stages:
TextEdit 48.73, T-TextEdit 30.01, FigureF1 50.0, ReadOrder 45.58, Heading
27.99 — identical between 12288 and 16384 because the two PASS documents
are byte-identical and the two truncated ones score empty either way.
Per-document official edit distances: handwritten 0.064 → **0.010**
(completion), paper 0.041 flat (byte-identical output — the side-effect
control holds), newspapers still truncated-dominated. Stage-level
populations differ across budgets, so cross-stage deltas are diagnostic,
never leaderboard claims.

## Reading

With the official suppression active, the 8192 budget was the binding
constraint for the two control documents (they complete at 12288 and never
look back). The newspapers are now pure budget cases: no loop remains, the
guard is consistent, and page yield per extra token has fallen to ~0 for
chn — dense-page documents need the official 39 321-class budget, which is
exactly the L1 question, decided separately in
`P2C_FIFTEEN_PAGE_DECISION.md`.

Evidence: `eval/results/Linux gfx1151 verified/p2c-combined/` (sanitized
receipts, per-stage gates, 8 replay reports, both scorer passes with
coverage, combined `p2c-evidence.json`). Raw outputs, token ids and scorer
trees stay untracked under `.b/` and in the external evidence archive.
