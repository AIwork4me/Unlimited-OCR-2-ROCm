# Independent review — P0 remediation + P2-C combined + 15-page decision + L1 spec

**Reviewer: independent agent, 2026-10-30, against branch
`feat/p2b-ngram-suppression` @ 3ed19dd (worktree wt-p18). All numbers
re-derived from raw artifacts (git objects, `.b/` run trees, scorer trees,
`p0-evidence/`); read-only; no GPU code executed. Full review text preserved
in the session record; verdicts and load-bearing findings below.**

## Verdicts

| Document | Verdict |
| --- | --- |
| `P0_PUBLICATION_REMEDIATION.md` (d0ffc1e) | **APPROVE-WITH-NOTES** |
| `P2C_COMBINED_RCA.md` (3ed19dd) | **APPROVE-WITH-NOTES** |
| `P2C_FIFTEEN_PAGE_DECISION.md` (3ed19dd) | **APPROVE-WITH-NOTES** |
| `L1_FIFTEEN_PAGE_SPEC.md` (3ed19dd) | **APPROVE-WITH-NOTES** |
| `test_publication_gates.py`, `test_p2c_combined_policy.py` | **APPROVE** (18/18 pass) |

Bottom line (reviewer): no fabricated or unverifiable PASS claims; the
containment, digest chains, gate behavior, replay/scorer evidence,
TRUNCATED discipline, BLOCKED decision and design-only L1 status all check
out against raw artifacts.

## Verified (reviewer-re-derived)

- P0: file absent at HEAD; slice digest 5f2211f8… == preserved copy ==
  historical blob 4d203a1:…; receipt counts exact (243/375/371/133 blocks;
  238/397/386/125 edges); gate PASS on `--scan eval` and correctly flags
  synthetic annotation-shaped JSON, a PNG, and a private path via `--paths`;
  gitignore semantics verified with `git check-ignore --no-index`; no other
  tracked annotation content; exposure window ~43 min (push reflog).
- P2-C: counts/termination identical to receipts; handwritten stop@10,043
  and paper stop@8,202 at BOTH budgets with identical `raw_sha256`
  (byte-identity); 0 forced mismatches/overshoots in all 8 replays; GPU
  peak 8.546 GiB at both stages; scorer table matches `scores.json` to two
  decimals; TRUNCATED never relabeled (`runner.py:106-110`, `ledger.py`
  demotion); 16384 stage verifiably started only after the 12288 gates
  passed; policy digests 2068efad…/5e7eec56… match the tracked files.
- Decision + L1: BLOCKED verdict follows from the evidence; context
  arithmetic 47,917 ≤ 131,072 (and `max_length=131072` confirmed in the
  frozen snapshot); guard-collision reachability confirmed against the
  frozen snapshot source (cap 20,000 at `modeling_unlimitedocr.py:1127`,
  force condition line 600, ban-before-guard order lines 1169–1172).

## Findings and disposition (all folded in the follow-up commit)

1. **(a) RCA resource paragraph misstated three measurements** — decode
   floor is 10.4 tok/s (not 12.6; rate falls with KV growth
   14.6→10.4 across 8192→16384), RSS peak 2.548 GiB (not 2.51), wall range
   635–1,587 s (not 745–1,540). FIXED in `P2C_COMBINED_RCA.md`; the
   decode-rate trend is now stated explicitly and the L1 timeout margin was
   re-derived from the 10.4 floor (≈1.85×, with a pre-flight re-derivation
   requirement) instead of the incorrect ≥2.3×.
2. (b) "identical per-metric populations" heading contradicted artifacts
   (eng text_block 284→287 across budgets). FIXED — scoped to the GT slice,
   cross-stage deltas labelled diagnostic-only.
3. (b) Loop-share attribution: 86.7–93.1 % is newspaper_chn only; eng was
   ~27.5 % (552×). FIXED in RCA and decision doc.
4. (b) P0 doc: synthetic fixtures are 3,315 B (not 1,817); "any binary
   image" overstated (suffix-based); whitelist is 4+15=19 entries (not 16).
   FIXED, with a gate-scope note (eval/ tree + staged files; tests/fixtures
   outside scope by design).
5. (c) Prefill max 4.79 s; "19–30 banning steps" applies to newspapers only
   (controls 1–2); KV estimate is ≈2.5 GiB (GiB/GB slip). All FIXED.
6. Reviewer notes GitHub-side facts (fork count, PR-ref retention) rest on
   host observation, and the SGLang 10-page/39,321 deployment semantics
   come from the pinned wheel/audit chain — both are documented as such in
   the P0 doc §5 and the L1 spec §7.

The 3,315 B fixture size, the decode-rate floor and the timeout margin
corrections change no gate outcome: no TIMEOUT occurred at 7,200 s, RSS
2.548 ≪ 4 GiB gate, and the L1 authorization checklist (§8) already
requires a fresh pre-flight before the single physical run.
