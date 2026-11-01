# 06 — L1.1-B Independent Review Record

- Reviewer: fresh read-only review agent; did NOT execute or observe the run; every conclusion re-derived from primary artifacts with independent hashing.
- Object: the ONE authorized L1.1-B physical GPU run (`paper_eng_64b5e1f4`, ledger `e9dde491…`, run tree `feat/l1-fifteen-page-native @ 39c3813`).
- Verdict: **CONFIRM** — zero P0/P1/P2 findings.

## What the reviewer re-derived (independently)

1. **Identity chain**: packet draft `64bae56a…` and packet receipt `8d80ae7b…` hash exactly as pinned (C1); committed receipt byte-identical to the packet (`cmp`); finalized policy `d95e5bd0…` differs from the draft in EXACTLY the two pre-declared receipt-reference fields; matches `EXPECTED_L11_B_POLICY_SHA256`; registry maps it to `L11_FIFTEEN_PAGE_NATIVE_B`.
2. **Run-tree safety**: `39c3813 = 3993fc4 + exactly 1 commit`; diff = the two policy/receipt files + the +10-line registry block + the new test file (300+/0−); the L1 identity `dd23a546…` unchanged.
3. **Page identity**: all 15/15 page files re-hashed — order-sensitive match against receipt AND policy (spec required only 3).
4. **Attempt integrity**: all 9 COMMIT.json hashes recomputed OK (report had said 8; the manifest seals 9 — corrected); `state PASS`; `finish_reason stop` in both native-report and metadata; `output_token_count 24050`; `raw_page_markers 15/15`; `token_ids.json` length 24050 (15 page tokens 128827, 0 EOS).
5. **Resource safety**: watchdog peaks 8.28 GiB (heartbeat) / 8.49 GiB (metadata) ≤ 16 GiB, breach null; native-report CUDA peak 8.617 GiB, RSS 2.696 GiB; wall 2,912.835 s ≪ 14,400 s; timestamps monotonic.
6. **Observer/replay agreement**: observer aborted=false, collision=null, anomalies=0, guard/ngram/normal all 24,051; replay steps 24,050, forced mismatches 0, first ban step 14,473; the 24,051-vs-24,050 relation PROVEN from the decode loop (terminal EOS step selected before append; EOS never persisted — 0 occurrences of token id 1).
7. **Official scorer**: strict == diag byte-identical; all 8 metrics match the report (TextEdit 99.6937, T-TextEdit 99.9795, TableTEDS 87.5683, FormulaCDM 99.2143, HeadingTEDS 99.1453, ReadOrderEdit 100.0, FigureF1 100.0, T-TableTEDS null, overall null); coverage = 7 tables / 7 display formulas / 0 merged tables; scorer selftest PASS 8/8; sealed copies byte-identical to the untracked originals; `config.json` (contains <MISSION_ROOT>/ private paths) correctly EXCLUDED from the sealed set; 0 private literals in the 9 sealed files; SHA256SUMS 9/9 verified.
8. **Prediction structure**: raw.txt carries exactly 15 `<PAGE>` markers (prediction.md is the marker-stripped converted rendering BY LEDGER DESIGN — a reviewer clarification, not a defect); raw sha256 `bce92196…` == native-report == metadata; token count consistent.
9. **Claim hygiene**: no parity/production/leaderboard/30-50-100-page/performance claims anywhere; ledger 1 PASS + 419 NOT_RUN; case A objectively BLOCKED (class max 14 pages).

## Disposition

The physical result (15/15 pages, natural stop at 24,050/39,321 tokens, zero forced-marker mismatches, zero supervision failures, 7 real numeric official metrics + null-by-rule T-TableTEDS/overall) stands as recorded. The run consumes the single L1.1-B authorization; any further GPU attempt (including a case-A substitute) requires a fresh gate review and new authorization.
