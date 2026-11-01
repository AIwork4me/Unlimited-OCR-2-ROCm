# Phase 0 Baseline Audit — Windows Delivery Closure Mission (2026-11-01)

Read-only verification of the exact local state before any publication or
benchmark-protocol work. Receipt: `.migration/windows-product-20261101/phase0-reverify.json`
(status PASS), produced by `.migration/windows-product-20261101/phase0_reverify.py`
before any new mission files were created in the main checkout.

## 0.1 Exact local state — PASS

| Item | Verified value | Status |
| --- | --- | --- |
| Main checkout branch/HEAD | `codex/overnight-closure-20261031` @ `254fd0c6433713b8602b8d5e425f5092a632b629` | PASS (identical to sealed `before.json`) |
| Original 42 dirty files | all 42 paths byte-identical (7 modified + 35 untracked) | PASS |
| Git index | SHA-256 `3ef5596b588e2ec1ddebd44815305655860c0262f66ae2de2dd923a4aebb3c57` | PASS (unchanged) |
| Historical 406 hashes | `.b/p230` (68) + `.b/p240` (338) = 406, zero mismatches | PASS |
| All prior seals (22 SHA256SUMS documents) | zero mismatches, byte-identical envelope hashes vs `.migration/publication-20261101/integrity-final.json` | PASS |
| Wheel/model/fixture/source identities | identical to prior receipt: model revision `9cf077c4b206e0e2a31879df937b0b0bcba3661c` (18 files), original torch wheel `415d2dbd…` (11,926 installed members re-hashed, zero mismatches), candidate `6a67d158…` separately verified (11,935 members) | PASS |
| Prior PR #28/#29 worktrees | heads unchanged, trees clean, both Draft/Open/unmerged | PASS |
| Golden v1 failure seal | byte-exact | PASS |
| Worktrees | 13 registered worktrees, all heads unchanged (note: `wd` advanced from `52d593e` in `before.json` to the delivery head as expected — the six delivery commits were created in it by the prior mission) | PASS |
| Remote `origin/main` | `52d593e0686176b3842d78fbac3ef76956ecf1ae` (unchanged) | PASS |

### Delivery branch chain — PASS

`delivery/windows-native-benchmark` in `.worktrees/wd`, all six commits on base
`52d593e0686176b3842d78fbac3ef76956ecf1ae`, parent links exact:

| # | Commit | Subject (from Git) |
| --- | --- | --- |
| 1 | `4abba4fce64eacaf4291e2ac30888a897963bd8b` | fix(windows): own benchmark workers and descendants with Job Objects |
| 2 | `c4310278e4529db20956745053135ab322ec3b04` | feat(windows): verify original wheels before fresh offline installation |
| 3 | `3556e3867a086437a43a645fa907689f4dd25260` | feat(windows): expose PowerShell benchmark entry and bind frozen annotations |
| 4 | `5436f80233477cf53f0e5eda4406c27a9e09c035` | fix(windows): contain inherited Hugging Face benchmark caches |
| 5 | `d2caa79d329278061f76ccd1a13a432e1fa98d64` | docs(windows): record clean-install GPU validation and full-benchmark blockers |
| 6 | `0ba45dd43baf2509de9495c15a19cdc382430601` | docs(windows): provide a scoped single-page user command |

Review-ready patch bodies (`.migration/windows-product-20261101/review-ready/0001-…0006-*.patch`)
are byte-identical to each commit's diff (verified, format-patch trailer excluded).

## 0.4 Mission ledger convention

New mission artifacts live under `docs/windows-delivery-closure/`. The exact-equality
receipt above was captured **before** this directory existed; from this point on,
integrity re-verification must assert the 42 original entries are an unchanged
*subset* of the dirty inventory (new untracked mission files are expected and are
enumerated in EXECUTION_STATUS.md).

**Gate 0 status: PASS — publication work may proceed.**
