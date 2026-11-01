# 01 — Publication Gate Independent Review Record

- Reviewer: fresh read-only review agent, not an author of the code; all conclusions re-derived from primary refs (no author claims trusted).
- Branch under review: `security-fix/publication-gate` (commits `aedb72c`, `d54e9b7`, `d3fc870`, final hardening commit) on top of `main @ beb38d19`.
- Verdict after two rounds: **GO** — branch safe to push as PR and merge.

## Round 1 (full review)

Verdict **GO**, zero P0/P1. Independently reproduced by the reviewer:

- Post-merge tree rebuilt from the 14 PR refs: 2,115 paths; the reviewer's own detector (not the module's) found **exactly 49 hits vs 49 whitelist entries** — 0 unwhitelisted hits, 0 no-hit entries, 0 credential hits.
- Both PR #24 defects confirmed and the fixes verified: `--scan .` anchoring (regression tests), staged whitelisted-file hash enforcement (fails under PR #24 semantics, passes here).
- All 49 hash pins verified against blobs the reviewer resolved independently (49/49 + 10 explicit `sha256sum` confirmations); `evidence.tar.gz` re-extracted: 489 members, all clean, matches its own recorded independent publication review (PASS).
- Test suite: 156 passed / 5 skipped (23 publication tests, including two real staged pollution-injection e2e tests in throwaway clones); ruff/mypy clean; `--scan ..` and external roots fail closed.
- Round-1 P2 findings (all inherited from PR #24, CI-backstopped): staged rename gap (`--diff-filter=ACM`), staged TOCTOU (worktree hashing), `--paths` silent PASS on directories/missing, lock.txt pinned to the pre-#14 version. Doc accuracy: §4.4 pin-drift claim not reproducible (0/19 drifted); two bullet counts off by one.

## Fixes folded (commit `d3fc870`)

- `scan_staged` rewritten to read INDEX blobs (`git cat-file :path`, `-z` listing without `--diff-filter`): rename smuggle now caught at the new path; TOCTOU closed (worktree scrub/delete/tamper all still flagged).
- `--paths` refuses directories/missing files (never silent PASS).
- lock.txt pin marked **transitional** (pre-#14 content merges via #11 and is byte-stable through the chain; the clean portable lock `549072ad…` arrives only with #14 — the pin update folds into the #14 merge step).
- Audit doc corrected (counts; drift claim removed; dual-version lock story).
- New regression tests: `test_staged_rename_smuggle_blocked`, `test_staged_scan_reads_index_not_worktree`.

## Round 2 (re-verification of the fixes)

Verdict **GO** — "d3fc870 is safe to push as PR". Adversarial probes in scratch clones: rename smuggle (incl. names with spaces) caught; TOCTOU closed in all three directions; staged deletions/gitlinks/mode-only changes/ITA entries all handled safely; reverse-TOCTOU (index tampered, clean worktree) flagged via index-byte hashing. Transitional pin sequence verified across all 15 refs. Suite 156+5, ruff/mypy clean.

Residual (accepted, defense-in-depth only, primary CI control intact):

- [P2] Non-UTF-8 staged/tracked filenames were silently skipped by the byte→str decode (`errors="replace"` corrupts the path so the blob lookup fails). **Fixed in the final hardening commit** before push: `os.fsdecode`/surrogateescape round-trip in `scan_staged` and `scan_tracked`; `--scan .`/`scan_tree` never had the hole.
- [note] `_index_blob_probe` moved to `tempfile.mkstemp` (no fixed-name /tmp pattern).

## Gate semantics vs PR #24 (no weakening)

Kept: structural annotation detector, exact-hash whitelist, receipt schema 2 (`build_gt_slice_receipt`/`validate_receipt`), `.gitignore` gt-slice containment, pre-commit hook entrypoint, staged/scan/paths CLI modes. Strengthened: nested + JSONL annotation detection, Windows private paths, host-identity keys, credential material, binary fail-closed, `scan_tracked` (tracked-tree authoritative scan), index-based staged verification, `--scan .` anchoring fix, `--paths` strictness. Removed: nothing.
