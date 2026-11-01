# 01 — Independent Publication Audit (Phase 1, P0 gate for all mainline merges)

- Audited at: 2026-11-01, Asia/Shanghai
- Auditor: independent session context; no prior agent PASS verdicts were reused as conclusions.
- Scope: every file the 14 mainline PRs (#7 #8 #9 #10 #11 #12 #13 #14 #16 #17 #18 #19 #20 #23) contribute on top of `main @ beb38d19`, plus the pre-existing `main` tree that the new publication gate must admit.
- Method: full blob-level scans against the resolved post-merge tree (2,115 unique paths: 1,588 on main + 543 incoming), with per-category classification; all `git` name lists built with `core.quotePath=false` after a quoted-path gap was found and fixed mid-audit.

## 1. Method and coverage guarantees

- Per-PR diffs were recomputed from remote refs (two-dot vs base head; base drift = 0 on all 15 audited PRs, so two-dot == PR content).
- The union of the 12 chain PRs equals the cumulative stack diff `main..#23` (522 paths, verified both directions); sibling PRs #10 (+22) and #14 (+2 files, 1 shared modification) were audited separately.
- Every blob in scope was classified by a prototype detector (annotation shape, JSONL annotation lines, nested annotation rows, private host paths POSIX+Windows, host-identity keys, credential material, binary content), and the 49 files with hits were read and individually dispositioned.
- Content spot-checks were run on the highest-risk artifacts (PR #10 evidence archive: all 489 members extracted and scanned; PR #19 real-sample receipts: walked for long text fields; PR #9 GT-adjacent fixtures: read).

## 2. Category classification (mission 1.1 taxonomy)

| Category | Finding | Disposition |
|---|---|---|
| Production source + tests | 13 `src/`, 12+ `tests/`, 20+ `scripts/` files — self-authored wrapper code | PUBLISH (repo MIT license) |
| Self-authored synthetic test data | `tests/fixtures/benchmark/synthetic.json` and 4 scorer `ground-truth.json` fixtures are `authored_synthetic` annotation-schema fixtures; G1 archive raw outputs are OCR of repo-owned fixture PNGs | PUBLISH — pinned by hash in gate whitelist (annotation-shape matches require the pins) |
| Third-party source | `_upstream/` (pre-existing main): unmodified Baidu postprocessor at pinned commit, MIT `LICENSE-Baidu.txt` included. Official scorer source is NOT vendored anywhere in the 14 PRs — only hash mappings (`producer-source-mapping.json`, `source-verification.json`) | PUBLISH (proper attribution/license); no new third-party source enters via the 14 PRs |
| Public benchmark metadata + hashes | `MPDocBench-page-index.json` (707 KB), `MarathonDocBench-page-index.json` (3.1 MB), `SlideVQA-page-index.json`, `manifests/*.jsonl` — paths, byte sizes, SHA-256s, page counts only; NO annotation text, NO images | PUBLISH (metadata/hashes; enables integrity verification without distributing dataset content) |
| Real dataset GT | NONE in the 543 incoming files. The only `page_info`/`layout_dets` payloads are the 5 synthetic fixtures above; the 204-byte `gt/bbox/*.jsonl` are trivial token fixtures with no annotation shape | BLOCKED BY DESIGN (gate) — verified absent |
| OCR raw output / transcripts | Raw text exists ONLY for repo-owned synthetic fixtures (G1 tarball, `multi/pages_N` manifests). Real-document evidence (#16/#18/#19/#20) carries receipts, digests (sha256 per file), scores and predeclarations — walked for long text fields; none present. `token_ids` arrays: 0 hits in the whole scope | PUBLISH as-is; real-doc raw outputs remain under untracked `.b/` |
| GPU/env/scorer diagnostics | Logs sanitized at generation time: `<repo>`, `<home>`, `<USERPROFILE>` placeholders (143 placeholder hits in the Windows-Chinese evidence set; scorer logs verified) | PUBLISH |
| Private paths / host identity | 22 files (detailed in §3): 5 historical Linux receipts, 2 prose docs, 6 CI/container provenance records, 3 synthetic redaction-test logs, 4 self-authored sanitizer tests, 2 docs — no credentials, hashed machine-id only (`machine_id_sha256`, one-way) | RETAINED behind hash-pinned whitelist; gate blocks all NEW occurrences |
| Credentials/secrets | 0 hits (`ghp_`/`gho_`/`github_pat_`/`hf_`+20, AWS AKIA, PEM markers) across all 2,115 files | CLEAN |
| Unclear provenance | `eval/g1/windows-gfx1151/evidence.tar.gz` (#10): 707 KB, 489 members, sealed; carries its own independent publication review (PASS: every member hash-verified, 16 PNG exclusions hash-bound); this audit independently re-extracted and re-scanned all members (synthetic fixture content, no user paths, no restricted data). `eval/rca/gfx1151-rocm7141/pdf-followup/input/Unlimited-OCR-2.pdf` (pre-existing main): user-supplied third-party research paper, provenance and non-relicensing statement documented in its README | PUBLISH with standing disclosure (see §6) |

## 3. The 49 whitelist entries (exact-hash-pinned, each with a reason)

Full inventory is embedded in `src/unlimited_ocr_2_rocm/publication.py` (`WHITELIST`):

- 5 annotation-shape fixtures — authored synthetic (`data_source: authored_synthetic`): 4 scorer self-test `ground-truth.json` (3,315 B each, identical hash) + `tests/fixtures/benchmark/synthetic.json`.
- 12 `tests/fixtures/*.png` — repo-owned synthetic OCR fixture images (required by the test suite).
- 1 `eval/g1/windows-gfx1151/evidence.tar.gz` — sealed G1 evidence (independent review PASS).
- 1 `Unlimited-OCR-2.pdf` — user-supplied third-party paper (disclosed provenance).
- 8 `eval/wheel-validation/.../logs/*.gz` — historical wheel-build logs.
- 7 provenance/container records with historical CI-runner/container paths (`eval/rca/gfx1151-rocm7141/provenance-*.json` ×5, `baseline-provenance.json`, `build-tool-preinstall.json`).
- 3 `*-redaction-red.txt` — synthetic redaction self-test content.
- 5 historical Linux validation receipts (`dataset-verification*.json` ×3, `page-repair-mp3lan_0165-43.json`, `mpdoc-sample/run-receipt.json`) — contain local home paths + host name; PR #17 review ruled "history intact, future receipts sanitized"; this audit concurs (no credential, no GT; evidentiary value is in hashes/metrics).
- 2 prose docs (`docs/validation/LINUX_DATASET_READINESS.md`, `docs/reviews/PR16_INDEPENDENT_REVIEW.md`) — historical host-layout prose.
- 4 `tests/unit/test_*.py` — sanitizer tests using synthetic path literals.

Notes:
- `requirements/linux-cp310.lock.txt` is dual-versioned across the merge sequence: the pre-#14 chain version (which contains the historical local wheelhouse URL that #14 exists to remove) is what `main` holds between the #11 and #14 merges, so it is pinned as an explicitly **transitional** whitelist entry; the #14 merge must be preceded by updating that pin to the clean portable lock (`549072ad…`), which is folded into the #14 merge step.
- `machine_id_sha256` receipt fields are one-way hashes; the gate does not flag them (documented decision — a hash cannot be dereferenced, and the prior review accepted them).

## 4. PR #24 gate extraction: reproduction of known defects and fixes

1. `--scan .` whitelist-anchoring defect (recorded as `scripts/publication_gate.py:80` in PR #24): `Path(".").resolve().relative_to(repo_root).as_posix()` evaluates to `"."`, producing `./`-prefixed report paths that miss every whitelist key — the command fails on exactly the whitelisted fixtures it should admit, which must not be disguised as a successful scan. **Reproduced** against the PR #24 code by inspection of the anchoring arithmetic and the recorded failure; **fixed** in the extraction (both `scan_tree` default-prefix derivation and the CLI anchor via the script location; `--scan .` and `--scan $PWD` are now equivalent), with a regression test (`test_gate_cli_scan_dot_anchors_whitelist` / `test_scan_repo_root_anchors_whitelist`).
2. `--staged` skipped whitelisted files entirely in PR #24 (a tampered whitelisted file passed stage-time checking and relied on a later CI tree scan). **Fixed**: staged whitelisted files are hash-verified at commit time (`test_staged_whitelisted_tamper_blocked`).
3. Detection strengthened beyond PR #24: recursive nested annotation rows, JSONL annotation lines, Windows home paths, host-identity keys, credential material, and binary files fail CLOSED (PR #24 admitted unknown opaque binaries silently). Every strengthening was verified to add zero new hits on the current scope, so the whitelist above is complete for the post-merge tree.
4. PR #24's whitelist referenced files that must exist independently of #24 (they all pre-exist on main or arrive via the 14 PRs); each of its 19 entries was re-verified against the post-merge tree — all 19 hashes are byte-identical to the new pins (0 drift), and the gate re-verifies every pin at each merge, so any future drift fails closed.

## 5. Historical GT exposure (mission 1.3 — audit and record only)

- Leak commit `4d203a1` (blob `4117e16f…`, 663,053 B GT slice of MPDocBench) was rewritten out of `feat/p2b-ngram-suppression` on 2026-11-01 under owner authorization (head now `4903976f`, tree unchanged; post-push server-side object search returned 0). Local refs still carrying pre-rewrite history are preserved untouched and are never pushed.
- GitHub Support ticket **#4934458 remains OPEN**; the ticket covers cached-view/PR-ref purge (cached raw view and blobs API were live at submission). **Ticket submitted ≠ data purged**; this mission does not modify PR #24 history, re-publish original files, or attempt GitHub-internal cache eviction. Standing local rules stay in force: no commits may be based on pre-rewrite SHAs.
- This audit found the leaked GT content class (annotation-shaped rows) to be structurally absent from all 14 PRs and from main; the gate exists precisely to keep it that way.

## 6. Verdict

**GO** for the Publication Safety Gate PR and for proceeding to the mainline merge sequence, under the following standing conditions:

1. Every future merge re-runs the gate (CI step `python scripts/publication_gate.py --scan .` + `pytest tests/unit/test_publication_gates.py`) on the merged tree; any whitelist hash drift fails the build.
2. The PDF (`Unlimited-OCR-2.pdf`) and the identity-bearing receipts remain hash-pinned disclosures; if the owner later wants them removed, that is a history-rewrite decision outside this mission's authority.
3. The whitelist must not be extended casually; new entries require an independent audit reason.
4. GitHub ticket #4934458 status is re-checked before the final report; it does not block merges (local + server head history are already clean).
