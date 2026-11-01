# 03 — Mainline Integration Report: Windows Regression + Linux gfx1151 (Phase 3)

- Date: 2026-11-01, Asia/Shanghai
- Integration tree for validation: `main @ 2efd2d1` (#7 → #8(→#31) → #9 → #10 → #11 → #12 → #13 merged) — all GPU/CPU validation below ran on this tree with the fresh environment.
- Phase-3 final tree: `main @ 5e3b5ee` (`17eb55f05c…`), i.e. 2efd2d1 + PR #14. The #14 delta is exactly three files (`git diff 2efd2d1..5e3b5ee --stat`): the clean portable lock, its reproduction doc, and the gate's lock.txt whitelist pin — **zero runtime code changes**, so the validation results below remain valid for the final Phase-3 tree (gate re-run PASS, 327 tests PASS on 5e3b5ee).
- Fresh environment: `envs/linux-gfx1151-rocm-fresh` (CPython 3.10.20, uv), built strictly per the #14 portable-lock procedure: verified wheels installed with `--no-deps` first, then the #14 lock (`linux-cp310.lock.txt` without the removed git-adapter line — the adapter removal IS the #14 fix), then `pip install --no-deps -e ./integration` from the audited checkout @ `2efd2d1`.

## Merge ledger (Phase 3 portion)

| PR | Head | Merged into main as | Tree verified |
|----|----|----|----|
| #10 | d8485be6 | 29e28dc | bf96f2d0 == local trial |
| #11 | 3ad8809e | a701b86 | d1056213 == local trial |
| #12 | 31b793e5 | 7b977ab | 3c31ee17 == local trial |
| #13 | 0ec78b0a | 2efd2d1 | 8ee1ae6e == local trial |
| #14 | 45d03a0 + pin update + branch-merge with main | 5e3b5ee (server merge commit 2535706 on the branch: identical tree 17eb55f0/parents to the locally verified merge; built via Git Data API because the git upload path was degraded) | 17eb55f0 == final main tree |

Incident recorded: PR #8 was merged into its stale base (`codex/benchmark-protocol`, merge commit `532d59d`) because `gh pr edit --base` failed silently (GitHub Projects-classic GraphQL deprecation). main was unaffected; recovery = PR #31 carrying the unchanged #8 head `afbafac9` into main (merged as `722f3f0`). All later base retargets use the REST API (`PATCH /pulls/N`) with explicit base verification before merge.

## Fresh environment identity (evidence: /tmp/audit/fresh_env2.log)

- torch 2.12.0+rocm7.2 (hip 7.2.53211, rocm 7.2.1) — wheel SHA 099e265d… (from linux-stack.txt, file verified before install)
- triton (triton-rocm) 3.7.0 — wheel SHA b7a9e90c…
- torchvision 0.27.0+rocm7.2 — wheel SHA 7c871ca2…
- `pip check`: **PASS** ("No broken requirements found"; pip 26.2.2 installed into the uv venv first)
- Wheel/source identity: package resolves to the audited checkout (`pip install --no-deps -e ./integration` @ `2efd2d1`); NO machine-local wheel URL and NO git editable adapter in the resolved environment (the #11-era lock's `-e git+…@8092ec2` adapter is present in the pre-#14 lock still on main — the exact defect #14 removes; the fresh env deliberately used the #14 lock content and confirmed the resolved stack matches the audited pins).

## GPU gates on the integrated tree (fresh env, model snapshot 9cf077c…)

Output dir: `eval/results/Linux gfx1151 integration-20261101/` (untracked working evidence).

| Gate | Result | Receipt |
|----|----|----|
| A — doctor (stack identity + real BF16 GPU matmul) | PASS (rc=0) | doctor.json |
| B — Triton compile+execute on gfx1151 | PASS | triton.json (`status: PASS`) |
| C — DASH sparse prefill vs independent math | PASS | sparse.json (`status: PASS`) |
| D — vision SDPA scoped MATH regression | PASS | vision_sdpa.json (`status: PASS`) |
| E — single-page OCR suite (6 cases) | PASS | single/*/manifest.json |
| F — multi-page native OCR suite (2/3/5 ×2 + table) | PASS | multi/*/manifest.json |

## Golden regression vs the #11-frozen evidence (byte-identical)

All 13 cases compared `raw_output` + `output_token_count` + `finish_reason` against `eval/results/Linux gfx1151 verified/` (the receipts merged via #11):

- single: chinese 82 tok, english 80, math 72, mixed 84, repeat_unicode_recovery 80, table 60 — **raw byte-identical ×6**, finish stop ×6
- multi: pages_2 ×2 (116), pages_3 ×2 (171), pages_5 ×2 (286), table_2_pages (94) — **raw byte-identical ×7**, finish stop ×7

## Windows-vs-Linux boundary statement (mission 3.1)

The Windows physical golden regression itself was executed on the Windows host in the G1 task (sealed evidence `eval/g1/windows-gfx1151/evidence.tar.gz`, independently reviewed PASS, merged via #10). This Linux session did NOT re-execute Windows GPU work and makes NO Windows re-validation claim; what this session re-proved is the Linux physical counterpart on the same frozen fixtures, byte-identical to the prior Linux evidence.

## #12 / #13 acceptance

- Dataset verification receipts (#12) and doc corrections (#13) audited in Phase 1 (host-identity receipts hash-pinned in the gate whitelist).
- KV-cache 7.5 GiB formula doc (#13) reviewed as documentation semantics only; no runtime change (its diff is docs + receipts).
- Confusion guard: the verified environment (this host, pinned stack) vs the compatibility matrix (e.g. torch 2.14/rocm7.14 line, SGLang) remain explicitly separated in the docs; no mixing introduced.

## Status

- Phase-3 merges #10/#11/#12/#13/#14: ALL DONE with per-step local trial-merge tree equality + publication gate PASS + full CPU suite green (327 passed / 7 skipped on the final 5e3b5ee tree).
- GitHub connectivity incident (2026-11-01): the git upload path (push) failed for ~1 h (GnuTLS curl 56 / connect timeouts) while the REST API stayed reachable; the #14 pin update and its branch-merge were therefore constructed server-side via the Git Data API with byte-identical tree/parents to the locally verified commits. One merge response was lost client-side while succeeding server-side (reconciled by re-querying state).
- KEEP EXPERIMENTAL; no production default changed (model.py change in #11 is platform pin dispatch only; profiles stay PREPARED_ONLY).
