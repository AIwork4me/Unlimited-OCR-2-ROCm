# Windows Draft PR Publication Record (Phase 3)

Published 2026-11-01 after all Phase 0/Phase 2 gates passed. Read back from
GitHub — not inferred from the push command.

## Published identity

| Field | Value |
| --- | --- |
| PR | **#33** — https://github.com/AIwork4me/Unlimited-OCR-2-ROCm/pull/33 |
| Title | Add reproducible Windows Radeon Native OCR installation and supervised benchmark runner |
| Draft | **true** (verified via API after creation; not Ready-for-Review) |
| Base | `main` @ `52d593e0686176b3842d78fbac3ef76956ecf1ae` |
| Head | `delivery/windows-native-benchmark` @ `a987dca00cf5ecf01f456c90462e188261f219d0` |
| Commits | 8 (six validated originals `4abba4f`…`0ba45dd`; reviewed follow-ups `c432861` whitelist fix, `a987dca` product-status docs) — verified commit-by-commit on GitHub |
| Files | 36, matching the audited inventory exactly; no PR #32 files (no `inference.py`/`completion.py`/`watchdog.py`) |
| Mergeable | MERGEABLE per GitHub |
| Body | full description incl. validated/not-validated split, status matrix, benchmark limitations, PR #32 compatibility section, commit structure |

## Pre-publication gates (all PASS before push)

- Phase 0 preservation re-verification (42 dirty files, index, 406 hashes, seals, identities) — PASS
- Quality battery at `a987dca`: pytest 389/8, ruff, format, mypy win+linux, build, pip check — PASS
- Exact Git source archive: 36 changed files byte-exact in sdist; 38 extracted-source tests; full-archive publication scan; `git diff --check` — PASS
- Privacy/provenance audit: introduced objects are all text; no private GT, raw predictions, token sequences, tensors, weights, secrets, private cache paths, machine identifiers or personal directories; `golden-validation.json` is metadata-only (hashes/counts/parity); no staged-index surprises — PASS
- Remote: branch absent before push (no conflict); `origin/main` unchanged at `52d593e` — verified immediately before push
- PR #32 head movement (`a01db7de`→`a720c49c`) recorded and compatibility re-verified pre-push (overlap unchanged; both merge orders → tree `7eb5c1cd`; 460/0/10)

## Hosted CI on the published head

| Check | Result | Run |
| --- | --- | --- |
| cpu (ubuntu-latest) | PASS | 41281751088 / 41281782148 |
| cpu (windows-latest) | PASS | 41281751088 (2m31s) / 41281782148 (2m35s) |
| scorer (ubuntu-latest, marathon) | PASS | both runs |
| scorer (windows-latest, qwen) | PASS | 41281751088 (2m40s) / 41281782148 (2m22s) |

All 8 check runs green; no fix attempts were needed. Note (per mission):
hosted scorer unit checks are not proof that a full Windows official scorer
runtime is installed and executable — XeLaTeX/Ghostscript/latexmlc remain
Phase 4 blockers. No separate hosted publication-gate/source-archive workflow
exists for this head; the equivalent scans ran locally (above) and the scanner
code ships in the PR.

## Reviewer findings / blockers

- None from hosted reviewers yet (Draft, just published).
- Known open items tracked in this ledger: Chinese-only authoritative report
  (PR body carries the English summary), watchdog not wired into the common
  runner (R5), fresh-host acquisition untested (documented in PR body).
- PR left in **Draft**; no merge, no release, no Ready-for-Review.
