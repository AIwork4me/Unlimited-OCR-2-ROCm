# Benchmark foundation implementation plan

**Goal:** implement issue #6 P0 evaluation infrastructure before any full GPU run.
**Spec:** issue #6 and the user's Benchmark Protocol Freeze mission.
**Architecture:** a standard-library manifest and append-only run ledger, with a
separate optional process that imports the hash-verified original official scorer.
The existing inference adapter and all historical evidence stay unchanged.
**Tech stack:** Python 3.10; existing Ruff/mypy/pytest/build CPU CI.

## Constraints and design decisions

- Work in the primary checkout, branch from merged main; preserve user files.
- No model/torch dependency in CPU CI; no GPU inference in this task.
- Frozen GT determines document IDs and page order. No directory-sort inference.
- Every input remains in summaries, including missing, interrupted and failed
  attempts. Only a verified PASS may be resumed without another attempt.
- Raw bytes and metadata commit together through directory rename; retries append.
- Dataset, protocol, environment and source hashes participate in run identity.
- Import original metric classes and official matching/task code; do not reimplement
  metrics. Missing CDM renders BLOCKED and null Overall, never a zero substitute.
- Record the original Qwen evaluator and Baidu Marathon patch as distinct profiles.
- Full MPDocBench image payload checksums cannot be asserted before acquisition.

## Tasks

- [x] Audit all current validation/RCA/wheel reports and draft #5; freeze source
  identities, GT hashes and metadata-only complete document inventories.
- [x] Write failing manifest tests (ordering, duplicates, traversal, wrong counts,
  corrupt/missing bytes); implement inventory and validation.
- [x] Write failing ledger tests (all states, retries, interruption, tampering,
  cross-dataset collision, config drift); implement atomic attempts and reports.
- [x] Write failing scorer adapter/fixture tests; import original scorer in a
  separate optional environment; retain actual dependency and CDM blockers.
- [x] Write protocol, matrix, execution guide and G1–G5 readiness mapping.
- [x] Run all six CPU checks, review complete diff, fix findings, create PR and
  inspect exact-head Windows/Ubuntu CI. Never merge or release.

## Review focus

Unsafe IDs/symlinks; Windows case collisions; shuffled/missing page metadata;
partial writes and corrupt PASS outputs; successful subset mistaken for full score;
CDM missing tools hidden behind zero; changed source/dependency identities.

## Execution record

User mission supplies execution authorization; no additional design approval is
needed. Existing user files were backed up under ignored `.migration/benchmark-start`
and stashed as `benchmark-preserve-user-files` before switching from #5 to main.
AGENTS.md is restored locally and excluded from this PR. Inline implementation;
a fresh reviewer checks the complete branch before delivery.

Delivered in PR #7. All six local checks pass (177 tests; five intentional
skips); the fresh reviewer cleared the import-integrity fix. Implementation
head `4c71757` passed both OS package jobs and both official-scorer integration
jobs. Final metadata/evidence commit checks are verified separately at delivery.
The acquired MPDocBench archive matches its official SHA256: 409 documents /
3,051 pages hashed; 11 gated SlideVQA decks / 84 pages remain BLOCKED. No GPU
benchmark, release, merge, wheel rebuild or upstream contact occurred.
