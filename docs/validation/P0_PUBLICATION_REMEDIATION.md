# P0 publication remediation — MPDocBench GT slice in PR #24

**Status: containment prepared locally; containment commit ready to push.**
**Historical rewrite: NOT performed — requires explicit user authorization.**
**Repo: PUBLIC (github.com/AIwork4me/Unlimited-OCR-2-ROCm), fork count 0 at
audit time.** Exposure window began 2026-10-30T10:14:57Z (commit 4d203a1
pushed ~10:20:39Z).

## 1. What was published

`eval/results/Linux gfx1151 verified/p2b-ngram/p2b-gt-slice.json` — a
663,053-byte JSON export of complete MPDocBench ground-truth annotations for
the four frozen P2-A documents:

- `layout_dets` for every page block: full transcription text, label,
  category, order, bounding boxes (e.g. handwritten doc: 243 blocks,
  newspaper_chn: 375, newspaper_eng: 371, paper_chn: 133);
- `page_info`: image paths, page geometry, annotations inventory;
- `extra.relation`: full read-order graphs (238/397/386/125 edges).

That is source-document text and layout data — licensed evaluation dataset
content, not hashes or coverage metadata. It entered the public tree through
commit `4d203a1` ("docs+evidence(p2b): physical n-gram suppression RCA on the
four frozen documents"), the single commit of open PR #24.

Per-file audit of all 20 files introduced by PR #24 (this commit's diff
against its base `feat/p2a-token-budget` @ ec38e30):

| Class | Files | Verdict |
| --- | --- | --- |
| GT annotations | `p2b-ngram/p2b-gt-slice.json` (663,053 B, 36,853 CJK chars) | **non-publishable dataset content** |
| Replay statistics (4) | `ngram-replay-*.json` (391–450 B) | clean — numeric counters only |
| Scorer evidence (4) | `score-*-coverage/-scores.json` | clean — counts + doc filenames only |
| Receipts (2) | `run-receipt.json`, `p2b-evidence.json` | clean — sanitized (gpu name nulled, no paths, no host id) |
| Policy/config (1) | `eval/experiments/p2b-ngram-suppression/policy.json` | clean |
| Code (5) | 2 scripts, 2 src modules, 1 test module | clean — no embedded data, no host paths |
| Docs (3) | RCA, source audit, policy identity audit | clean — repetition quotes ≤ 8 CJK chars / short ASCII fragments (diagnostic fragments, PR #17 precedent) |

No image content, no raw token arrays, no OCR output files, no private host
information in any other PR #24 file. Other `ground-truth.json` files already
tracked in the repo (`eval/benchmarks/g05/scorer/{preliminary,final}/…`,
3,315 B each) are synthetic scorer self-test fixtures, not dataset content;
they are hash-pinned in the publication whitelist.

## 2. Commits and refs containing the content

| Location | Ref/commit | Reachable |
| --- | --- | --- |
| Local worktree `wt-p18` branch | `feat/p2b-ngram-suppression` = 4d203a1 | yes (will advance with containment commit) |
| Remote branch | `origin/feat/p2b-ngram-suppression` = 4d203a1 | yes until containment push |
| GitHub PR ref | `refs/pull/24/head` = 4d203a1 | **yes — survives branch deletion** |
| GitHub PR merge ref | `refs/pull/24/merge` = 481cce9 | yes while PR open |
| Forks | fork count 0 | none known |

No other commit in any branch ever touched the file
(`git log --all -- <path>` → only 4d203a1; `git branch -r --contains 4d203a1`
→ only `origin/feat/p2b-ngram-suppression`).

## 3. Containment performed (local, reversible, no history rewrite)

1. **Preservation outside Git** — the exact committed bytes plus every P2-B
   scorer input/output were archived at `../p0-evidence/` (outside all repos):
   the slice (`sha256 5f2211f8337e…`, byte-identical to the committed copy and
   to the `.b/` worker copy), `wt-p18-b/` (prepare manifest, runs, control
   predictions, export diagnostics), `scorer-runs/` (both official scorer
   passes), and a 1,881-file `SHA256MANIFEST.txt`.
2. **Replacement by identity receipt** — `p2b-gt-slice.json` deleted from the
   tree; `p2b-gt-slice-receipt.json` committed in its place carrying only:
   schema/label, dataset, experiment, run id, policy digest, full-GT digest
   (`0a579dca…`, independently confirmed by the run's prepare manifest),
   real-slice digest (`5f2211f8…`), per-document page/block/relation counts,
   claim limit, and a note. Anyone holding the licensed dataset can rebuild
   the slice and verify it against the receipt; the annotations themselves are
   no longer redistributed.
3. **Generation is scripted, not hand-edited** — `scripts/sanitize_gt_slice.py`
   → `unlimited_ocr_2_rocm.publication.build_gt_slice_receipt`, self-checked by
   `validate_receipt` (exact key set, 64-hex digests, bounded ASCII prose,
   counts only).

## 4. Prevention added

- **Structural detector** (`src/unlimited_ocr_2_rocm/publication.py`): flags
  any annotation-shaped export (list of rows with `page_info`+`layout_dets`),
  any image-suffixed file (extension-based), any Unix user home-directory
  path (a slash-home or slash-Users prefix followed by a username). Scope:
  the tree scan covers `eval/`; the hook scans staged files; other trees
  (e.g. `tests/fixtures/`) are outside the gate's published-evidence scope
  by design. Not a filename blocklist — it fires on any future slice regardless of
  name or location.
- **Whitelist, hash-pinned with reasons**: 4 synthetic scorer fixtures + 16
  historical files that predate the gate (GitHub-Actions
  runner-home provenance, build-container default-user paths, synthetic
  redaction-test macOS user-path strings, and the validation-host home-path
  receipts already reviewed in PR #17 with the decision
  "history left intact; future receipts sanitized"). Any edit to a whitelisted
  file re-trips the gate.
- **`.gitignore`**: `eval/results/**/*gt-slice.json` and
  `eval/results/**/*gt-slice-*.json`, re-including `*-gt-slice-receipt.json`.
- **Versioned hook** `.githooks/pre-commit` → `scripts/publication_gate.py
  --staged` (enable per clone with `git config core.hooksPath .githooks`);
  enforced centrally regardless of local hooks by
  `tests/unit/test_publication_gates.py` (10 tests: detector positives/negatives,
  whole-`eval/` tree scan, whitelist hash enforcement, receipt validation,
  gitignore semantics via `git check-ignore --no-index`, CLI end-to-end).

## 5. Minimal remediation plan for the PR #24 history

Ordered by escalation; steps 1–2 are prepared and need only a normal push
(no history rewrite, no authorization question beyond the standing no-merge
rule). Steps 3+ rewrite public history and are **blocked on explicit user
authorization** per the mission's hard rules.

1. **Containment commit (ready)** — push the local commit that replaces the
   slice with the receipt, adds the gate, and documents this plan. Removes the
   content from the branch head and from the PR's file/commit views as
   rendered by default. ~45-minute exposure window so far, 0 known forks.
2. **Keep PR #24 open** (per no-auto-merge) until independent review of this
   remediation lands; the PR base (`docs/p1-ngram-source-audit`, PR #23) is
   unaffected.
3. **Optional, authorized-only: purge history.** Because 4d203a1 is the PR
   head and the branch's only commit beyond its base, a surgical rewrite is
   unusually clean: rebase `feat/p2b-ngram-suppression` onto `ec38e30`
   dropping the file from the commit (or amend the commit to the receipt
   version), then `git push --force-with-lease`. The local authoritative
   commit stays intact in `wt-p18` history. This makes the branch ref clean
   but does NOT clean GitHub's PR refs (below).
4. **GitHub cached-reference risk (applies even after any force-push).**
   - `refs/pull/24/head` is immutable from the client side and retains
     4d203a1 indefinitely, even after the branch is deleted or rewritten;
     anyone who knows the PR number can
     `git fetch origin refs/pull/24/head`. The PR "Files changed" view also
     keeps rendering the old diff for the old head until the PR is closed,
     and closed PRs keep their refs.
   - Additional server-side caches that do not honor ref updates on request:
     raw.githubusercontent.com blob URLs for the old commit (e.g.
     `raw.githubusercontent.com/AIwork4me/Unlimited-OCR-2-ROCm/4d203a1/<path>`)
     can remain resolvable, and fork/PR network objects may persist in
     GitHub's object database until garbage collection.
   - **Assessment: GitHub Support escalation is warranted.** The content is a
     licensed-dataset redistribution concern in a public repo. Support can
     remove cached views/objects under `refs/pull/*` and run early GC on
     unreachable objects — actions no repository permission can perform.
     Mitigating factors for the ticket: 0 forks, exposure < 1 hour before
     containment, single file, four documents out of a 420-document corpus,
     no full-corpus redistribution. Suggested request: purge
     `refs/pull/24/head` + `refs/pull/24/merge` and cached raw views of
     commit 4d203a1 after the history rewrite (if authorized), or of the
     blob in any case.
5. **Do not** rewrite any other branch/PR: the file exists nowhere else.

## 6. CI

Full local CI (ruff check + format, mypy, complete pytest suite) runs before
the containment commit is pushed; CI on GitHub re-runs the publication gate
via the new unit tests on both CI operating systems.
