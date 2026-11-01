# Final independent scorer and dataset review

**Final decision: PASS, zero unresolved findings.** All scorer controls and both complete dataset byte-identity scopes have passed the independent audits described below. The initially reported renderer cleanup defect and narrow provenance/reproducibility gaps were fixed and rechecked. Full model/GPU quality scoring remains NOT RUN, and this review makes no resource-cap acceptance claim. Exact final receipt hashes appear in the final section.

The earlier sections preserve the review chronology; their pending states are superseded by the final closure decision.

# Independent scorer and dataset review (initial pass)

Reviewed from the primary repository root against HEAD `d842ce6`. The assignment's `7d842ce6` does not resolve; `d842ce6` is PR7's recorded base. No source edits, network/download requests, model execution, GPU work, or background process changes were performed. Review artifacts are confined to `.b/g05-review/`.

## Finding

**[P2] Ensure renderer process groups are cleaned up on interruption and unexpected observer errors.** `scripts/g05_scorer_closure.py:260` starts a new session but its only cleanup branch is `except subprocess.TimeoutExpired` at lines 282-286. `KeyboardInterrupt`, and errors from process inspection/proof writing, escape that block while leaving the isolated child and its TeX descendants running. This is material for the deliberately nonterminating timeout fixture. Put group termination/wait in `finally`, handle an already-exited leader safely, and record that active group members are absent after the timeout.

Concrete reproduction: `.b/g05-review/interruption-repro.json` injects `KeyboardInterrupt` from the first child's `wait()`. The exception escapes and no `killpg` cleanup is requested. The reproduction uses an injected process object and starts no real OS process. The scorer owner acknowledged this finding and is implementing the cleanup and rerunning controls. Status: **open pending owner fix/evidence**.

## Checks and evidence

- `.b/benchmark-cpu/Scripts/python.exe -m pytest tests/test_g05_dataset_download.py tests/test_g05_scorer_closure.py tests/test_g05_slidevqa.py -q --basetemp=.b/g05-review/data-pytest`: **17 passed, 1 skipped in 9.42s**. The optional SlideVQA module skips because clean CPU environment has no pyarrow; this matches the intended no-heavy-CPU-CI policy.
- `git diff d842ce6` for benchmark-lock.json, both frozen manifests, scorer requirement lock, existing scorer.py and worker.py is empty.
- Independently rehashed all **85** locked files in each local Qwen and Marathon source profile; no mismatches. Original source identities, Marathon's three-file official patch, and unchanged dependency-lock identity are preserved.
- Executed only Marathon's local `manifest()` binding checks: 100 documents, 18,272 unique ordered pages, 11,805,490,064 expected payload bytes, lock-authenticated GT/scope digests and per-document ordered published LFS hashes all agree. No network or payload acquisition was initiated.
- Existing raw Qwen and Marathon pipeline evidence each reports all eight known answers PASS, all eight full-pipeline metric percentages equal 100, overall 100, and exactly one displayed-formula sample. This is synthetic scorer evidence, not a model score. Compact independent results: `.b/g05-review/source-and-evidence-check.json`.
- Qwen raw case evidence inspected: exact 1.0 and mismatch 0.4 both have positive-size PNGs, nonwhite content and five valid glyph boxes on GT and prediction. Error control has separate retained renderer-boundary diagnostic; timeout observes actual prediction xetex process before the outer deadline. Final public export and rerun remain pending.
- Downloader checks exercise actual local HTTP Range honored/ignored, invalid Content-Range, short/overlong content, SHA mismatch, replacement of existing corruption, exact LFS OID validation/retry, and authorization removal on a redirect to a different host. No false successful page acceptance found.
- SlideVQA extractor binds GT, reviewed downloader source and all 12 shard hashes, selects only exact frozen deck URLs, preserves ordered last-match overwrite behavior, refuses null expected pages, uses `datasets.Image.decode_example` then Pillow default JPEG save, and rereads generated outputs. Reviewed local serialization receipt binds its script/test/raw-output hashes and strict versions. No alternate deck/page substitution found.
- MPDoc original verifier binds ZIP hash/size, GT/scope, historical inventory digest and published receipt; compares every original ZIP page to that inventory and verifies extracted byte identity. The separate 409-document/3,051-original-page claim remains explicit and does not claim complete 420-document acquisition.
- Dataset receipt exports contain hashes, relative paths, provenance/access constraints and counts; no source payload or credentials observed. Credentials use an unredirected header, signed CDN URLs stay in memory, and receipt failures use exception type/status rather than exception text.

## Deliberately incomplete review gates

At this pass, public scorer export is not yet present and docs/G05_SCORER.md says pending. Final scorer receipt/hash/sanitization/reproduction audit must be done after export, including the cleanup rerun. Marathon acquisition and separate watcher verification remain active; the full-page receipt has not yet arrived and no complete acquisition claim is approved by this review. SlideVQA remains credential-gated, and full MPDoc reconciliation must follow any successful authorized acquisition. These are known pending work, not newly discovered defects.

## Cleanup fix follow-up

Owner added `active_group_processes()`/`cleanup_process_group()` and calls cleanup in the child supervisor's `finally`, preserving the original exception while terminating the isolated group and checking for remaining non-zombie members. Independent repeat of the injected interruption now observes killpg followed by wait before KeyboardInterrupt propagates: `.b/g05-review/interruption-fixed-repro.json`. The reported P2 is **fixed in source**; final real renderer rerun/public export is still pending review. A Linux-only regression test now exercises an exited leader with a still-running descendant.

## Additional independent serialization execution

Reused `.b/benchmark-scorer/Scripts/python.exe`, appended the existing CPU environment only for pytest, and asserted the installed versions before running the two optional local Parquet cases with `--basetemp=.b/g05-review/serialization-pytest`: **2 passed in 1.34s**. Observed versions: datasets 5.1.0, pyarrow 25.0.1, Pillow 10.4.0. This independently exercises real Parquet decoding/default JPEG bytes and null-page rejection without network, source payloads, model imports, or new package installation.

## Authorized SlideVQA and full MPDoc receipt follow-up

The parent completed authorized SlideVQA acquisition and full MPDoc reconciliation while preserving the original 401 and original 3,051-page evidence. Independently reviewed `.b/g05-data/mpdoc-full-reconcile.py` and checked its SHA256 against `MPDocBench-full-scope.json`. Then separately reread and hashed the original ZIP, **all 3,135 actual local pages**, and **all 12 actual pinned Parquet shards**. Verified the full ordered GT inventory has 420 documents/3,135 unique pages, all 3,051 original page hashes remain unchanged, the 84 supplemental outputs equal the frozen missing-page set, and the SlideVQA index records 60 matching source rows with the expected official last-match semantics. Shard payload totals **1,393,939,599 bytes**.

Receipt/index/producer links, GT and scope hashes, original inventory/archive identity, supplemental source/index identity, and retained first-gate receipt hash all match. Compact result: `.b/g05-review/mpdoc-full-scope-independent-audit.json` (**PASS**, 3.72 seconds). No correctness/security defect found in these completed byte-identity claims.

**[P3] Serialization runtime provenance completeness:** docs/G05_DATASETS.md requires recording Python as well as dataset/Pillow versions, but SlideVQA.json currently contains only package versions. Parent was asked to record the actual serialization Python runtime in a supplemental receipt, preferably with executable hash and Pillow JPEG codec version. This does not change the already-verified page/shard hashes or reopen the authorization gate.

Marathon's original writer/watcher stopped at 9,500 completed attempts and the parent explicitly owns one replacement writer under `.b/g05-data/parent_runner.py`. No completed Marathon claim is reviewed until the final independent verification receipt arrives.

## Final scorer export audit

The owner reran both full pipelines and both four-case controls after the cleanup fix and published final scorer evidence. Independently rehashed **455 retained raw artifacts** and **309 public text copies** against the new artifact manifest, verified current scorer/test producer hashes, and scanned all **314 published files** for host-user path/credential patterns: no failures or hits. Twenty derived TeX filenames are intentionally sanitized in public `local_raw_path`; their actual retained counterparts were resolved through the private raw-evidence manifest and rehashed successfully.

All public scorer file paths are at most **100 repository-relative characters**. Long logical names are mapped to short `artifacts/<id>` paths with raw/public hashes preserved. Raw evidence remains unchanged. This checks the publication path repair; a complete package build remains the parent's integration responsibility.

Both summary profiles pass all eight known answers and all eight full-pipeline metrics, with overall 100 and one formula sample. Exact/mismatch are 1.0/0.4, error is EXPECTED_RENDER_ERROR, timeout is EXPECTED_TIMEOUT with actual prediction-renderer start observed, and every control records no active survivors after cleanup. Commands list all four executions with zero exit codes. The cleanup regression receipt observes an exited leader's active descendant before cleanup and no active descendant afterward. The P2 cleanup finding is **resolved and verified**.

Read the completed reproduction recipe: explicit process overrides are available from inventory, sanitized fontconfig/wrapper can be expanded for the retained runtime, source and frozen package reconstruction are described, and every run uses fresh outputs. The completed 11-entry failure catalog has no missing receipt links, preserves earlier invalid fixture assumptions, and explicitly labels the upstream native 30-second Timer path NOT EXERCISED. No full-dataset/GPU/model-quality claim is inferred.

Independent compact audit result: `.b/g05-review/scorer-public-export-audit.json` (**PASS**). No unresolved P1/P2 scorer/data correctness or security finding at this stage. Full Marathon completion and the narrow dataset provenance/reproduction follow-ups remain pending their final receipts.

## Dataset reproducibility follow-ups resolved

Added tracked `scripts/g05_reconcile_mpdoc.py` is independently AST-equivalent to the preserved private producer after excluding only producer-location and supplemental runtime metadata fields. Its SHA256 is `9512903dc05e0b41a5bb6835d24db92c83f417294676075e9bdb51b11c8361c4`; the original private producer hash remains unchanged. Reconciliation was rerun and all receipt/index links match; full-scope receipt SHA256 is `8e6f9db70110eb0d66029653cd32c6dfebe80153538781943c1a73d46b634a80`.

Independently launched the unchanged serialization interpreter and compared actual `sys.version`, venv/base executable SHA256, Pillow imaging-binary SHA256 and JPEG/libjpeg-turbo versions against `SlideVQA-runtime.json`: all match. Runtime receipt SHA256 is `0089d683ef4bf494d906053fd5f8897ba700dc643facc3fc6ed4b36acd19ee3f`, linked by the full-scope receipt and bound to the successful unchanged acquisition receipt. Actual runtime is CPython 3.10.20, JPEG 8.0, libjpeg-turbo 3.0.3. The P3 provenance issue and tracked-producer reproducibility follow-up are **resolved**. Audit: `.b/g05-review/dataset-reproducibility-audit.json`.

Owner's final scorer `verification.json` agrees with the independent 455-raw/309-public-artifact counts and zero-survivor controls. The native 30-second timer remains explicitly untested; no replacement claim was introduced. All reviewed scorer and completed MPDoc data work now have **zero unresolved P1/P2 findings**. Marathon's final completion/independent-reread receipt remains the only pending scorer/data review gate.


## Final Marathon closure and review decision

**PASS — zero unresolved findings.** Scorer synthetic controls and both complete dataset byte-identity scopes are closed. MPDocBench is 420 documents/3,135 pages; MarathonDocBench is 100 documents/18,272 pages/11,805,490,064 image bytes. Full GPU inference and full-dataset quality scoring remain NOT RUN; resource-cap acceptance is outside this scorer/data review and must remain blocked until physically measured.

For Marathon, revalidated the frozen GT and scope identities, metadata hash, every annotation/order/ordered-LFS per-document digest, all 18,272 index hashes and sizes against the frozen official LFS identities, and the existence/size of every actual page. Verified current and all three retained earlier producer hashes. Parent-owned acquisition and independent verification both exited zero; retained verification stdout exactly agrees with the successful public receipt and stderr is empty. The separate verification process already reread all 11.8 GB of actual page bytes; this review validated that process, source, complete resulting hash index and its official identity binding without a redundant second full reread.

The published Marathon index is exactly byte-identical to the retained verified index: 3,152,533 bytes. Export receipt correctly binds public index, retained source index and acquisition/verification receipt. The earlier 9,500-attempt interrupted run is preserved as separate provenance; it is not the completed-run evidence.

The scorer's actual imported benchmark-module hashes (including ledger.py) and postprocessor hashes in both final worker selftests match the current source that the scorer/data PR will inherit from PR8. Official scorer source, frozen benchmark lock, scope and requirement locks remain unchanged. Published payload is metadata/text/hash evidence only.

Final compact audits: `.b/g05-review/marathon-final-independent-audit.json` and `.b/g05-review/final-closure-receipt-hashes.json`.

Exact final receipt hashes:

| Receipt (under eval/benchmarks/g05/) | SHA256 |
| --- | --- |
| datasets/MarathonDocBench.json | `961ddaca50692277843b29f5037f1f851a67d387176521c8c536992f51a311b1` |
| datasets/MarathonDocBench-export.json | `ced27e70d87e613801d9d60bac26c5f01413654baa6f19f7572e66b294ca9177` |
| datasets/MarathonDocBench-page-index.json | `62f1ba97d87a9f2bb3726921349e5d95a145b9614dc744239621aab780093563` |
| datasets/MPDocBench-full-scope.json | `8e6f9db70110eb0d66029653cd32c6dfebe80153538781943c1a73d46b634a80` |
| datasets/SlideVQA.json | `c7796705bc379befcf24ccbd7d68648881b986fc65e7eee3bdaab8bf952ac283` |
| datasets/SlideVQA-runtime.json | `0089d683ef4bf494d906053fd5f8897ba700dc643facc3fc6ed4b36acd19ee3f` |
| scorer/summary.json | `e8e87aaafd1b92cfcc9def5e612730a869301612fe46aa7d8c59af930303ba1c` |
| scorer/artifact-manifest.json | `04ac654ed9578a7232c67d855dd11b293fd6800ee5954d3de6b839e9619cecc7` |
| scorer/verification.json | `4c8acad4b1a461c59a57797a66124538c7ccde4e00134aaf55499f2469c8abb5` |
