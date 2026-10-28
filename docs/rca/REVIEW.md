# Independent RCA review and dispositions

Review scope: origin/main `966265c` through evidence commit `7fdadff`. One independent fresh-context reviewer inspected code, numerical semantics, source links, original private artifacts and published sanitized bytes. The review performed no GPU work and changed no files. It independently executed the three CPU reference/statistics/fixture tests with the pinned torch interpreter: all passed.

The reviewer verified 401 SHA256SUMS entries, 400 original/published digest pairs, 232 original process-stream digests, 113 experiment summaries, seven producer-source mappings and eight upstream-source digests. No historical PASS contained a HIP error. No Critical finding. The reviewer found L4 justified for the packaged runtime/image contract on this exact stack, conditional on correcting the full-shape narrative.

## Important findings, fixed in one pass

1. Final classification previously reused a HIP-error flag computed before downstream copy/arithmetic/cast. The fix scans all recorded checkpoints again before readback/classification. `test_final_classification_includes_new_downstream_hip_error` was observed RED (missing scan), then GREEN. Earlier errors remain latched even if a later API consumes them.
2. Matrix reruns could truncate timeout/crash logs if no child JSON existed. The runner now rejects a nonempty output directory before launch, refuses preexisting streams, and uses exclusive file creation. `test_matrix_refuses_previous_failure_logs_without_child_json` and `test_process_runner_cannot_truncate_existing_streams` were observed RED (exit 0/overwritten streams), then GREEN. [Expected pre-fix output](../../eval/rca/gfx1151-rocm7141/review-red.stdout.log) is retained as non-acceptance evidence.
3. The large-shape A/B/A description mixed the primary OCR baseline and the secondary environment. The report and handoff now distinguish primary-baseline FAIL from secondary B PASS → restored-secondary FAIL. Exact same-environment A/B/A remains established for scalar FLASH and three repeated small EFFICIENT cases. No GPU results were changed or rerun to retrofit the narrative.

The full CPU suite after these fixes passed: 68 passed, five intentional skips (two physical opt-ins, three torch-optional CPU-reference tests in the torch-free CI venv). Final verification also runs all tests in the pinned torch environment, including those reference tests and the existing physical OCR contract; see [final command manifest](../../eval/rca/gfx1151-rocm7141/checks-final/process-manifest.json).

## Deferred minor

The provenance collector uses locale-default decoding for config.h text; its UTF-8 copyright character displays incorrectly in historical JSON. ASCII build identifiers and raw file SHA-256 remain correct. Existing evidence is retained without rewriting this display text. This does not change the causal inference or acceptance status. Explicit UTF-8 text decoding is a follow-up improvement.

## Review boundaries and rulings

- Fresh GPU reproduction was excluded from the independent review; root investigator's actual process evidence and final physical tests supply it. Cost if wrong: physical reproducibility would need another engineer to challenge it; full commands/fixtures are public.
- Rebuilt upstream fixes, other platforms, training and backward remain untested. No support or fix claim is made for them; the scoped production adapter remains. Cost of this limit: official fixed-wheel adoption needs new validation.
- Publication integrity and hosted CI were pending during review. The root investigator must verify both after pushing, before delivery. This review does not substitute for those checks.

No second review round was dispatched. The three substantive corrections were validated with failing regressions followed by the full suite, and original producer hashes/logs are preserved.

## Subsequent whole-PDF and publication reviews

The later user-requested PDF follow-up received a separate bounded read-only methodology review during generation. It verified one native 15-page request in each cohort, identical pinned model inventories, first-SAM efficient SDPA HIP 0→1, all 540 adapted HIP-clean SDPA calls, 15 SAM/15 CLIP completions, 12 native DASH calls and unchanged primary-runtime/image files. It required precise labels for normalized-source versus CRLF-file hashes, and resolution of the README's extra terminal LF. Both are preserved in the [source-byte audit](../../eval/rca/gfx1151-rocm7141/pdf-followup/whole-document/source-byte-audit.json). Final EOS/count/hash verification was performed after generation by the root investigator, not attributed to the mid-generation review.

Before publishing the follow-up and the explicitly user-authorized original PDF, a fresh-context reviewer checked all **113 export records**, **112 retained text-source originals and exact transformations**, and **555 inventory entries**. Exact PDF byte identity, privacy/credential scans, reproduction preparation/producer compilation, input/output claims and document links were checked. No Critical, Important or Minor findings. The reviewer ran no GPU/model code and made no test-suite claims. Separately, the root investigator executed the documented preparation and original CPU renderer: all 15 PNG hashes reproduced exactly. The review was bounded to this publication supplement; it did not repeat or replace the original causal RCA audit.
