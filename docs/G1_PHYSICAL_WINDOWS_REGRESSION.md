# G1 physical Windows regression

Date: 2026-10-29 (Asia/Shanghai). This report covers the original published-wheel
Windows gfx1151 Native Transformers MATH path. It does not validate Linux,
experimental AUTO/fused execution, SGLang serving or a full dataset benchmark.

| Check | Gate | Actual scope |
| --- | --- | --- |
| Original environment and physical BF16 preflight | PASS | Windows 11 build 26200, Radeon 8060S/gfx1151, driver 32.0.22032.6002; exact mandatory trio, Triton and model identities; all 15 installed and loaded Torch native members match the original wheel. |
| Standalone native DASH numerical probe | PASS | Existing immutable 160/173-token BF16 probes pass the original FP32 math reference and fixed 0.03 absolute/relative tolerances. |
| Existing 1/2/3/5-page fixture acceptance | PASS | 13 cases per arm, all 26 pass the unchanged CER <=0.02, important-number and page-order gates with canonical blocks and 12 recorded native DASH prefill calls per request. |
| Before/after native output equality | PASS | All 13 pairs match exact raw UTF-8 bytes, prompt/output IDs, native finish/page fields, kernel records, scoped vision events and guard traces. Total normal raw bytes: 4,484 per arm. |
| Actual MATH dispatcher control | PASS | Separate instrumented English control observes 72 `aten::_scaled_dot_product_attention_math` calls, no fused SDPA calls, zero HIP error and exact output/ID parity with the ordinary after run. |
| Bounded output and context length | PASS | Both arms preserve native `length`: eight output IDs at output budget 8; separately eight IDs below budget 64 when total context is lowered to prompt+8. No length is reinterpreted as stop. |
| Page-guard limitation | PASS | Lowered per-page cap 8 produces native stop with incomplete content; it is a negative diagnostic, not accepted OCR. Guard-forced terminal EOS is also observed in ordinary two-page/table fixtures. |
| Controlled physical interruption and retry | PASS | After the third completed decoder forward, a disclosed KeyboardInterrupt records FAIL/INTERRUPTED. State restores, HIP remains clean and the subsequent two-page retry exactly matches the normal request. |
| Raw preservation and atomic ledger failure contract | PASS | CPU replay of captured GPU raw bytes: seven documents, fourteen pages, eight attempts; length/missing/malformed/interrupted and injected rename-failure outcomes retain raw/seals/history. Corrupt derivatives are explicitly synthetic, not GPU corruption. |
| External hard-kill recovery / OOM envelope | NOT RUN | Controlled in-process cancellation and ledger replay do not establish these separate physical failure paths. |
| 15/30/50/100-page physical requests | NOT RUN | Production five-image/8192-token/pixel caps remain unchanged; a separate reviewed resource policy is required. |
| Full MPDocBench / Marathon quality runs | NOT RUN | 0/420 and 0/100 benchmark documents inferred here. |
| Local CPU checks | PASS | Ruff check/format, mypy, build and pip check; final pytest: 271 passed, 7 explicitly skipped GPU/torch checks in the isolated CPU environment. |
| Final PR #7/#8/#9 Windows/Ubuntu CI | PASS | Each unchanged head has eight successful CPU/scorer checks; exact snapshots are retained in the archive. These are CPU CI results. |

## Identity and comparison method

The actual original environment is the retained
`.worktrees/phase1-windows-gfx1151/.venv-repro`, invoked through the existing
`C:/uocr` compatibility junction while the working directory stays at the
primary root. Original Torch wheel SHA256:
`415d2dbd4b6ab002cff1c54795fdd66322abd9a46bf982cb64eebb4493ac2e7c`
(114,472,348 bytes). `.b/o` instead contains the source-built candidate
`6a67d15861538ee217afdccd1ccc09ff318d7f44bff29a10f74c0d51730e6e1e`;
it was identified and left untouched, not used as an original-wheel baseline.
No wheel was rebuilt, replaced or distributed.

The mandatory versions remain Torch 2.12.0+rocm7.14.1, torchvision
0.27.0+rocm7.14.1 and torchaudio 2.11.0+rocm7.14.1. Native dependencies are
Transformers 4.52.1, triton-windows 3.7.1.post27, huggingface-hub 0.36.0 and
tokenizers 0.21.4 on CPython 3.10.20. Observed HIP runtime and driver API versions
are 71460850; every checked `hipPeekAtLastError` is zero and never clears the
error state. Source/config/tokenizer/index and both weight shards match model
revision `9cf077c4b206e0e2a31879df937b0b0bcba3661c`.

Before uses the exact `d842ce6` inference caller. Its five imported helpers
(model/output/compat/config/doctor) are byte-identical to the after helpers.
After uses PR #8's inference/completion behavior inherited unchanged by PR #9.
The reviewed diagnostic producer was committed at
`a68e88a658ab6a6958c05c15d6ddf9caf66e9cb3` before either arm; runner SHA256
`5983da029cb62ee19d3221d7c9c2700931212fee6c2895b2f33c36e60b027e51`.
Each arm loads once in its own fresh process. An identical passive infer observer
returns the original result object, retaining the dictionary the before tuple
interface discards. The guard observer calls the original function once and
reads its scalar state without reading or changing logits.

Six single-page cases retain automatic crop; seven multi cases retain base mode.
The single inputs have 899 prompt IDs (893 image IDs); two/three/five-page base
prompts contain 553/827/1375 IDs. Whole-document context is preserved; no pagewise
OCR concatenation is used. All normal native finishes are stop. Seven multi
wrapper fields improve from `upstream_tuple_no_finish_field` to the observed
native stop; that is expected metadata retention, not a generated-output change.

| Normal case | Output IDs per arm | Exact raw/ID equality |
| --- | ---: | --- |
| English / Chinese / mixed / table / math | 80 / 82 / 84 / 60 / 72 | PASS, each |
| Unicode-path recovery repeat | 80 | PASS |
| 2 pages, repeats 1 and 2 | 116 / 116 | PASS |
| 3 pages, repeats 1 and 2 | 171 / 171 | PASS |
| 5 pages, repeats 1 and 2 | 286 / 286 | PASS |
| Two-page cross-page table | 94 | PASS |

The two-page repeats and cross-page table have three PAGE markers for two
content pages, ending in an empty terminal delimiter. The unchanged guard then
forces EOS for excess markers. Other normal cases have an unmodified final
guard branch. The separate low page-cap control demonstrates why stop and page
counts alone cannot prove semantic completeness. Interruption exposes no native
returned partial IDs/text; absence of a returned result is recorded without
inventing partial output.

## Evidence, memory and remaining gates

The sealed [evidence index](../eval/g1/windows-gfx1151/index.json) binds original
raw hashes, sanitized member hashes and the deterministic evidence archive.
It includes both complete arm receipts and native outputs, negative controls,
supervisor logs, original-wheel/provider identities, numerical and MATH controls,
comparison, fresh committed analyzer receipt, ledger replay and independent
reviews. Original private raw artifacts remain under `.b/g1/` and
`.b/g1-physical/`; no historical evidence is rewritten. Restricted benchmark
images, source GT, credentials, weights and wheels are excluded.

The [independent publication review](../eval/g1/PUBLICATION_REVIEW.md) verifies
all 505 source bindings, 489 members and 71 exact raw-text copies, including
deterministic reproduction and Git blob equality. The archive retains its
original 269-test CPU snapshot and committed publisher identity. Two later
publisher regression cases raise the final CPU result to 271 passed / 7 skipped;
the separate [final validation receipt](../eval/g1/VALIDATION.json) binds those
source hashes without rewriting the sealed archive.

Five-page GPU allocator peaks are 9,222,452,224 bytes allocated and
11,284,774,912 reserved in both arms (8.589 / 10.510 GiB). These are APU shared
memory allocator readings, not dedicated VRAM or total system usage. Actual
worker RSS and process-lifetime peak working set are separate in the
[resource strategy](G1_LONG_DOCUMENT_RESOURCE_STRATEGY.md). The parent host
stream sampled the Windows venv redirector, not the model worker; those RSS
samples are excluded from model-memory claims. Per-request exact host peak
sampling remains a future worker requirement. Cold/warm JIT and observation
overhead make these runs unsuitable for a before/after speed claim.

The five-page fixture generated only 286 IDs under a 5120 ceiling, so it does
not stress 5120, 8192 or the full context. The separately reviewed resource
strategy proposes bounded 5/15/30/50/100 tiers without implementing them.
MPDocBench has 249 documents above five pages. Every Marathon document is
103–298 pages, so even a future 100-page test tier cannot close full Marathon.
Further resource profiles, physical envelopes, cancellation supervision and
predeclared real-corpus quality gates remain required.

PR #7 is ready within its infrastructure scope; #8 now has this narrow original-
wheel physical regression evidence; #9's source/scorer/dataset closure is ready
on that stack. These are scoped technical readiness findings, not automatic
merge approval or full benchmark acceptance. All remain separate from draft
#5's experimental fused evidence. See [integration review](G1_INTEGRATION_REVIEW.md)
for exact ancestry, final CI and formal-review status. MATH remains default;
model algorithms and all current production caps are unchanged.
