# Independent SAM/CLIP numerical reference RCA

**KEEP EXPERIMENTAL. Independent reference and boundary measurements complete; no arithmetic defect established.**

| Subsystem | Arithmetic-defect classification | Reason |
| --- | --- | --- |
| SAM MATH / efficient | INSUFFICIENT EVIDENCE | Matched-input independent FP32/FP64 errors measured; no applicable quantitative contract violation established |
| CLIP MATH / flash | INSUFFICIENT EVIDENCE | Two matched-input pairs at fixed SAM measured; same unresolved contract limitation |

The measured differences are consistent with ordinary floating-point variation and BF16 quantization. This is a descriptive interpretation, not proof that every backend arithmetic contract is satisfied. No observed error is turned into a tolerance or OCR acceptance rule. Exact OCR/backend parity remains failed on the original fixture. No default, kernel, model, weight, tie-breaking rule or production pipeline was changed.

## Identity and audit

Baseline PR#26 /37f46d357769757faa48bee54639fba286b7cbe1. Existing isolation261, priorRCA379, mixed249 and historical406 seals are preserved. Candidate wheel SHA2566a67d15861538ee217afdccd1ccc09ff318d7f44bff29a10f74c0d51730e6e1e; immutable model/code9cf077c4b206e0e2a31879df937b0b0bcba3661c. Original wheel415d2dbd4b6ab002cff1c54795fdd66322abd9a46bf982cb64eebb4493ac2e7c is audited and unused for new inference. Installed wheel members, loaded modules, reviewed model/config/tokenizer/weight/source and fixture identity remain checked by the existing loader. Windows Radeon8060S/gfx1151 is the only tested platform; no other hardware is qualified. Version contract remains torch2.12.0/vision0.27.0/audio2.11.0 +rocm7.14.1.

The installed SDPA documentation permits fused-backend numerical differences. MATH's source promotes BF16 inputs to FP32 when its reduced-precision flag is false, scales Q and K separately, applies safe softmax and V, then casts output back. The actual flag is false in every new worker. Available local PyTorch source is60dd50f143fd9b10ac23cde8cdd34be61232b139. Source inspection and dispatch do not identify every native kernel instruction.

Contextual primary documentation: [SDPA](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html) and [numerical accuracy](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html). These current online docs are not substituted for the frozen installed source/runtime. No universal quantitative accuracy bound for these exact BF16 kernels was established here.

## Reference contract and independent validation

NumPy CPU FP32 and FP64 independently evaluate `softmax(Q Kᵀ / sqrt(d) + captured_bias) V`, with max-subtraction before exponentiation, explicit accumulator dtype and bounded head chunks. They do not call SDPA. The captured BF16 values are converted exactly to FP32/FP64; SAM's existing materialized relative-position bias is used directly, not recomputed. CLIP has no mask. Actual dimension64 gives scale0.125; dropout0, causalFalse and GQAFalse. Unsupported or incomplete invocation metadata fails closed. Boolean-mask/safe-softmax behavior has analytic tests, although the actual SAM bias is additive and finite.

FP64 is an independent high-precision approximation, not exact-real arithmetic or a bitwise emulator of MATH's scaling order/fused reduction. BF16-rounded FP64 comparisons are descriptive. FP32/FP64 discrepancy, absolute/relative errors, RelativeL2, zero/nonfinite handling, error quantiles and first differing coordinates are in attention-reference.json. Near-zero reference values can dominate maximum pointwise relative error; no denominator floor is used to hide this.

The independent reviewer separately coded unchunked NumPy FP64 equations and obtained bitwise-identical full SAM and both CLIP-input-set reference arrays. The reviewer also recomputed all eight full-vocabulary FP64 lm_head projections using CPU torch matmul, obtaining bitwise equality, and separately checked16 top-two dots with math.fsum. These independent checks find no reference-equation discrepancy.

## Matched attention results

| Condition | Subsystem | Max absolute vs FP64 | Mean absolute | RelativeL2 |
| --- | --- | --- | --- | --- |
| MM | sam | 0.00390596013931 | 0.000104103015851 | 0.00164912450932 |
| MM | clip | 0.00384924637779 | 8.2230617187e-05 | 0.00164645390133 |
| AA | sam | 0.0046572802223 | 0.000104648362147 | 0.00165878338325 |
| AA | clip | 0.00407352617319 | 8.57324194541e-05 | 0.00168107386827 |
| AM | sam | 0.0046572802223 | 0.000104648362147 | 0.00165878338325 |
| AM | clip | 0.00382832314686 | 8.21963291583e-05 | 0.00164205548597 |
| MA | sam | 0.00390596013931 | 0.000104103015851 | 0.00164912450932 |
| MA | clip | 0.00409710435456 | 8.56474697891e-05 | 0.00168356864124 |

SAM MM→AA has bitwise-equal Q/K/V/mask. CLIP MM→MA and AM→AA each hold SAM fixed and have bitwise-equal Q/K/V and invocation options. Shapes, strides, actual GPU storage offsets, dtype and device match for each input pair. Historical captures lacked original GPU offsets; those old offsets remain unmeasured. New captures record them directly. Output layout equality is separate from logical-value equality. All attention observations are finite. The older AM-MATH versus MA-flash CLIP pair remains an own-input comparison only, because its QKV differ.

## First differing operation and propagation

| Pair | Captured boundary | Different elements | Max absolute difference |
| --- | --- | --- | --- |
| MM→AA | sam.first_sdpa.output | 368433 | 0.0078125 |
| MM→MA | clip.first_sdpa.output | 68054 | 0.0078125 |
| MM→MA | clip.norm1 | 0 | 0 |
| MM→MA | clip.attention_projected | 128168 | 0.00390625 |
| MM→MA | clip.residual | 53577 | 0.0078125 |
| MM→MA | clip.norm2 | 59358 | 0.03125 |
| MM→MA | clip.mlp | 217312 | 0.0078125 |
| MM→MA | clip.block0.output | 127681 | 0.0078125 |
| AM→AA | clip.first_sdpa.output | 68587 | 0.0078125 |
| AM→AA | clip.norm1 | 0 | 0 |
| AM→AA | clip.attention_projected | 130124 | 0.00390625 |
| AM→AA | clip.residual | 54232 | 0.0078125 |
| AM→AA | clip.norm2 | 59595 | 0.03125 |
| AM→AA | clip.mlp | 221820 | 0.0078125 |
| AM→AA | clip.block0.output | 130882 | 0.0078125 |

For each matched CLIP pair, norm1 and projected QKV agree before the first SDPA return differs. Divergence is then observed through attention output projection, residual sum, norm2, MLP and block0 output. This isolates the SDPA return from preceding QKV projection and following output-projection effects at the first measured call. SAM's first SDPA return is likewise different with identical inputs. No smaller native instruction is identified. These are earliest directly measured differing operations, not a claim that every unobserved operation was instrumented.

## Common-prefix logits7 and10

Four new fresh owned workers run MM,AA,AM,MA once each, original eager QuickGELU unchanged. They use a diagnostic clone of the immutable generation function. Actual unchanged argmax is recorded first; frozen MM tokens0..9 are then fed consistently into generation, guard, position and cache state. Step10 is unforced. This is teacher-forced diagnosis, not a replacement natural-OCR parity test or a new repeatability qualification. Model source/weights and production generation remain unchanged.

All selected shared capture points checked at prefill/step7 match their existing natural controls exactly; this does not assert equality of previously unmeasured full prefill or cache states. New raw BF16 logits are cast to FP32 exactly as in the actual loop; guard7/10 is unmodified and full-vocabulary selection logits match the cast raw logits. Captured GPU argmax agrees with independent inspection. Numeric token IDs and complete arrays remain private; token-selection.json retains full-vocabulary hashes, hashed token identities and scalar top-two/margin summaries. Local token-logits-complete-private.json retains the requested exact IDs and values.

| Condition | Step | GPU top1−top2 | GPU max ties | FP64 projection margin | Same argmax |
| --- | --- | --- | --- | --- | --- |
| MM | 7 | 0.125 | 1 | 0.0204975425731 | True |
| MM | 10 | 0.125 | 1 | 0.108381299418 | True |
| AA | 7 | 0 | 2 | 0.0144579459447 | False |
| AA | 10 | 0.125 | 1 | 0.129230304425 | True |
| AM | 7 | 0 | 2 | 0.00308986823075 | False |
| AM | 10 | 0.125 | 1 | 0.145587217005 | True |
| MA | 7 | 0 | 2 | 0.0320298190927 | False |
| MA | 10 | 0.125 | 1 | 0.107614827342 | True |

At step7, all four fixed-hidden-state FP64 projections have the same unique winner as MM. Rounding their observed competing top-two projection values to BF16 exactly reproduces all GPU values. MM keeps a unique BF16 winner; AA/AM/MA become ties and the unchanged argmax rule selects the other token. At step10, all conditions have unique GPU maxima, margin0.125; all agree with their own FP64 projection argmax. All eight boundaries' observed top-two values match the BF16-rounded independent FP64 projection. This supports final-projection quantization/tie amplification at token7, not an SDPA arithmetic-defect conclusion. FP64 projections condition on already captured BF16 hidden states and BF16 weights; the decoder itself was not re-executed in higher precision.

First downstream hidden-state differences already exist at decoder prefill/step0 in the prior and new captures. A common prefix removes differing fed token IDs as an explanation for differences at step10, while backend-dependent prefill/KV values remain present. No logits or tie rule were modified to force parity.

## Verified KV-cache contract

Transformers4.52.1 DynamicCache is inspected and source-hash bound. Supported `len(cache)`, `cache[layer]`, and `get_seq_length(layer)` replace the old unsupported layers API. Twelve K/V pairs are captured at prefill and steps7/10. Every layer has sequence lengths899/906/909. All192 K/V append-prefix comparisons pass exactly. Cache object identity is stable within each request; observed K/V storage-pointer changes are summarized as allocation observations, not a total-memory or lifetime guarantee beyond these snapshots. Per-layer shape/stride/offset/dtype/hash and position contracts are in cache-contract.json; private pointers are omitted. The old unsupported collector remains explicitly unmeasured at its old identity.

## Failures, verification and limits

Two failed attempts are preserved before the four completed diagnostic requests: v1 imported Transformers too early and hit the unchanged fresh-module guard; v2 used decorator-wrapper globals and failed cloning before inference. Each terminated with zero active descendants and completed no OCR inference. New v3 identity follows reviewed fixes and CPU regression tests. No module-cache guard was bypassed or compiler cache reset. The first retained reference export is preserved as exploratory after producer formatting; admitted retained-v2 references have archived producers and independent numerical validation. An analysis-only generic-dispatch filter was corrected after all GPU workers completed; its declared source is archived with a separate relocation/provenance record. No GPU rerun was used to repair CPU analysis.

Nine local experimental CPU tests and all six repository checks pass; pytest340 passed/7 skipped. Four completed diagnostic requests pass physical/functional/stop/HIP/DASH, actual per-subsystem dispatch, immutable parameter/buffer, source/options/prompt and Job cleanup gates. These functional checks do not convert forced-prefix outputs into native OCR parity evidence. APU allocations are shared system memory, not dedicated VRAM. Instrumented timings are not performance evidence.

No confirmed defect means no defect-specific minimum reproducer or upstream issue is submitted. The reference equations/tests are model-free. A justified defect classification would still need an applicable accuracy/precision contract and a reproducible violation, not a tolerance fitted to these errors. No production fix, default change, kernel modification, automatic merge, release, 120-request campaign or real-document test is part of this draft. **KEEP EXPERIMENTAL.**

Independent read-only review and final preservation receipts accompany this report. The next highest-value action is to establish an authoritative quantitative accuracy contract for the exact efficient/flash kernels before attempting defect attribution; do not change OCR acceptance or force MATH/AUTO token parity.
