# G1 native long-document resource strategy

This is a source-audited design and a prospective physical-test policy, dated
2026-10-29. It changes no production code or limits. The current wrapper still
rejects more than **five images**, more than **8192 new tokens**, or more than
**16,000,000 source pixels per page**. Native `max_length=131072` and
`max_page_tokens=20000` remain unchanged. Stages above five pages require a
separate resource-policy PR, review, CPU checks and physical admission gates.
They are **BLOCKED**, not executed by this document.

The purpose is to determine a measured, bounded envelope for one native
Transformers document request on the physical Windows gfx1151 device, with
the pinned BF16 model, DASH and its original local-attention behavior intact.
It does not claim SGLang window/cache parity, published benchmark equivalence,
support on another GPU, or accuracy from a resource-only completion.

## Immutable source audit

The audit read local source as text and JSON; it did not import remote model
code, torch or a tokenizer and did not run GPU work. All **16** locked
source/config/tokenizer/index files and **both** weight shards matched
`src/unlimited_ocr_2_rocm/upstream-lock.json` before source trust. Model and
code revision: `9cf077c4b206e0e2a31879df937b0b0bcba3661c`; GitHub documentation
revision: `4d0bebab8714e0f3b0c811cd44bf422ddb49daae`. The local hash receipt is
`.b/g1-physical/resource/source-hash-facts.json`. Hashing weights is an
integrity check, not execution or GPU validation.

The [native completion contract](NATIVE_COMPLETION_CONTRACT.md) retains the
termination/structure acceptance boundary. Relevant immutable source:

| Source | Audit finding |
| --- | --- |
| `modeling_unlimitedocr.py:531–580` | `_uocr_build_inputs` uses base size 1024, crop size 640, patch size 16 and downsample ratio 4. A base page contributes `16*(16+1)+1=273` image tokens. Global views are BF16 `[pages,3,1024,1024]`; no-crop mode also allocates a dummy BF16 `[1,3,1024,1024]` crop tensor. |
| `modeling_unlimitedocr.py:1038–1089` | `infer` opens all ordered images as RGB. Empty prompt becomes `Multi page parsing.`; missing placeholders become `<PAGE><image>` once per page. One page with automatic crop can add dynamic 640px tiles; base and automatic single-page runs are different resource/input identities. |
| `modeling_unlimitedocr.py:1110–1123` | `infer_multi` delegates once to `infer` with an ordered image list, defaults `crop_mode=False`, requests the native dictionary internally, then returns only text and token count. The wrapper's direct dictionary call preserves the delegated inference semantics and authoritative completion fields. |
| `modeling_unlimitedocr.py:583–605` | The page guard suppresses EOS before the expected marker count, can force another PAGE at the per-page cap, and can force EOS at the last cap or after excess markers. `stop` is not proof of content completeness. |
| `modeling_unlimitedocr.py:1125–1205` | One prefill and token-by-token decode use `DynamicCache`. Only the final hidden position goes through `lm_head`; no full prompt-by-vocabulary logits are required. EOS consumes a decode iteration but is omitted from returned IDs. The output budget and total context limit can independently terminate with `length`. |
| `modeling_unlimitedocr.py:1149–1154,1196–1198` | The official inference path temporarily sets `_ring_window` and `sliding_window` to `None` and configures DASH, restoring state in `finally`. This is original source behavior, not permission to change attention or disable DASH/SR-SWA. Cache memory must not be estimated as a bounded ring. |
| `modeling_unlimitedocr.py:709–789` | Base vision inference visits pages sequentially, accumulating projected features before decoder prefill. Original RGB pages and all global input tensors can coexist. This is one document decode, even though the encoder visits pages in a loop. |
| `modeling_deepseekv2.py:296–335,433–451` | Inference preserves pinned checkpoint routing parameters and page-scoped prefill. Triton prefill is preferred, but the official function has a dense fallback if its kernel returns no result. A completed fallback cannot establish successful native sparse-kernel validation. |
| `modeling_deepseekv2.py:365–376,433–445,489–537` | Router index K is retained in the cache as FP32 and concatenated during decode. K/V cache remains full sequence length. Temporary old/new buffers and allocator reuse can increase peaks above payload size. |
| `modeling_deepseekv2.py:455–486` | Sparse decode constructs masks over the whole cache, casts full K/V to FP32, and performs masked attention. A sparse read domain does not make cache storage constant in document length. |
| `deepencoder.py:315–365,612–675,817–839` | SAM and CLIP have distinct SDPA paths and intermediates. Keep the existing scoped GPU MATH compatibility adapter and record its events; a decoder-only cache formula cannot predict the vision peak. |
| `config.json` | BF16; 12 decoder layers; hidden size 1280; 10 attention and 10 KV heads; head dimension 128; router index dimension 128; sparse block size 32; local blocks 4 (128-token local window); `dash_topk_blocks=12`; default inference `dash_topm=16`; configured context 131072. |

Primary immutable source links: [input/generation implementation](https://huggingface.co/baidu/Unlimited-OCR-2/blob/9cf077c4b206e0e2a31879df937b0b0bcba3661c/modeling_unlimitedocr.py),
[decoder implementation](https://huggingface.co/baidu/Unlimited-OCR-2/blob/9cf077c4b206e0e2a31879df937b0b0bcba3661c/modeling_deepseekv2.py),
[configuration](https://huggingface.co/baidu/Unlimited-OCR-2/blob/9cf077c4b206e0e2a31879df937b0b0bcba3661c/config.json).
These links identify the reviewed bytes; local verified files are the audit
evidence. Mutable repository heads are not accepted sources for execution.

## Resource model and its limits

For batch size one and full cached token count `T`, source-derived payloads are:

```text
BF16 decoder K/V = T * 12 layers * 2(K,V) * 10 KV heads * 128 dims * 2 bytes
                = T * 61,440 bytes = T * 60 KiB
FP32 router K   = T * 12 layers * 128 dims * 4 bytes
                = T * 6,144 bytes = T * 6 KiB
combined payload = T * 67,584 bytes = T * 66 KiB
one layer's two FP32 decode K/V copies = T * 10,240 bytes
one FP32 prefill route-score tensor = 4 * P * ceil(P/32) bytes
BF16 base global pixel tensor = pages * 6 MiB
```

`P` is actual prompt token count, including image IDs, PAGE markers, BOS and
instruction tokens. Default base layout has `273*pages` image tokens and one
PAGE marker per page; instruction tokenization must be counted with the locked
tokenizer. The analytical table reserves 32 tokens for BOS/instruction, so
`P_allowance=274*pages+32`. This is a declared allowance, not a measured prompt
count or a general guarantee for custom prompts. Require actual prompt IDs and
`P_actual <= P_allowance` before accepting the planned envelope. A historical
default-prompt 15-page request had `P_actual=4115`; do not substitute that count
for a new run's metadata.

Header-only inspection of both verified safetensors shards found 2,819 tensors:
3,343,978,240 BF16 elements and 257 I64 elements, totaling 6,687,958,536 bytes
(6.229 GiB) of serialized tensor payload. This is not measured GPU model-load
allocation. The calculations exclude loaded-library storage, duplication during
load, vision and MoE activations, attention masks/biases, allocator fragmentation,
cache concatenation overlaps and host image storage. FP32 copies are shown for
one decoder layer; multiplying them by all layers would also misrepresent their
lifetimes. A configured 131072-token position limit establishes no memory-safety
claim. At that length, K/V plus router payload alone is 8.25 GiB.

The CPU-only reproducible calculation and full-precision JSON are local:
`.b/g1-physical/resource/calculate_resource_envelope.py` and
`.b/g1-physical/resource/analytical-envelope.json`. They import only the standard
library and read weight headers as JSON, without materializing tensors.

| Ordered base pages | Image IDs | Prompt allowance | Decode iteration ceiling | Prompt + decode envelope | K/V + router payload (GiB) | One-layer FP32 K/V copies (GiB) | One prefill score tensor (MiB) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 1365 | 1402 | 8192 | 9594 | 0.604 | 0.091 | 0.235 |
| 15 | 4095 | 4142 | 16384 | 20526 | 1.292 | 0.196 | 2.054 |
| 30 | 8190 | 8252 | 32768 | 41020 | 2.582 | 0.391 | 8.122 |
| 50 | 13650 | 13732 | 49152 | 62884 | 3.958 | 0.600 | 22.525 |
| 100 | 27300 | 27432 | 98304 | 125736 | 7.914 | 1.199 | 89.785 |

All rows are analytical estimates, not physical validation. The 100-page row
keeps 5,336 tokens below the configured context limit; it is deliberately a
bounded moderate-density probe. Dense 100-page output can exceed this budget
and must be reported as truncated. EOS needs an iteration, so a successful
`stop` has fewer output IDs than the decode ceiling. No budget is a promise of
semantic completion, quality or throughput.

## Separately reviewable policy decision

Retain all current wrapper caps during G1. The recommended future policy is an
explicit, opt-in experimental profile for one serialized document request, with
a frozen input manifest, page count, base layout, output ceiling and resource
budget. The normal profile keeps its existing limits. Currently the direct CLI
checks the image count before model loading, while token-budget and pixel
checks occur in `run_native` after the model has loaded. The future policy must
perform every resource admission check before model loading; do not claim that
all current limit failures avoid GPU allocation. The policy PR must test 6-page rejection under the normal
profile, each experiment profile's bounds, context admission, rejection before
GPU work, failure preservation and watchdog cleanup. It must not modify the
official model/decoder, prompt defaults, page guard, native context limit,
precision or crop processing. The 16-million-source-pixel limit stays in every
profile. No concurrent calls in one model/process: inference changes global
DASH/page-scope state temporarily.

Retain `require_native_environment` before loading and at observed native kernel
boundaries. It already rejects unexpected `DASH_STAGE0` and other unreviewed
overrides; record the effective `head_cls` checkpoint routing and environment
identity rather than trusting configuration alone. A future worker must preserve
this gate, the exact base/crop mode and all frozen source identities.

Alternatives considered:

| Choice | Decision and reason |
| --- | --- |
| Remove image/token limits globally | Rejected. Position capacity and sparse routing do not establish peak memory, bounded latency or completion. It also changes the ordinary API's safety contract. |
| Independent page OCR then concatenate | Rejected. It changes cross-page context, page state, completion and benchmark identity. |
| Raise context above 131072 or emulate SGLang page windows | Rejected for this route. Native and SGLang have different cache/window/ngram behavior; an implementation would need a separate semantic and hardware validation project. |
| Force multi-page dynamic tiling | Deferred as a separately named experiment. It changes image tokens and input identity; base-mode measurements cannot admit it. |
| Explicit progressive native base profiles | Recommended. Each step retains one ordered request and native completion metadata, has a fixed resource envelope, and can be rejected independently. |

## Predeclared physical admission and stop rules

These are conservative proposed experiment budgets, chosen before new long
document runs. They are not hardware capacity measurements. Their adoption
requires the future policy PR; do not silently loosen them after a failure.

| Pages | Maximum GPU allocated (GiB) | Maximum GPU reserved (GiB) | Maximum process RSS (GiB) | Load + inference wall deadline |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 16 | 20 | 20 | 1800 s |
| 15 | 16 | 20 | 24 | 3600 s |
| 30 | 20 | 24 | 28 | 7200 s |
| 50 | 24 | 28 | 32 | 14400 s |
| 100 | 32 | 36 | 48 | 28800 s |

For admission, use the smaller of the table allocation limit and 75% of the
doctor-reported PyTorch-visible GPU budget. Use the smaller of the reservation
limit and 85% of that visible budget. Before each process, require available
system memory of at least `max(12 GiB, 25% of system_total_bytes)`; during a run,
stop when sampled available memory falls below `max(8 GiB, 15% of total)`.
These system and allocator readings overlap on an APU: never sum GPU allocated,
reserved and RSS into a purported physical-memory total, and never label them
dedicated VRAM. Private commit/Windows commit limits are additional diagnostics,
not interchangeable with RSS or GPU allocation.

Before admitting each higher tier, obtain fresh-process model-load and previous
tier peaks. Predict allocation conservatively as:

```text
load_allocated + planned_KV_router_payload
+ max(4 GiB, 2 * max(0, previous_peak_allocated - load_allocated))
+ 1.25 * planned_one_layer_FP32_KV_copy
+ planned_BF16_global_pixel_tensor
```

The factor, 4-GiB floor and overlap reserve are policy safety margins, not a
fitted model. Require this estimate below the effective allocation limit, and
the estimate plus 4 GiB below the effective reservation limit. The previous
peak may already contain caches/inputs; intentional double counting makes this
a conservative admission screen. It cannot certify safety. Host-memory
admission also needs an input-dimension-based RGB storage estimate and a measured
previous-tier host peak; otherwise leave the higher tier BLOCKED. At most
16 million pixels means roughly 45.8 MiB/page for one RGB payload, before PIL
and temporary copies. One hundred such pages can therefore retain multiple GiB
on the host even in base mode.

Sample process RSS, private commit and system available memory at 250 ms with a
CPU parent watchdog. Retain raw timestamped samples. Capture allocator peaks in
the child; the current API exposes them on completion/error but has no live
cross-process allocation counter. An external watchdog cannot guarantee a live
GPU-allocation abort based only on an exit manifest. Do not describe this
proposed budget as a hard allocator quota. A future worker may report live
counters at safe boundaries only after review; it must not reset peaks or
change returned tensors. If reliable monitoring/cleanup is absent, do not start
a higher tier.

Stop progression after any OOM, timeout, HIP error, process crash, nonfinite
diagnostic, source/stack mismatch, threshold exceedance, unexplained memory
growth, native `length`, inconsistent metadata, missing/misordered pages,
malformed blocks or failed ground truth. Retain the exact failed attempt, its
exit status, stdout/stderr and any available output. Do not increase limits,
retry with cropped inputs, substitute a model, switch device/backend, disable
DASH or turn a timeout into success. After HIP/OOM failure, use a fresh process
and re-run the identity/health gate before any authorized new attempt; never
reuse potentially poisoned state.

The outer watchdog must kill and await the whole child process tree on timeout,
record cleanup, and preserve the failure even if the child cannot write a final
manifest. Native `_uocr_generate` exposes no partial returned text/IDs until it
returns. Empty partial artifacts after a killed request are absence of evidence,
not proof that zero tokens were generated. A run can remain `NOT_RUN` or
`BLOCKED`; both are distinct from a recorded attempted failure.

User cancellation has the same whole-process-tree cleanup requirement. Record
the attempted request as `FAIL` with reason `CANCELLED` or `INTERRUPTED`, preserve
the pending attempt, exit status and already-written bytes, and stop progression.
Kill and await only the owned worker tree, checking PID identity to avoid a
reused PID; retain cleanup evidence if the worker cannot write its manifest.
Do not treat cancellation as successful EOS or automatically start a higher
tier. A later authorized retry uses a new attempt directory, the same frozen
input/configuration and a fresh identity/health check; it preserves earlier
failures. This is a proposed supervisor contract, not a new ledger enum here.

## Inputs, progression and acceptance

Freeze manifests before inference: ordered file names and SHA-256, dimensions,
source/data license and revision, RGB conversion, image/layout mode, prompt,
model/code/config/tokenizer/weight hashes, output budget, exact argv, environment
identity, ground truth, scorer identity and thresholds. Inputs must contain
unique page IDs and cross-page references; repeated copies of one page test
allocation only and cannot prove page order or content preservation. Freeze
light text, mixed Chinese/English, table, equation and multi-column cases.
Keep native base layout fixed throughout a series. Distinguish single automatic
crop (public single route), one-page base (diagnostic multi layout), multi base,
and any separately approved dynamic-tile series.

1. **G1 1/2/3/5 baselines:** run the existing single fixtures and ordered native
   multi fixtures on the approved physical stack. Keep the current 1024-token
   per-page smoke budget (`smoke-multi` multiplies it by count), and add an
   explicitly named 8192-token five-page request only if its input manifest and
   acceptance thresholds are frozen first. A brief fixture's observed peak does
   not measure the full 8192-token envelope. Preserve fresh-process and repeated
   loaded-model results separately; two repeats are not two independent loads.
2. **5-page resource anchor:** exercise the approved document classes with one
   whole request each, two fresh-process repetitions, and a repeated-process
   sequence to observe retained allocation. Do not pad decode with fake tokens
   or suppress EOS to reach the budget. A naturally short completion proves
   only its actual measured token count. Dense inputs that hit 8192 are retained
   truncations; no per-page concatenation is allowed.
3. **15 pages:** after the separate policy review, freeze unique ordered inputs
   with moderate reference output sized below 75% of the 16384-token ceiling.
   This reference sizing is a screening estimate, not a generated-output bound.
   Admit only after the five-page memory screen passes. Execute two fresh
   processes; keep expected/actual prompt, image and output counts.
4. **30, then 50, then 100 pages:** repeat the same gates independently at each
   tier. Both previous-tier repetitions must complete within budgets, match
   pages/order/ground truth and preserve complete evidence. Do not skip a tier.
   Freeze output/reference density for each declared ceiling; report dense
   documents outside it as unsupported or truncated with full denominators.

Completion requires authoritative native `finish_reason=stop`, consistent
token/page metadata, valid native block grammar, the ordered nonempty content
page count, no unacceptable kernel/backend event and all resource limits.
For synthetic fixtures, keep the existing predeclared normalized CER <=0.02,
exact important numbers and exact page order. For real datasets, retain their
official scorer/threshold identity and distinguish resource completion from
scored accuracy. The page guard can force stop and is not a semantic acceptance
test. Explicitly retain guard-observability limits; do not invent a guard-hit
field that official metadata does not expose.

## Existing measurement fields and G1 receipt status

`doctor.memory_snapshot()` currently reports `process_rss_bytes`,
`process_peak_rss_bytes`, `system_total_bytes`, `system_available_bytes`,
`system_memory_percent`, and, when torch/GPU is available,
`gpu_allocated_bytes`, `gpu_reserved_bytes`, `gpu_peak_allocated_bytes`,
`gpu_peak_reserved_bytes`, `gpu_free_total_bytes=[free,total]`. Device
`pytorch_visible_budget_bytes` is a runtime-visible budget, not dedicated VRAM.

`run_native` saves before/after snapshots and resets GPU peaks per invocation.
Windows process `peak_wset` is a lifetime high-water mark; it is not reset by
the GPU peak reset. `memory_after` can be far below a transient peak. A repeated
model process therefore needs samples and fresh-process comparisons to explain
RSS growth. Source `time`/`prefill_time` and synchronized wrapper wall time have
different boundaries; report each with its name. Load time is separate.

The following appended measurements come from the completed original-wheel
physical `before` and `after` arms under `.b/g1/`. Each arm loads one model
process and runs the fixture sequence; the within-arm repeats reuse that
loaded process. They are not fresh-process repetitions of each case. All 13
normal cases in each arm passed; corresponding raw bytes, prompt/output IDs
and the mode/count fields below agree. No cold/warm JIT timing comparison or
speed claim is made. Exact byte values and SHA-256 bindings to all 26 source
summaries are retained in `.b/resource/physical-measurement-facts.json`.

| Case | Pages / layout | Prompt IDs | Image IDs | Output IDs | Decode ceiling |
| --- | --- | ---: | ---: | ---: | ---: |
| `english` | 1 / single auto-crop | 899 | 893 | 80 | 1024 |
| `chinese` | 1 / single auto-crop | 899 | 893 | 82 | 1024 |
| `mixed` | 1 / single auto-crop | 899 | 893 | 84 | 1024 |
| `table` | 1 / single auto-crop | 899 | 893 | 60 | 1024 |
| `math` | 1 / single auto-crop | 899 | 893 | 72 | 1024 |
| `repeat_unicode_recovery` | 1 / single auto-crop | 899 | 893 | 80 | 1024 |
| `pages_2_run_1` | 2 / ordered base | 553 | 546 | 116 | 2048 |
| `pages_2_run_2` | 2 / ordered base | 553 | 546 | 116 | 2048 |
| `pages_3_run_1` | 3 / ordered base | 827 | 819 | 171 | 3072 |
| `pages_3_run_2` | 3 / ordered base | 827 | 819 | 171 | 3072 |
| `pages_5_run_1` | 5 / ordered base | 1375 | 1365 | 286 | 5120 |
| `pages_5_run_2` | 5 / ordered base | 1375 | 1365 | 286 | 5120 |
| `table_2_pages` | 2 / ordered base | 553 | 546 | 94 | 2048 |

Memory values below are GiB (`bytes / 2^30`), rounded to three decimals.
`A/R` means per-invocation GPU allocated/reserved high-water marks retained in
the in-worker `summary.memory_after`; `RSS/peak` means that worker's RSS at
that snapshot and its **process-lifetime** Windows peak working set. The
latter is not a per-request peak, and reservation can retain allocations from
earlier cases even though GPU peak counters reset per request. The wrapper's
own exit snapshots are also retained in the facts receipt; host RSS can move
between that boundary and the harness's later summary snapshot. These are
actual model-worker readings, never launcher RSS.

| Case | Before GPU A/R | After GPU A/R | Before worker RSS/peak | After worker RSS/peak |
| --- | ---: | ---: | ---: | ---: |
| `english` | 8.573 / 8.908 | 8.573 / 8.908 | 9.294 / 9.315 | 9.253 / 9.274 |
| `chinese` | 8.573 / 10.498 | 8.573 / 10.498 | 11.080 / 11.102 | 11.038 / 11.060 |
| `mixed` | 8.573 / 10.498 | 8.573 / 10.498 | 11.080 / 11.102 | 11.039 / 11.061 |
| `table` | 8.573 / 10.498 | 8.573 / 10.498 | 11.081 / 11.103 | 11.040 / 11.062 |
| `math` | 8.573 / 10.498 | 8.573 / 10.498 | 11.081 / 11.103 | 11.040 / 11.062 |
| `repeat_unicode_recovery` | 8.573 / 10.498 | 8.573 / 10.498 | 11.081 / 11.103 | 11.041 / 11.062 |
| `pages_2_run_1` | 8.568 / 10.498 | 8.568 / 10.498 | 11.049 / 11.122 | 10.993 / 11.065 |
| `pages_2_run_2` | 8.568 / 10.498 | 8.568 / 10.498 | 11.050 / 11.122 | 10.996 / 11.065 |
| `pages_3_run_1` | 8.574 / 10.498 | 8.574 / 10.498 | 11.091 / 11.137 | 11.033 / 11.079 |
| `pages_3_run_2` | 8.574 / 10.498 | 8.574 / 10.498 | 11.091 / 11.145 | 11.036 / 11.090 |
| `pages_5_run_1` | 8.589 / 10.510 | 8.589 / 10.510 | 11.116 / 11.173 | 11.066 / 11.122 |
| `pages_5_run_2` | 8.589 / 10.510 | 8.589 / 10.510 | 11.117 / 11.174 | 11.069 / 11.125 |
| `table_2_pages` | 8.568 / 10.510 | 8.568 / 10.510 | 11.118 / 11.174 | 11.069 / 11.125 |

These five-page fixtures generated only **286 output IDs**, with 1375 prompt
IDs and a requested ceiling of 5120. They did **not** stress 5120 generated
tokens, the 8192-token wrapper limit, or the analytical five-page envelope.
The measured single-page auto-crop case uses 893 image IDs, so it cannot be
treated as a 273-image-ID single base case. The two-page table fixture is a
separate content/layout case from the ordered text-page fixtures. Its empty
terminal PAGE and the text two-page fixtures' excess-marker guard branch are
retained in the native/guard receipts; native stop alone remains insufficient
to prove semantic completion.

**Watchdog lesson:** the `.b/g1-physical/{before-process,after-process}/host-memory.jsonl`
streams sampled the roughly
4-MB Python redirector/launcher PID, not the roughly 10–12-GB model worker.
Those launcher-only RSS samples cannot calibrate model peak RSS, host growth
or higher-tier admission. In-worker snapshots above are valid observations,
but cannot reconstruct unobserved between-snapshot host peaks or establish
live watchdog coverage. Before any future tier, identify and bind the actual
model-worker PID, creation time, executable/argv and owned descendant tree;
sample that worker and account separately for its owned descendants. Confirm
the sampled identity stays live, and prove owned-tree cancellation/timeout
cleanup with PID-reuse checks. A launcher PID must not stand in for worker
coverage. This monitoring prerequisite remains unmet by the launcher-only
stream; these new receipts therefore do not admit a higher tier.

| Future physical series | Status |
| --- | --- |
| 15/30/50/100 pages under future resource profiles | BLOCKED / NOT_RUN in G1; analytical design only; requires separate reviewed policy and actual-worker monitoring |

The immutable [historical whole-PDF trace](rca/FULL_PDF_PIPELINE.md) records one
15-page scoped-MATH native request with 4115 prompt IDs, 13836 output IDs and
native stop, allocation peak 9,298,788,352 bytes, reservation peak 9,542,041,600
bytes and process RSS peak 9,560,043,520 bytes. It used a diagnostic producer,
39321 decode ceiling and tracing; accuracy was unscored. It is historical
evidence for that exact run, not a new G1 acceptance result, current wrapper
support above five pages, a calibrated memory model or general 15-page safety.
Its original paths/hashes must remain unchanged.

## Exact next commands and required future diagnostics

Run from the primary repository root. These commands are prospective; this
source audit executed only the standalone CPU calculation above. The `C:\b`
compatibility junction remains intact and resolves into this repository. Use
the approved existing environment; no installation, driver/security/PATH change
or candidate-wheel manipulation is part of this policy.

```powershell
# CPU-only regeneration, after independently verifying the locked source hashes.
python .b/g1-physical/resource/calculate_resource_envelope.py

# Physical commands: only the coordinating physical worker may execute these.
$ResourcePython = 'C:/uocr/.worktrees/phase1-windows-gfx1151/.venv-repro/Scripts/python.exe'
$ResourceRuntime = Join-Path (Get-Location) '.b/g1-physical/resource/runtime'
New-Item -ItemType Directory -Force "$ResourceRuntime/tmp", "$ResourceRuntime/triton" | Out-Null
$env:TEMP = 'C:\b\g1-physical\resource\runtime\tmp'
$env:TMP = $env:TEMP
$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
$env:TRITON_CACHE_DIR = 'C:\b\g1-physical\resource\runtime\triton'
& $ResourcePython -m unlimited_ocr_2_rocm doctor --output .b/g1-physical/resource/doctor-next.json
if ($LASTEXITCODE -ne 0) { throw 'Stack/BF16 gate failed; retain receipt and stop' }
& $ResourcePython -m unlimited_ocr_2_rocm smoke-single --offline --cache-dir .b/h --cases english chinese mixed table math --max-new-tokens 1024 --output .b/g1-physical/resource/single-next
if ($LASTEXITCODE -ne 0) { throw 'Single gate failed; retain receipt and stop' }
& $ResourcePython -m unlimited_ocr_2_rocm smoke-multi --offline --cache-dir .b/h --pages 2 3 5 --max-new-tokens 1024 --output .b/g1-physical/resource/multi-next
if ($LASTEXITCODE -ne 0) { throw 'Multi gate failed; retain receipt and stop' }
```

Each output must be fresh; preserve existing receipts and choose a new suffix
for a new attempt. Capture the argument array, stdout/stderr, exit code and
watchdog samples around these commands. The smoke command runs repeated cases
inside one loaded process and an additional two-page table case; it does not
replace the fresh-process repetitions required for resource calibration.

No runnable 15/30/50/100 acceptance command exists under the unchanged wrapper
caps. Do not call official `infer` directly to bypass them. The next exact
engineering actions for the separate policy PR are: implement reviewed
CPU-only manifest/preflight inspection, explicit opt-in tier selection and a
supervised fresh-process worker; prove the default caps still reject before
model load; then document its actual `--help`/argv and invoke tiers in the
order 15, 30, 50, 100 with the ceilings above. Proposed command names must not
be presented as already available tools.

The complete dataset scopes are document counts, not page counts. The frozen
MPDocBench manifest has 420 documents / 3,135 pages, with 2–30 pages per document;
249 documents exceed the current five-image limit. Marathon has 100 documents /
18,272 pages, with 103–298 pages per document: all 100 exceed even the proposed
100-page initial test tier. The 5/15/30/50/100 progression therefore cannot close
full Marathon evaluation. Further whole-document profiles and physical evidence
covering the actual 103–298-page range remain separate blockers.

At 298 base pages, the declared prompt allowance is `274*298+32=81684` tokens,
leaving at most 49,388 tokens under the unchanged 131,072 context ceiling before
any extra safety reserve. That arithmetic is not a proven sufficient decode
budget. Dense output may exceed it, and the 100-page ceiling of 98,304 new tokens
cannot be reused for 298 pages. Full 420/100 evaluation also needs reviewed
output budgets, all document outcomes in the denominator, physical Windows
acceptance and scored quality gates; the acquired payloads alone do not close it.

Required future worker diagnostics are actual prompt IDs and image-token count;
full native dictionary and raw UTF-8 output; expected/actual/parsed page counts;
load/before/after memory and retained peaks; timestamped host samples; native
DASH and scoped vision events; synchronized timings; producer/package/loaded
provider hashes; and raw failure/cleanup receipts. Missing diagnostics keep the
relevant claim unvalidated. Linux, W7900/gfx1100, gfx1200 and every other
hardware identity remain unvalidated until physically tested under their own
reviewed profile. This strategy begins no Phase 2 work and authorizes no release,
stable tag, feature-PR merge or publication.
