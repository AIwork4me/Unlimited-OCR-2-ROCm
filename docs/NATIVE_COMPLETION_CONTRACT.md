# Native document completion contract

Reviewed design, 2026-10-29. CPU validation does not validate a GPU or a
benchmark score. The pinned model and all existing resource limits remain fixed.
The coordinating reviewer independently read the pinned infer/infer_multi,
page-guard and generation paths and approved the direct dictionary adapter
design before implementation. An independent implementation review then found
three failure-evidence/structure issues; the corrections below have CPU
regressions. Final independent re-review passed all 64 focused CPU tests with no
remaining P1/P2 findings; see the [review receipt](../eval/benchmarks/g05/native/review.md).

## Reviewed immutable source

The model snapshot is `baidu/Unlimited-OCR-2` revision
`9cf077c4b206e0e2a31879df937b0b0bcba3661c`. Source references below are line
numbers in that snapshot, not the mutable model repository head.

| File | SHA-256 | Relevant source |
| --- | --- | --- |
| `modeling_unlimitedocr.py` | `f6df813cef23cc28b1daca40fd8cb20740d05a238e7b9dfa2334531fdcf39f1f` | input construction 531–580; page guard 583–605; infer 1038–1108; infer_multi 1110–1123; complete generation loop 1125–1205 |
| `modeling_deepseekv2.py` | `24a15dfd47578ab24a7a4d4e977dc5d7a23dfef0a2a4c4bf73fcf3e2ce7acaf2` | native DASH configuration 296–336; prefill 433–451; sparse decode 455–537 |
| `config.json` | `607d130a70d5af5a35329aa237acd68651f45504c62cc7ced981066f4e781126` | BF16, DASH, page token 128827, maximum position embeddings 131072 |
| `generation_config.json` | `dcd0479f3a12e5e4413d956197f8a4074408e4901834b38c7e036519134e05af` | EOS token ID 1 |
| `tokenizer_config.json` | `ceba7d221f265cca4c98b5e0a1d48c10107270b7b29c12396610d6d0fd64b126` | EOS token spelling; no automatic EOS append |

These hashes are frozen by `src/unlimited_ocr_2_rocm/upstream-lock.json`.
`load_model` verifies source/config/tokenizer and both weight shards before
loading local reviewed code. This investigation read local source without
executing model imports or `trust_remote_code`.

## Actual termination semantics

`infer_multi` performs one document call to `infer`, with an ordered image list,
`crop_mode=False`, and `return_dict=True`. Its final tuple discards the native
completion dictionary. Completion cannot be inferred from that tuple: the
generation loop may hit `max_length` below `max_new_tokens`.

`_uocr_generate` uses its own DynamicCache loop, not Transformers `generate`.
It starts with `finish_reason='length'`. Only selecting the tokenizer's EOS
sets `finish_reason='stop'`, before appending EOS. Exhausting the decode loop or
reaching the total context boundary retains `length`. Generated `token_ids` and
decoded text exclude EOS; EOS therefore consumes a decode iteration but is not
included in the output-token count. No stop may be fabricated from a count,
page marker, text suffix, or nonempty output.

The native page guard can force EOS when more than the expected page markers
have appeared, or when the final page hits `max_page_tokens` (default 20000).
It can force another page marker at an earlier page cap and suppress EOS before
the expected count. Thus `stop` proves the loop selected EOS; it does not prove
all page contents are complete or correct. Preserve native page counts, compare
them with ordered input count and parsed output pages, and reject malformed or
missing pages. An empty segment after a final `<PAGE>` may be a delimiter; a
nonempty final segment is a page. The adapter does not observe which guard
branch selected EOS, so semantic content completeness and guard exhaustion
remain unproven even when native stop and page counts agree. The current
8192-token output limit is below the default 20000-token per-page guard, but
extra-page-marker forcing is still possible.

## Minimal adapter

For multi mode, invoke the exact delegated official call directly:
`model.infer(tokenizer, image_file=ordered_image_list, crop_mode=False,
save_results=False, eval_mode=True, return_dict=True,
max_new_tokens=requested_budget)`. This retains the dictionary already returned
by the official implementation. Single mode retains its scalar image input and
automatic crop policy. The public CLI remains unchanged. Evidence must identify
the actual `infer` entry point and the multi/base-mode compatibility mapping.

No decoder/generation function, tensors, logits, cache, weights, tokenizer,
prompt, temperature, DASH routing, native local attention, image size, crop
policy, BF16 setting, or vision MATH compatibility is modified. The native
temporary ring/sliding-window changes and restoration remain official code.
There is no method replacement or global observer state. Each invocation covers
the whole document; no per-page generation or chunking is introduced.

The adapter validates dictionary metadata without reinterpreting native finish:
text is a string; output IDs are nonnegative integers; finish is `stop` or
`length`; counts agree with the native ID lists and requested inputs. Preserve
available raw text before validating metadata or converting Markdown. A native
`length` is always truncated, even below the requested output-token budget.
Malformed metadata, missing pages, invalid blocks, interruption, OOM and timeout
cannot become a successful completion. The existing append-only ledger owns
atomic attempts, retry history, raw preservation, and complete denominators.
Direct inference also requires parseable page blocks and the expected content
page count, even without ground truth. The official converter has a `__raw__`
fallback and repairs some malformed syntax; a truthy parse alone cannot prove
validity. `require_complete_pages` checks the original block headers against
a canonical header grammar and the official parsed blocks, rejecting fallback
blocks, unmatched markers, malformed headers and substantive preambles. In
particular a repaired `<LEVEL1}` header is rejected; a `<LEVEL1>` header may
appear before or after the coordinates. The converter itself is unchanged.
Empty block content is structurally allowed and does not establish semantic
content completeness. An unknown finish remains a failure.
The full valid native JSON result is retained as `native_generation`; raw text
is saved before validation or Markdown conversion. Invalid values, including
unsupported types, cycles and nonfinite floats, receive explicit diagnostic
tags instead of arbitrary string representations, and cause a failure even in
unexpected extra fields. A malformed result cannot suppress its failure
manifest through JSON serialization; diagnostic output contains no JSON NaN or
Infinity constants. This protection is local to native evidence and does not
change the shared doctor serializer. Invalid UTF-8 mapping keys, including
nested keys, are retained in explicit key/value entry lists so escaped-looking
valid keys remain distinct and no key is silently replaced. A `KeyboardInterrupt`
writes an
explicit `FAIL`/`INTERRUPTED` manifest and propagates the interruption. If the
official loop has not returned, it exposes no partial IDs/text to preserve;
the wrapper records that failure without inventing partial output.

Because the pinned loop excludes EOS from returned IDs, `stop` with exactly
`max_new_tokens` output IDs is inconsistent: selecting EOS requires another
loop iteration. The adapter rejects that metadata. `max_new_tokens - 1` IDs
followed by EOS is accepted; `length` at or below the budget remains `length`.

`completion_metadata` validates generation fields and diagnostic serializability;
the ledger owns the final document acceptance boundary. Any benchmark caller
must preserve raw text, retain metadata, and submit the native finish truthfully.
The ledger now calls the same `require_complete_pages` validator before a
`PASS` can commit. Rejected headers commit `FAIL` with the raw bytes and metadata
retained. This shared validator has no default character limit, so generic
ledger documents above one million characters retain their existing support.
Native inference explicitly retains its existing one-million-character
readable-output bound. None of these structure checks establish semantic
completeness or a benchmark score; ground truth and the official scorer remain
independent requirements.

## Resource-limit decision

Keep the wrapper limits of five pages, 8192 new tokens, and 16 million pixels per
page unchanged. Native `max_length=131072` and `max_page_tokens=20000` remain
unchanged. The configured position limit is not evidence of practical memory
safety. Base images are padded to 1024 square; the default input builder emits
273 image tokens per base page plus page markers, BOS and instruction tokens.
Even a five-page request can exceed 8192 output tokens for dense pages, and must
then remain an explicit truncation. Documents above five pages remain blocked
before GPU work; benchmark coverage must record that limitation. Removing caps
requires a separately reviewed resource policy and physical device evidence.

## Validation and next physical checks

CPU tests precede production changes for exact multi call mapping, authoritative below-budget `length`,
unknown/malformed finish metadata, token/count mismatch, and preservation of
raw text when validation fails. Integrate the adapter with the real ledger in
CPU tests for missing pages, malformed blocks, truncation, interruption,
resumption, and preservation of prior committed artifacts on failure.
An optional local test verifies the pinned source hash and executes only its
extracted `infer_multi` AST against a CPU boundary double, proving effective
kwargs/text/token-count equivalence without importing the remote module or
torch. CI skips that characterization when the local snapshot is absent;
all adapter and ledger tests run without downloads.

After review corrections, the focused adapter/inference/ledger suite passed
64 tests, including the original review reproducers and an output exceeding
one million characters at the generic ledger boundary. The full suite passed
237 tests with six skips (one absent optional pyarrow dependency, two GPU
opt-ins, and three torch-dependent RCA tests). Ruff, formatting, mypy and pip
check passed. Exact source/test hashes and checks are recorded locally under
`.b/native-completion/`.
Parent integration/build verification and independent re-review remain
separate; these receipts are not public GPU evidence.

After review and CPU checks, a physical Windows gfx1151 run must capture raw
outputs, native finish fields, source/weight hashes, input order/page hashes,
timing, GPU allocated/reserved peaks, process RSS and complete failures. Use the
existing opt-in worker and approved stack; no model or stack changes. Compare
the retained native text/token counts against the `infer_multi` compatibility
contract. The commands below are prospective and were not run during adapter
development. Run from the primary repository root. The preserved `C:\b`
junction refers into this root; it is not a separate development directory.
Select a fresh output path for each repetition.

```powershell
$NativePython = 'C:\b\o\Scripts\python.exe'
$RuntimeRoot = Join-Path (Get-Location) '.b/native-completion-runtime'
New-Item -ItemType Directory -Force "$RuntimeRoot/tmp", "$RuntimeRoot/triton" | Out-Null
$env:TEMP = "$RuntimeRoot/tmp"
$env:TMP = "$RuntimeRoot/tmp"
$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
$env:TRITON_CACHE_DIR = 'C:\b\native-completion-runtime\triton'
& $NativePython -m unlimited_ocr_2_rocm doctor --output .b/native-completion-runtime/doctor.json
if ($LASTEXITCODE -ne 0) { throw 'Physical stack/BF16 gate failed' }
& $NativePython -m unlimited_ocr_2_rocm smoke-multi --offline --cache-dir .b/h --pages 2 3 5 --max-new-tokens 1024 --output .b/native-completion-runtime/multi-first
if ($LASTEXITCODE -ne 0) { throw 'Physical native multi-page validation failed' }
```

Linux and every other GPU identity remain unvalidated. The current production
loader enforces the Windows stack and `triton-windows`; no Linux inference
command can honestly pass that profile. A separate reviewed Linux dependency
and loader profile plus physical hardware are required before a Linux native
inference command can be proposed. Linux native inference is **BLOCKED** until
both exist. No Linux wheel/version pins are inferred from the Windows profile.

These Linux hardware/provenance commands are explicit next steps for the actual
Linux GPU host and have not been run. Start from this repository root, choose a
fresh evidence directory, and retain command failures as well as successful
outputs. Tool absence is evidence to resolve, not GPU validation.

```sh
Evidence=.b/native-completion/linux-provenance-first
test ! -e "$Evidence" || exit 1
mkdir -p "$Evidence"
uname -a > "$Evidence/uname.stdout.log" 2> "$Evidence/uname.stderr.log"
echo $? > "$Evidence/uname.exit-code.txt"
rocminfo > "$Evidence/rocminfo.stdout.log" 2> "$Evidence/rocminfo.stderr.log"
echo $? > "$Evidence/rocminfo.exit-code.txt"
rocm-smi > "$Evidence/rocm-smi.stdout.log" 2> "$Evidence/rocm-smi.stderr.log"
echo $? > "$Evidence/rocm-smi.exit-code.txt"
```

These commands inventory hardware and runtime provenance only. They do not
validate native kernels, BF16 model execution, or any benchmark score. A Linux
or WSL CPU run does not validate W7900/gfx1100 or another physical GPU. CPU
contract checks can run separately on Linux without implying GPU validation:

```sh
python -m pytest tests/unit/test_native_completion.py tests/unit/test_benchmark_ledger.py --basetemp=.b/native-completion-cpu-tests -q
```
