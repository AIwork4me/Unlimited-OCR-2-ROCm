# Official Transformers whole-PDF trace on Windows gfx1151

**Result:** the exact 15-page test PDF completes in **one official native `infer_multi` request** with the existing SAM/CLIP scoped GPU MATH adapter. A separately instantiated unadapted official model fails at the first SAM SDPA. This corroborates the previously confirmed **L4 packaged AOTriton runtime/image contract defect**; the PDF run is not a new images-only causal experiment. This follow-up changes no production code, model source, weights, stack version, driver or decoder attention.

中文结论：原始 15 页 PDF、完整 OCR 输出和逐层追踪均已公开。未适配时，第一页 SAM 的默认 efficient SDPA 首次产生 HIP 错误；保留局部 GPU MATH 后整份 PDF 正常结束，原生 DASH 保持启用。这里验证的是全档推理完成，OCR 准确率尚未验收。

## Input and method

[Exact test input](../../eval/rca/gfx1151-rocm7141/pdf-followup/input/Unlimited-OCR-2.pdf): 1,515,878 bytes, 15 A4 pages, SHA-256 **`1eb5885f4ddfa4ec6fbff375092377f30b5d6a5e034ec4936213f125bf2f9444`**. The user explicitly requested publishing this original for reproducibility. Its byte identity differs from the public model-repository paper; the committed fixture is authoritative.

The [official README at 4d0bebab](https://github.com/baidu/Unlimited-OCR/blob/4d0bebab8714e0f3b0c811cd44bf422ddb49daae/README.md) documents native AutoModel/infer_multi and a `pdf_to_images` helper in its PDF/SGLang client example. We AST-selected that helper unchanged, executed it with **PyMuPDF 1.27.2.2 at 300 DPI** in an isolated renderer venv, then supplied all its ordered PNG paths to native Transformers. The rest of the SGLang example was not executed. This is the user's selected Transformers route; it does not pretend infer_multi itself accepts a PDF pathname.

Each actual PNG is 2481×3508. Both fresh-process cohorts verify the same 15 PNG hashes, open them in order, and call infer_multi **once** with `crop_mode=False`. Native construction produces **4,115 prompt tokens, 4,095 image tokens (15×273)**, and 15 global 1024×1024 image views. No five-page wrapper batching or per-page generation is used. CPU PDF rasterization is preprocessing; the BF16 model remains on the physical GPU.

Model/code revision is **9cf077c4b206e0e2a31879df937b0b0bcba3661c**. Code/config/tokenizer/weight inventories match the lock before both loads and in the post-run verification. Python 3.10.20; torch 2.12.0+rocm7.14.1; torchvision 0.27.0+rocm7.14.1; torchaudio 2.11.0+rocm7.14.1; HIP 7.14.60850; Windows 11; Radeon 8060S / gfx1151; driver 32.0.22032.6002.

Observers return original results and never explicitly clear HIP errors. Recorded boundaries are PDF read → Image.open → input builder → infer_multi/infer → generation → SAM/CLIP → actual ATen SDPA → decoder layers/native DASH → decode → original raw output. CPU profiler identifies the first actual backend of each encoder; it is not a claim of GPU kernel-symbol capture.

## Actual outcomes

| Condition | First error / completion | Recorded scope |
|---|---|---|
| Unadapted diagnostic, `eaadf93f-28d4-4bb2-b575-04dce902596e` | First SAM SDPA: HIP 0→1; later contiguous exception; exit 1 | All 15 images read, input construction complete, SAM page 1 entered; CLIP/DASH not reached |
| Existing scoped MATH, `8c816946-cf2f-4b09-9ebd-7e7fb815c962` | `finish_reason=stop`; exit 0; 13,836 tokens; 15 nonempty segments | 15 SAM + 15 CLIP completions, 540 HIP-clean SDPA calls, 12 decoder layers, 12 synchronized native DASH prefill calls with sequence 4115 |

Unadapted first call: official `deepencoder.py:836`, BF16 Q/K/V **[25,12,196,64]**, stride [150528,12544,64,1], additive bias [25,12,196,196], profiler-observed **EFFICIENT_ATTENTION**. Immediately after SDPA HIP=1; synchronization returns normally. The later `deepencoder.py:890` window_unpartition/permute/contiguous is the delayed reporting site. [Raw first failure](../../eval/rca/gfx1151-rocm7141/pdf-followup/whole-document/unadapted/first-failure.json) and [complete failed stderr](../../eval/rca/gfx1151-rocm7141/pdf-followup/whole-document/unadapted.stderr.log).

The MATH run took **1,300.985 s including loading/tracing**; original generation time 1,276.409 s and prefill 11.960 s. GPU allocation peak 9,298,788,352 bytes (8.66 GiB), reservation peak 9,542,041,600 bytes (8.89 GiB), process RSS peak 9,560,043,520 bytes (8.90 GiB). APU allocator counters are not dedicated VRAM or total system consumption. This instrumented run is not a throughput benchmark.

Native greedy decoding, max_length=131072, max_new_tokens=39321 (0.3×131072), native page guard, max_page_tokens=20000 and default no-repeat-ngram=0 are retained. DASH/SR-SWA remains enabled. [Original output](../../eval/rca/gfx1151-rocm7141/pdf-followup/whole-document/math/raw.txt) SHA-256 **`2fde62d850f3e07cb914c792e818f9378cbea287fb80423ab6beec44130cf1c4`**; [readable output](../../eval/rca/gfx1151-rocm7141/pdf-followup/whole-document/math/readable.md), [full manifest](../../eval/rca/gfx1151-rocm7141/pdf-followup/whole-document/math/manifest.json), [JSONL trace](../../eval/rca/gfx1151-rocm7141/pdf-followup/whole-document/math/trace.jsonl), [warnings/stderr](../../eval/rca/gfx1151-rocm7141/pdf-followup/whole-document/math.stderr.log).

A normalized character-5-gram set-Dice comparison against all PDF text-layer pages finds the same-index page as the best match for **15/15** output segments (scores 0.677–0.994). The text layer was never model input. This is a page-order diagnostic, not manual ground truth, CER/WER or an acceptance threshold. **OCR accuracy remains unscored**; the final status is `INFERENCE_COMPLETED_NOT_ACCURACY_SCORED`. [All scores and definition](../../eval/rca/gfx1151-rocm7141/pdf-followup/whole-document/full-order-diagnostic.json).

## Root cause and earlier scope correction

“0.13 images” means the **AOTriton precompiled GPU-kernel resource family**. The packaged 0.12.50 runtime expects loose AKS2 files while the provided 0.13 family uses ZIP storage. The [prior source/binary and images-only A/B evidence](SDPA_ROOT_CAUSE.md#16-causal-explanation) proves the exact-stack defect. This PDF follow-up rechecks all five DLL hashes, eight image-payload files and runtime-header bytes against the original primary environment, unchanged. MATH avoids the incompatible fused-image loader while retaining GPU computation. The adapter remains a workaround; no rebuilt/corrected official wheel was tested.

The earlier [144-DPI page-1 investigation](../../eval/rca/gfx1151-rocm7141/pdf-followup/first-page/historical-RESULTS.md) was insufficient for the user's whole-document requirement. Pillow rejecting a PDF only describes that wrapper input reader. Real first-page replay still provides valid narrower evidence: SAM DEFAULT→EFFICIENT and CLIP DEFAULT→FLASH both HIP=1; MATH has zero failing/nonfinite elements against CPU FP32 with atol=rtol=.03 (max absolute errors .00390625 and .0019503235816955566). That preprocessing differs from the later native global-view request; do not mix its [54,12,196,64] SAM shape with the full-document [25,12,196,64] shape. The corrected full request above processes all 15 pages.

An independent read-only review checked whole-document methodology and immutable inputs during generation. It identified two provenance nits, now explicitly recorded without altering original evidence: the local README copy had one extra terminal LF; its separate canonical copy matches official Git blob **8782f6e14af3fc0b052e2bfc245dd88e33756084**. The function digest is normalized-LF compilation-source identity; the saved CRLF file has a separate byte digest. Reader ASTs match. [Source-byte audit](../../eval/rca/gfx1151-rocm7141/pdf-followup/whole-document/source-byte-audit.json). Final completion, token IDs, hashes, file invariants and HIP cleanliness were asserted by the root investigator after EOS. [Verified summary](../../eval/rca/gfx1151-rocm7141/pdf-followup/whole-document/summary.json).

## Reproduce the whole-document cohorts

Use the physical Windows gfx1151 exact-stack environment from [repository setup](../../README.md#setup-and-validation). Hosted CPU CI does not run these GPU steps. Set `$OcrPython` and `$CacheDir` to that environment and its already verified model cache. Download using the repository's pinned downloader if needed; never execute a mutable remote revision. This diagnostic deliberately instantiates an unadapted **separate process**; it does not remove the production adapter.

From the repository root, use a fresh `.local/pdf-pipeline-rca` (the preparation refuses an existing directory). The archived scripts are the actual producer sources; configure only the explicit PDF/cache paths in local copies:

```powershell
$OcrPython = "C:\uocr\b\Scripts\python.exe"
$CacheDir = "C:\uocr\cache"
& $OcrPython -m unlimited_ocr_2_rocm download --cache-dir $CacheDir
@'
import hashlib, pathlib, shutil, sys
root = pathlib.Path.cwd()
art = root / 'eval/rca/gfx1151-rocm7141/pdf-followup/whole-document'
out = root / '.local/pdf-pipeline-rca'
assert not out.exists(), 'Use a fresh diagnostic directory; preserve old runs'
pdf = root / 'eval/rca/gfx1151-rocm7141/pdf-followup/input/Unlimited-OCR-2.pdf'
assert hashlib.sha256(pdf.read_bytes()).hexdigest() == '1eb5885f4ddfa4ec6fbff375092377f30b5d6a5e034ec4936213f125bf2f9444'
readme = art / 'upstream/official-README.canonical.md'
assert hashlib.sha256(readme.read_bytes()).hexdigest() == '43a888efc4069811d0f33d0ca248c2533aa206846d17b73d5b7d5e00029bd71e'
(out / 'upstream').mkdir(parents=True)
shutil.copyfile(readme, out / 'upstream/official-README.md')
renderer = (art / 'render_official_pdf.py.txt').read_text(encoding='utf-8')
lines = renderer.splitlines()
hits = [i for i, line in enumerate(lines) if line.startswith('pdf=pathlib.Path(')]
assert len(hits) == 1
lines[hits[0]] = 'pdf=pathlib.Path(' + repr(str(pdf)) + ')'
renderer = '\n'.join(lines) + '\n'
compile(renderer, 'render_official_pdf.py', 'exec')
(out / 'render_official_pdf.py').write_text(renderer, encoding='utf-8')
trace = (art / 'trace_official_native.py.txt').read_text(encoding='utf-8')
old = "cache=pathlib.Path.home()/'.cache/unlimited-ocr2-phase1/models'"
assert trace.count(old) == 1
trace = trace.replace(old, 'cache=pathlib.Path(' + repr(str(pathlib.Path(sys.argv[1]).resolve())) + ')')
compile(trace, 'trace_official_native.py', 'exec')
(out / 'trace_official_native.py').write_text(trace, encoding='utf-8')
print('Prepared original reader/trace with explicit input and cache paths')
'@ | & $OcrPython - $CacheDir
if ($LASTEXITCODE -ne 0) { throw 'Preparation failed' }

& $OcrPython -m venv .local/pdf-pipeline-rca/renderer-venv
$RendererPython = ".local/pdf-pipeline-rca/renderer-venv/Scripts/python.exe"
& $RendererPython -m pip install --only-binary=:all: --no-deps --require-hashes -r eval/rca/gfx1151-rocm7141/pdf-followup/renderer-requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Renderer install failed' }
& $RendererPython .local/pdf-pipeline-rca/render_official_pdf.py
if ($LASTEXITCODE -ne 0) { throw 'Rendering failed' }

# Expected exit 1 on the original defective package; run in its own process.
& $OcrPython .local/pdf-pipeline-rca/trace_official_native.py --mode unadapted
$UnadaptedExit = $LASTEXITCODE
# Expected exit 0; preserves scoped MATH and native DASH.
& $OcrPython .local/pdf-pipeline-rca/trace_official_native.py --mode math
$MathExit = $LASTEXITCODE
Write-Output "Unadapted=$UnadaptedExit; MATH=$MathExit"
```

Each mode refuses an existing output directory and writes its own full manifest/trace/raw output. Preserve stdout/stderr as UTF-8 streams in addition to those files. Do not interpret a process crash, timeout, nonzero HIP value, missing final manifest, token-budget finish, wrong page count or nonempty-page failure as success. The original instrumented full request took about 22 minutes. New machines may differ; compare fixed input/model/package identities and actual operations, not run IDs or timings.

For a download-light kernel reproducer, use the existing [weights-free SDPA instructions](../../experiments/sdpa_rca/README.md). Full model inference requires the separately downloaded pinned official weights; none are stored in Git. [All published artifacts, history, exclusions and checksums](../../eval/rca/gfx1151-rocm7141/pdf-followup/README.md).

[Remote publication/byte-integrity and hosted CPU CI receipt](PDF_PUBLICATION_VERIFICATION.json) is bound to the immutable evidence commit. It verifies the public PDF and all research evidence after upload; it does not substitute for the recorded physical GPU run.
