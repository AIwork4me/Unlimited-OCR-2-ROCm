# Benchmark protocol freeze — 2026-10-29

Scope: [issue #6](https://github.com/AIwork4me/Unlimited-OCR-2-ROCm/issues/6),
P0 infrastructure only. **No full GPU dataset run or quality/performance claim.**
The protocol rules and source/annotation identities below are frozen. The
local MPDocBench/SlideVQA image payloads and CDM environment remain acquisition gates;
this is not a claim that the complete execution supply chain is ready.

## Sources and dataset identity

The machine-readable authority is
[`eval/benchmarks/benchmark-lock.json`](../eval/benchmarks/benchmark-lock.json).
It freezes individual evaluator files, Baidu client/config documentation,
postprocessing, three Marathon patches and the inspected SGLang wheel members.
No mutable HEAD, `master` or `latest` may enter an accepted run.

| Source | Immutable revision / artifact |
| --- | --- |
| [Baidu inference client and setup](https://github.com/baidu/Unlimited-OCR/tree/48ff432fb68a18ed91f2f70021f29028b1dde354) | `48ff432fb68a18ed91f2f70021f29028b1dde354` |
| [Official Qwen MPDocBench scorer](https://github.com/Tongyi-Zhiwen/Qwen-Doc/tree/4e5aee32280febb7b0844237ba087f837e3c8905/MPDocBench) | `4e5aee32280febb7b0844237ba087f837e3c8905` |
| [Marathon data and official patch](https://huggingface.co/datasets/baidu/MarathonDocBench/tree/fa05fccbe82ee6ec628006a1d58bb153954c3ee3) | `fa05fccbe82ee6ec628006a1d58bb153954c3ee3` |
| [HF native model/code/tokenizer](https://huggingface.co/baidu/Unlimited-OCR-2/tree/9cf077c4b206e0e2a31879df937b0b0bcba3661c) | `9cf077c4b206e0e2a31879df937b0b0bcba3661c` |
| Customized SGLang | `0.5.20.dev10+g8d62f5fb9c`; wheel SHA256 `a3f3e2c94e09de90b61b43d01475fd5d1aa818d16eec05204b8079c09c2ba70d`, 15,281,032 bytes |

| Dataset | Mandatory denominator | Annotation SHA256 |
| --- | --- | --- |
| Public MPDocBench-Parse | **420 documents / 3,135 pages** | `0a579dca6cf875a41fdab0975efee6d4e0fea999e0fa6c3ca95ad08b948f40df` |
| MarathonDocBench | **100 documents / 18,272 pages** | `7db0e659c76661ce87d4e1b3d47e010edbd48adfb446e3e2ee37260c60adf51c` |

Both counts were recomputed from the downloaded immutable annotation files.
MPDocBench's paper describes 433 documents / 3,246 pages; the public release
removed 13 documents. That paper denominator is not interchangeable with 420.
The checked-in metadata-only JSONL scopes contain every actual document ID,
page count, ordered page-path digest and annotation-record digest. Marathon
also records an ordered digest of published per-page SHA256 values. They contain
no document text, annotations or image bytes. Complete page manifests are
generated locally from the frozen GT, using its explicit ordered `images_list`.

Marathon's immutable HF metadata lists all 18,272 PNG objects with LFS SHA256;
their total payload is **11,805,490,064 bytes**. Those published hashes describe
remote objects; local image acquisition/byte verification is **NOT RUN**.
MPDocBench's official ModelScope image repository is frozen at
`48428421ee985717fbda3a6521165aaedfc42077`. Its 600,152,009-byte archive has
published SHA256 `f9a4c05d586b11a749e60ce13565090fdfbf8f4387fd35e5511a4a1a3d682d22`.
The separate official subset downloader references 11 SlideVQA decks. Freeze
`NTT-hil-insight/SlideVQA` at `e780ea258d734958befbfaed19fa059584642ca8`;
its access is **gated**. The lock records all required deck IDs, test-shard
hashes and the audited downloader hash. Do not execute its mutable
`load_dataset` call: an authorized acquisition must use the frozen data
revision and explicitly pin Pillow's JPEG serialization. Full local image
verification remains BLOCKED pending acquisition and complete reconciliation.

The MPDocBench toolkit is Apache-2.0; ModelScope image-dataset metadata declares
**CC-BY-NC-SA-4.0**, and the Qwen README restricts documents to research use.
Marathon annotations/tool references are Apache-2.0, while its source-document
redistribution is explicitly not granted. Baidu code is MIT; preserve its
notice and existing third-party notices. Downloaded scorer copies retain the
official license. Do not commit or upload dataset images, source PDFs, weights,
private outputs, or install logs containing credentials/home paths.

## Eight official leaderboard metrics

Use the original metric classes, original matching/normalization and original
`end2end_eval` task. Do not use this project's synthetic CER helper as a scorer.

| Leaderboard field | Official task → metric → aggregate | Direction on raw output |
| --- | --- | --- |
| TextEdit | `text_block → Edit_dist → ALL_page_avg` | lower |
| T-TextEdit | `merged_text_block → Edit_dist → ALL_page_avg` | lower |
| TableTEDS | `table → TEDS → all` | higher |
| T-TableTEDS | `merged_table → TEDS → all` | higher |
| FormulaCDM | `display_formula → CDM → all` | higher |
| ReadOrderEdit | `reading_order → Edit_dist → ALL_page_avg` | lower |
| FigureF1 | `figure → FigureF1 → all` | higher |
| HeadingTEDS | `head → HeadTEDS → all` | higher |

For edit distance, the official code sums edit counts and `max(len(gt),len(pred))`
within each document identifier (its historical field name is `image_name`),
then averages document ratios. Preserve `edit_whole` and `edit_sample_avg` as
diagnostics; they are different aggregates and cannot replace `ALL_page_avg`.
TEDS includes cell content, not only structure; preserve the separately emitted
structure-only result. CDM renders and matches formula characters, with its
original three-decimal sample F1. FigureF1 uses original page attribution,
greedy IoU matching at 0.5, and document F1. HeadingTEDS includes the virtual
root and normalized title edit costs. Continuation metrics use original GT
`truncated` relations and merged content. No matching/quality threshold changes.

Convert raw edit distances to `100*(1-distance)`, other similarities to
`100*similarity`. **Overall is the arithmetic mean of all eight percentages**,
as documented by Marathon's pinned card. Keep raw official output alongside
these presentation fields. Missing, NaN, infinite or out-of-range values make
Overall **null/BLOCKED**; never average seven available values or turn an
unavailable CDM renderer into a measured zero. BLEU, METEOR, Relation_F1 and
TEDS_structure_only are auxiliary metrics, not replacements for these eight.

## Scorer environment and profiles

Use a separate CPython **3.10** CPU environment, never the OCR venv.
Before import, reject unlisted source/bytecode/native import artifacts and
materialize only the locked files into a fresh verified source copy for each
scorer process. Disable bytecode writes in it and its child processes; a hash
check of `.py` alone does not prove that Python loaded those bytes.
[`requirements/benchmark-scorer.lock.txt`](../requirements/benchmark-scorer.lock.txt)
pins the tested transitive Python closure;
[`benchmark-scorer.txt`](../requirements/benchmark-scorer.txt) identifies the
direct integration choices. The frozen upstream commit has **no top-level
`MPDocBench/requirements.txt`**, despite its README command. These pins are
our tested environment, not fabricated upstream version declarations.
Core pins: NumPy 1.26.4, pandas 2.2.3, SciPy 1.13.1, scikit-image 0.20.0,
Levenshtein 0.27.1, apted 1.0.3, lxml 5.3.0, Pillow 10.4.0,
pylatexenc 2.10, evaluate 0.4.3, mmeval 0.2.1, func_timeout 4.3.5.
No torch/model installation is required by this CPU environment.

Two evaluator profiles are distinct identities:

- `qwen`: unchanged Qwen files and `configs/end2end.yaml` metric mapping,
  `quick_match`, original 30-second text matching timeout and fallback.
- `marathon`: the above plus exactly the three hash-pinned Baidu dataset
  patches. Full official settings are `MPDOC_EVAL_NPROC=32`,
  `MPDOC_MATCH_TIMEOUT=1800`, `MPDOC_DOC_BUDGET=0`, `MPDOC_TEDS_NPROC=32`,
  `MPDOC_TEDS_TIMEOUT=300`. Record CDM concurrency explicitly; do not inherit
  an undocumented override. CPU fixtures use serial matching/TEDS and do not
  certify full-scale multiprocessing behavior.

The patch also sanitizes malformed table span attributes and changes parts of
normalization. Treat it as an exact official variant, not proven bitwise parity
with vanilla on arbitrary output. Its hard document-budget implementation uses
`fork`; the documented full profile disables that budget. Its 300-second TEDS
guard uses SIGALRM and is not enforced on Windows. Run full Marathon scoring in
a validated Linux CPU environment even for GPU predictions produced on Windows;
this does not validate Linux GPU inference. `latexmlc` is an additional unpinned
dependency of the scorer's LaTeX-table conversion branch; pin and self-test it
before scoring any configuration that exercises that branch. The authored HTML
table fixture does not certify LaTeX-table conversion.

Do not silently use the patch for MPDocBench or the vanilla timeout for a
Marathon leaderboard comparison. The original scorer can continue past missing
files, fall back on matching timeout, score TEDS exceptions/timeouts as zero,
and return CDM zero when rendering fails. Preserve stdout/stderr, sample
outputs, timeout/error catalogs and denominator reconciliation. An evaluator
exit code alone is insufficient acceptance evidence.

CDM requires Node, ImageMagick with working PDF rasterization/Ghostscript,
XeLaTeX/TeX packages including `xeCJK`, and Source Han Sans SC fonts. The pinned
README suggests Node 16.13.1, an unpinned ImageMagick 7.1.1 checkout and an
unpinned `texlive-full`; it does **not** supply an exact tested TeX/font closure.
Observed on this host: Node **24.16.0**, ImageMagick **7.1.2-26**, kpathsea
**6.4.2**; executable hashes are in the new readiness receipt. **`xelatex` is
absent.** TeX/font/Ghostscript closure and the original POSIX-shell CDM calls
on Windows remain BLOCKED. No global install/security/PATH change was made.
Freeze a working portable/local renderer and all font/package/policy hashes,
then pass the positive formula known-answer render before accepting CDM.

## Model and inference protocol

The existing package
[`upstream-lock.json`](../src/unlimited_ocr_2_rocm/upstream-lock.json) freezes
all reviewed Python/config/tokenizer and both weight shards; verify every hash
before `trust_remote_code`. Tokenizer identity is LlamaTokenizerFast from that
snapshot. Weight SHA256 values are `4c40b740f209fd7682040d4ebf19409208acde5c401ccd2dde426d0f41542f18`
and `9490b82fa47f89e5110eb5577f8ab1b6694c12121905dbc086ddb10207a4ba0c`.
Native Transformers is **4.52.1**, Hub **0.36.0**, tokenizers **0.21.4**;
the decoder, DASH/SR-SWA, BF16 and original routing remain intact.

Native single `infer` defaults to dynamic 640px tiling for sufficiently large
images; native `infer_multi` defaults to `crop_mode=False`, one padded 1024px
global view per page. Open original published page images, convert RGB through
the original processor, retain input dimensions/bytes/hash and exact arguments.
Do not rescale, recrop, rerender PDFs or switch image modes without a separately
named experiment. All pages go in **one ordered document request**; independent
page OCR plus concatenation is not equivalent cross-page inference.

The pinned [official benchmark client](https://github.com/baidu/Unlimited-OCR/blob/48ff432fb68a18ed91f2f70021f29028b1dde354/infer.py)
is SGLang-specific, not the native wrapper:

- Prompt: `"<PAGE><image>" * num_pages + "Multi page parsing."`.
  Greedy `temperature=0.0`, `skip_special_tokens=False`.
- Default `crop_mode=True` (gundam). Exact per-PDF `{gundam,base}.jsonl`
  token tables, if present, must be hashed. Output budget is
  `max_model_len - image_tokens - 700`; if this is below `gt_text_tokens+1`
  for multiple pages, the official client selects base. Without a table the
  budget is `int(0.3*max_model_len)` = 39,321 / 235,929 for the two benchmarks.
  Those per-PDF tables are **not present in the audited source/data trees**.
- Context limits: 131,072 (MPDocBench), 786,432 (Marathon). One refit after a
  reported prompt-length rejection uses `max_model_len-input_tokens-64`.
  Retain the original rejection, both payloads, actual mode and final limit.
- Custom `DeepseekOCRNoRepeatNGramLogitProcessor`, ngram size **35**, window
  **1024** for multiple pages and **128** for one page. Record exact serialized
  processor code/hash; do not substitute a generic repetition penalty.
- Official page-window defaults: `SGLANG_UOCR_EANABLE_DOC_LIMIT=1` (upstream
  spelling), gundam/base windows **10**, per-page cap **20000**, reusing previous
  window KV tails. Whole-document single-window mode is a separate identity.
  Window boundaries affect continuation semantics; record every window and
  cap/auto-truncation event. Never call a truncated window successful.

The customized SGLang wheel was downloaded and statically inspected, not
installed or executed. Its METADATA requires **torch==2.13.0**,
flashinfer `[cu13]`, flash-attn-4 and other NVIDIA packages. It cannot be installed
unmodified into the preserved torch **2.12.0+rocm7.14.1** environment. HIP parity
and dependencies remain BLOCKED; generic SGLang is not a substitute.

Native Transformers and SGLang have different cache/window/token behavior.
The current production wrapper still allows at most **five images** and
`max_new_tokens <=8192`; its multi tuple lacks an authoritative finish reason.
This task removes neither cap and changes neither inference nor default MATH.
Future long-document adaptation must expose original finish/page/window fields
and prove semantics before full GPU evaluation. Published Baidu scores cannot
be treated as comparable until all these identities agree.

## Output, errors and reproducibility

Use the original V2 converter, already vendored byte-for-byte at SHA256
`fb38738cd5f09625fb1095d201533b6059a85820bca99c4262347cc7b078a782`.
Its bytes match the newly audited client revision. Keep raw UTF-8 separately.
Preserve PAGE positions, BLOCK/FORK fields, heading levels, HTML tables,
cross/append and MERGE union-find semantics. Figure tags must remain
`![](pageN_x1_y1_x2_y2.jpg)` on the 1000-coordinate grid. Do not normalize away
errors before scoring or replace original postprocessing with a local parser.

Each run binds dataset revision and complete local manifest hash, protocol/CLI
argument arrays, model/tokenizer/weight hashes, crop/token/ngram/window policy,
platform/device/backend/precision and frozen environment receipt into its ID.
Record UTC timestamps, git HEAD/dirty state, OS/Python/driver/ROCm/HIP/Triton,
actual imported and loaded binary/provider hashes, wheel SHA256, cache state,
seeds/concurrency, finish reasons, page/token counts, memory, raw errors and
timeouts. Sanitize exported paths/secrets without rewriting original evidence.

For Windows preserve torch **2.12.0+rocm7.14.1**, torchvision
**0.27.0+rocm7.14.1**, torchaudio **2.11.0+rocm7.14.1**, gfx1151.
Historical official wheel SHA256 `415d2dbd4b6ab002cff1c54795fdd66322abd9a46bf982cb64eebb4493ac2e7c`
and local candidate SHA256 `6a67d15861538ee217afdccd1ccc09ff318d7f44bff29a10f74c0d51730e6e1e`
are separate identities. Candidate installation must pass the existing complete
native/image/loaded-provider gate with no historical gfx115x image overlays.
No wheel rebuilding or redistribution. No CPU/model substitution or disabled
DASH/SR-SWA. MATH stays the default. APU allocation/reservation is shared system
memory accounting, not dedicated VRAM or total system consumption.

Document outcomes: PASS, FAIL, TIMEOUT, OOM, TRUNCATED; untouched inputs are
NOT_RUN. PASS means structurally complete inference/artifact preservation, not
correct content or a benchmark quality gate. Truncation, empty/malformed output,
wrong page count, nonfinite GPU evidence and missing finish data cannot PASS.
At most one empty trailing PAGE is reported separately as the existing native
end delimiter; blank middle/omitted/duplicated pages remain failures.

Attempt directories commit raw output and metadata together by atomic rename
and file fsync; committed files have hash seals. Pending attempts are visible
INTERRUPTED failures. Resume retries every non-PASS document and appends history;
only intact verified PASS artifacts may be reused. Runs/configurations never
share output namespaces. A single producing process owns each document.
Directory rename is a same-filesystem publication boundary, not a guarantee
against every power-loss/storage fault; revalidate all hashes after recovery.

Export exactly one official `.md` per expected document. Failed/unrun inputs
get explicit empty predictions with an accompanying full accounting receipt;
raw partial output remains in their attempt histories. Keep full original
scorer sample outputs and per-document metric coverage. A PARTIAL run must
retain its 420/100 denominator and error counts. Do not silently drop failures,
retry-select a better score, relax thresholds, infer omitted-page correctness
from a marker count alone, or label a successful subset FULL PASS.

No benchmark-quality acceptance threshold has been authorized by this task.
The old synthetic CER/number/structure thresholds remain unchanged and do not
establish real-corpus quality. A reviewed predeclared quality gate and all eight
working metrics are prerequisites to the next full-dataset mission.
