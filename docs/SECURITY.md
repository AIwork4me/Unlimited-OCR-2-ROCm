# Security notes for pinned Unlimited-OCR-2

This document records static findings for Hugging Face revision `9cf077c4b206e0e2a31879df937b0b0bcba3661c`. It does not claim the model was executed, the environment was validated, or that all vulnerabilities were found. The audited files and hashes are in [UPSTREAM_AUDIT.md](UPSTREAM_AUDIT.md).

## Critical: generated coordinates reach Python `eval`

The upstream postprocessor is unsafe for untrusted documents/model output. `re_match()` captures detection content; `draw_bounding_boxes()` passes it to `extract_coordinates_and_label()`, which executes `eval(ref_text[2])`. The primary detection regex accepts arbitrary text between `<|det|>` and `<|/det|>`. A generated expression can therefore run Python with the inference process's permissions before any coordinate unpacking occurs. The surrounding `try/except` catches errors after execution; it is not a security boundary. [Regex and eval, lines 98–127](https://huggingface.co/baidu/Unlimited-OCR-2/blob/9cf077c4b206e0e2a31879df937b0b0bcba3661c/modeling_unlimitedocr.py#L98-L127), [call from drawing, lines 149–170](https://huggingface.co/baidu/Unlimited-OCR-2/blob/9cf077c4b206e0e2a31879df937b0b0bcba3661c/modeling_unlimitedocr.py#L149-L170).

Reachable public path:

```text
infer(save_results=True, output_path=...)
  -> _uocr_save_results
  -> re_match
  -> process_image_with_refs
  -> draw_bounding_boxes
  -> extract_coordinates_and_label
  -> eval(generated_detection_text)
```

`infer_multi(save_results=True)` delegates into the same path. This is a source-level vulnerability; no payload was executed during this audit. OCR text can be adversarial or simply malformed, even when the model source and weights are trusted. [Save call](https://huggingface.co/baidu/Unlimited-OCR-2/blob/9cf077c4b206e0e2a31879df937b0b0bcba3661c/modeling_unlimitedocr.py#L1102-L1124), [save implementation](https://huggingface.co/baidu/Unlimited-OCR-2/blob/9cf077c4b206e0e2a31879df937b0b0bcba3661c/modeling_unlimitedocr.py#L1207-L1228).

The minimal safe initial policy is `save_results=False` and independent UTF-8 storage of the returned string or result dictionary. This preserves vision, MoE, page positions, DASH routing and custom GPU kernels. `eval_mode=True` only disables streaming; it does **not** disable unsafe saving. Avoid calling the drawing helpers directly until a security patch is in place. Setting an empty output path also bypasses saving, but explicitly setting `save_results=False` is clearer.

## Minimal postprocessor patch boundary

No upstream source patch was made by this audit. If bounding boxes/crops are required, the smallest acceptable change is confined to detection parsing and validation:

1. Replace `eval` with a non-executing parser such as `ast.literal_eval`; never use `eval(..., {"__builtins__": {}})` as a substitute.
2. Bound input length before parsing. Reject an excessively long/deep result or too many boxes; literal parsing prevents code execution but does not eliminate resource exhaustion.
3. Require a flat four-number box or a bounded list of four-number boxes. Exclude booleans and non-numeric/container objects; require finite values. Check `0 <= x1 < x2 <= 999` and `0 <= y1 < y2 <= 999` before scaling/cropping, or explicitly define a documented clipping policy.
4. Treat invalid detection data as plain raw output/skip that detection; never retry through `eval`. Bound crop dimensions/count before allocating images.
5. Exercise valid single/multiple boxes, malformed data, expression/call syntax, non-finite/oversized coordinates, and resource bounds using inert fixtures. These checks can target an isolated parser without loading model weights or running upstream model code.

Record a patch manifest with upstream revision, original file hash, patch hash and the exact security-only diff. Preserve the pristine downloaded model source. Apply a patch in a separately identified local snapshot, not by editing a shared Hugging Face cache in place. Do not describe a patched source tree as byte-identical to upstream.

## Remote code and supply-chain boundary

`trust_remote_code=True` executes repository Python with user privileges. Safetensors restricts weight serialization; it does not make accompanying Python safe. Even `AutoConfig` maps to the modeling entrypoint here and is not an inert JSON-only inspection. Pin code, config, tokenizer and weights to the same exact revision, and record local hashes. The entrypoint directly references the other nine Python files, including both Triton kernels; keep the complete dependency closure. [Top-level mapping](https://huggingface.co/baidu/Unlimited-OCR-2/blob/9cf077c4b206e0e2a31879df937b0b0bcba3661c/config.json#L1-L11), [direct imports](https://huggingface.co/baidu/Unlimited-OCR-2/blob/9cf077c4b206e0e2a31879df937b0b0bcba3661c/modeling_unlimitedocr.py#L1-L59).

Loader dependency discovery is not a sandbox. `try` wrappers do not remove relative files from the load graph; see the version-specific recursive-copy explanation in [UPSTREAM_AUDIT.md](UPSTREAM_AUDIT.md).

The full-file static scan found no direct `subprocess`, shell execution, socket client, dynamic `exec`, inline assembly, or network request invocation in the ten upstream Python files. `requests` is imported but the inspected `infer` path opens local images. This is a bounded observation about these files, not a guarantee about imported packages or future revisions.

Import side effects are real: the base reads/hashes its own file for logging, reads environment controls, optionally wraps flex attention with `torch.compile`, and the overlay mutates base attention classes globally. A public `disable_torch_init()` helper globally replaces `Linear.reset_parameters` and `LayerNorm.reset_parameters`; the audited `infer` path does not call it. Use a dedicated process and avoid invoking that helper in a shared application. [Base import behavior](https://huggingface.co/baidu/Unlimited-OCR-2/blob/9cf077c4b206e0e2a31879df937b0b0bcba3661c/_sparse_base_impl.py#L51-L195), [init helper](https://huggingface.co/baidu/Unlimited-OCR-2/blob/9cf077c4b206e0e2a31879df937b0b0bcba3661c/modeling_unlimitedocr.py#L1028-L1035).

An alternate SAM builder has `torch.load(checkpoint)` without an explicit serialization policy. The ordinary constructor passes no checkpoint, so this call is not part of the selected load path. Do not supply arbitrary pickled checkpoint files to the helper. [Alternate checkpoint branch](https://huggingface.co/baidu/Unlimited-OCR-2/blob/9cf077c4b206e0e2a31879df937b0b0bcba3661c/deepencoder.py#L1044-L1055).

Use the fixed requested package versions and trusted, recorded wheel origins; do not run upstream's generic `pip install torch==2.13.0 ...` inside the target ROCm environment. No SGLang wheel is needed for this Transformers route. Wheel authenticity, dependency vulnerabilities, driver components and compiler executables require separate provenance/validation; model-source review cannot certify them.

## Documents, outputs and resource limits

Treat generated Markdown, formulas, links and embedded markup as untrusted content. Save raw results before any optional transformation. A viewer should disable active HTML/scripts and automatic remote fetches; raw OCR must never be interpolated into executable code or shell commands. The upstream crop filenames use numeric page/image indices rather than model labels, but `output_path` is caller-controlled and files are overwritten if names already exist. Select a dedicated output directory in the application layer. [Output writes](https://huggingface.co/baidu/Unlimited-OCR-2/blob/9cf077c4b206e0e2a31879df937b0b0bcba3661c/modeling_unlimitedocr.py#L1207-L1228).

Bound local image size, decoded pixel count, page count, crop count, prompt length and generated tokens before accepting arbitrary jobs. Do not disable Pillow's image-decompression protections. The model loads all page images and uses growing caches in the HF path; it does not impose a complete service resource budget. Even a short generated response can require a large prefill. Use explicit output-token limits and retain failure diagnostics rather than retrying indefinitely. [Image load and tile control](https://huggingface.co/baidu/Unlimited-OCR-2/blob/9cf077c4b206e0e2a31879df937b0b0bcba3661c/modeling_unlimitedocr.py#L1069-L1101), [cache growth](https://huggingface.co/baidu/Unlimited-OCR-2/blob/9cf077c4b206e0e2a31879df937b0b0bcba3661c/modeling_deepseekv2.py#L484-L529).

Generation toggles global inference/page-scope state and temporarily changes config, restoring it in `finally`. This is not a thread-isolation guarantee: serialize inference calls per process/model or use separate workers. Exception handling should preserve the original compiler/launch traceback. The native requirement keeps `dash_use_triton=True`; falling back or disabling it must be explicitly labeled a diagnostic, never reported as custom-kernel success. [State switch](https://huggingface.co/baidu/Unlimited-OCR-2/blob/9cf077c4b206e0e2a31879df937b0b0bcba3661c/modeling_deepseekv2.py#L296-L335), [generation cleanup](https://huggingface.co/baidu/Unlimited-OCR-2/blob/9cf077c4b206e0e2a31879df937b0b0bcba3661c/modeling_unlimitedocr.py#L1150-L1205).

## License preservation

Preserve the pinned [Baidu MIT license](https://github.com/baidu/Unlimited-OCR/blob/4d0bebab8714e0f3b0c811cd44bf422ddb49daae/LICENSE), source copyright headers and third-party notices when copying or patching helpers. The base source explicitly contains Apache-2.0 terms; the gather source acknowledges flash-linear-attention adaptation. A security patch does not remove attribution obligations. No separate postprocessor license was present in the inspected files. See the licensing section of [UPSTREAM_AUDIT.md](UPSTREAM_AUDIT.md).
