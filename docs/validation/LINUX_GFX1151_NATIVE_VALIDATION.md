# Linux gfx1151 native validation — physical GPU results

Date: 2026-10-29 (Asia/Shanghai). First physical native-Linux ROCm validation
of Baidu Unlimited OCR 2 on the **primary Linux target**
(Ryzen AI MAX+ PRO 395 / Radeon 8060S / gfx1151, HP ZBook Ultra G1a,
Ubuntu 24.04.4 + OEM kernel 6.17). Evidence root:
`eval/results/Linux gfx1151 verified/`. Environment identity:
`docs/validation/LINUX_GFX1151_ENVIRONMENT_DECISION.md`,
freeze `requirements/linux-cp310.lock.txt`.

All PASS values below were produced by the commands recorded in
`scripts/run_linux_gfx1151_validation.sh` on the pinned stack; nothing is
inferred from matrix listings or Windows evidence.

## Stack identity (Gate A — doctor PASS)

- CPython 3.10 venv `../envs/linux-gfx1151-rocm`; `torch 2.12.0+rocm7.2`,
  `torchvision 0.27.0+rocm7.2`, `triton-rocm 3.7.0` — all three wheels
  SHA-256-verified before install (`wheel-sha256.json`, all verified: true).
- Runtime identity observed in-process: `torch.version.hip = 7.2.53211`,
  `torch.version.rocm = 7.2.1`, device `AMD Radeon 8060S`,
  `gcnArchName gfx1151`, arch list includes gfx1151; no `HSA_OVERRIDE_*`,
  CPU fallback refused by the profile gate.
- Hardware evidence: CPU model string `AMD RYZEN AI MAX+ PRO 395 w/ Radeon
  8060S`, kernel `6.17.0-1033-oem`, `/dev/kfd` accessible, render node
  `/dev/dri/renderD128`, machine-id hash recorded in `doctor.json`.
- Real BF16 GPU matmul vs independent CPU FP32 reference: PASS,
  max abs error `4.76837158203125e-07` — the same cancellation-residue value
  documented on the Windows stack (same 32×32 integer-valued fixture).
- Model snapshot: all 18 lock files (code+tokenizer+weights) SHA-256-verified
  against `upstream-lock.json` before load; both weight shards re-verified
  after parallel-range download (`4c40b740…`, `9490b82f…`).

## Kernel-level gates (PASS)

| Gate | Evidence | Result |
| --- | --- | --- |
| B — Triton compile+execute | `triton.json` | PASS: triton 3.7.0, `GPUTarget(backend='hip', arch='gfx1151', warp_size=32)`, amdgcn SHA `b8868418f377c5f8…`, 0.262 s compile+run, exact numeric match |
| C — DASH sparse prefill kernel vs independent math | `sparse.json` | PASS on the frozen `sparse_gather_e2e_triton.py` (checkpoint head layout, remote-bias + partial-block cases) |
| D — vision SDPA scoped MATH | `vision_sdpa.json` | PASS: 3 SAM block forwards + 54×12×196×64 SDPA reference, max abs err 0.0039 (rtol/atol 0.03), `hipPeekAtLastError == 0` via the bundled `libamdhip64`, global attention flags restored, 4 recorded events |

## OCR regression (Gates E/F — suite PASS)

Single-page (`single/suite.json`): `english`, `chinese`, `mixed`, `table`,
`math` and `repeat_unicode_recovery` — **all PASS**.
Multi-page native one-request (`multi/suite.json`): `pages_2` ×2, `pages_3`
×2, `pages_5` ×2, `table_2_pages` — **all PASS**.

Per-run completion contract (PR #8 fields, first physical exercise on any
platform): `finish_reason = stop` on every run; output tokens ≤ budget with
token counts **identical to Windows** on all compared cases
(80/82/84/60/72 single; 116/171/286/286/94 multi); `expected_pages` ==
input pages; `num_pages_out` == `<PAGE>` count in ids and text (the guard's
trailing page-flip before forced EOS appears on the 2-page cases on both
platforms and is accepted by the reviewed contract semantics); timing fields
(`time`, `prefill_time`) valid and recorded.

PAGE/BLOCK structure: `require_complete_pages` passed on every run — canonical
`<BLOCK>label[<LEVELn>][bbox]<FORK>` headers, no `__raw__` fallbacks, no stray
pre-block text. Original DASH kernel execution: ≥12
`triton_gather_attention_e2e` kernel executions recorded per run (exactly 12,
one per decoder layer), dense-prefill fallback refused at runtime, 12-layer
`DashAttention` `use_triton` configuration verified at load.

## Comparison with the reviewed Windows evidence

Windows anchor: `eval/results/Windows 中文 verified/` (Phase 1, source-built
`torch-2.12.0+rocm7.14.1`, triton-windows 3.7.1.post27).

| Aspect | Result |
| --- | --- |
| Normalized OCR text | **byte-identical** on every compared case (5 single + 5 multi) |
| Raw outputs | text content identical; bbox coordinates differ by ±1 integer unit on 8/10 cases (coordinate-regression rounding across stacks; `table` single raw byte-identical) |
| Output token counts / finish | identical tokens, `stop` on both platforms, 12/12 kernels |
| Wall time (same fixtures, different host+stack) | Linux faster on every case, e.g. 5-page 14.5 s vs 21.2 s, single chinese 4.4 s vs 6.8 s — observation only; no performance claim beyond these fixtures |
| Correctness gate | CER/status PASS thresholds unchanged; both suites PASS |

No Windows evidence was modified. These Linux results do not validate
Windows, gfx1100/W7900, other dtypes, AUTO mode, SGLang, or >5-page requests.

## Reproduction

```
VENV_PYTHON=../envs/linux-gfx1151-rocm/bin/python \
HF_CACHE=../hf-cache scripts/run_linux_gfx1151_validation.sh
```

Fresh process per gate; fresh output dirs (attempt 1 of `single`, which
failed before model load on the missing-torchvision import, is preserved at
`single_attempt1_torchvision_missing/`).
