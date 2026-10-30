# Windows external termination and recovery — 2026-10-30

**Strict restart-parity outcome: FAIL.** External process termination and a
checkpoint-relative hard timeout both terminate their owned Windows Job Object
with zero active processes left. Subsequent original-wheel GPU OCR succeeds
functionally, but its first cold request differs from the ordered golden output.
A fresh no-kill control reproduces the same difference. This is not overall P3
qualification or evidence of a termination-specific corruption regression.

Producer: `c01dbd8fd5af256565fc5a13fd56e9cbd2471e93`.
Production base: `82951c73b28bf735f5655e6f5c3306cd170a21c2`.
Original published wheel SHA256:
`415d2dbd4b6ab002cff1c54795fdd66322abd9a46bf982cb64eebb4493ac2e7c`.
This work uses the original wheel, independently of candidate-wheel P2.
Real documents remain **BLOCKED_DATA**; only original two-page synthetic inputs
are used. No existing PDF, restricted dataset, production default or wheel is
modified. Raw text/token arrays, images and scope JSON remain private/local.

## Actual matrix

| Test | Outcome | Evidence scope |
| --- | --- | --- |
| External termination during live GPU decoding | PASS | Synchronized checkpoint after three real decoder forwards; owned job terminated; worker exit 1; no persisted native-return artifact; zero active job processes. |
| Hard inference deadline and cleanup | PASS | 100 ms after the supervisor observes the same live checkpoint, then terminate/await job; zero active processes. This is not an end-to-end 100-ms timeout; boot/checkpoint/recovery has a separate 600-second bound. |
| New-process functional GPU recovery | PASS | One fresh process after each fault; two sequential requests each. All four requests have correct text/numbers, canonical pages/blocks, native stop, HIP-zero and twelve BF16 DASH calls. |
| Exact ordered-golden restart parity | FAIL | First request after each fault differs in raw/output IDs by a one-unit bbox coordinate; second request matches exactly. Prompt IDs and token count 116 match throughout. |
| Fresh no-kill control | Informational | Same two-request sequence: identical cold raw hash to both post-fault processes, then identical warm/golden hash. Both functional checks pass. |
| Original failure/retry ledger campaign | PARTIAL / FAIL | Two-document denominator retained: external-kill FAIL, timeout NOT_RUN because the first recovery rejection stopped the campaign. No final PASS result was written. |
| Independent timeout followup | Separately recorded | Timeout and its recovery were executed in new output directories; never folded into the rejected campaign's ledger. |
| Local CPU checks | PASS | 333 passed / 8 skipped, including six focused diagnostic checks; Ruff/format, production mypy, build and pip check exit 0. |

The initial campaign's failure remains intact. The followup no-kill control
and timeout do not retroactively pass the ordered-golden acceptance rule.
Six total recovery/control requests passed functional acceptance; three first
requests fail ordered-golden exact equality and three second requests match.
Removing only bbox coordinates gives equal content, labels, structure and page
order. This localization is diagnostic, not a tolerance relaxation.

The observed same-invocation cold-control agreement isolates the discrepancy
from the mere presence of a preceding kill. The low-level numerical/kernel or
initialization cause remains unproven. No production fix, warmup workaround,
new tolerance or semantic completeness claim is justified from this result.
The earlier Windows P0 receipt applies to its sealed thirteen-case ordered
regression in the retained environment; it does not certify all cold/order
contexts, fresh installation or broad release/repository compliance.

## Failure and identity contracts

Every child rehashes the original wheel and all installed native members,
checks pip, loads only the locked offline model, and verifies all fifteen
loaded Torch members plus observed SDK hashes against the qualified receipt.
All five actual child identities and model/source bindings were independently
reviewed. No mutable model code, source substitution or CPU fallback is used.

The supervisor creates a kill-on-close job, assigns a suspended child before
resuming it, records its PID/creation identity and awaits termination of the
job. A 32 GiB available-system-memory floor precedes launch; an 8 GiB runtime
floor stops it. A native result captured before termination rejects the fault
test. Actual returned raw is retained; unavailable partial text/IDs are never
invented. A pending ledger attempt cannot be read as PASS.

Before any aggregate PASS, the diagnostic checks the actual committed ledger
status/counts and exact `[FAIL, PASS]` / `[TIMEOUT, PASS]` histories. Synthetic
CPU regressions caught late-return and downgraded-ledger false-PASS paths before
physical execution. Independent review approved the frozen source, then
reconstructed all physical results and the metrics-only publication.

These tests do not qualify physical OOM, concurrent serving, system driver
recovery, arbitrary long paths/documents, >5 pages, other hardware/OS, all
resource-admission failure paths or complete P3 readiness. APU allocator usage
is not dedicated VRAM or total system memory. No performance claim follows
from fault-test wall times.

## Reproduction

From the primary repository root, use the separate retained worktree and a
new repository-local output directory with a genuine matching scoped receipt:

```powershell
python .worktrees/r240/experiments/windows_reliability/physical.py `
  --output .b/p240/reliability-new `
  --p0-receipt .b/p240/windows-p0-pass.json --execute-gpu
```

The source/production/fixture gates require a clean, committed producer.
Existing directories are refused. This command currently reproduces the strict
recovery rejection; do not add warmups or replace the reference to force PASS.
Keep all failed outputs. Private originals and the separately bounded no-kill /
timeout followup driver are under `.b/p240/`. Published metrics and hashes:
`eval/windows-reliability/physical.json`. No raw OCR or fixture payload is included.
