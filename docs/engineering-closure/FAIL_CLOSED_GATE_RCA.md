# Fail-closed experiment admission RCA and API

KEEP EXPERIMENTAL. This is an opt-in prospective dispatcher, with no production
activation. Historical private runners and their failed/unadmitted receipts stay
unchanged; this change does not intercept arbitrary Python process launches.

## Demonstrated root cause

The retained overnight incident receipt states that the prepared mixed AUTO
process launched in the same orchestration before the math analyzer's returned
strict FAIL was inspected. The analyzer writes its strict verdict but the generic
supervisor gates only process exit/owned-child cleanup, not semantic correctness.
Six math requests across four processes have exit0/functional PASS while strict bbox fails.
There was no executable dependency linking the analyzer's verdict to the next
supervisor invocation. Two later completed requests remain
UNADMITTED_AFTER_CORRECTNESS_FAIL. Their disposition is never changed here.

Evidence supports missing semantic-result inspection and an absent atomic
authorization boundary. It does not establish a stale receipt or concurrent GPU
race as the historical cause. Retained source lacks the surrounding interactive
tool-orchestration program; exact orchestration ordering is supported by the
incident record and process/result timestamps, not reconstructed source code.

An exit-only mutation control starts CPU fake workers after strict FAIL, PARTIAL
and absent qualification. Tests expecting real launch refusal fail under that
control, then pass with the new gate. This reproduces the unsafe mechanism
without executing a GPU request or altering a historical producer.

## Minimal solution

`benchmark.admission.Gate` reuses benchmark digest/file-hash/safe-ID helpers and
the existing Windows Job supervisor. A standard-library SQLite transaction
provides cross-process serialization; one table stores the frozen stage and
receipt. One ledger on a local filesystem belongs to one physical-device owner.
No new external dependency or general-purpose scheduling framework is introduced.

Transitions are PREPARED→RUNNING→EVALUATING→PASS/FAIL. Process exit0 reaches
EVALUATING, never PASS. A trusted evaluator must bind a sealed receipt to the
exact execution attempt. Incomplete/partial/incorrect results cannot authorize a
dependent stage. Exceptions/interruption before evaluation leave durable
RUNNING/EVALUATING or explicit FAIL; restart never promotes them.

Before spawn, admission checks immutable source/interpreter/wheel/fixture/policy/
artifact hashes, supported target, current complete strict PASS and no active
owned children. The hashed policy binds each stage to exact argv, cwd, environment
and dependency. Executable and entry script must be covered by source hashes;
the controller supplies the complete imported-source/runtime closure. Receipt
identity includes stage, command, attempt nonce, source/policy identity, predecessor
receipt and gate source hash. A prior receipt cannot qualify another execution.

Phase1 atomically reserves RUNNING and the admission receipt identity. Phase2
rechecks the same receipt/artifacts and process accounting immediately before
spawn, holding the SQLite write lock until worker exit/owned-tree cleanup. No
pre-admission token can be reused. The reserved owner waits at most five seconds
for a short transaction in the handoff gap. Other controllers fail closed on a busy ledger
or unfinished work. The Windows Job establishes ownership while suspended,
then resumes and drains descendants; no console broadcast termination occurs.

## API and evidence contract

Identity is a JSON object with exactly `sources`, `wheels`, `fixtures`, `policy`,
`artifacts`, `target`. Each first-five field is a nonempty absolute-path→SHA256
map. `policy` contains exactly one hashed JSON file; `target` is
`windows/gfx1151`. Artifacts must include the authenticated model/source/weight,
runtime/hardware and experiment evidence required by the prospective protocol.
The gate rehashes files; it does not itself load models or rediscover GPU identity.

The policy JSON maps stage names to exactly these four fields:

```python
{
    "correctness": {
        "argv": [python_executable, verified_worker_script, *frozen_arguments],
        "cwd": repository_root,
        "env": frozen_environment,
        "dependency": None,
    },
    "performance": {
        "argv": [python_executable, verified_worker_script, *other_frozen_arguments],
        "cwd": repository_root,
        "env": frozen_environment,
        "dependency": "correctness",
    },
}
```

Only an independently authorized initial qualification uses a null dependency
and `prepare(..., initial_qualification=True)`. Performance/population expansion
must use its declared predecessor. Freeze policy before qualifying; changing it
invalidates prior PASS. A trusted controller calls `prepare(stage, identity,
argv)`, then `dispatch(stage, dependency, identity, owned_job, timeout=seconds,
cwd=root, env=frozen_environment, stdout=log, stderr=log)`.

After completed execution and cleanup, the evaluator writes a new sealed JSON:

```python
{
    "identity": digest(identity),
    "execution": gate.execution(stage),
    "status": "PASS",            # or FAIL/PARTIAL; never normalize a failure
    "complete": True,
    "strict_correctness": True,
    "artifacts": verified_result_path_to_sha256,
}
```

It calls `evaluate(stage, receipt_path, receipt_sha256)`. Actual evaluator checks
are protocol-specific and must be reviewed; the gate authenticates their bound
receipt and artifact integrity, not correctness by assertion. Never manufacture
a PASS from functional completion or an old aggregate. Receipts/ledgers/env stay
private and may contain secrets. Use allowlisted summaries for publication.

For prospective dependent runs, the sole new CLI path is:

```powershell
python experiments/admission/run.py --ledger LEDGER --stage STAGE `
  --qualification QUALIFICATION --identity IDENTITY_JSON --output NEW_OUTPUT `
  --timeout SECONDS --execute-gpu
```

All paths/commands must already be frozen by the authorized coordinator; the CLI
cannot bootstrap a qualification or bypass its required predecessor. The explicit
GPU flag is not approval for this mission to run an experiment. CPU tests use
only fake workers. Original ungated controllers must not be used prospectively.

## Alternatives and limits

Prompt-only sequencing and exit-code checks caused the gap. An unlocked PASS
file permits check/spawn races. An in-process lock alone does not serialize
controllers. Existing append-only document ledger state alone cannot bind
cross-experiment dependencies, so it is not overloaded with GPU authorization.

This is a cooperative-owner correctness boundary, not a hostile-code sandbox.
The owner must enforce one shared ledger, authenticated evaluators, frozen files,
complete source/runtime closure, and exclusive physical hardware ownership.
Different ledgers, arbitrary legacy launchers or hostile filesystem/DB writers
can bypass that ownership policy; no claim of OS-wide GPU isolation is made.
The final check cannot eliminate adversarial file replacement between hash and
exec. Cleanup failure remains RUNNING and requires investigation; logs retain
the original attempt context, but a cleanup exception may mask the primary error.
There is no automatic crash recovery, retry, stale-lock deletion or PASS repair.
Retain the old ledger/failed attempt and require a newly reviewed protocol before
resuming. Source changes intentionally invalidate existing ledgers/receipts.
