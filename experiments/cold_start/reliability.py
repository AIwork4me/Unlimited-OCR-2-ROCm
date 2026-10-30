"""New strict ledger attached to actual D jobs; historical campaigns stay immutable."""

from __future__ import annotations

import json
import shutil
import sys

import run as driver
from report import process_gate
from worker import BASE, ORIGINAL, ROOT, sha, write

sys.path.insert(0, str(ROOT / "src"))
from unlimited_ocr_2_rocm.benchmark.ledger import Run  # noqa: E402
from unlimited_ocr_2_rocm.benchmark.manifest import inventory  # noqa: E402


def failure_reason(exact, functional, process_ok):
    if not process_ok:
        return "PROCESS_COMPLETION_CLEANUP_FAIL"
    if not functional:
        return "FUNCTIONAL_RECOVERY_FAIL"
    return "" if exact else "EXACT_GOLDEN_PARITY_FAIL"


def fault_contract(receipt):
    event = receipt.get("fault_checkpoint", {})
    return (
        receipt.get("exit_code", 0) != 0
        and event.get("completed_decoder_forwards") == 3
        and event.get("synchronized") is True
        and event.get("hip_zero") is True
        and receipt.get("native_return_before_kill") is False
        and receipt.get("accounting_after_cleanup", {}).get("active_processes") == 0
    )


def main():
    if sys.argv[1:] != ["--execute-gpu"] or ROOT != __import__("pathlib").Path.cwd().resolve():
        raise ValueError("Explicit GPU opt-in from primary root required")
    out = ROOT / ".b/cold-start-rca/D-ledger-attempt1"
    out.mkdir(exist_ok=False)
    write(
        out / "producer-identities.json",
        {str(p.relative_to(ROOT)): sha(p) for p in (ROOT / "experiments/cold_start").glob("*.py")},
    )
    scope = []
    for doc in ("no_kill", "post_kill"):
        images = []
        for page in (1, 2):
            name = f"images/{doc}/page_{page}.png"
            dest = out / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / f"tests/fixtures/page_{page}.png", dest)
            images.append(name)
        scope.append(dict(page_info=dict(image_path=doc + ".pdf", images_list=images)))
    write(out / "scope.json", scope)
    manifest = inventory(out / "scope.json", out, "new-strict-cold-start-reliability", BASE, 2, 4)
    ledger = Run.create(
        out / "runs",
        manifest,
        dict(
            source_commit=BASE,
            wheel_sha256=ORIGINAL,
            ledger_producer_sha256=sha(__file__),
            acceptance="EXACT_ORDERED_GOLDEN; warm requests diagnostic only",
        ),
        dict(device="Windows gfx1151", historical_campaigns="IMMUTABLE"),
    )
    golden = json.loads(
        (ROOT / ".b/i3010/physical2/gpu/pages_2_run_1/diagnostics/native-001.json").read_text(
            "utf-8"
        )
    )
    attempts = {}

    def record_ocr(doc, folder):
        path = folder / "worker/request-0/summary.json"
        if not path.exists():
            ledger.finish(doc, attempts[doc], "", "FAIL", dict(error="NO_COMPLETED_NATIVE_RESULT"))
            return
        summary = json.loads(path.read_text("utf-8"))
        native = summary.get("native", {})
        exact = all(native.get(k) == golden[k] for k in ("text", "token_ids", "prompt_token_ids"))
        functional = summary.get("status") == "PASS" and summary.get("physical_contract") == "PASS"
        process_ok = process_gate(folder)
        reason = failure_reason(exact, functional, process_ok)
        state = "FAIL" if reason else "PASS"
        ledger.finish(
            doc,
            attempts[doc],
            native.get("text", ""),
            state,
            dict(
                error=reason,
                process_completion_cleanup=process_ok,
                exact_golden=exact,
                functional_ocr=physical_functional(summary),
                finish_reason=native.get("finish_reason"),
                output_tokens=len(native.get("token_ids", [])),
                max_tokens=2048,
                memory=summary.get("memory_after", {}),
                source="ACTUAL_NEW_GPU_REQUEST_0",
                warm_request="DIAGNOSTIC_ONLY_NOT_LEDGER_CREDIT",
            ),
        )

    def observed_write(path, value):
        if path.name == "predeclared.json":
            value = dict(
                value,
                actual_entrypoint="experiments/cold_start/reliability.py",
                ledger_producer_sha256=sha(__file__),
                ledger_acceptance="STRICT_GOLDEN_REQUEST_0; warm requests diagnostic only",
            )
        write(path, value)
        if path.name == "finished.json":
            name = path.parent.name
            if name == "no_kill":
                record_ocr("no_kill", path.parent)
            elif name == "fault":
                returned = list(
                    (path.parent / "worker").glob("request-*/diagnostics/native-*.json")
                )
                raw = json.loads(returned[0].read_text("utf-8")).get("text", "") if returned else ""
                ledger.finish(
                    "post_kill",
                    attempts["post_kill"],
                    raw,
                    "FAIL",
                    dict(
                        error="EXTERNAL_TERMINATION"
                        if fault_contract(value) and not returned
                        else "FAULT_CONTRACT_FAIL",
                        native_return_available=bool(returned),
                        partial_native_return="full native return retained"
                        if returned
                        else "unavailable",
                        actual_process_exit=value.get("exit_code"),
                    ),
                )
            elif name == "recovery":
                record_ocr("post_kill", path.parent)

    driver.write = observed_write
    original_load = driver.load

    def witnessed_load(name, path):
        module = original_load(name, path)
        original_spawn = module.Job.spawn

        def witnessed_spawn(job, command, *a, **kw):
            worker_out = __import__("pathlib").Path(command[command.index("--output") + 1])
            doc = "no_kill" if worker_out.parent.name == "no_kill" else "post_kill"
            attempts[doc] = ledger.begin(doc)
            return original_spawn(job, command, *a, **kw)

        module.Job.spawn = witnessed_spawn
        return module

    driver.load = witnessed_load
    sys.argv = [
        str(ROOT / "experiments/cold_start/run.py"),
        "--phase",
        "D",
        "--output",
        str(ROOT / ".b/cold-start-rca/D-attempt1"),
        "--execute-gpu",
    ]
    try:
        driver.main()
    finally:
        write(out / "ledger-summary.json", ledger.summary())


def physical_functional(summary):
    return summary.get("status") == summary.get("physical_contract") == "PASS"


if __name__ == "__main__":
    main()
