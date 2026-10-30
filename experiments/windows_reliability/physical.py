"""Opt-in original-wheel external termination and fresh-process recovery evidence."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

WORK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORK / "experiments/p2_ab"))
from protocol import file_hash, publish  # noqa: E402
from windows_job import Job  # noqa: E402
from worker import execution_gate  # noqa: E402

ORIGINAL = "415d2dbd4b6ab002cff1c54795fdd66322abd9a46bf982cb64eebb4493ac2e7c"
BASE = "82951c73b28bf735f5655e6f5c3306cd170a21c2"


def qualify_fault(event: Any, exit_code: int, active: int) -> bool:
    return bool(
        isinstance(event, dict)
        and event.get("completed_decoder_forwards") == 3
        and event.get("hip_zero") is True
        and exit_code != 0
        and active == 0
    )


def has_native_return(output: Path) -> bool:
    return (
        any(output.glob("native-returned.json*"))
        or any(output.glob("request-*/diagnostics/native-*"))
        or any(output.glob("request-*/wrapper/raw.txt*"))
        or any(output.glob("request-*/summary.json"))
        or (output / "normal-finish.json").exists()
    )


def ledger_pass(summary: dict[str, Any]) -> bool:
    documents = {row.get("doc_id"): row for row in summary.get("documents", [])}
    if (
        summary.get("status") != "COMPLETE"
        or summary.get("expected_documents") != 2
        or summary.get("expected_pages") != 4
        or summary.get("counts", {}).get("PASS") != 2
        or set(documents) != {"external_kill", "timeout"}
    ):
        return False
    return all(
        [a.get("state") for a in documents[doc]["attempts"]] == [state, "PASS"]
        for doc, state in (("external_kill", "FAIL"), ("timeout", "TIMEOUT"))
    )


def helper() -> Any:
    path = WORK / "experiments/native_regression/runner.py"
    spec = importlib.util.spec_from_file_location("reliability_sealed_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def child(output: Path, fault: bool, receipt: Path) -> None:
    import hashlib

    root = Path.cwd()
    execution_gate({"base_source_commit": BASE}, receipt)
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(WORK / "src"))
    runner = helper()
    wheel = root / ".b/wheels/torch-2.12.0+rocm7.14.1-cp310-cp310-win_amd64.whl"
    if file_hash(wheel) != ORIGINAL:
        raise ValueError("Original published artifact required")
    expected = {}
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.lower().endswith((".dll", ".pyd")):
                sha = hashlib.sha256(archive.read(name)).hexdigest()
                if file_hash(Path(sys.prefix) / "Lib/site-packages" / name) != sha:
                    raise ValueError("Original installed native identity mismatch")
                expected[Path(name).name.lower()] = sha
    pip = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, check=False)
    publish(output / "pip-check.json", dict(exit_code=pip.returncode))
    if pip.returncode:
        raise ValueError("Original environment dependency check failed")
    from unlimited_ocr_2_rocm import inference
    from unlimited_ocr_2_rocm.model import load_model

    native = load_model(root / ".b/h", offline=True)
    loaded = runner.loaded_modules()
    actual = {Path(m["path"]).name.lower(): m for m in loaded}
    for name, sha in expected.items():
        if (
            name not in actual
            or actual[name]["sha256"] != sha
            or not Path(actual[name]["path"]).resolve().is_relative_to(Path(sys.prefix).resolve())
        ):
            raise ValueError("Original loaded native identity mismatch")
    sealed = json.loads((root / ".b/i3010/physical2/preflight/report.raw.json").read_text("utf-8"))
    sealed_modules = {Path(m["path"]).name.lower(): m["sha256"] for m in sealed["loaded_modules"]}
    if any(m["sha256"] != sealed_modules.get(Path(m["path"]).name.lower()) for m in loaded):
        raise ValueError("Loaded SDK/native module differs from the qualified original receipt")
    publish(
        output / "identity.json",
        dict(wheel_sha256=ORIGINAL, installed=expected, loaded=loaded, model=native.evidence),
    )
    hip = runner.open_loaded_hip(loaded)
    module = __import__(native.model.__class__.__module__, fromlist=["_"])
    fixtures = WORK / "tests/fixtures"
    truths = json.loads((fixtures / "ground_truth.json").read_text(encoding="utf-8"))["cases"]
    case = next(c for c in runner.normal_cases() if c["name"] == "pages_2_run_1")
    if fault:
        import torch

        original = native.model.model.forward
        calls = 0

        def observed(*args: Any, **kwargs: Any) -> Any:
            nonlocal calls
            result = original(*args, **kwargs)
            calls += 1
            if calls == 3:
                torch.cuda.synchronize()
                check = runner.hip_probe(hip)
                publish(
                    output / "inference-live.json",
                    dict(
                        completed_decoder_forwards=3,
                        hip_zero=check["status"] == "PASS",
                        native_result_returned=False,
                    ),
                )
            return result

        native.model.model.forward = observed
        original_infer = native.model.infer

        def returned(*args: Any, **kwargs: Any) -> Any:
            result = original_infer(*args, **kwargs)
            publish(output / "native-returned.json", result)
            return result

        native.model.infer = returned
    comparisons = []
    for repeat in range(1 if fault else 2):
        summary = runner.run_case(
            native, module, inference, case, truths, fixtures, output / f"request-{repeat}", hip=hip
        )
        golden = runner.read_case(root / ".b/i3010/physical2/gpu/pages_2_run_1")
        comparisons.append(runner.compare_case(golden, summary))
    publish(
        output / "normal-finish.json",
        dict(
            comparisons=comparisons,
            status="PASS" if all(c["status"] == "PASS" for c in comparisons) else "FAIL",
        ),
    )


def launch(output: Path, receipt: Path, mode: str) -> dict[str, Any]:
    import psutil

    root = Path.cwd()
    if psutil.virtual_memory().available < 32 * 1024**3:
        raise MemoryError("32 GiB system-available admission floor")
    output.mkdir()
    temporary = output / "tmp"
    temporary.mkdir()
    env = dict(
        os.environ,
        PYTHONUTF8="1",
        PYTHONIOENCODING="utf-8",
        TEMP=str(temporary),
        TMP=str(temporary),
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        TRITON_CACHE_DIR="C:/b/g1/triton",
    )
    command = [
        "C:/uocr/.worktrees/phase1-windows-gfx1151/.venv-repro/Scripts/python.exe",
        "-u",
        str(Path(__file__)),
        "--child",
        "--output",
        str(output / "worker"),
        "--p0-receipt",
        str(receipt),
    ]
    if mode != "recovery":
        command.append("--fault")
    record: dict[str, Any] = dict(
        mode=mode,
        state="RUNNING",
        command=command,
        deadline_seconds=600,
        producer_sha256=file_hash(Path(__file__)),
    )
    publish(output / "started.json", record)
    job, process = Job(), None
    started, terminate_at = time.monotonic(), None
    event = None
    try:
        with (
            (output / "stdout.log").open("wb") as stdout,
            (output / "stderr.log").open("wb") as stderr,
        ):
            process = job.spawn(command, cwd=root, env=env, stdout=stdout, stderr=stderr)
            record.update(pid=process.pid, created=psutil.Process(process.pid).create_time())
            while process.poll() is None:
                marker = output / "worker/inference-live.json"
                if marker.exists() and event is None:
                    event = json.loads(marker.read_text(encoding="utf-8"))
                    terminate_at = time.monotonic() + (0.10 if mode == "timeout" else 0)
                if (
                    mode != "recovery"
                    and terminate_at is not None
                    and time.monotonic() >= terminate_at
                ):
                    record["fault_triggered"] = True
                    job.terminate()
                    break
                if time.monotonic() - started > 600:
                    raise TimeoutError("Worker never reached the required physical checkpoint")
                if psutil.virtual_memory().available < 8 * 1024**3:
                    raise MemoryError("8 GiB system-available stop floor")
                time.sleep(0.02)
            process.wait(timeout=30)
            if mode == "recovery":
                finish = json.loads(
                    (output / "worker/normal-finish.json").read_text(encoding="utf-8")
                )
                record["qualified"] = (
                    process.returncode == 0
                    and finish["status"] == "PASS"
                    and job.accounting()["active_processes"] == 0
                )
            else:
                record["qualified"] = (
                    record.get("fault_triggered") is True
                    and qualify_fault(
                        event, process.returncode, job.accounting()["active_processes"]
                    )
                    and not has_native_return(output / "worker")
                )
            record.update(
                state="PASS" if record["qualified"] else "FAIL", exit_code=process.returncode
            )
    except BaseException as error:
        record.update(state="FAIL", error=repr(error))
        raise
    finally:
        record["job_before_final_cleanup"] = job.accounting()
        if record["job_before_final_cleanup"]["active_processes"]:
            job.terminate()
        record["job_after_cleanup"] = job.accounting()
        job.close()
        if process is not None:
            process.wait(timeout=30)
        record.update(
            seconds=time.monotonic() - started,
            checkpoint=event,
            producer_commit=subprocess.check_output(
                ["git", "-C", str(WORK), "rev-parse", "HEAD"], cwd=root
            )
            .decode()
            .strip(),
        )
        publish(output / "finished.json", record)
    if record["state"] != "PASS":
        raise RuntimeError("External failure/recovery not qualified; retain attempt")
    return record


def parent(output: Path, receipt: Path) -> None:
    import shutil

    root = Path.cwd()
    execution_gate({"base_source_commit": BASE}, receipt)
    if subprocess.check_output(["git", "-C", str(WORK), "status", "--porcelain"], cwd=root):
        raise ValueError("Freeze diagnostic producers in a clean commit before physical execution")
    if subprocess.check_output(
        [
            "git",
            "-C",
            str(WORK),
            "diff",
            BASE,
            "HEAD",
            "--",
            "src",
            "requirements",
            "tests/fixtures",
        ],
        cwd=root,
    ):
        raise ValueError("Original qualified production/dependency/fixture source changed")
    if output.exists() or not output.resolve().is_relative_to(root.resolve()):
        raise ValueError("Use a fresh repository-local output")
    output.mkdir(parents=True)
    sys.path.insert(0, str(WORK / "src"))
    from unlimited_ocr_2_rocm.benchmark.ledger import Run
    from unlimited_ocr_2_rocm.benchmark.manifest import inventory

    rows = []
    for mode in ("external_kill", "timeout"):
        images = []
        for page in (1, 2):
            relative = f"images/{mode}/page_{page}.png"
            dest = output / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(WORK / f"tests/fixtures/page_{page}.png", dest)
            images.append(relative)
        rows.append(dict(page_info=dict(image_path=mode + ".pdf", images_list=images)))
    publish(output / "synthetic-scope.json", rows)
    manifest = inventory(
        output / "synthetic-scope.json",
        output,
        "Windows-physical-external-termination-synthetic",
        BASE,
        2,
        4,
    )
    run = Run.create(
        output / "runs",
        manifest,
        {"source_commit": BASE, "original_wheel": ORIGINAL},
        {"physical_target": "Windows gfx1151", "p0_receipt_sha256": file_hash(receipt)},
    )
    results = {}
    for mode in ("external_kill", "timeout"):
        attempt = run.begin(mode)
        publish(output / (mode + "-pending.json"), run.attempts(mode))
        fault = launch(output / mode, receipt, mode)
        state = "TIMEOUT" if mode == "timeout" else "FAIL"
        run.finish(
            mode,
            attempt,
            "",
            state,
            dict(
                error="HARD_TIMEOUT" if mode == "timeout" else "EXTERNAL_TERMINATION",
                partial_native_result="unavailable",
                physical_checkpoint=fault["checkpoint"],
                process_pid=fault["pid"],
            ),
        )
        recovery = launch(output / (mode + "-recovery"), receipt, "recovery")
        request = output / (mode + "-recovery") / "worker/request-0"
        summary = json.loads((request / "summary.json").read_text(encoding="utf-8"))
        native = summary["native"]
        retry = run.begin(mode)
        run.finish(
            mode,
            retry,
            native["text"],
            "PASS",
            dict(
                finish_reason=native["finish_reason"],
                output_tokens=len(native["token_ids"]),
                max_tokens=2048,
                memory=summary["memory_after"],
                source="ACTUAL_NEW_GPU_RECOVERY",
            ),
        )
        results[mode] = dict(fault=fault, recovery=recovery, attempts=run.attempts(mode))
    final_ledger = run.summary()
    accepted = ledger_pass(final_ledger)
    publish(output / "ledger-summary.json", final_ledger)
    publish(
        output / "result.json",
        dict(
            status="PASS" if accepted else "FAIL",
            source_commit=BASE,
            wheel_sha256=ORIGINAL,
            real_documents="BLOCKED_DATA",
            results=results,
        ),
    )
    if not accepted:
        raise RuntimeError("Actual committed ledger/history did not satisfy failure/retry contract")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--p0-receipt", type=Path, required=True)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--fault", action="store_true")
    parser.add_argument("--execute-gpu", action="store_true")
    args = parser.parse_args()
    if args.child:
        child(args.output.resolve(), args.fault, args.p0_receipt.resolve())
    elif args.execute_gpu:
        parent(args.output.resolve(), args.p0_receipt.resolve())
    else:
        parser.error("Explicit --execute-gpu required")
