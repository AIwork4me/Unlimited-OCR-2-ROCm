"""Prepare local synthetic A/B inputs; GPU execution requires a matching P0 PASS."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from protocol import (
    BASE,
    CANDIDATE,
    REFERENCE,
    definitions,
    digest,
    file_hash,
    publish,
    read_plan,
    schedule,
    summarize,
    validate_timeout,
)
from worker import execution_gate

WORK = Path(__file__).resolve().parents[2]


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=root)


def prepare(profile_path: Path, output: Path, repeats: int, seed: int) -> None:
    root = Path.cwd().resolve()
    if output.exists():
        raise ValueError("Use a fresh output directory")
    if not WORK.is_relative_to(root) or not output.resolve().is_relative_to(root):
        raise ValueError("Run from the primary root; keep all outputs inside it")
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if profile.get("wheel_sha256") != CANDIDATE:
        raise ValueError("Both arms require the exact candidate artifact")
    if file_hash(Path(profile["wheel_path"])) != CANDIDATE:
        raise ValueError("Candidate wheel bytes do not match")
    output.mkdir(parents=True)
    refs = output / "references"
    refs.mkdir()
    for commit, source, name in (
        (REFERENCE, "experiments/native_ocr/validate.py", "validate.py"),
        (REFERENCE, "src/unlimited_ocr_2_rocm/compat.py", "compat.py"),
        (BASE, "experiments/native_regression/runner.py", "observer.py"),
    ):
        (refs / name).write_bytes(git(root, "show", commit + ":" + source))
    gt_path = WORK / "tests/fixtures/ground_truth.json"
    ground_truth = json.loads(gt_path.read_text(encoding="utf-8"))["cases"]
    cases = {}
    for name, keys in definitions().items():
        images = [gt_path.parent / ground_truth[key]["image"] for key in keys]
        hashes = [file_hash(p) for p in images]
        if hashes != [ground_truth[key]["sha256"] for key in keys]:
            raise ValueError("Original synthetic fixture hash mismatch")
        cases[name] = dict(
            images=list(map(str, images)),
            hashes=hashes,
            truth_keys=keys,
            max_new_tokens=1024 * len(keys),
        )
    sources = list((WORK / "src").rglob("*.py")) + list(Path(__file__).parent.glob("*.py"))
    sources += list(refs.glob("*.py")) + [profile_path.resolve(), gt_path]
    sources += [WORK / "src/unlimited_ocr_2_rocm/upstream-lock.json"]
    plan = dict(
        schema=1,
        workload="SYNTHETIC_ONLY",
        real_documents="BLOCKED_DATA",
        root=str(root),
        worktree=str(WORK),
        references=str(refs.resolve()),
        identity_profile=str(profile_path.resolve()),
        cache_dir=str(root / ".b/h"),
        ground_truth=str(gt_path),
        ground_truth_sha256=file_hash(gt_path),
        source_commit=git(root, "-C", str(WORK), "rev-parse", "HEAD").decode().strip(),
        source_dirty=bool(git(root, "-C", str(WORK), "status", "--porcelain")),
        base_source_commit=BASE,
        repeats=repeats,
        seed=seed,
        generation=dict(
            temperature=0,
            dtype="bfloat16",
            model_lock=file_hash(sources[-1]),
            exact_synthetic_raw_prompt_output_id_parity=True,
            bf16_atol=0.03,
            bf16_rtol=0.03,
        ),
        cases=cases,
        schedule=schedule(list(cases), repeats, seed),
        source_hashes={str(p.resolve()): file_hash(p) for p in sources},
    )
    publish(output / "plan.json", plan)
    publish(
        output / "public-preparation.json",
        dict(
            status="PREPARED_ONLY",
            synthetic_gpu="NOT_RUN",
            real_documents="BLOCKED_DATA",
            wheel_sha256=CANDIDATE,
            model_revision="9cf077c4b206e0e2a31879df937b0b0bcba3661c",
            source_commit=plan["source_commit"],
            source_dirty=plan["source_dirty"],
            protocol_sha256=digest(plan),
            cases=len(cases),
            trials=len(plan["schedule"]),
            measured_pairs_per_case=repeats,
            diagnostic_trials=2 * len(cases),
            warmup_trials=2 * len(cases),
            input_hashes={name: value["hashes"] for name, value in cases.items()},
        ),
    )
    print("PREPARED_ONLY: synthetic plan frozen; real documents BLOCKED_DATA")


def supervise(command: list[str], output: Path, deadline: float, env: dict[str, str]) -> None:
    validate_timeout(deadline)
    import psutil
    from windows_job import Job

    if psutil.virtual_memory().available < 32 * 1024**3:
        raise MemoryError("Require 32 GiB system available before model load")
    output.mkdir()
    started = time.monotonic()
    record: dict[str, Any] = dict(state="RUNNING", command=command, deadline_seconds=deadline)
    publish(output / "started.json", record)
    owned: dict[int, float] = {}
    child = None
    job = None
    try:
        job = Job()
        with (
            (output / "stdout.log").open("wb") as stdout,
            (output / "stderr.log").open("wb") as stderr,
        ):
            child = job.spawn(
                command,
                env=env,
                cwd=Path.cwd(),
                stdout=stdout,
                stderr=stderr,
            )
            process = psutil.Process(child.pid)
            owned[child.pid] = process.create_time()
            while child.poll() is None:
                try:
                    for descendant in process.children(recursive=True):
                        owned[descendant.pid] = descendant.create_time()
                except psutil.NoSuchProcess:
                    pass
                if time.monotonic() - started > deadline:
                    record["state"] = "TIMEOUT"
                    raise TimeoutError("Bounded synthetic worker timeout")
                if psutil.virtual_memory().available < 8 * 1024**3:
                    raise MemoryError("System available fell below the 8 GiB stop threshold")
                time.sleep(0.25)
            record.update(
                state="PASS" if child.returncode == 0 else "FAIL", exit_code=child.returncode
            )
    except BaseException as error:
        if record["state"] == "RUNNING":
            record["state"] = "FAIL"
        record["error"] = repr(error)
        raise
    finally:
        try:
            if job is not None:
                record["job_before_cleanup"] = job.accounting()
                if record["job_before_cleanup"]["active_processes"]:
                    if record["state"] == "PASS":
                        record.update(state="FAIL", error="Worker exited with live descendants")
                    job.terminate()
                record["job_after_cleanup"] = job.accounting()
            if child is not None:
                child.wait(timeout=30)
        except BaseException as error:
            record.update(state="FAIL", cleanup_error=repr(error))
            raise
        finally:
            if job is not None:
                job.close()
            record.update(observed_processes=owned, seconds=time.monotonic() - started)
            publish(output / "finished.json", record)
    if record["state"] != "PASS":
        raise RuntimeError("Synthetic worker failed; retain all attempts")


def execute(plan_path: Path, output: Path, receipt: Path, timeout: float) -> None:
    validate_timeout(timeout)
    plan = read_plan(plan_path, Path.cwd().resolve(), WORK)
    execution_gate(plan, receipt)
    if plan["source_dirty"] or git(Path.cwd(), "-C", str(WORK), "status", "--porcelain"):
        raise ValueError("Commit and freeze reviewed producers before physical execution")
    if output.exists() or not output.resolve().is_relative_to(Path(plan["root"])):
        raise ValueError("Use a fresh repository-local execution output")
    output.mkdir(parents=True)
    profile = json.loads(Path(plan["identity_profile"]).read_text(encoding="utf-8"))
    python = Path(profile["prefix"]) / "Scripts/python.exe"
    temporary = output / "tmp"
    temporary.mkdir()
    env = dict(
        os.environ,
        PYTHONUTF8="1",
        PYTHONIOENCODING="utf-8",
        TEMP=str(temporary.resolve()),
        TMP=str(temporary.resolve()),
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        TRITON_CACHE_DIR=str((output / "triton").resolve()),
    )
    trials = []
    for phase in ("diagnostic", "measurement"):
        worker_output = output / phase
        command = [
            str(python),
            "-u",
            str(Path(__file__).with_name("worker.py")),
            "--plan",
            str(plan_path.resolve()),
            "--output",
            str(worker_output.resolve()),
            "--phase",
            phase,
            "--p0-receipt",
            str(receipt.resolve()),
            "--execute-gpu",
        ]
        supervise(command, output / (phase + "-process"), timeout, env)
        for item in plan["schedule"]:
            if (item["phase"] == "diagnostic") == (phase == "diagnostic"):
                trial = json.loads((worker_output / item["id"] / "trial.json").read_text("utf-8"))
                if trial.get("input_sha256") != digest(plan["cases"][item["case"]]["hashes"]):
                    raise ValueError("Trial inputs do not match the frozen plan")
                trials.append(trial)
        if phase == "diagnostic":
            diagnostic_plan = [r for r in plan["schedule"] if r["phase"] == "diagnostic"]
            diagnostic_summary = summarize(diagnostic_plan, trials)
            # Diagnostics have no timed sample pairs; only validation is needed here.
            if diagnostic_summary["status"] != "PASS":
                publish(output / "public-summary.json", diagnostic_summary)
                raise ValueError("Diagnostic gate failed; no measurement child launched")
    summary = summarize(plan["schedule"], trials)
    publish(output / "public-summary.json", summary)
    if summary["status"] != "PASS":
        raise ValueError("A/B acceptance failed; no speedup claim")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--identity-profile", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--repeats", type=int, default=6)
    prep.add_argument("--seed", type=int, default=0)
    run = commands.add_parser("execute")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--p0-receipt", type=Path, required=True)
    run.add_argument("--timeout-seconds", type=float, default=1800)
    run.add_argument("--execute-gpu", action="store_true", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.identity_profile, args.output.resolve(), args.repeats, args.seed)
    else:
        execute(args.plan, args.output.resolve(), args.p0_receipt, args.timeout_seconds)


if __name__ == "__main__":
    main()
