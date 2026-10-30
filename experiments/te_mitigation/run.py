"""Fresh owned processes; schedules are frozen before model execution."""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/cold_start"))
from windows_job import Job  # noqa: E402
from worker import CANDIDATE, ORIGINAL, load, sha, write  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--options", default="default,te-off")
    p.add_argument("--suite", action="store_true")
    p.add_argument("--backend", action="store_true")
    p.add_argument("--tensors", action="store_true")
    p.add_argument("--execute-gpu", action="store_true", required=True)
    args = p.parse_args()
    admission = {}
    if args.backend:
        if args.options != "eager" or args.suite:
            raise ValueError("Backend scope requires reviewed eager two-page controls")
        for name in ("eager-p0", "suite-1"):
            path = ROOT / ".b/te-e2e" / name / "qualification.json"
            result = json.loads(path.read_text("utf-8"))
            if not (
                result["all_physical_and_functional_gates"]
                and result["option_exact_repeatability"]["eager"]
            ):
                raise ValueError("Backend admission failed")
            admission[path.relative_to(ROOT).as_posix()] = sha(path)
    if Path.cwd().resolve() != ROOT or not args.output.resolve().is_relative_to(ROOT / ".b/te-e2e"):
        raise ValueError("Root/local output required")
    args.output.mkdir(parents=True, exist_ok=False)
    schedule = [
        dict(option=option, repeat=r, modes=modes)
        for option in args.options.split(",")
        for modes in (
            ("MATH,MATH", "AUTO,AUTO", "MATH,AUTO", "AUTO,MATH") if args.backend else (None,)
        )
        for r in (0, 1)
    ]
    helper = load("te_declared_cases", ROOT / "experiments/native_regression/runner.py")
    cases = helper.normal_cases()
    if not args.suite:
        cases = [next(c for c in cases if c["name"] == "pages_2_run_1")] * 2
    truth = json.loads((ROOT / "tests/fixtures/ground_truth.json").read_text("utf-8"))["cases"]
    inputs = {"tests/fixtures/ground_truth.json"}
    for case in cases:
        inputs.update("tests/fixtures/" + truth[key]["image"] for key in case["truth_keys"])
    sources = [
        "experiments/te_mitigation/worker.py",
        "experiments/te_mitigation/run.py",
        "experiments/cold_start/worker.py",
        "experiments/cold_start/windows_job.py",
        "experiments/native_regression/runner.py",
        "experiments/te_mitigation/analyze.py",
        "src/unlimited_ocr_2_rocm/upstream-lock.json",
        "docs/superpowers/plans/2026-10-30-te-mitigation-e2e.md",
    ]
    if args.backend:
        sources.append("docs/rca/MATCHED_CONTEXT_BACKEND_PROTOCOL_20261030.md")
    lock = json.loads((ROOT / "src/unlimited_ocr_2_rocm/upstream-lock.json").read_text("utf-8"))
    write(
        args.output / "predeclared.json",
        dict(
            protocol_version=3,
            wheel_sha256=CANDIDATE if args.backend else ORIGINAL,
            model_revision=lock["model_revision"],
            model_manifest_sha256=sha(ROOT / "src/unlimited_ocr_2_rocm/upstream-lock.json"),
            admission_receipts=admission,
            cases=cases,
            source_hashes={p: sha(ROOT / p) for p in sources},
            input_hashes={p: sha(ROOT / p) for p in sorted(inputs)},
            schedule=schedule,
            suite=args.suite,
            backend=args.backend,
            tensors=args.tensors,
            worker_sha256=sha(ROOT / "experiments/te_mitigation/worker.py"),
            controller_sha256=sha(__file__),
            protocol_sha256=sha(ROOT / "docs/superpowers/plans/2026-10-30-te-mitigation-e2e.md"),
            frozen_helper_sha256=sha(ROOT / "experiments/cold_start/worker.py"),
        ),
    )
    for item_index, item in enumerate(schedule):
        label = f"{item['option']}-{item['modes'] or 'MATH'}-{item['repeat']}"
        folder = args.output / label
        folder.mkdir()
        tmp = folder / "tmp"
        tmp.mkdir()
        python = ROOT / (
            ".b/o/Scripts/python.exe"
            if args.backend
            else ".worktrees/phase1-windows-gfx1151/.venv-repro/Scripts/python.exe"
        )
        command = [
            str(python),
            "-u",
            str(ROOT / "experiments/te_mitigation/worker.py"),
            "--declaration",
            str((args.output / "predeclared.json").resolve()),
            "--item",
            str(item_index),
            "--output",
            str(folder.resolve() / "data"),
            "--option",
            item["option"],
            "--wheel",
            "candidate" if args.backend else "original",
            "--execute-gpu",
        ]
        if item["modes"]:
            command.extend(["--modes", item["modes"]])
        if args.suite:
            command.append("--suite")
        if args.tensors:
            command.append("--tensors")
        env = dict(
            os.environ,
            PYTHONUTF8="1",
            PYTHONIOENCODING="utf-8",
            PYTHONHASHSEED="0",
            TEMP=str(tmp.resolve()),
            TMP=str(tmp.resolve()),
            HF_HUB_OFFLINE="1",
            HF_DATASETS_OFFLINE="1",
            TRITON_CACHE_DIR=str(ROOT / ".b/g1/triton"),
            PYTORCH_KERNEL_CACHE_PATH=str(ROOT / ".b/te-e2e/kernels"),
            TORCH_EXTENSIONS_DIR=str(ROOT / ".b/te-e2e/extensions"),
        )
        job = Job()
        record = {}
        print("START", label, flush=True)
        try:
            with (
                (folder / "stdout.log").open("wb") as stdout,
                (folder / "stderr.log").open("wb") as stderr,
            ):
                process = job.spawn(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
                record = dict(pid=process.pid, exit_code=process.wait(timeout=1200))
        finally:
            job.terminate()
            record["accounting"] = job.accounting()
            job.close()
            write(folder / "finished.json", record)
        print("END", label, record.get("exit_code"), flush=True)
        if record.get("exit_code") != 0:
            raise RuntimeError("Rejected process; retain failure")


if __name__ == "__main__":
    main()
