"""Sequential bounded Windows jobs. Explicit GPU invocation only."""

import argparse
import os
import time
from pathlib import Path

from worker import ROOT, load, sha, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["A", "trace", "deep", "jit", "B", "D"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-gpu", action="store_true", required=True)
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT or not args.output.resolve().is_relative_to(
        ROOT / ".b/cold-start-rca"
    ):
        parser.error("Primary-root cwd and repository-local experiment output required")
    args.output.mkdir(parents=True, exist_ok=False)
    jobs = load("cold_start_jobs", ROOT / "experiments/cold_start/windows_job.py")
    if args.phase == "A":
        schedule = [
            ("A1", "pages_2", "MATH,MATH", False),
            ("A2", "pages_2", "MATH,MATH", False),
            ("A3", "pages_2", "MATH", False),
            ("A4", "pages_2", "MATH", True),
        ]
    elif args.phase in ("trace", "deep", "jit"):
        schedule = [("T1", "pages_2", "MATH,MATH", False), ("T2", "pages_2", "MATH,MATH", False)]
    elif args.phase == "D":
        schedule = [
            ("no_kill", "pages_2", "MATH,MATH", False),
            ("fault", "pages_2", "MATH", False),
            ("recovery", "pages_2", "MATH,MATH", False),
        ]
    else:
        schedule = [
            (f"{case}-{pair}-{r}", case, modes, False)
            for case in ("chinese", "pages_2", "mixed")
            for pair, modes in (
                ("AA", "MATH,MATH"),
                ("BB", "AUTO,AUTO"),
                ("AB", "MATH,AUTO"),
                ("BA", "AUTO,MATH"),
            )
            for r in (1, 2)
        ]
    write(
        args.output / "predeclared.json",
        dict(
            schedule=schedule,
            worker_sha256=sha(ROOT / "experiments/cold_start/worker.py"),
            driver_sha256=sha(__file__),
            helper_hashes={
                str(p.relative_to(ROOT)): sha(p)
                for p in (
                    ROOT / "experiments/cold_start/windows_job.py",
                    ROOT / "experiments/cold_start/fault.py",
                    ROOT / "experiments/native_regression/runner.py",
                )
            },
            seed=0,
            policy="Exact parity only; no OCR bbox tolerance; no speed measurements",
        ),
    )
    for name, case, modes, init in schedule:
        output = args.output / name
        output.mkdir()
        tmp = output / "tmp"
        tmp.mkdir()
        python = ROOT / (
            ".b/o/Scripts/python.exe"
            if args.phase == "B"
            else ".worktrees/phase1-windows-gfx1151/.venv-repro/Scripts/python.exe"
        )
        command = [
            str(python),
            "-u",
            str(ROOT / "experiments/cold_start/worker.py"),
            "--output",
            str(output / "worker"),
            "--wheel",
            "candidate" if args.phase == "B" else "original",
            "--case",
            case,
            "--modes",
            modes,
            "--execute-gpu",
        ]
        if init:
            command.append("--init")
        if args.phase in ("trace", "deep", "jit"):
            command.append("--trace")
        if args.phase in ("deep", "jit"):
            command.append("--deep-trace")
        if args.phase == "jit":
            command.append("--jit-unoptimized")
        if name == "fault":
            command[2] = str(ROOT / "experiments/cold_start/fault.py")
        env = dict(
            os.environ,
            TEMP=str(tmp),
            TMP=str(tmp),
            PYTHONUTF8="1",
            PYTHONIOENCODING="utf-8",
            PYTHONHASHSEED="0",
            HF_HUB_OFFLINE="1",
            HF_DATASETS_OFFLINE="1",
            TRITON_CACHE_DIR=str(ROOT / ".b/g1/triton"),
        )
        job = jobs.Job()
        start = time.monotonic()
        record = dict(command=command)
        print("START", name, flush=True)
        try:
            with (
                (output / "stdout.log").open("wb") as stdout,
                (output / "stderr.log").open("wb") as stderr,
            ):
                process = job.spawn(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
                record["pid"] = process.pid
                if name == "fault":
                    marker = output / "worker/kill-marker.json"
                    while process.poll() is None and not marker.exists():
                        if time.monotonic() - start > 900:
                            raise TimeoutError("No synchronized fault checkpoint")
                        time.sleep(0.02)
                    if not marker.exists():
                        raise RuntimeError("Fault child exited before checkpoint")
                    record["fault_checkpoint"] = __import__("json").loads(marker.read_text("utf-8"))
                    if record["fault_checkpoint"].get("hip_zero") is not True:
                        raise RuntimeError("Fault checkpoint HIP state is not clean")
                    job.terminate()
                    record["native_return_before_kill"] = any(
                        (output / "worker").glob("request-*/diagnostics/native-*.json")
                    )
                record["exit_code"] = process.wait(timeout=900)
        finally:
            record["accounting_before_cleanup"] = job.accounting()
            job.terminate()
            record["accounting_after_cleanup"] = job.accounting()
            job.close()
            record["seconds"] = time.monotonic() - start
            write(output / "finished.json", record)
        print("END", name, record["exit_code"], flush=True)
        if record["exit_code"] and name != "fault":
            raise RuntimeError("Failed worker; stop campaign and retain failure")
        if name == "fault" and (record["exit_code"] == 0 or record["native_return_before_kill"]):
            raise RuntimeError("External termination contract failed")


if __name__ == "__main__":
    main()
