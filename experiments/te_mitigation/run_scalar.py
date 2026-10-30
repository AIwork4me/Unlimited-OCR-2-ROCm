"""Two prospective correction prototypes in two isolated original-wheel processes."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/cold_start"))
from windows_job import Job  # noqa: E402
from worker import sha, write  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-gpu", action="store_true", required=True)
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT or not args.output.resolve().is_relative_to(ROOT / ".b/te-e2e"):
        raise ValueError("Primary root/local evidence required")
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [
        "experiments/te_mitigation/scalar.py",
        "experiments/te_mitigation/run_scalar.py",
        "experiments/clip_mlp/replay.py",
        "experiments/cold_start/windows_job.py",
        ".b/mlp/capture-1/captures.json",
        ".b/mlp/capture-1/call-01/activation_input.pt",
    ]
    declaration = args.output / "predeclared.json"
    write(
        declaration,
        dict(
            hashes={p: sha(ROOT / p) for p in paths},
            repeats=2,
            jit_logging="config-only PYTORCH_JIT_LOG_LEVEL=>>cuda_codegen",
            purpose="Separate scalar-only and explicit-boundary prototypes",
        ),
    )
    for repeat in range(2):
        folder = args.output / str(repeat)
        folder.mkdir()
        tmp = folder / "tmp"
        tmp.mkdir()
        command = [
            str(ROOT / ".worktrees/phase1-windows-gfx1151/.venv-repro/Scripts/python.exe"),
            "-u",
            str(ROOT / "experiments/te_mitigation/scalar.py"),
            "--declaration",
            str(declaration.resolve()),
            "--output",
            str((folder / "data").resolve()),
            "--execute-gpu",
        ]
        env = dict(
            os.environ,
            PYTHONUTF8="1",
            PYTHONHASHSEED="0",
            TEMP=str(tmp.resolve()),
            TMP=str(tmp.resolve()),
            PYTORCH_KERNEL_CACHE_PATH=str(ROOT / ".b/te-e2e/kernels"),
            PYTORCH_JIT_LOG_LEVEL=">>cuda_codegen",
        )
        job, record = Job(), {}
        print("START scalar", repeat, flush=True)
        try:
            with (
                (folder / "stdout.log").open("wb") as stdout,
                (folder / "stderr.log").open("wb") as stderr,
            ):
                process = job.spawn(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
                record = dict(pid=process.pid, exit_code=process.wait(timeout=300))
        finally:
            job.terminate()
            record["accounting"] = job.accounting()
            job.close()
            write(folder / "finished.json", record)
        print("END scalar", repeat, record.get("exit_code"), flush=True)
        if record.get("exit_code") != 0:
            raise RuntimeError("Retain failed scalar experiment")


if __name__ == "__main__":
    main()
