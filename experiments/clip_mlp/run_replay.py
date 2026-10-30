"""Bounded sequential model-free GPU processes with predeclared conditions."""

from __future__ import annotations

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
    parser.add_argument("--controls", default="normal,no-opt")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--prefix", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument("--execute-gpu", action="store_true", required=True)
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT or not args.output.resolve().is_relative_to(ROOT / ".b/mlp"):
        raise ValueError("Primary root/local fresh output required")
    args.output.mkdir(parents=True, exist_ok=False)
    schedule = [
        (control, position, repeat)
        for control in args.controls.split(",")
        for position in ("cold", "warm")
        for repeat in range(args.repeats)
    ]
    write(
        args.output / "predeclared.json",
        dict(
            schedule=schedule,
            producer_sha256=sha(__file__),
            worker_sha256=sha(ROOT / "experiments/clip_mlp/replay.py"),
            capture=".b/mlp/capture-1",
            scope="Frozen BF16 activation; no weights; warm initialization call retained",
            original_operator_prefix=args.prefix,
        ),
    )
    for control, position, repeat in schedule:
        out = args.output / f"{control}-{position}-{repeat}"
        out.mkdir()
        tmp = out / "tmp"
        tmp.mkdir()
        env = dict(
            os.environ,
            PYTHONUTF8="1",
            PYTHONIOENCODING="utf-8",
            PYTHONHASHSEED="0",
            TEMP=str(tmp.resolve()),
            TMP=str(tmp.resolve()),
            PYTORCH_KERNEL_CACHE_PATH=str(ROOT / ".b/mlp/kernel-cache"),
            TORCH_EXTENSIONS_DIR=str(ROOT / ".b/mlp/extensions"),
        )
        command = [
            str(ROOT / ".worktrees/phase1-windows-gfx1151/.venv-repro/Scripts/python.exe"),
            "-u",
            str(ROOT / "experiments/clip_mlp/replay.py"),
            "--capture",
            str(ROOT / ".b/mlp/capture-1"),
            "--output",
            str(out.resolve() / "worker"),
            "--control",
            control,
            "--position",
            position,
            "--execute-gpu",
            "--prefix",
            str(args.prefix),
        ]
        print("START", out.name, flush=True)
        job = Job()
        record = {}
        try:
            with (
                (out / "stdout.log").open("wb") as stdout,
                (out / "stderr.log").open("wb") as stderr,
            ):
                process = job.spawn(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
                record = dict(pid=process.pid, exit_code=process.wait(timeout=300))
        finally:
            job.terminate()
            record["accounting"] = job.accounting()
            job.close()
            write(out / "finished.json", record)
        print("END", out.name, record["exit_code"], flush=True)
        if record["exit_code"]:
            raise RuntimeError("Replay failed; retain evidence")


if __name__ == "__main__":
    main()
