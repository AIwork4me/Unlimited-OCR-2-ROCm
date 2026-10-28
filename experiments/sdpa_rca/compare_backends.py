"""Launch one clean interpreter per matrix entry; preserve original child streams."""

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from reporting import sha256, write_json


def run_case(command, stem, cwd, timeout=300):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    stdout, stderr = stem.with_suffix(".stdout.log"), stem.with_suffix(".stderr.log")
    start = time.monotonic()
    utc = datetime.now(timezone.utc).isoformat()
    timed_out = False
    with stdout.open("wb") as out, stderr.open("wb") as err:
        try:
            process = subprocess.run(
                command,
                cwd=cwd,
                stdout=out,
                stderr=err,
                env=dict(os.environ, PYTHONIOENCODING="utf-8"),
                timeout=timeout,
            )
            code = process.returncode
        except subprocess.TimeoutExpired:
            code, timed_out = 124, True
    return {
        "command": command,
        "exit_code": code,
        "timed_out": timed_out,
        "process_run_id": str(uuid.uuid4()),
        "utc": utc,
        "wall_seconds": time.monotonic() - start,
        "stdout_sha256": sha256(stdout),
        "stderr_sha256": sha256(stderr),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=Path, help="JSON list of {name, args} conditions")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    cases = (
        json.loads(args.cases.read_text(encoding="utf-8-sig"))
        if args.cases
        else [
            {"name": backend.lower(), "args": ["--backend", backend]}
            for backend in (
                "DEFAULT",
                "MATH",
                "FLASH_ATTENTION",
                "EFFICIENT_ATTENTION",
                "CUDNN_ATTENTION",
            )
        ]
    )
    args.output.mkdir(parents=True, exist_ok=True)
    if len({case["name"] for case in cases}) != len(cases):
        parser.error("Duplicate case names would overwrite evidence")
    manifest = []
    for case in cases:
        stem = args.output / case["name"]
        if stem.with_suffix(".json").exists():
            raise FileExistsError(f"Refusing to overwrite producing evidence: {stem}")
        command = [
            args.python,
            str(Path(__file__).with_name("repro_sdpa.py")),
            "--output",
            str(stem.with_suffix(".json")),
            *case["args"],
        ]
        result = run_case(command, stem, root)
        result["case"] = case["name"]
        path = stem.with_suffix(".json")
        report = json.loads(path.read_text()) if path.exists() else {}
        result.update(
            {
                k: report.get(k)
                for k in ("run_id", "status", "backend_observed", "fixture_sha256", "script_sha256")
            }
        )
        manifest.append(result)
        write_json(args.output / "process-manifest.json", manifest)
        print(
            case["name"],
            result["exit_code"],
            result["status"],
            result["backend_observed"],
            flush=True,
        )
    # 0 means every child completed and was recorded; per-condition acceptance is in JSON.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
