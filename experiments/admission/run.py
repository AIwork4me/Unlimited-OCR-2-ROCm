"""Prospective dependent experiment entry point; every launch goes through Gate."""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from unlimited_ocr_2_rocm.benchmark.admission import Gate  # noqa: E402
from unlimited_ocr_2_rocm.benchmark.windows_job import NativeJob  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--qualification", required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument("--execute-gpu", action="store_true", required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(Path.cwd().resolve())
    args.output.mkdir(exist_ok=False, parents=True)
    job = NativeJob()
    try:
        with (
            (args.output / "stdout.log").open("xb") as out,
            (args.output / "stderr.log").open("xb") as err,
        ):
            Gate(args.ledger).dispatch(
                args.stage,
                args.qualification,
                json.loads(args.identity.read_text(encoding="utf-8")),
                job,
                timeout=args.timeout,
                cwd=Path.cwd(),
                env=dict(os.environ, PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1"),
                stdout=out,
                stderr=err,
            )
    finally:
        job.close()


if __name__ == "__main__":
    main()
