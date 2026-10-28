"""Run from an installed editable package or the repository root."""

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unlimited_ocr_2_rocm.doctor import (  # noqa: E402
    bf16_probe,
    collect_environment,
    require_stack,
    write_json,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("eval/results/doctor.json"))
    args = parser.parse_args()
    report = {}
    try:
        report = collect_environment()
        require_stack(report)
        report["bf16"] = bf16_probe()
        report["status"] = "PASS"
    except Exception:
        report["status"] = "FAIL"
        report["traceback"] = traceback.format_exc()
    write_json(args.output, report)
    print(f"Gate A: {report['status']} — {args.output}")
    if report["status"] != "PASS":
        print(report["traceback"], file=sys.stderr)
        sys.exit(1)
