"""CI gate for original seven metrics and pipeline; retain explicit CDM readiness.

Exit zero means available integration checks passed. The retained report still
says BLOCKED unless all eight metrics, including real CDM, pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from unlimited_ocr_2_rocm.benchmark.cli import main as benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=["qwen", "marathon"], default="qwen")
    args = parser.parse_args()
    lock = "eval/benchmarks/benchmark-lock.json"
    common = ["--source", str(args.source), "--lock", lock, "--profile", args.profile]
    benchmark(["fetch-scorer", *common])
    code = benchmark(["score", *common, "--output", str(args.output)])
    report = json.loads((args.output / "report.json").read_text(encoding="utf-8"))
    fixture = report.get("selftest", {})
    checks = fixture.get("checks", {})
    if len(checks) != 8 or fixture.get("pipeline", {}).get("status") != "PASS":
        raise RuntimeError("Original scorer/pipeline did not complete; inspect retained report")
    for name, check in checks.items():
        if name != "FormulaCDM" and check["status"] != "PASS":
            raise RuntimeError(f"Known-answer check failed: {name}: {check}")
    cdm = checks["FormulaCDM"]
    if cdm["status"] == "PASS":
        return code
    if cdm.get("reason") != "CDM_RENDERER_UNAVAILABLE" or report["overall"] is not None:
        raise RuntimeError(f"Unexpected CDM failure: {cdm}")
    print("Seven original metrics and cross-page pipeline PASS; CDM/Overall BLOCKED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
