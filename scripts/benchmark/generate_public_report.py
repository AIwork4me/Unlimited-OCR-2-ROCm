"""Generate the public benchmark report from frozen evidence files.

Pipeline (mission workflow E):

    ledger run receipt  ->  official scorer result  ->  eight metrics + Overall
      ->  comparability audit  ->  Radeon-minus-Baidu delta
      ->  public-results.json  ->  README scorecard rows

Every input is a frozen evidence file with a pinned sha256; no Radeon score is
ever typed by hand. The builder refuses denominators that do not match the
frozen dataset scope and refuses FULL labels that the accounting does not
support. Until a completed run exists, the honest output is result_status
NOT_RUN with null scores — never a placeholder that looks like a result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
try:
    from unlimited_ocr_2_rocm.benchmark.public_report import build_public_report
    from unlimited_ocr_2_rocm.config import file_sha256
except ImportError:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(_REPO_ROOT / "src"))
    from unlimited_ocr_2_rocm.benchmark.public_report import build_public_report
    from unlimited_ocr_2_rocm.config import file_sha256


def _load_frozen(path: Path | None) -> tuple[dict | None, str | None]:
    if path is None:
        return None, None
    return json.loads(path.read_text("utf-8")), file_sha256(path)


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        return "UNKNOWN"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", default="MPDocBench", choices=("MPDocBench", "MarathonDocBench")
    )
    parser.add_argument(
        "--run-receipt", type=Path, default=None, help="ledger summary receipt (sanitized)"
    )
    parser.add_argument(
        "--scorer-result", type=Path, default=None, help="official scorer metrics JSON"
    )
    parser.add_argument(
        "--official-reference", type=Path, required=True, help="locked official reference JSON"
    )
    parser.add_argument(
        "--comparability", type=Path, required=True, help="comparability audit JSON"
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args(argv)

    run_summary, run_receipt_sha = _load_frozen(args.run_receipt)
    scorer_result, scorer_sha = _load_frozen(args.scorer_result)
    reference, _ = _load_frozen(args.official_reference)
    comparability, _ = _load_frozen(args.comparability)

    report = build_public_report(
        dataset=args.dataset,
        run_summary=run_summary,
        scorer_result=scorer_result,
        official_reference=reference,
        comparability=comparability,
        run_id=(run_summary or {}).get("run_id"),
        git_sha=_git_sha(),
        evidence_sha256=hashlib.sha256(
            json.dumps(
                {
                    "run_receipt_sha256": run_receipt_sha,
                    "scorer_result_sha256": scorer_sha,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        if (run_receipt_sha or scorer_sha)
        else None,
    )
    report["evidence_inputs"] = {
        "run_receipt_sha256": run_receipt_sha,
        "scorer_result_sha256": scorer_sha,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=1) + "\n", "utf-8"
    )
    if args.output_md:
        lines = [
            f"# {args.dataset} — Radeon vs official reference",
            "",
            f"- Result status: **{report['result_status']}**",
            f"- Comparability: **{report['comparability_status']}**",
            f"- Radeon Overall: **{report['overall']}**",
            f"- Official Overall: {report['official_overall']}",
            f"- Delta: {report['delta']}",
            "",
            f"> {report['claim_limit']}",
            "",
        ]
        args.output_md.write_text("\n".join(lines), "utf-8")
    print(json.dumps({"status": report["result_status"], "output": str(args.output_json)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
