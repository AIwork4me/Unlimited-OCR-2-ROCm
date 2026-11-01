"""Scorer-bridge CLI: export (Windows) / verify + report (Linux).

Option B boundary: predictions + hash-bound manifest are the only artifact
that crosses hosts; ground truth never does. Run ``export`` on the inference
host with sanitized producer provenance; run ``verify`` and ``report`` on the
Linux scorer host against its local, hash-frozen ground truth.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unlimited_ocr_2_rocm.benchmark import bridge


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export", help="build the sanitized transfer artifact")
    export.add_argument(
        "--ledger-summary",
        type=Path,
        required=True,
        help="run summary JSON: doc_id -> {state, text}",
    )
    export.add_argument("--campaign-id", required=True)
    export.add_argument(
        "--provenance", type=Path, required=True, help="JSON object of producer digest fields"
    )
    export.add_argument("--destination", type=Path, required=True)
    export.add_argument(
        "--no-strict", action="store_true", help="deviation from frozen policy; not for campaigns"
    )

    verify = sub.add_parser("verify", help="Linux-side pre-scoring gate")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--predictions", type=Path, required=True)
    verify.add_argument("--ground-truth", type=Path, required=True)

    report = sub.add_parser("report", help="eight-metric + Overall comparison report")
    report.add_argument(
        "--metric-result", type=Path, required=True, help="the Linux scorer's metric_result.json"
    )
    report.add_argument(
        "--verification", type=Path, required=True, help="JSON receipt from the verify step"
    )
    report.add_argument(
        "--failure-catalog",
        type=Path,
        required=True,
        help="failure catalog from the export manifest",
    )
    report.add_argument("--output", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "export":
        documents = json.loads(args.ledger_summary.read_text(encoding="utf-8"))
        provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
        receipt = bridge.export_predictions(
            documents,
            campaign_id=args.campaign_id,
            provenance=provenance,
            destination=args.destination,
            strict=not args.no_strict,
        )
        print(json.dumps(receipt, indent=1, sort_keys=True))
        return 0
    if args.command == "verify":
        receipt = bridge.verify_predictions(args.manifest, args.predictions, args.ground_truth)
        print(json.dumps(receipt, indent=1, sort_keys=True))
        return 0
    verification = json.loads(args.verification.read_text(encoding="utf-8"))
    catalog = json.loads(args.failure_catalog.read_text(encoding="utf-8"))
    metric_result = json.loads(args.metric_result.read_text(encoding="utf-8"))
    result = bridge.comparison_report(
        metric_result, manifest_verification=verification, failure_catalog=catalog
    )
    args.output.write_text(json.dumps(result, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps({"overall": result["overall"], "status": result["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
