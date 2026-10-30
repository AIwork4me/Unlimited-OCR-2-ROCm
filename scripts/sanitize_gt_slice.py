"""Derive the publishable identity receipt for a scorer GT slice.

Reads the real annotation slice (kept under ``.b/`` or the external evidence
archive — never committed) and emits a sanitized receipt with hashes, counts
and approved metadata only. Exit code is non-zero unless the receipt passes
unlimited_ocr_2_rocm.publication.validate_receipt, so the output can be
committed directly as replacement evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from unlimited_ocr_2_rocm.publication import build_gt_slice_receipt  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--slice", required=True, type=Path, help="real GT slice (annotation bytes)"
    )
    parser.add_argument("--receipt", required=True, type=Path, help="sanitized receipt output")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--run-id", required=True, help="ledger run id (64-hex)")
    parser.add_argument("--policy-sha256", required=True, help="frozen policy digest (64-hex)")
    parser.add_argument(
        "--ground-truth-sha256", required=True, help="full licensed GT digest (64-hex)"
    )
    parser.add_argument(
        "--claim-limit",
        required=True,
        help="approved claim-limit prose (ASCII) carried on the receipt",
    )
    args = parser.parse_args(argv)

    receipt = build_gt_slice_receipt(
        args.slice,
        dataset=args.dataset,
        experiment=args.experiment,
        run_id=args.run_id,
        policy_sha256=args.policy_sha256,
        ground_truth_sha256=args.ground_truth_sha256,
        claim_limit=args.claim_limit,
    )
    if args.receipt.exists():
        raise SystemExit(f"refusing to overwrite existing receipt: {args.receipt}")
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"receipt": str(args.receipt), "slice_sha256": receipt["slice_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
