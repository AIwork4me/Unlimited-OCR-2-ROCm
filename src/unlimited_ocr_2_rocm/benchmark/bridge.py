"""Windows inference -> Linux official scorer bridge (Option B, Phase 4.5).

The official scorer stays unchanged on Linux. Only predictions — model
outputs, portable by nature — cross the boundary, wrapped in a hash-bound
manifest. Windows keeps inference evidence; Linux keeps scoring evidence; each
side re-derives identity from hashes rather than trusting the other.

Boundary rules implemented here (WINDOWS_LINUX_SCORER_BRIDGE_DESIGN.md):

- Ground truth never crosses in either direction; the Linux side verifies the
  prediction inventory against its *local* GT.
- Failed documents (TRUNCATED/TIMEOUT/OOM/INTERRUPTED/FAIL) are exported as
  explicit empty files under strict mode — they stay in the denominator and in
  the failure catalog; never silently excluded.
- The export is sanitized by the standard publication detector before it may
  leave the producing host; a violation refuses the whole export.
- A Linux scorer result is Linux scorer evidence. Reports carry explicit scope
  labels and never claim a native Windows scorer pass; Overall stays null
  while any of the eight official metrics is missing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .. import publication
from .ledger import json_bytes, write_bytes
from .manifest import file_hash
from .scorer import METRICS, aggregate, metric_values

MANIFEST_NAME = "predictions-manifest.json"
PREDICTIONS_DIRNAME = "predictions"
#: Ledger states that export as empty files under strict mode.
NON_PASS_STATES = {"TRUNCATED", "TIMEOUT", "OOM", "INTERRUPTED", "FAIL"}


def _sha256(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _doc_ids_from_ground_truth(ground_truth: Path) -> set[str]:
    rows = json.loads(ground_truth.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Ground truth must be a non-empty list of annotation rows")
    return {Path(str(row["page_info"]["image_path"])).stem for row in rows}


def export_predictions(
    documents: dict[str, dict[str, Any]],
    *,
    campaign_id: str,
    provenance: dict[str, str],
    destination: Path,
    strict: bool = True,
) -> dict[str, Any]:
    """Write the transfer artifact: flat prediction directory + manifest.

    ``documents`` maps doc id -> ``{"state": ledger state, "text": markdown}``.
    Under ``strict`` (frozen policy) any non-PASS document — or one without
    text — exports as an explicit empty file and is listed in the failure
    catalog. Every byte is sanitized by the publication detector before
    anything is written; a violation refuses the export with nothing on disk.
    """
    if destination.exists():
        raise FileExistsError(destination)
    if not campaign_id or not provenance:
        raise ValueError("campaign id and provenance digests are required")
    staged: list[tuple[str, bytes | None]] = []
    failure_catalog: dict[str, str] = {}
    files: dict[str, str] = {}
    for doc_id in sorted(documents):
        record = documents[doc_id]
        state = str(record.get("state", "FAIL"))
        text = record.get("text")
        if strict and (state != "PASS" or not isinstance(text, str)):
            if state == "PASS":
                state = "MISSING_OUTPUT"
            text = ""
            failure_catalog[doc_id] = state
        body = (text if isinstance(text, str) else "").encode("utf-8")
        name = f"{doc_id}.md"
        staged.append((f"{PREDICTIONS_DIRNAME}/{name}", body))
        files[name] = _sha256(body)
    manifest = {
        "schema": 1,
        "campaign_id": campaign_id,
        "strict_mode": strict,
        "document_count": len(documents),
        "files": files,
        "failure_catalog": failure_catalog,
        "provenance": dict(sorted(provenance.items())),
    }
    manifest_bytes = json_bytes(manifest)
    staged.append((MANIFEST_NAME, manifest_bytes))
    violations = publication.scan_staged_files(staged)
    if violations:
        raise ValueError(f"Publication scan refused the export: {violations[:4]}")
    predictions_dir = destination / PREDICTIONS_DIRNAME
    predictions_dir.mkdir(parents=True)
    for relative, staged_body in staged:
        assert staged_body is not None  # only this function appends, always with bytes
        write_bytes(destination / relative, staged_body)
    return {
        "destination": str(destination),
        "campaign_id": campaign_id,
        "document_count": len(documents),
        "empty_exports": sorted(failure_catalog),
        "manifest_sha256": _sha256(manifest_bytes),
    }


def verify_predictions(
    manifest_path: Path,
    predictions_dir: Path,
    ground_truth: Path,
) -> dict[str, Any]:
    """Linux-side pre-scoring gate: hashes, inventory, GT binding.

    Refuses (raises ValueError) on any manifest/file drift or on inventory
    inequality against the local ground truth. Ground truth bytes are never
    read across the boundary — only their document identity is compared.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != 1:
        raise ValueError("Unknown predictions manifest schema")
    expected = manifest.get("files")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Predictions manifest has no file inventory")
    present = {path.name: file_hash(path) for path in predictions_dir.glob("*.md")}
    if present != expected:
        drift = {
            "missing": sorted(set(expected) - set(present)),
            "unexpected": sorted(set(present) - set(expected)),
            "hash_mismatch": sorted(
                name for name, sha in present.items() if expected.get(name) not in (None, sha)
            ),
        }
        raise ValueError(f"Prediction inventory drift: {drift}")
    gt_ids = _doc_ids_from_ground_truth(ground_truth)
    manifest_ids = {name.removesuffix(".md") for name in expected}
    if manifest_ids != gt_ids:
        raise ValueError(
            "Prediction inventory must equal the local ground-truth inventory: "
            f"missing={sorted(gt_ids - manifest_ids)} "
            f"extra={sorted(manifest_ids - gt_ids)}"
        )
    return {
        "verified": True,
        "manifest_sha256": file_hash(manifest_path),
        "ground_truth_sha256": file_hash(ground_truth),
        "documents": len(expected),
        "campaign_id": manifest.get("campaign_id"),
        "provenance": manifest.get("provenance", {}),
    }


def comparison_report(
    metric_result: dict[str, Any],
    *,
    manifest_verification: dict[str, Any],
    failure_catalog: dict[str, str],
) -> dict[str, Any]:
    """Official eight-metric + Overall report from a Linux scorer result.

    ``metric_result`` is the original scorer's ``metric_result.json`` value,
    parsed verbatim — this function computes nothing the scorer did not. The
    report binds the scoring to the verified prediction manifest identity and
    keeps failed documents in the denominators.
    """
    if not manifest_verification.get("verified"):
        raise ValueError("Comparison report requires a verified predictions manifest")
    values = metric_values(metric_result)
    missing = sorted(name for name, value in values.items() if value is None)
    attempted = int(manifest_verification.get("documents", 0))
    failed = len(failure_catalog)
    return {
        "schema": 1,
        "metrics": values,
        "metric_count": len(METRICS),
        "missing_metrics": missing,
        "overall": aggregate(values),
        "documents": {
            "attempted": attempted,
            "exported_empty_strict": failed,
            "scored": attempted,
        },
        "failure_catalog": dict(sorted(failure_catalog.items())),
        "prediction_identity": {
            "manifest_sha256": manifest_verification.get("manifest_sha256"),
            "campaign_id": manifest_verification.get("campaign_id"),
            "ground_truth_sha256": manifest_verification.get("ground_truth_sha256"),
            "provenance": manifest_verification.get("provenance", {}),
        },
        "scope": {
            "inference_platform": "windows-radeon-native",
            "scorer_platform": "linux-cpu-official",
            "windows_native_scorer": "BLOCKED",
            "note": (
                "Linux scorer evidence bound to Windows inference evidence only "
                "through the shared predictions-manifest hash; this is never a "
                "Windows-native scorer pass."
            ),
        },
        "status": "COMPLETE" if not missing else "INCOMPLETE_METRICS_NULL_OVERALL",
    }
