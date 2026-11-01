"""Option B scorer bridge: export / verify / report with synthetic fixtures."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from unlimited_ocr_2_rocm.benchmark import bridge

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "scorer_selftest_metric_result.json"
PROVENANCE = {
    "source_commit": "0" * 40,
    "torch_wheel_sha256": "415d2dbd4b6ab002cff1c54795fdd66322abd9a46bf982cb64eebb4493ac2e7c",
    "model_revision": "9cf077c4b206e0e2a31879df937b0b0bcba3661c",
    "policy_digest": "a" * 64,
    "ledger_seal": "b" * 64,
}
DOCUMENTS = {
    "doc_pass": {"state": "PASS", "text": "# Heading\n\nparagraph\n"},
    "doc_truncated": {"state": "TRUNCATED", "text": "partial output"},
    "doc_oom": {"state": "OOM", "text": None},
}


def ground_truth(path: Path, doc_ids: list[str]) -> Path:
    rows = [
        {"page_info": {"image_path": f"images/{doc_id}.pdf"}, "annotations": []}
        for doc_id in doc_ids
    ]
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def test_export_writes_one_file_per_document_with_strict_empty_failures(tmp_path):
    receipt = bridge.export_predictions(
        DOCUMENTS,
        campaign_id="synthetic-campaign-1",
        provenance=PROVENANCE,
        destination=tmp_path / "artifact",
    )
    predictions = tmp_path / "artifact" / "predictions"
    names = sorted(p.name for p in predictions.glob("*.md"))
    assert names == ["doc_oom.md", "doc_pass.md", "doc_truncated.md"]
    assert (predictions / "doc_pass.md").read_text(encoding="utf-8").startswith("# Heading")
    assert (predictions / "doc_truncated.md").read_text(encoding="utf-8") == ""
    assert (predictions / "doc_oom.md").read_text(encoding="utf-8") == ""
    manifest = json.loads(
        (tmp_path / "artifact" / bridge.MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert manifest["document_count"] == 3
    assert manifest["strict_mode"] is True
    assert manifest["failure_catalog"] == {
        "doc_truncated": "TRUNCATED",
        "doc_oom": "OOM",
    }
    assert manifest["provenance"] == PROVENANCE
    assert receipt["empty_exports"] == ["doc_oom", "doc_truncated"]
    assert len(receipt["manifest_sha256"]) == 64


def test_export_refuses_private_paths_in_predictions(tmp_path):
    smuggled = {"doc_pass": {"state": "PASS", "text": "see C:/Users/person/private/notes.md"}}
    with pytest.raises(ValueError, match="Publication scan refused the export"):
        bridge.export_predictions(
            smuggled,
            campaign_id="synthetic-campaign-2",
            provenance=PROVENANCE,
            destination=tmp_path / "artifact",
        )
    assert not (tmp_path / "artifact").exists()  # fail-closed: nothing written


def test_export_refuses_existing_destination_and_missing_provenance(tmp_path):
    (tmp_path / "artifact").mkdir()
    with pytest.raises(FileExistsError):
        bridge.export_predictions(
            DOCUMENTS,
            campaign_id="c",
            provenance=PROVENANCE,
            destination=tmp_path / "artifact",
        )
    with pytest.raises(ValueError, match="provenance"):
        bridge.export_predictions(
            DOCUMENTS,
            campaign_id="c",
            provenance={},
            destination=tmp_path / "artifact2",
        )


def test_verify_passes_and_binds_hashes(tmp_path):
    receipt = bridge.export_predictions(
        DOCUMENTS,
        campaign_id="c3",
        provenance=PROVENANCE,
        destination=tmp_path / "artifact",
    )
    verification = bridge.verify_predictions(
        tmp_path / "artifact" / bridge.MANIFEST_NAME,
        tmp_path / "artifact" / "predictions",
        ground_truth(tmp_path / "gt.json", ["doc_pass", "doc_truncated", "doc_oom"]),
    )
    assert verification["verified"] is True
    assert verification["documents"] == 3
    assert verification["campaign_id"] == "c3"
    assert verification["provenance"] == PROVENANCE
    assert verification["manifest_sha256"] == receipt["manifest_sha256"]
    assert len(verification["ground_truth_sha256"]) == 64


def test_verify_refuses_tampered_prediction(tmp_path):
    bridge.export_predictions(
        DOCUMENTS,
        campaign_id="c4",
        provenance=PROVENANCE,
        destination=tmp_path / "artifact",
    )
    target = tmp_path / "artifact" / "predictions" / "doc_pass.md"
    target.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory drift"):
        bridge.verify_predictions(
            tmp_path / "artifact" / bridge.MANIFEST_NAME,
            tmp_path / "artifact" / "predictions",
            ground_truth(tmp_path / "gt.json", list(DOCUMENTS)),
        )


def test_verify_refuses_inventory_inequality_vs_local_gt(tmp_path):
    bridge.export_predictions(
        DOCUMENTS,
        campaign_id="c5",
        provenance=PROVENANCE,
        destination=tmp_path / "artifact",
    )
    # Local GT has an extra document: predictions must not silently skip it.
    with pytest.raises(ValueError, match="local ground-truth inventory"):
        bridge.verify_predictions(
            tmp_path / "artifact" / bridge.MANIFEST_NAME,
            tmp_path / "artifact" / "predictions",
            ground_truth(
                tmp_path / "gt.json",
                ["doc_pass", "doc_truncated", "doc_oom", "doc_missing"],
            ),
        )


def test_comparison_report_real_scorer_values_keep_overall_null(tmp_path):
    """Real eight-metric control: the frozen Linux scorer selftest output.

    ``scorer_selftest_metric_result.json`` is the original scorer's verbatim
    ``metric_result.json`` from the CI-qualified synthetic selftest (seven real
    values; FormulaCDM null without the Linux CDM renderer). The bridge must
    pass the real values through unchanged and null Overall on the gap.
    """
    receipt = bridge.export_predictions(
        DOCUMENTS,
        campaign_id="c6",
        provenance=PROVENANCE,
        destination=tmp_path / "artifact",
    )
    verification = bridge.verify_predictions(
        tmp_path / "artifact" / bridge.MANIFEST_NAME,
        tmp_path / "artifact" / "predictions",
        ground_truth(tmp_path / "gt.json", list(DOCUMENTS)),
    )
    metric_result = json.loads(FIXTURE.read_text(encoding="utf-8"))
    report = bridge.comparison_report(
        metric_result,
        manifest_verification=verification,
        failure_catalog={"doc_truncated": "TRUNCATED", "doc_oom": "OOM"},
    )
    assert report["metrics"]["TextEdit"] == 100.0
    assert report["metrics"]["TableTEDS"] == 100.0
    assert report["metrics"]["FigureF1"] == 100.0
    assert report["metrics"]["FormulaCDM"] is None
    assert report["missing_metrics"] == ["FormulaCDM"]
    assert report["overall"] is None
    assert report["status"] == "INCOMPLETE_METRICS_NULL_OVERALL"
    assert report["documents"] == {"attempted": 3, "exported_empty_strict": 2, "scored": 3}
    assert report["prediction_identity"]["manifest_sha256"] == receipt["manifest_sha256"]
    assert report["scope"]["windows_native_scorer"] == "BLOCKED"
    assert report["scope"]["scorer_platform"] == "linux-cpu-official"


def test_comparison_report_requires_verified_manifest(tmp_path):
    with pytest.raises(ValueError, match="verified"):
        bridge.comparison_report({}, manifest_verification={"verified": False}, failure_catalog={})


def test_cli_round_trip(tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps(DOCUMENTS), encoding="utf-8")
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps(PROVENANCE), encoding="utf-8")
    cli = [
        sys.executable,
        str(Path(__file__).resolve().parents[2] / "scripts" / "scorer_bridge.py"),
    ]
    export = subprocess.run(
        [
            *cli,
            "export",
            "--ledger-summary",
            str(summary),
            "--campaign-id",
            "cli-1",
            "--provenance",
            str(provenance),
            "--destination",
            str(tmp_path / "artifact"),
        ],
        capture_output=True,
        timeout=60,
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    assert export.returncode == 0, export.stderr
    gt = ground_truth(tmp_path / "gt.json", list(DOCUMENTS))
    verify = subprocess.run(
        [
            *cli,
            "verify",
            "--manifest",
            str(tmp_path / "artifact" / bridge.MANIFEST_NAME),
            "--predictions",
            str(tmp_path / "artifact" / "predictions"),
            "--ground-truth",
            str(gt),
        ],
        capture_output=True,
        timeout=60,
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    assert verify.returncode == 0, verify.stderr
    (tmp_path / "verification.json").write_text(verify.stdout.decode("utf-8"), encoding="utf-8")
    manifest = json.loads(
        (tmp_path / "artifact" / bridge.MANIFEST_NAME).read_text(encoding="utf-8")
    )
    (tmp_path / "catalog.json").write_text(
        json.dumps(manifest["failure_catalog"]), encoding="utf-8"
    )
    report = subprocess.run(
        [
            *cli,
            "report",
            "--metric-result",
            str(FIXTURE),
            "--verification",
            str(tmp_path / "verification.json"),
            "--failure-catalog",
            str(tmp_path / "catalog.json"),
            "--output",
            str(tmp_path / "report.json"),
        ],
        capture_output=True,
        timeout=60,
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    assert report.returncode == 0, report.stderr
    result = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert result["overall"] is None
    assert result["metrics"]["ReadOrderEdit"] == 100.0
