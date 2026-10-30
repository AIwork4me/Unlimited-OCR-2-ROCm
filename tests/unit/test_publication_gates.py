"""Publication gates: dataset annotation content must never reach the repo tree."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from unlimited_ocr_2_rocm.config import file_sha256
from unlimited_ocr_2_rocm.publication import (
    build_gt_slice_receipt,
    looks_like_dataset_annotations,
    publication_findings,
    scan_tree,
    validate_receipt,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _annotation_row(doc_id: str = "synthetic_doc_00000000", text: str = "synthetic title") -> dict:
    return {
        "page_info": {
            "image_path": f"{doc_id}.pdf",
            "images_list": ["p1.jpg"],
            "annotations_list": [{}],
        },
        "layout_dets": [{"label": "title", "text": text, "bbox": [1, 2, 3, 4]}],
        "extra": {"relation": [[1, 2]]},
    }


def test_annotation_shape_detected() -> None:
    assert looks_like_dataset_annotations([_annotation_row()])
    assert not looks_like_dataset_annotations([])
    assert not looks_like_dataset_annotations({"page_info": {}, "layout_dets": []})
    assert not looks_like_dataset_annotations([{"unrelated": True}])


def test_gt_slice_file_flagged(tmp_path: Path) -> None:
    path = tmp_path / "p2b-gt-slice.json"
    path.write_text(json.dumps([_annotation_row()], ensure_ascii=False), encoding="utf-8")
    assert any("dataset annotation" in f for f in publication_findings(path))


def test_private_host_path_flagged(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    leaked = "/home" + "/amd/worker"  # assembled: keep the literal out of this file
    path.write_text(json.dumps({"note": f"ran under {leaked}"}), encoding="utf-8")
    assert any("private host path" in f for f in publication_findings(path))


def test_image_file_flagged(tmp_path: Path) -> None:
    assert publication_findings(tmp_path / "page_1.png") != []


def test_published_eval_tree_is_clean() -> None:
    assert scan_tree(REPO_ROOT / "eval", prefix="eval") == []


def test_whitelisted_fixture_hash_enforced(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps([_annotation_row()]), encoding="utf-8")
    assert scan_tree(tmp_path) != []
    # A synthetic row at the pinned hash is admitted; a different hash is not.
    from unlimited_ocr_2_rocm import publication

    original = dict(publication.WHITELIST)
    try:
        publication.WHITELIST.clear()
        publication.WHITELIST["fixture.json"] = {
            "sha256": file_sha256(fixture),
            "reason": "test fixture",
        }
        assert scan_tree(tmp_path) == []
        fixture.write_text(json.dumps([_annotation_row(text="edited")]), encoding="utf-8")
        violations = scan_tree(tmp_path)
        assert violations and "hash changed" in violations[0][1][0]
    finally:
        publication.WHITELIST.clear()
        publication.WHITELIST.update(original)


def test_receipt_builder_and_validator(tmp_path: Path) -> None:
    slice_path = tmp_path / "slice.json"
    slice_path.write_text(json.dumps([_annotation_row()], ensure_ascii=False), encoding="utf-8")
    receipt = build_gt_slice_receipt(
        slice_path,
        dataset="MPDocBench",
        experiment="SYNTHETIC_TEST",
        run_id="a" * 64,
        policy_sha256="b" * 64,
        ground_truth_sha256="c" * 64,
        claim_limit="Synthetic test receipt; not benchmark evidence.",
    )
    assert receipt["documents"] == {
        "synthetic_doc_00000000": {"pages": 1, "layout_dets": 1, "relation_edges": 1}
    }
    assert receipt["slice_sha256"] == file_sha256(slice_path)
    assert "\u4e00" not in json.dumps(receipt)
    validate_receipt(receipt)
    with pytest.raises(ValueError):
        receipt["note"] = "汉字 GT 原文"
        validate_receipt(receipt)


def test_committed_receipt_is_sanitized() -> None:
    receipt_path = (
        REPO_ROOT / "eval/results/Linux gfx1151 verified/p2b-ngram/p2b-gt-slice-receipt.json"
    )
    validate_receipt(json.loads(receipt_path.read_text(encoding="utf-8")))


def test_gitignore_covers_gt_slices() -> None:
    # Probes live under a dedicated directory so they can never collide with
    # real evidence files; every probe is created and removed by this test.
    def ignored(path: str) -> bool:
        probe = REPO_ROOT / path
        if probe.exists():
            raise AssertionError(f"probe path already exists: {probe}")
        probe.parent.mkdir(parents=True, exist_ok=True)
        probe.write_text("{}", encoding="utf-8")
        try:
            # --no-index tests the pattern semantics only: an already-tracked
            # file would otherwise be skipped by the ignore machinery.
            result = subprocess.run(
                ["git", "-C", str(REPO_ROOT), "check-ignore", "--no-index", "-q", path],
                capture_output=True,
            )
        finally:
            probe.unlink()
        return result.returncode == 0

    assert ignored("eval/results/__pubgate_probe__/dir/p2b-gt-slice.json")
    assert ignored("eval/results/__pubgate_probe__/p2x-gt-slice-v2.json")
    assert not ignored("eval/results/__pubgate_probe__/p2b-gt-slice-receipt.json")


def test_gate_cli_blocks_annotation_file(tmp_path: Path) -> None:
    path = tmp_path / "p2b-gt-slice.json"
    path.write_text(json.dumps([_annotation_row()], ensure_ascii=False), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/publication_gate.py"), "--paths", str(path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "dataset annotation" in result.stdout
