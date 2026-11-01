"""Publication gates: dataset annotation content must never reach the repo tree."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from unlimited_ocr_2_rocm.config import file_sha256
from unlimited_ocr_2_rocm.publication import (
    build_gt_slice_receipt,
    looks_like_dataset_annotations,
    publication_findings,
    scan_staged,
    scan_tracked,
    scan_tree,
    validate_receipt,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "publication_gate.py"


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


def _write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def test_annotation_shape_detected() -> None:
    assert looks_like_dataset_annotations([_annotation_row()])
    assert not looks_like_dataset_annotations([])
    assert not looks_like_dataset_annotations({"page_info": {}, "layout_dets": []})
    assert not looks_like_dataset_annotations([{"unrelated": True}])
    # partial rows (only one of the two keys) are not the signature
    partial = [_annotation_row()]
    del partial[0]["layout_dets"]
    assert not looks_like_dataset_annotations(partial)


def test_annotation_rows_nested_inside_a_dict_are_detected() -> None:
    wrapped = {"runs": [{"meta": 1}], "payload": [_annotation_row()]}
    path = _write(Path(tempfile.gettempdir()) / "pubgate_nested.json", wrapped)
    try:
        assert any("dataset annotation" in f for f in publication_findings(path))
    finally:
        path.unlink()


def test_gt_slice_file_flagged(tmp_path: Path) -> None:
    path = _write(tmp_path / "p2b-gt-slice.json", [_annotation_row()])
    assert any("dataset annotation" in f for f in publication_findings(path))


def test_gt_slice_jsonl_flagged(tmp_path: Path) -> None:
    path = tmp_path / "p2b-gt-slice.jsonl"
    path.write_text(
        json.dumps([_annotation_row()], ensure_ascii=False) + "\n" + json.dumps({"flat": 1}) + "\n",
        encoding="utf-8",
    )
    assert any("dataset annotation" in f for f in publication_findings(path))


def test_private_host_path_flagged(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    leaked = "/home" + "/amd/worker"  # assembled: keep the literal out of this file
    path.write_text(json.dumps({"note": f"ran under {leaked}"}), encoding="utf-8")
    assert any("private host path" in f for f in publication_findings(path))


def test_windows_host_path_flagged(tmp_path: Path) -> None:
    back = chr(92)
    for leaked in ("C:" + back + "Users" + back + "operator", "C:" + "/" + "Users" + "/operator"):
        path = tmp_path / "receipt.txt"
        path.write_text(f"ran under {leaked}", encoding="utf-8")
        assert any("private host path" in f for f in publication_findings(path))


def test_host_identity_field_flagged(tmp_path: Path) -> None:
    # Key is assembled so this source file never contains the literal pattern.
    key = "host" + "name"
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps({key: "validation-host-01"}), encoding="utf-8")
    assert any("host identity" in f for f in publication_findings(path))


def test_credential_material_flagged(tmp_path: Path) -> None:
    path = tmp_path / "env.txt"
    path.write_text("GITHUB_TOKEN=ghp_" + "a" * 30, encoding="utf-8")
    assert any("credential" in f for f in publication_findings(path))


def test_image_file_flagged(tmp_path: Path) -> None:
    assert publication_findings(tmp_path / "page_1.png") != []


def test_opaque_binary_fail_closed(tmp_path: Path) -> None:
    blob = tmp_path / "payload.bin"
    blob.write_bytes(b"\x00\x01\x02rest")
    assert any("binary content" in f for f in publication_findings(blob))


def test_published_eval_tree_is_clean() -> None:
    assert scan_tree(REPO_ROOT / "eval", prefix="eval") == []


def test_scan_repo_root_anchors_whitelist() -> None:
    # Regression for the PR #24 defect: scanning "." must anchor whitelist
    # lookups at the repo root, not under a "./" prefix.
    violations = scan_tree(REPO_ROOT)
    assert violations == []


def test_tracked_publication_tree_is_clean() -> None:
    # The authoritative full-tree gate: exactly the committed content.
    assert scan_tracked(REPO_ROOT) == []


def test_whitelisted_fixture_hash_enforced(tmp_path: Path) -> None:
    fixture = _write(tmp_path / "fixture.json", [_annotation_row()])
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
        _write(fixture, [_annotation_row(text="edited")])
        violations = scan_tree(tmp_path)
        assert violations and "hash changed" in violations[0][1][0]
    finally:
        publication.WHITELIST.clear()
        publication.WHITELIST.update(original)


def test_receipt_builder_and_validator(tmp_path: Path) -> None:
    slice_path = _write(tmp_path / "slice.json", [_annotation_row()])
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
        receipt["note"] = "GT \u539f\u6587 must not pass"
        validate_receipt(receipt)


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
    path = _write(tmp_path / "p2b-gt-slice.json", [_annotation_row()])
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--paths", str(path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "dataset annotation" in result.stdout


def test_gate_cli_scan_dot_anchors_whitelist() -> None:
    # The PR #24 CLI quirk: `--scan .` flagged whitelisted fixtures because
    # the derived prefix was ".". It must PASS on this repository.
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--scan", "."],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stdout
    assert json.loads(result.stdout)["publication_gate"] == "PASS"


def _make_temp_clone() -> Path:
    clone = Path(tempfile.mkdtemp(prefix="pubgate_e2e_")) / "clone"
    subprocess.run(
        ["git", "clone", "--quiet", "--no-hardlinks", str(REPO_ROOT), str(clone)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(clone), "config", "user.email", "pubgate@example.invalid"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(clone), "config", "user.name", "Publication Gate Test"],
        check=True,
        capture_output=True,
    )
    return clone


def test_staged_pollution_injection_blocked() -> None:
    """Real contamination injection: stage a GT-shaped slice in a throwaway
    clone and require both the module API and the CLI hook entrypoint to
    refuse it."""
    clone = _make_temp_clone()
    try:
        slice_dir = clone / "eval" / "results" / "pollution-probe"
        slice_dir.mkdir(parents=True)
        _write(slice_dir / "injected-payload.json", [_annotation_row(doc_id="injected_doc")])
        subprocess.run(["git", "-C", str(clone), "add", "-A"], check=True, capture_output=True)

        violations = scan_staged(clone)
        assert violations and violations[0][0].endswith("injected-payload.json")

        result = subprocess.run(
            [sys.executable, str(clone / "scripts" / "publication_gate.py"), "--staged"],
            capture_output=True,
            text=True,
            cwd=clone,
        )
        assert result.returncode == 1
        assert "dataset annotation" in result.stdout
    finally:
        shutil.rmtree(clone.parent, ignore_errors=True)


def test_staged_whitelisted_tamper_blocked() -> None:
    """Editing a whitelisted file must fail the staged scan (PR #24 skipped
    whitelisted files at stage time entirely). Uses tests/fixtures/page_1.png,
    which exists on the baseline tree (the synthetic.json fixture arrives only
    with PR #7)."""
    clone = _make_temp_clone()
    try:
        target = REPO_ROOT / "tests" / "fixtures" / "page_1.png"
        staged = clone / "tests" / "fixtures" / "page_1.png"
        staged.write_bytes(target.read_bytes() + b"\n")
        subprocess.run(["git", "-C", str(clone), "add", "-A"], check=True, capture_output=True)
        violations = scan_staged(clone)
        assert violations and "hash changed" in violations[0][1][0]
    finally:
        shutil.rmtree(clone.parent, ignore_errors=True)


def test_clean_staged_scan_passes() -> None:
    assert scan_staged(REPO_ROOT) == []
