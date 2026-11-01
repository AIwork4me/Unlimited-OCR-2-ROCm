"""Publication gates: dataset annotation content must never reach the repo tree.

Fusion of the two audited suites: the PR #30 line (e2e staged-scan security
tests, `--scan .` anchoring regression, host-identity/credential detectors)
and the L1 migration line (fail-closed JSON/JSONL parsing, duplicate-key
rejection, nested-row node budget, index-side staged scanning, Windows path
coverage). Neither suite's assertions were weakened to make them fit.
"""

from __future__ import annotations

import hashlib
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


def test_windows_lowercase_and_mixed_separator_paths_flagged(tmp_path: Path) -> None:
    lowered = "c:" + chr(92) + "users" + chr(92) + "dev"
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps({"note": f"ran under {lowered}"}), encoding="utf-8")
    assert any("private host path" in f for f in publication_findings(path))
    mixed = "C:" + "/Users" + "/dev/worker"  # assembled: keep the literal out of this file
    text_path = tmp_path / "log.txt"
    text_path.write_text(f"dir {mixed}\n", encoding="utf-8")
    assert any("private host path" in f for f in publication_findings(text_path))


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


def test_binary_like_non_json_evidence_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "payload.dat"
    path.write_bytes(b"\x00some packed annotation bytes\x00")
    findings = publication_findings(path)
    assert findings  # binary-like evidence is not silently skipped


def test_malformed_json_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text('{"experiment": "unterminated', encoding="utf-8")
    findings = publication_findings(path)
    assert findings and "unreadable" in findings[0]


def test_malformed_jsonl_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text('{"a": 1}\n{"broken": \n', encoding="utf-8")
    findings = publication_findings(path)
    assert findings and "unreadable" in findings[0]


def test_duplicate_json_keys_fail_closed(tmp_path: Path) -> None:
    smuggle = (
        '{"rows": [{"page_info":{"image_path":"d.pdf","images_list":["p1.jpg"],'
        '"annotations_list":[{}]},"layout_dets":[{"label":"title","text":"GT"}]}],'
        ' "rows": {"clean": true}}'
    )
    path = tmp_path / "dup.json"
    path.write_text(smuggle, encoding="utf-8")
    findings = publication_findings(path)
    assert findings and "duplicate" in findings[0]
    jsonl = tmp_path / "dup.jsonl"
    jsonl.write_text('{"a": 1, "a": 2}\n', encoding="utf-8")
    assert any("duplicate" in f for f in publication_findings(jsonl))


def test_ndjson_annotation_rows_flagged(tmp_path: Path) -> None:
    path = tmp_path / "slice.ndjson"
    path.write_text(json.dumps(_annotation_row(), ensure_ascii=False) + "\n", encoding="utf-8")
    assert any("dataset annotation" in f for f in publication_findings(path))


def test_deeply_nested_annotation_rows_flagged(tmp_path: Path) -> None:
    wrapped: dict = {"leaf": _annotation_row()}
    for _ in range(6):
        wrapped = {"level": [wrapped]}
    path = tmp_path / "deep.json"
    path.write_text(json.dumps(wrapped, ensure_ascii=False), encoding="utf-8")
    assert any("dataset annotation" in f for f in publication_findings(path))


def test_type_mangled_rows_flagged_nested_and_top(tmp_path: Path) -> None:
    row = {
        "page_info": {"image_path": 12345, "images_list": ["p1.jpg"], "annotations_list": [{}]},
        "layout_dets": [{"label": "title", "text": "GT text", "bbox": [1, 2, 3, 4]}],
    }
    top = tmp_path / "mangled.json"
    top.write_text(json.dumps([row]), encoding="utf-8")
    assert any("dataset annotation" in f for f in publication_findings(top))
    wrapped = tmp_path / "mangled-wrapped.json"
    wrapped.write_text(json.dumps({"rows": [row]}), encoding="utf-8")
    assert any("dataset annotation" in f for f in publication_findings(wrapped))
    str_page_info = {
        "page_info": "doc.pdf",
        "layout_dets": [{"label": "title", "text": "GT text"}],
    }
    loose = tmp_path / "mangled-str.json"
    loose.write_text(json.dumps({"rows": [str_page_info]}), encoding="utf-8")
    assert any("dataset annotation" in f for f in publication_findings(loose))


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


def test_scan_tree_covers_deeply_nested_evidence_paths(tmp_path: Path) -> None:
    deep = tmp_path / "eval" / "results" / "exp" / "sub" / "deeper"
    deep.mkdir(parents=True)
    (deep / "p2b-gt-slice.json").write_text(
        json.dumps([_annotation_row()], ensure_ascii=False), encoding="utf-8"
    )
    violations = scan_tree(tmp_path / "eval", prefix="eval")
    assert len(violations) == 1
    rel, findings = violations[0]
    assert rel == "eval/results/exp/sub/deeper/p2b-gt-slice.json"
    assert any("dataset annotation" in f for f in findings)


def test_scan_tree_excludes_git_and_b_workspaces(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".b").mkdir()
    (tmp_path / "__pycache__").mkdir()
    for directory in (".git", ".b", "__pycache__"):
        (tmp_path / directory / "p2b-gt-slice.json").write_text(
            json.dumps([_annotation_row()]), encoding="utf-8"
        )
    assert scan_tree(tmp_path) == []


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


def test_gate_cli_scan_reports_nested_violation(tmp_path: Path) -> None:
    deep = tmp_path / "eval" / "sub"
    deep.mkdir(parents=True)
    (deep / "p2b-gt-slice.json").write_text(
        json.dumps([_annotation_row()], ensure_ascii=False), encoding="utf-8"
    )
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--scan", str(tmp_path)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 1
    assert "dataset annotation" in result.stdout


# ---------------------------------------------------------------------------
# Staged-scan security: index-side blobs, fail-closed gaps, rename/typechange.
# ---------------------------------------------------------------------------


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


def test_staged_rename_smuggle_blocked() -> None:
    """git mv of a GT-shaped file must still be scanned at its new path
    (PR #24 filtered rename entries out of the staged listing)."""
    clone = _make_temp_clone()
    try:
        slice_dir = clone / "eval" / "results" / "pollution-probe"
        slice_dir.mkdir(parents=True)
        _write(slice_dir / "benign-name.json", [_annotation_row(doc_id="rename_doc")])
        subprocess.run(["git", "-C", str(clone), "add", "-A"], check=True, capture_output=True)
        subprocess.run(
            ["git", "-C", str(clone), "commit", "-q", "-m", "setup"],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(clone), "mv", str(slice_dir / "benign-name.json"), "moved.json"],
            check=True,
            capture_output=True,
        )
        violations = scan_staged(clone)
        assert violations and violations[0][0] == "moved.json"
    finally:
        shutil.rmtree(clone.parent, ignore_errors=True)


def test_staged_scan_reads_index_not_worktree() -> None:
    """Stage a GT-shaped file, then scrub the worktree copy: the staged scan
    must still flag it (PR #24 hashed the mutable worktree file)."""
    clone = _make_temp_clone()
    try:
        slice_dir = clone / "eval" / "results" / "pollution-probe"
        slice_dir.mkdir(parents=True)
        target = slice_dir / "index-probe.json"
        _write(target, [_annotation_row(doc_id="index_doc")])
        subprocess.run(["git", "-C", str(clone), "add", "-A"], check=True, capture_output=True)
        target.unlink()
        violations = scan_staged(clone)
        assert violations and violations[0][0].endswith("index-probe.json")
    finally:
        shutil.rmtree(clone.parent, ignore_errors=True)


def test_clean_staged_scan_passes() -> None:
    assert scan_staged(REPO_ROOT) == []


def test_staged_whitelist_hash_mismatch_flagged() -> None:
    from unlimited_ocr_2_rocm import publication

    rel = "eval/benchmarks/g05/scorer/final/pipelines/marathon-pipeline/ground-truth.json"
    tampered = json.dumps([_annotation_row(text="tampered")]).encode("utf-8")
    original = dict(publication.WHITELIST)
    try:
        violations = publication.scan_staged_files([(rel, tampered)])
        assert violations and "hash changed" in violations[0][1][0]
        # Exact pinned bytes pass the staged hash verification.
        pinned = json.dumps([_annotation_row()]).encode("utf-8")
        publication.WHITELIST.clear()
        publication.WHITELIST["pinned.json"] = {
            "sha256": hashlib.sha256(pinned).hexdigest(),
            "reason": "test fixture",
        }
        assert publication.scan_staged_files([("pinned.json", pinned)]) == []
    finally:
        publication.WHITELIST.clear()
        publication.WHITELIST.update(original)


def test_staged_whitelist_hash_verified_by_default() -> None:
    """The real whitelist entry must verify staged BYTES (index blobs, not worktree)."""
    from unlimited_ocr_2_rocm import publication

    rel = "eval/benchmarks/g05/scorer/final/pipelines/marathon-pipeline/ground-truth.json"
    reintroduced = json.dumps([_annotation_row(text="reintroduced GT")]).encode("utf-8")
    violations = publication.scan_staged_files([(rel, reintroduced)])
    assert violations and "hash changed" in violations[0][1][0]


def test_staged_missing_blob_fail_closed() -> None:
    from unlimited_ocr_2_rocm import publication

    violations = publication.scan_staged_files([("eval/x/new.json", None)])
    assert violations and "fail-closed" in violations[0][1][0]


def test_staged_name_status_parsing_covers_renames_and_typechanges() -> None:
    from unlimited_ocr_2_rocm import publication

    raw = (
        b"A\0eval/new.json\0M\0eval/edited.json\0R100\0eval/old.json\0"
        b"eval/renamed.json\0T\0eval/link.json\0D\0eval/gone.json\0"
    )
    entries = publication.parse_staged_name_status(raw)
    # Deleted files never enter the repo; renames report BOTH endpoints —
    # the destination must exist in the index, the rename source may not.
    assert entries == [
        ("eval/new.json", True),
        ("eval/edited.json", True),
        ("eval/old.json", False),
        ("eval/renamed.json", True),
        ("eval/link.json", True),
    ]
