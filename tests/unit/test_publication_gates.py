"""Publication gates: dataset annotation content must never reach the repo tree."""

from __future__ import annotations

import hashlib
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


# ---------------------------------------------------------------------------
# Security hardening (Task A): staged whitelist hashes, nested/JSONL rows,
# Windows private paths, coverage, reintroduction, fail-closed evidence.
# ---------------------------------------------------------------------------


def test_nested_annotation_rows_flagged(tmp_path: Path) -> None:
    wrapped = {"meta": {"experiment": "x", "rows": [_annotation_row()]}}
    path = tmp_path / "nested-export.json"
    path.write_text(json.dumps(wrapped, ensure_ascii=False), encoding="utf-8")
    assert any("dataset annotation" in f for f in publication_findings(path))


def test_deeply_nested_annotation_rows_flagged(tmp_path: Path) -> None:
    wrapped: dict = {"leaf": _annotation_row()}
    for _ in range(6):
        wrapped = {"level": [wrapped]}
    path = tmp_path / "deep.json"
    path.write_text(json.dumps(wrapped, ensure_ascii=False), encoding="utf-8")
    assert any("dataset annotation" in f for f in publication_findings(path))


def test_jsonl_annotation_rows_flagged(tmp_path: Path) -> None:
    path = tmp_path / "slice.jsonl"
    rows = [_annotation_row(), {"ok": 1}]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )
    assert any("dataset annotation" in f for f in publication_findings(path))


def test_ndjson_annotation_rows_flagged(tmp_path: Path) -> None:
    path = tmp_path / "slice.ndjson"
    path.write_text(json.dumps(_annotation_row(), ensure_ascii=False) + "\n", encoding="utf-8")
    assert any("dataset annotation" in f for f in publication_findings(path))


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


def test_windows_private_path_flagged(tmp_path: Path) -> None:
    leaked = "C:" + chr(92) + "Users" + chr(92) + "dev" + chr(92) + "worker"
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps({"note": f"ran under {leaked}"}), encoding="utf-8")
    assert any("private host path" in f for f in publication_findings(path))
    text_path = tmp_path / "log.txt"
    text_path.write_text(f"worker dir {leaked}\n", encoding="utf-8")
    assert any("private host path" in f for f in publication_findings(text_path))


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


def test_staged_index_blob_toctou_closed() -> None:
    """F3: verification hashes the INDEX BLOB; a restored worktree cannot help."""
    from unlimited_ocr_2_rocm import publication

    rel = "eval/benchmarks/g05/scorer/final/pipelines/marathon-pipeline/ground-truth.json"
    index_blob = json.dumps([_annotation_row(text="REINTRODUCED REAL GT")]).encode("utf-8")
    # The caller passes what the INDEX holds; whatever the worktree contains
    # is irrelevant to this verdict.
    violations = publication.scan_staged_files([(rel, index_blob)])
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


def test_windows_lowercase_and_mixed_separator_paths_flagged(tmp_path: Path) -> None:
    lowered = "c:" + chr(92) + "users" + chr(92) + "dev"
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps({"note": f"ran under {lowered}"}), encoding="utf-8")
    assert any("private host path" in f for f in publication_findings(path))
    mixed = "C:" + "/Users" + "/dev/worker"  # assembled: keep the literal out of this file
    text_path = tmp_path / "log.txt"
    text_path.write_text(f"dir {mixed}\n", encoding="utf-8")
    assert any("private host path" in f for f in publication_findings(text_path))


def test_binary_like_non_json_evidence_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "payload.dat"
    path.write_bytes(b"\x00some packed annotation bytes\x00")
    findings = publication_findings(path)
    assert findings  # binary-like evidence is no longer silently skipped


def test_gate_cli_scan_reports_nested_violation(tmp_path: Path) -> None:
    deep = tmp_path / "eval" / "sub"
    deep.mkdir(parents=True)
    (deep / "p2b-gt-slice.json").write_text(
        json.dumps([_annotation_row()], ensure_ascii=False), encoding="utf-8"
    )
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/publication_gate.py"), "--scan", str(tmp_path)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 1
    assert "dataset annotation" in result.stdout
