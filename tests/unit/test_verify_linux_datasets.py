"""CPU unit tests for the read-only Linux dataset verifier."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def verifier():
    path = Path(__file__).parents[2] / "scripts/verify_linux_datasets.py"
    assert path.exists(), "read-only Linux verifier has not been implemented"
    spec = importlib.util.spec_from_file_location("verify_linux_datasets", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def png_bytes(size: tuple[int, int] = (4, 4)) -> bytes:
    import io

    from PIL import Image

    image = Image.new("RGB", size, color=(128, 64, 32))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def canonical_digest(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


DATASET = "SyntheticDocBench"


def build_dataset(tmp_path: Path, page_count: int = 4) -> dict:
    """Create a minimal frozen-identity layout plus matching on-disk images.

    doc_01 holds pages 1-3 (multi-page document), doc_02 holds page 4.
    """
    root = tmp_path / "data" / DATASET
    layout = {"doc_01": page_count - 1, "doc_02": 1}
    pages = []
    rows = []
    counter = 0
    for doc_id, count in layout.items():
        paths = []
        for _ in range(count):
            counter += 1
            rel = f"images/{doc_id}/page_{counter}.png"
            payload = png_bytes()
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_bytes(payload)
            pages.append({"path": rel, "bytes": len(payload), "sha256": sha256_bytes(payload)})
            paths.append(rel)
        rows.append(
            {
                "id": doc_id,
                "pages": count,
                "ordered_paths_sha256": canonical_digest(paths),
                "annotation_sha256": "0" * 64,
            }
        )
    manifest = tmp_path / "manifests" / f"{DATASET}.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8"
    )
    lock = {
        "datasets": {
            DATASET: {
                "documents": len(rows),
                "pages": page_count,
                "scope_sha256": sha256_bytes(manifest.read_bytes()),
            }
        }
    }
    return {
        "data_root": tmp_path / "data",
        "pages": pages,
        "rows": rows,
        "manifest": manifest,
        "lock": lock,
    }


def run(fixed: dict, **overrides):
    module = verifier()
    pages = overrides.pop("pages", None) or fixed["pages"]
    rows = overrides.pop("rows", None) or fixed["rows"]
    lock = overrides.pop("lock", None) or fixed["lock"]
    return module.verify_dataset(
        data_root=fixed["data_root"],
        dataset=DATASET,
        pages=pages,
        rows=rows,
        lock_entry=lock["datasets"][DATASET],
        manifest_path=fixed["manifest"],
        **overrides,
    )


def test_valid_synthetic_inventory_passes(tmp_path):
    fixed = build_dataset(tmp_path)
    report = run(fixed, decode=True)
    assert report["status"] == "PASS"
    assert report["expected_pages"] == 4
    assert report["verified_pages"] == 4
    assert report["missing"] == []
    assert report["mismatched"] == []
    assert report["extra"] == []
    assert report["document_digests"]["mismatched"] == []


def test_missing_page_blocks(tmp_path):
    fixed = build_dataset(tmp_path)
    (fixed["data_root"] / DATASET / "images/doc_02/page_4.png").unlink()
    report = run(fixed)
    assert report["status"] == "BLOCKED"
    assert report["missing"] == ["images/doc_02/page_4.png"]
    assert report["verified_pages"] == 3


def test_corrupted_sha256_blocks(tmp_path):
    fixed = build_dataset(tmp_path)
    target = fixed["data_root"] / DATASET / "images/doc_02/page_4.png"
    target.write_bytes(png_bytes((8, 8)))
    report = run(fixed)
    assert report["status"] == "BLOCKED"
    assert report["mismatched"][0]["path"] == "images/doc_02/page_4.png"
    assert report["mismatched"][0]["expected_sha256"] != report["mismatched"][0]["actual_sha256"]


def test_size_mismatch_recorded_with_hash(tmp_path):
    fixed = build_dataset(tmp_path)
    target = fixed["data_root"] / DATASET / "images/doc_01/page_1.png"
    target.write_bytes(target.read_bytes()[:-4])
    report = run(fixed)
    assert report["status"] == "BLOCKED"
    entry = report["mismatched"][0]
    assert entry["path"] == "images/doc_01/page_1.png"
    assert entry["expected_bytes"] - entry["actual_bytes"] == 4


def test_truncated_image_fails_decode(tmp_path):
    fixed = build_dataset(tmp_path)
    target = fixed["data_root"] / DATASET / "images/doc_01/page_1.png"
    target.write_bytes(target.read_bytes()[:32])
    report = run(fixed, decode=True)
    assert report["status"] == "BLOCKED"
    assert report["mismatched"][0]["decode_error"]


def test_wrong_page_order_detected(tmp_path):
    fixed = build_dataset(tmp_path)
    pages = list(reversed(fixed["pages"]))
    report = run(fixed, pages=pages)
    assert report["status"] == "BLOCKED"
    assert report["document_digests"]["mismatched"]


def test_wrong_annotation_hash_flagged(tmp_path):
    fixed = build_dataset(tmp_path)
    rows = [dict(r) for r in fixed["rows"]]
    rows[0]["annotation_sha256"] = "1" * 64
    report = run(fixed, rows=rows)
    assert report["status"] == "BLOCKED"
    assert "annotation" in json.dumps(report).lower()


def test_duplicate_paths_in_index_rejected(tmp_path):
    fixed = build_dataset(tmp_path)
    pages = fixed["pages"] + [dict(fixed["pages"][0])]
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        run(fixed, pages=pages)


def test_case_collision_in_index_rejected(tmp_path):
    fixed = build_dataset(tmp_path)
    page = dict(fixed["pages"][0])
    page["path"] = page["path"].replace("page_1", "PAGE_1")
    with pytest.raises(ValueError, match="[Cc]ase"):
        run(fixed, pages=fixed["pages"] + [page])


def test_path_escape_in_index_rejected(tmp_path):
    fixed = build_dataset(tmp_path)
    pages = [dict(fixed["pages"][0])]
    pages[0]["path"] = "../outside.png"
    with pytest.raises(ValueError, match="escape|traversal|absolute"):
        run(fixed, pages=pages)


def test_symlinked_page_rejected(tmp_path):
    fixed = build_dataset(tmp_path)
    outside = tmp_path / "outside.png"
    outside.write_bytes(png_bytes())
    target = fixed["data_root"] / DATASET / "images/doc_01/page_1.png"
    target.unlink()
    target.symlink_to(outside)
    report = run(fixed)
    assert report["status"] == "BLOCKED"
    assert any("symlink" in json.dumps(report).lower() for _ in [0])


def test_unexpected_extra_files_flagged(tmp_path):
    fixed = build_dataset(tmp_path)
    extra = fixed["data_root"] / DATASET / "images/doc_01/unexpected.png"
    extra.write_bytes(png_bytes())
    report = run(fixed)
    assert report["status"] == "BLOCKED"
    assert report["extra"] == ["images/doc_01/unexpected.png"]


def test_scope_hash_mismatch_rejected(tmp_path):
    fixed = build_dataset(tmp_path)
    lock = json.loads(json.dumps(fixed["lock"]))
    lock["datasets"][DATASET]["scope_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="scope"):
        run(fixed, lock=lock)


def test_page_count_mismatch_rejected(tmp_path):
    fixed = build_dataset(tmp_path)
    lock = json.loads(json.dumps(fixed["lock"]))
    lock["datasets"][DATASET]["pages"] = 3
    with pytest.raises(ValueError, match="page"):
        run(fixed, lock=lock)


def test_interrupted_verification_writes_no_receipt(tmp_path, monkeypatch):
    fixed = build_dataset(tmp_path)
    module = verifier()
    output = tmp_path / "receipt.json"

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(module, "verify_single_page", interrupted)
    with pytest.raises(KeyboardInterrupt):
        module.main(
            [
                "--data-root",
                str(fixed["data_root"]),
                "--benchmark-lock",
                str(build_lock_file(tmp_path, fixed)),
                "--dataset",
                DATASET,
                "--output",
                str(output),
                "--page-source",
                str(build_index_file(tmp_path, fixed)),
                "--manifest",
                str(fixed["manifest"]),
            ]
        )
    assert not output.exists()


def test_main_writes_receipt_for_valid_dataset(tmp_path):
    fixed = build_dataset(tmp_path)
    module = verifier()
    output = tmp_path / "receipt.json"
    module.main(
        [
            "--data-root",
            str(fixed["data_root"]),
            "--benchmark-lock",
            str(build_lock_file(tmp_path, fixed)),
            "--dataset",
            DATASET,
            "--output",
            str(output),
            "--page-source",
            str(build_index_file(tmp_path, fixed)),
            "--manifest",
            str(fixed["manifest"]),
        ]
    )
    receipt = json.loads(output.read_text("utf-8"))
    assert receipt["status"] == "PASS"
    assert receipt["datasets"][0]["dataset"] == DATASET


def build_index_file(tmp_path: Path, fixed: dict) -> Path:
    path = tmp_path / "page-index.json"
    path.write_text(json.dumps({"pages": fixed["pages"]}), encoding="utf-8")
    return path


def build_lock_file(tmp_path: Path, fixed: dict) -> Path:
    path = tmp_path / "benchmark-lock.json"
    path.write_text(json.dumps(fixed["lock"]), encoding="utf-8")
    return path
