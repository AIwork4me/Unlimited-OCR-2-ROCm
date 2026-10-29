import copy
import json
from pathlib import Path

import pytest

from unlimited_ocr_2_rocm.benchmark.manifest import inventory, validate_manifest


def dataset(tmp_path: Path):
    docs = []
    for name, count in (("z", 2), ("a", 10)):
        pages = []
        for i in range(1, count + 1):
            path = f"images/{name}/page_{i}.png"
            target = tmp_path / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(f"{name}:{i}".encode())
            pages.append(path)
        docs.append({"page_info": {"image_path": f"{name}.pdf", "images_list": pages}})
    gt = tmp_path / "gt.json"
    gt.write_text(json.dumps(docs), encoding="utf-8")
    return gt


def test_inventory_uses_numeric_page_order_and_sorted_document_ids(tmp_path):
    gt = dataset(tmp_path)
    manifest = inventory(gt, tmp_path, "synthetic", "a" * 40, 2, 12)
    assert [d["id"] for d in manifest["documents"]] == ["a", "z"]
    assert manifest["documents"][0]["pages"][9]["path"].endswith("page_10.png")
    assert validate_manifest(manifest, tmp_path)["status"] == "PASS"
    assert "layout_dets" not in json.dumps(manifest)


def test_missing_and_changed_bytes_never_validate(tmp_path):
    gt = dataset(tmp_path)
    manifest = inventory(gt, tmp_path, "synthetic", "a" * 40, 2, 12)
    (tmp_path / "images/a/page_1.png").unlink()
    (tmp_path / "images/z/page_1.png").write_bytes(b"changed")
    report = validate_manifest(manifest, tmp_path)
    assert report["status"] == "BLOCKED"
    assert len(report["errors"]) == 2
    assert report["expected_pages"] == 12


@pytest.mark.parametrize("mutation", ["duplicate", "case", "traversal", "order", "count"])
def test_manifest_rejects_ambiguous_or_unsafe_input(tmp_path, mutation):
    gt = dataset(tmp_path)
    manifest = inventory(gt, tmp_path, "synthetic", "a" * 40, 2, 12)
    bad = copy.deepcopy(manifest)
    if mutation == "duplicate":
        bad["documents"][1] = bad["documents"][0]
    elif mutation == "case":
        bad["documents"][1]["id"] = "A"
    elif mutation == "traversal":
        bad["documents"][0]["pages"][0]["path"] = "../outside.png"
    elif mutation == "order":
        bad["documents"][0]["pages"].reverse()
    else:
        bad["expected_pages"] = 11
    with pytest.raises(ValueError):
        validate_manifest(bad)


def test_official_denominators_cannot_be_overridden(tmp_path):
    gt = dataset(tmp_path)
    with pytest.raises(ValueError, match="denominator"):
        inventory(gt, tmp_path, "MPDocBench", "a" * 40, 2, 12)


def test_missing_images_create_blocked_inventory_without_fabricated_hashes(tmp_path):
    gt = dataset(tmp_path)
    (tmp_path / "images/a/page_1.png").unlink()
    manifest = inventory(gt, tmp_path, "synthetic", "a" * 40, 2, 12)
    assert manifest["documents"][0]["pages"][0]["sha256"] is None
    assert validate_manifest(manifest)["status"] == "BLOCKED"


def test_symlink_escape_is_rejected(tmp_path):
    gt = dataset(tmp_path)
    manifest = inventory(gt, tmp_path, "synthetic", "a" * 40, 2, 12)
    link = tmp_path / "images/a/page_1.png"
    link.unlink()
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.write_bytes(b"outside")
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("host cannot create symlinks")
    with pytest.raises(ValueError, match="escape"):
        validate_manifest(manifest, tmp_path)
