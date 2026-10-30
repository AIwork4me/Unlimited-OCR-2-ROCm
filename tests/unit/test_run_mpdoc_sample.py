"""CPU unit tests for the real-document MPDocBench sample runner."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def runner():
    path = Path(__file__).parents[2] / "scripts/run_mpdoc_sample.py"
    assert path.exists(), "real-document runner has not been implemented"
    spec = importlib.util.spec_from_file_location("run_mpdoc_sample", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_gt(tmp_path: Path) -> Path:
    doc = {
        "page_info": {
            "image_path": "doc_a.pdf",
            "images_list": ["images/doc_a/page_1.jpg", "images/doc_a/page_2.jpg"],
        },
        "layout_dets": [],
    }
    path = tmp_path / "gt.json"
    path.write_text(json.dumps([doc]), encoding="utf-8")
    return path


def fake_data(tmp_path: Path, second_page_bytes: bytes = b"page2") -> Path:
    root = tmp_path / "data"
    payloads = (
        ("images/doc_a/page_1.jpg", b"page1"),
        ("images/doc_a/page_2.jpg", second_page_bytes),
    )
    for rel, payload in payloads:
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    return root


def load_index(index_path: Path) -> dict:
    return {p["path"]: p for p in json.loads(index_path.read_text())["pages"]}


def fake_index(tmp_path: Path, root: Path, mutate=None) -> Path:
    pages = []
    payloads = (
        ("images/doc_a/page_1.jpg", b"page1"),
        ("images/doc_a/page_2.jpg", b"page2"),
    )
    for rel, payload in payloads:
        pages.append(
            {"path": rel, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        )
    if mutate:
        mutate(pages)
    index = tmp_path / "page-index.json"
    index.write_text(json.dumps({"pages": pages}), encoding="utf-8")
    return index


def test_load_ground_truth_and_pages(tmp_path):
    module = runner()
    gt = module.load_ground_truth(fake_gt(tmp_path))
    assert "doc_a" in gt
    assert module.document_pages(gt["doc_a"]) == [
        "images/doc_a/page_1.jpg",
        "images/doc_a/page_2.jpg",
    ]


def test_verify_page_hashes_accepts_exact_bytes(tmp_path):
    module = runner()
    root = fake_data(tmp_path)
    index_path = fake_index(tmp_path, root)
    index = load_index(index_path)
    module.verify_page_hashes(root, [p["path"] for p in index.values()], index)


def test_verify_page_hashes_rejects_corrupted_page(tmp_path):
    module = runner()
    root = fake_data(tmp_path, second_page_bytes=b"tampered")
    index_path = fake_index(tmp_path, root)
    index = load_index(index_path)
    with pytest.raises(ValueError, match="hash mismatch"):
        module.verify_page_hashes(root, ["images/doc_a/page_2.jpg"], index)


def test_verify_page_hashes_rejects_unknown_page(tmp_path):
    module = runner()
    root = fake_data(tmp_path)
    index_path = fake_index(tmp_path, root)
    index = load_index(index_path)
    with pytest.raises(ValueError, match="frozen index"):
        module.verify_page_hashes(root, ["images/doc_a/page_9.jpg"], index)
