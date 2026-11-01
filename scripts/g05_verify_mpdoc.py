"""Independently recheck original MPDocBench ZIP bytes and frozen page hashes."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from g05_acquire_datasets import DATA, ROOT, digest, require, write_json


def main() -> None:
    lock_path = ROOT / "eval/benchmarks/benchmark-lock.json"
    expected = json.loads(lock_path.read_text("utf-8"))["datasets"]["MPDocBench"]
    archive = ROOT / ".b/benchmark-data/MPDocBench_data.zip"
    require(
        archive.stat().st_size == expected["image_archive_bytes"], "archive byte count mismatch"
    )
    require(digest(archive) == expected["image_archive_sha256"], "archive identity mismatch")
    ground_truth = ROOT / ".b/benchmark-audit/qwen/MPDocBench/MPDocBench.json"
    require(
        digest(ground_truth) == expected["ground_truth_sha256"], "ground truth identity mismatch"
    )
    scope = ROOT / "eval/benchmarks/manifests/MPDocBench.jsonl"
    require(digest(scope) == expected["scope_sha256"], "scope identity mismatch")
    gt = json.loads(ground_truth.read_text("utf-8"))
    paths = [page for doc in gt for page in doc["page_info"]["images_list"]]
    require(len(gt) == expected["documents"] == 420, "document count mismatch")
    require(len(paths) == len(set(paths)) == expected["pages"] == 3135, "page count mismatch")
    prior_path = ROOT / ".b/benchmark-audit/mpdoc-archive-inventory-final.json"
    prior = json.loads(prior_path.read_text("utf-8"))
    published_path = ROOT / "eval/benchmarks/acquisition/MPDocBench.json"
    published = json.loads(published_path.read_text("utf-8"))
    require(
        digest(prior_path) == published["page_hash_index_sha256"],
        "original inventory differs from published receipt",
    )
    page_hashes = {}
    missing = []
    with zipfile.ZipFile(archive) as source:
        names = set(source.namelist())
        for name in paths:
            member = name.removeprefix("images/")
            if member not in names:
                missing.append(name)
                continue
            content = source.read(member)  # ZipFile independently validates CRC as well.
            actual = hashlib.sha256(content).hexdigest()
            require(actual == prior["page_sha256"][name], f"original page hash mismatch: {name}")
            page_hashes[name] = actual
            output = DATA / "MPDocBench" / name
            output.parent.mkdir(parents=True, exist_ok=True)
            if not output.exists() or digest(output) != actual:
                output.write_bytes(content)
            require(digest(output) == actual, f"extracted page hash mismatch: {name}")
    require(page_hashes == prior["page_sha256"], "page hash index mismatch")
    require(missing == prior["missing_pages"], "missing page inventory mismatch")
    require(len(page_hashes) == published["archive_verified_pages"] == 3051, "page count mismatch")
    require(len(missing) == published["not_acquired_pages"] == 84, "missing page count mismatch")
    documents = {Path(name).parent.name for name in page_hashes}
    require(
        len(documents) == published["archive_verified_documents"] == 409, "document count mismatch"
    )
    index = DATA / "mpdoc-independent-page-index.json"
    write_json(index, {"page_sha256": page_hashes, "missing_pages": missing})
    receipt = {
        "schema": 1,
        "dataset": "MPDocBench",
        "status": "BLOCKED_84_GATED_SLIDEVQA_PAGES",
        "archive_revision": expected["image_archive_revision"],
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": digest(archive),
        "expected_documents": len(gt),
        "expected_pages": len(paths),
        "independently_verified_documents": len(documents),
        "independently_verified_original_pages": len(page_hashes),
        "missing_documents": published["not_acquired_documents"],
        "missing_pages": missing,
        "scope_sha256": digest(scope),
        "ground_truth_sha256": digest(ground_truth),
        "prior_inventory_sha256": digest(prior_path),
        "published_acquisition_receipt_sha256": digest(published_path),
        "page_hash_index_sha256": digest(index),
        "page_hash_index": ".b/g05-data/mpdoc-independent-page-index.json",
        "producer_sha256": digest(Path(__file__)),
        "archive_layout": "Original <document>/page_n.jpg extracted byte-for-byte into images/",
        "verification": (
            "Original ZIP SHA256, each ZIP page SHA256 against immutable inventory, "
            "then extracted-page reread"
        ),
        "image_metadata_license": "CC-BY-NC-SA-4.0",
        "redistribution": "No images or ground truth transcriptions in Git",
    }
    write_json(ROOT / "eval/benchmarks/g05/datasets/MPDocBench.json", receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != "missing_pages"}))


if __name__ == "__main__":
    main()
