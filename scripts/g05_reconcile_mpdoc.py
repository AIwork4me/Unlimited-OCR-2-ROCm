"""Evidence-only full-scope reread; never downloads or changes image bytes."""

import hashlib
import json
from pathlib import Path

ROOT = Path.cwd()
PUBLIC = ROOT / "eval/benchmarks/g05/datasets"
LOCAL = ROOT / ".b/g05-data"


def sha(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def check(condition, reason):
    if not condition:
        raise ValueError(reason)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


lock = read(ROOT / "eval/benchmarks/benchmark-lock.json")["datasets"]["MPDocBench"]
gt_path = ROOT / ".b/benchmark-audit/qwen/MPDocBench/MPDocBench.json"
check(sha(gt_path) == lock["ground_truth_sha256"], "ground truth identity mismatch")
scope_path = ROOT / "eval/benchmarks/manifests/MPDocBench.jsonl"
check(sha(scope_path) == lock["scope_sha256"], "scope identity mismatch")
gt = read(gt_path)
paths = [page for document in gt for page in document["page_info"]["images_list"]]
check(len(gt) == lock["documents"] == 420, "full document denominator mismatch")
check(len(paths) == len(set(paths)) == lock["pages"] == 3135, "full page denominator mismatch")

archive_path = ROOT / ".b/benchmark-data/MPDocBench_data.zip"
check(sha(archive_path) == lock["image_archive_sha256"], "original archive SHA256 mismatch")
check(archive_path.stat().st_size == lock["image_archive_bytes"], "original archive size mismatch")
original_path = ROOT / ".b/benchmark-audit/mpdoc-archive-inventory-final.json"
prior_receipt = read(ROOT / "eval/benchmarks/acquisition/MPDocBench.json")
check(
    sha(original_path) == prior_receipt["page_hash_index_sha256"],
    "prior inventory identity mismatch",
)
original = read(original_path)
check(len(original["page_sha256"]) == 3051, "original page denominator mismatch")
check(len(original["missing_pages"]) == 84, "missing page denominator mismatch")

slide_receipt_path = PUBLIC / "SlideVQA.json"
slide_receipt = read(slide_receipt_path)
check(slide_receipt["status"] == "VERIFIED", "authorized SlideVQA acquisition is incomplete")
check(slide_receipt["revision"] == lock["slidevqa"]["revision"], "SlideVQA revision mismatch")
check(slide_receipt["verified_decks"] == 11, "SlideVQA deck denominator mismatch")
check(slide_receipt["verified_pages"] == 84, "SlideVQA page denominator mismatch")
check(
    slide_receipt["test_shard_sha256"] == lock["slidevqa"]["test_shards"], "shard manifest mismatch"
)
slide_index_path = PUBLIC / "SlideVQA-page-index.json"
check(
    sha(slide_index_path) == slide_receipt["page_hash_index_sha256"],
    "SlideVQA page index identity mismatch",
)
slide = read(slide_index_path)
supplement = {page["path"]: page for page in slide["pages"]}
check(len(slide["pages"]) == len(supplement) == 84, "supplemental page count mismatch")
check(
    set(supplement) == set(original["missing_pages"]), "supplement differs from frozen missing set"
)
check(
    not set(supplement).intersection(original["page_sha256"]), "supplement overlaps original pages"
)
check(set(supplement).union(original["page_sha256"]) == set(paths), "combined scope mismatch")

metadata = read(ROOT / ".b/benchmark-audit/slidevqa-meta.json")
check(metadata["sha"] == lock["slidevqa"]["revision"], "shard metadata revision mismatch")
by_path = {item["rfilename"]: item for item in metadata["siblings"]}
shard_records = []
for name, expected_hash in sorted(lock["slidevqa"]["test_shards"].items()):
    path = LOCAL / "SlideVQA" / name
    check(sha(path) == expected_hash, f"independent shard hash mismatch: {name}")
    check(
        path.stat().st_size == by_path[name]["lfs"]["size"],
        f"independent shard size mismatch: {name}",
    )
    shard_records.append({"path": name, "sha256": expected_hash, "bytes": path.stat().st_size})

records = []
original_verified = 0
supplement_verified = 0
for name in paths:
    path = LOCAL / "MPDocBench" / name
    actual = sha(path)
    if name in original["page_sha256"]:
        check(actual == original["page_sha256"][name], f"original page changed: {name}")
        source = "original_archive_bytes"
        original_verified += 1
    else:
        check(actual == supplement[name]["output_sha256"], f"serialized page changed: {name}")
        check(
            path.stat().st_size == supplement[name]["output_bytes"],
            f"serialized page size mismatch: {name}",
        )
        source = "official_slidevqa_jpeg_serialization"
        supplement_verified += 1
    records.append({"path": name, "sha256": actual, "bytes": path.stat().st_size, "source": source})
check(original_verified == 3051 and supplement_verified == 84, "combined verified count mismatch")
index_path = PUBLIC / "MPDocBench-page-index.json"
write(index_path, {"ground_truth_sha256": lock["ground_truth_sha256"], "pages": records})
receipt = {
    "schema": 1,
    "dataset": "MPDocBench",
    "status": "VERIFIED_FULL_SCOPE",
    "expected_documents": 420,
    "expected_pages": 3135,
    "independently_verified_documents": len(gt),
    "independently_verified_pages": len(records),
    "independently_verified_original_pages_unchanged": original_verified,
    "independently_verified_slidevqa_pages": supplement_verified,
    "missing_documents": [],
    "missing_pages": [],
    "archive_sha256": sha(archive_path),
    "archive_bytes": archive_path.stat().st_size,
    "ground_truth_sha256": sha(gt_path),
    "scope_sha256": sha(scope_path),
    "original_page_inventory_sha256": sha(original_path),
    "slidevqa_revision": lock["slidevqa"]["revision"],
    "slidevqa_receipt_sha256": sha(slide_receipt_path),
    "slidevqa_page_index_sha256": sha(slide_index_path),
    "independently_verified_test_shards": shard_records,
    "page_hash_index": "eval/benchmarks/g05/datasets/MPDocBench-page-index.json",
    "page_hash_index_sha256": sha(index_path),
    "data_root": ".b/g05-data/MPDocBench",
    "producer": "scripts/g05_reconcile_mpdoc.py",
    "producer_sha256": sha(Path(__file__)),
    "serialization_runtime_receipt": "SlideVQA-runtime.json",
    "serialization_runtime_receipt_sha256": sha(PUBLIC / "SlideVQA-runtime.json"),
    "verification": (
        "Separate process reread of all 3135 GT-defined page bytes, "
        "original ZIP and all 12 frozen SlideVQA shards"
    ),
    "source_constraints": (
        "Retained MPDocBench image metadata and pinned NTT evaluation terms apply; "
        "no inferred legal permission"
    ),
    "redistribution": (
        "Public files contain only IDs, counts, versions and hashes; "
        "restricted images, Parquet and GT remain ignored"
    ),
}
write(PUBLIC / "MPDocBench-full-scope.json", receipt)
print(
    json.dumps(
        {
            key: value
            for key, value in receipt.items()
            if key != "independently_verified_test_shards"
        }
    )
)
