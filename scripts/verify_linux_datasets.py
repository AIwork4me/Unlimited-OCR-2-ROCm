"""Read-only verifier for the Linux gfx1151 local benchmark datasets.

Checks the local ``Data/`` trees byte-for-byte against the frozen benchmark
lock, the frozen per-document manifests and the frozen per-page hash indexes.
The verifier never writes inside the data root; the only output is the
metadata receipt named by ``--output``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

READ_BLOCK = 1024 * 1024

MISMATCH_FIELDS = (
    "path",
    "expected_bytes",
    "actual_bytes",
    "expected_sha256",
    "actual_sha256",
    "decode_error",
)

DATASET_CONFIG = {
    "MPDocBench": {
        "page_index": "eval/benchmarks/g05/datasets/MPDocBench-page-index.json",
        "manifest": "eval/benchmarks/manifests/MPDocBench.jsonl",
        "page_digest_field": "archive_page_hashes_sha256",
        # The 11 SlideVQA decks joined later under an authorized frozen JPEG
        # serialization; their archive digest is intentionally not comparable.
        "page_digest_skip_marker": "official_slidevqa_jpeg_serialization",
    },
    "MarathonDocBench": {
        "page_index": "eval/benchmarks/g05/datasets/MarathonDocBench-page-index.json",
        "manifest": "eval/benchmarks/manifests/MarathonDocBench.jsonl",
        "page_digest_field": "published_page_hashes_sha256",
        "page_digest_skip_marker": None,
    },
}


def canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(READ_BLOCK), b""):
            checksum.update(block)
    return checksum.hexdigest()


def check_page_index_sanity(pages: list[dict]) -> None:
    seen: set[str] = set()
    folded: dict[str, str] = {}
    for page in pages:
        rel = page["path"]
        if os.path.isabs(rel) or ".." in Path(rel).parts:
            raise ValueError(f"page index path escapes the dataset root: {rel}")
        if rel in seen:
            raise ValueError(f"duplicate page path in index: {rel}")
        seen.add(rel)
        low = rel.casefold()
        if low in folded and folded[low] != rel:
            raise ValueError(f"case-insensitive path collision in index: {folded[low]} vs {rel}")
        folded[low] = rel


def document_of(page_path: str) -> str:
    parts = Path(page_path).parts
    if len(parts) < 3:
        raise ValueError(f"page path outside a document directory: {page_path}")
    return parts[-2]


def verify_single_page(dataset_root: Path, page: dict, decode: bool) -> dict:
    """Verify one page. Pure read; returns a per-page result record."""
    rel = page["path"]
    result = {
        "path": rel,
        "expected_bytes": page["bytes"],
        "expected_sha256": page["sha256"],
    }
    target = dataset_root / rel
    if target.is_symlink() or any((dataset_root / part).is_symlink() for part in _cumulative(rel)):
        result["status"] = "symlink"
        result["symlink"] = True
        return result
    if not target.exists():
        result["status"] = "missing"
        return result
    result["actual_bytes"] = target.stat().st_size
    result["actual_sha256"] = file_digest(target)
    ok = (
        result["actual_bytes"] == result["expected_bytes"]
        and result["actual_sha256"] == result["expected_sha256"]
    )
    if decode:
        # Decode even when bytes already mismatch: the extra evidence shows
        # whether the payload is merely altered or outright unreadable.
        try:
            from PIL import Image  # noqa: PLC0415

            with Image.open(target) as image:
                image.load()
        except Exception as error:  # noqa: BLE001 - record any decoder failure
            result["status"] = "mismatched"
            result["decode_error"] = f"{type(error).__name__}: {error}"
            return result
    result["status"] = "verified" if ok else "mismatched"
    return result


def _cumulative(rel: str) -> list[Path]:
    parts = Path(rel).parts
    return [Path(*parts[: i + 1]) for i in range(len(parts) - 1)]


def find_extra_files(dataset_root: Path, known: set[str]) -> list[str]:
    """List files under the dataset root that the frozen index does not define."""
    extras = []
    for current, _dirs, files in os.walk(dataset_root):
        for name in files:
            rel = os.path.relpath(Path(current) / name, dataset_root).replace(os.sep, "/")
            if rel not in known:
                extras.append(rel)
    return sorted(extras)


def verify_dataset(
    *,
    data_root: Path,
    dataset: str,
    pages: list[dict],
    rows: list[dict],
    lock_entry: dict,
    manifest_path: Path,
    decode: bool = False,
    page_digest_field: str | None = None,
    page_digest_skip_marker: str | None = None,
    workers: int = 1,
) -> dict:
    """Verify one dataset against its frozen identities. Read-only."""
    started = time.time()
    manifest_bytes = manifest_path.read_bytes()
    if file_digest(manifest_path) != lock_entry.get("scope_sha256"):
        raise ValueError(
            f"scope mismatch: manifest {manifest_path} does not match the frozen scope hash"
        )
    file_rows = [json.loads(line) for line in manifest_bytes.decode("utf-8").splitlines() if line]
    if len(file_rows) != lock_entry.get("documents"):
        raise ValueError(
            f"document count mismatch: manifest has {len(file_rows)}, "
            f"lock expects {lock_entry.get('documents')}"
        )
    row_mismatches = reconcile_rows(rows, file_rows)
    check_page_index_sanity(pages)
    if len(pages) != lock_entry.get("pages"):
        raise ValueError(
            f"page count mismatch: index has {len(pages)}, lock expects {lock_entry.get('pages')}"
        )

    dataset_root = data_root / dataset
    if workers > 1:
        from functools import partial

        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(
                pool.map(partial(_worker, str(dataset_root), decode=decode), pages, chunksize=16)
            )
    else:
        results = [verify_single_page(dataset_root, page, decode) for page in pages]

    missing = [r["path"] for r in results if r["status"] == "missing"]
    mismatched = [
        {k: v for k, v in r.items() if k in MISMATCH_FIELDS}
        for r in results
        if r["status"] == "mismatched"
    ]
    symlink_violations = [r["path"] for r in results if r["status"] == "symlink"]
    verified = [r for r in results if r["status"] == "verified"]
    extras = find_extra_files(dataset_root, {p["path"] for p in pages})

    doc_report = verify_document_digests(
        pages=pages,
        results=results,
        rows=file_rows,
        page_digest_field=page_digest_field,
        page_digest_skip_marker=page_digest_skip_marker,
    )

    verified_bytes = sum(r["expected_bytes"] for r in verified)
    mismatched_bytes = sum(m.get("actual_bytes", 0) - m["expected_bytes"] for m in mismatched)
    verified_lines = sorted(f"{r['path']}:{r['expected_sha256']}" for r in verified)
    index_digest = hashlib.sha256("\n".join(verified_lines).encode()).hexdigest()
    blocked = bool(
        missing
        or mismatched
        or extras
        or symlink_violations
        or row_mismatches
        or doc_report["mismatched"]
    )
    return {
        "schema": 1,
        "dataset": dataset,
        "status": "BLOCKED" if blocked else "PASS",
        "read_only": True,
        "expected_documents": lock_entry["documents"],
        "expected_pages": lock_entry["pages"],
        "verified_documents": len(file_rows) - len(doc_report["mismatched"]),
        "verified_pages": len(verified),
        "verified_bytes": verified_bytes,
        "expected_bytes_total": sum(p["bytes"] for p in pages),
        "mismatched_bytes_delta": mismatched_bytes,
        "missing": missing,
        "mismatched": mismatched,
        "symlink_violations": symlink_violations,
        "extra": extras,
        "manifest_row_mismatches": row_mismatches,
        "document_digests": doc_report,
        "page_index_digest": index_digest,
        "scope_sha256": lock_entry["scope_sha256"],
        "decode_checked": decode,
        "verification_seconds": round(time.time() - started, 3),
    }


def reconcile_rows(rows: list[dict], file_rows: list[dict]) -> list[dict]:
    """Flag caller-provided manifest rows that differ from the scope-bound file."""
    mismatches = []
    for index, (row, file_row) in enumerate(zip(rows, file_rows, strict=False)):
        fields = [
            key
            for key in sorted(set(row) | set(file_row))
            if canonical_digest(row.get(key)) != canonical_digest(file_row.get(key))
        ]
        if fields:
            mismatches.append({"line": index, "id": file_row.get("id"), "fields": fields})
    if len(rows) != len(file_rows):
        mismatches.append(
            {"line": None, "id": None, "fields": [f"row_count:{len(rows)}!={len(file_rows)}"]}
        )
    return mismatches


def verify_document_digests(
    *,
    pages: list[dict],
    results: list[dict],
    rows: list[dict],
    page_digest_field: str | None,
    page_digest_skip_marker: str | None,
) -> dict:
    by_doc: dict[str, list[dict]] = {}
    for page, result in zip(pages, results, strict=True):
        doc_pages = by_doc.setdefault(document_of(page["path"]), [])
        doc_pages.append({**page, "status": result["status"]})
    row_by_id = {row["id"]: row for row in rows}
    ok = 0
    mismatched = []
    skipped = []
    for doc_id, doc_pages in by_doc.items():
        row = row_by_id.get(doc_id)
        if row is None:
            mismatched.append({"id": doc_id, "field": "manifest_row_absent"})
            continue
        if canonical_digest([p["path"] for p in doc_pages]) != row["ordered_paths_sha256"]:
            mismatched.append({"id": doc_id, "field": "ordered_paths_sha256"})
            continue
        if page_digest_field and row.get(page_digest_field):
            marker_skipped = page_digest_skip_marker and any(
                p.get("source") == page_digest_skip_marker for p in doc_pages
            )
            if marker_skipped:
                skipped.append(doc_id)
            else:
                digest = canonical_digest([p["sha256"] for p in doc_pages])
                if digest != row[page_digest_field]:
                    mismatched.append({"id": doc_id, "field": page_digest_field})
                    continue
        ok += 1
    missing_docs = [row["id"] for row in rows if row["id"] not in by_doc]
    for doc_id in missing_docs:
        mismatched.append({"id": doc_id, "field": "pages_absent"})
    return {"ok": ok, "mismatched": mismatched, "page_digest_skipped": sorted(skipped)}


def _worker(dataset_root: str, page: dict, decode: bool) -> dict:
    return verify_single_page(Path(dataset_root), page, decode)


def host_identity() -> dict:
    return {
        "hostname": platform.node(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--benchmark-lock", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--decode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--page-source", type=Path, help="override page index path (testing)")
    parser.add_argument("--manifest", type=Path, help="override manifest path (testing)")
    args = parser.parse_args(argv)

    lock = json.loads(args.benchmark_lock.read_text("utf-8"))
    repo_root = args.benchmark_lock.parent.parent.parent
    started = time.time()
    receipts = []
    for dataset in args.dataset:
        config = DATASET_CONFIG.get(dataset)
        entry = lock["datasets"][dataset]
        page_source = args.page_source or (repo_root / config["page_index"])
        manifest_path = args.manifest or (repo_root / config["manifest"])
        pages = json.loads(page_source.read_text("utf-8"))["pages"]
        rows = [json.loads(line) for line in manifest_path.read_text("utf-8").splitlines() if line]
        report = verify_dataset(
            data_root=args.data_root,
            dataset=dataset,
            pages=pages,
            rows=rows,
            lock_entry=entry,
            manifest_path=manifest_path,
            decode=args.decode,
            page_digest_field=config["page_digest_field"] if config else None,
            page_digest_skip_marker=config["page_digest_skip_marker"] if config else None,
            workers=args.workers,
        )
        report["dataset_revision"] = entry.get("revision", entry.get("image_archive_revision"))
        report["page_index_source"] = str(page_source)
        receipts.append(report)
        print(
            f"{dataset}: {report['status']} "
            f"pages {report['verified_pages']}/{report['expected_pages']} "
            f"missing={len(report['missing'])} mismatched={len(report['mismatched'])} "
            f"extra={len(report['extra'])}"
        )
    receipt = {
        "schema": 1,
        "host": host_identity(),
        "data_root": str(args.data_root),
        "benchmark_lock_sha256": file_digest(args.benchmark_lock),
        "producer_sha256": file_digest(Path(__file__)),
        "verification_seconds_total": round(time.time() - started, 3),
        "datasets": receipts,
        "status": "PASS" if all(r["status"] == "PASS" for r in receipts) else "BLOCKED",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n", "utf-8")
    return 0 if receipt["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
