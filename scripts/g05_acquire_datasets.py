"""Acquire the frozen Marathon page payload without executing dataset code.

All downloads are revision-pinned, byte-counted and SHA256 checked before rename.
The separate verify command rereads every file and emits a deterministic hash index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / ".b/g05-data"
REVISION = "fa05fccbe82ee6ec628006a1d58bb153954c3ee3"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_digest(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_frozen_scope(gt: list[dict], by_path: dict, rows: dict) -> None:
    """Bind local metadata's LFS identities to the lock-authenticated scope."""
    require(len(rows) == len(gt), "scope document count mismatch")
    for document in gt:
        info = document["page_info"]
        row = rows[info["image_path"].removesuffix(".pdf")]
        paths = info["images_list"]
        require(len(paths) == row["pages"], "scope page count mismatch")
        require(canonical_digest(paths) == row["ordered_paths_sha256"], "page order mismatch")
        require(
            canonical_digest(document) == row["annotation_sha256"], "annotation identity mismatch"
        )
        require(
            canonical_digest([by_path[path]["lfs"]["sha256"] for path in paths])
            == row["published_page_hashes_sha256"],
            "published LFS hashes differ from frozen scope",
        )


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def download_file(
    url: str,
    dest: Path,
    size: int,
    sha256: str,
    attempts: int = 5,
    token: str | None = None,
) -> str:
    """Resume a partial transfer, but never accept an unverified final file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == size and digest(dest) == sha256:
        return "already_verified"
    partial = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(attempts):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset >= size:
                partial.unlink()
                offset = 0
            headers = {"User-Agent": "Unlimited-OCR-2-ROCm-frozen-dataset-audit"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = urllib.request.Request(url, headers=headers)
            if token:
                # urllib does not copy unredirected headers to signed CDN actions.
                request.add_unredirected_header("Authorization", "Bearer " + token)
            with urllib.request.urlopen(request, timeout=60) as response:
                if response.status == 206:
                    expected = f"bytes {offset}-{size - 1}/{size}"
                    if response.headers.get("Content-Range") != expected:
                        raise ValueError("unexpected Content-Range")
                elif response.status == 200:
                    offset = 0
                else:
                    raise ValueError(f"unexpected status {response.status}")
                with partial.open("ab" if offset else "wb") as stream:
                    while block := response.read(1024 * 1024):
                        stream.write(block)
                        if stream.tell() > size:
                            raise ValueError("payload exceeds expected size")
            if partial.stat().st_size != size:
                raise ValueError("size mismatch")
            if digest(partial) != sha256:
                partial.unlink()
                raise ValueError("sha256 mismatch")
            partial.replace(dest)
            return "downloaded_verified"
        except (OSError, ValueError, urllib.error.URLError) as exc:
            if attempt + 1 == attempts:
                raise
            delay = min(60, 2 ** (attempt + 1))
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 429:
                delay = max(delay, min(300, int(exc.headers.get("Retry-After", "60"))))
            time.sleep(delay)
    raise AssertionError("unreachable")


def manifest() -> tuple[list[dict], dict]:
    lock = json.loads((ROOT / "eval/benchmarks/benchmark-lock.json").read_text("utf-8"))
    expected = lock["datasets"]["MarathonDocBench"]
    metadata = json.loads((ROOT / ".b/benchmark-audit/marathon-meta.json").read_text("utf-8"))
    require(metadata["sha"] == expected["revision"] == REVISION, "revision mismatch")
    gt_path = ROOT / ".b/benchmark-audit/marathon/MarathonDocBench.json"
    require(digest(gt_path) == expected["ground_truth_sha256"], "ground truth identity mismatch")
    gt = json.loads(gt_path.read_text("utf-8"))
    paths = [page for doc in gt for page in doc["page_info"]["images_list"]]
    by_path = {item["rfilename"]: item for item in metadata["siblings"]}
    scope_path = ROOT / "eval/benchmarks/manifests/MarathonDocBench.jsonl"
    require(digest(scope_path) == expected["scope_sha256"], "frozen scope identity mismatch")
    rows = [json.loads(line) for line in scope_path.read_text("utf-8").splitlines()]
    validate_frozen_scope(gt, by_path, {row["id"]: row for row in rows})
    pages = [by_path[path] for path in paths]
    require(len(gt) == expected["documents"] == 100, "document count mismatch")
    require(len(paths) == len(set(paths)) == expected["pages"] == 18272, "page count mismatch")
    require(
        sum(item["lfs"]["size"] for item in pages) == expected["page_payload_bytes"],
        "total payload byte count mismatch",
    )
    for item in pages:
        path = Path(item["rfilename"])
        require(
            not path.is_absolute()
            and ".." not in path.parts
            and ":" not in item["rfilename"]
            and "\\" not in item["rfilename"],
            "unsafe payload path",
        )
    return pages, expected


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", "utf-8")
    temporary.replace(path)


def batch_download_urls(
    pages: list[dict],
    endpoint: str = "https://huggingface.co/datasets/baidu/MarathonDocBench.git/info/lfs/objects/batch",
    attempts: int = 5,
) -> dict[str, str]:
    """Resolve only the exact frozen OIDs through the official Git LFS service.

    Signed download actions remain in memory and are never written to receipts.
    """
    objects = {item["lfs"]["sha256"]: item["lfs"]["size"] for item in pages}
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(
            {
                "operation": "download",
                "transfers": ["basic"],
                "ref": {"name": REVISION},
                "objects": [{"oid": oid, "size": size} for oid, size in objects.items()],
            }
        ).encode("utf-8"),
        headers={
            "Content-Type": "application/vnd.git-lfs+json",
            "Accept": "application/vnd.git-lfs+json",
        },
    )
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                result = json.load(response)
            break
        except (OSError, urllib.error.URLError) as exc:
            if attempt + 1 == attempts:
                raise
            failure = {
                "stage": "lfs_batch",
                "attempt": attempt + 1,
                "error_type": type(exc).__name__,
            }
            if isinstance(exc, urllib.error.HTTPError):
                failure["http_status"] = exc.code
            print(json.dumps(failure), flush=True)
            time.sleep(min(60, 2 ** (attempt + 1)))
    actions = {}
    for item in result["objects"]:
        oid = item.get("oid")
        require(
            oid in objects and item.get("size") == objects[oid] and oid not in actions,
            "LFS batch identity mismatch",
        )
        require("error" not in item, "LFS batch object unavailable")
        url = item["actions"]["download"]["href"]
        parsed = urllib.parse.urlparse(url)
        host = parsed.hostname or ""
        require(
            parsed.scheme == "https"
            and (host == "huggingface.co" or host.endswith(".hf.co"))
            and not parsed.username
            and not parsed.password,
            "LFS action is not an official HTTPS download host",
        )
        actions[oid] = url
    require(set(actions) == set(objects), "LFS batch identity mismatch")
    return {item["rfilename"]: actions[item["lfs"]["sha256"]] for item in pages}


def acquire(workers: int, transport: str = "resolve") -> None:
    pages, expected = manifest()
    started = time.monotonic()
    counts: dict[str, int] = {}
    failures = []
    batch_size = 256 if transport == "lfs-batch" else len(pages)
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(pages), batch_size):
            batch = pages[start : start + batch_size]
            urls = batch_download_urls(batch) if transport == "lfs-batch" else {}
            futures = {}
            for item in batch:
                name = item["rfilename"]
                url = urls.get(name) or (
                    "https://huggingface.co/datasets/baidu/MarathonDocBench/resolve/"
                    + REVISION
                    + "/"
                    + urllib.parse.quote(name)
                )
                future = pool.submit(
                    download_file,
                    url,
                    DATA / "MarathonDocBench" / name,
                    item["lfs"]["size"],
                    item["lfs"]["sha256"],
                )
                futures[future] = name
            for future in as_completed(futures):
                completed += 1
                try:
                    result = future.result()
                    counts[result] = counts.get(result, 0) + 1
                except Exception as exc:
                    # Never persist exception messages: redirect URLs may carry signatures.
                    failure = {"page": futures[future], "error_type": type(exc).__name__}
                    if isinstance(exc, urllib.error.HTTPError):
                        failure["http_status"] = exc.code
                    failures.append(failure)
                if completed % 100 != 0 and completed != len(pages):
                    continue
                progress = {
                    "completed": completed,
                    "expected_pages": expected["pages"],
                    "counts": counts,
                    "transport": transport,
                    "failures": failures,
                    "elapsed_seconds": round(time.monotonic() - started, 1),
                }
                write_json(DATA / "marathon-download-progress.json", progress)
                print(
                    json.dumps({k: v for k, v in progress.items() if k != "failures"}), flush=True
                )
    if failures:
        raise SystemExit(f"{len(failures)} pages failed; see sanitized progress receipt")


def verify() -> None:
    pages, expected = manifest()
    records = []
    failures = []
    total = 0
    for item in pages:
        name = item["rfilename"]
        path = DATA / "MarathonDocBench" / name
        if not path.exists():
            failures.append({"page": name, "reason": "missing"})
            continue
        size, actual = path.stat().st_size, digest(path)
        if size != item["lfs"]["size"] or actual != item["lfs"]["sha256"]:
            failures.append({"page": name, "reason": "size_or_hash_mismatch"})
            continue
        total += size
        records.append({"path": name, "bytes": size, "sha256": actual})
    index = DATA / "marathon-independent-page-index.json"
    write_json(index, {"revision": REVISION, "pages": records})
    verified_paths = {record["path"] for record in records}
    gt = json.loads((ROOT / ".b/benchmark-audit/marathon/MarathonDocBench.json").read_text("utf-8"))
    documents = sum(
        all(path in verified_paths for path in document["page_info"]["images_list"])
        for document in gt
    )
    receipt = {
        "schema": 1,
        "dataset": "MarathonDocBench",
        "repository": expected["repository"],
        "revision": REVISION,
        "status": "VERIFIED" if not failures else "INCOMPLETE",
        "expected_documents": expected["documents"],
        "expected_pages": expected["pages"],
        "expected_page_payload_bytes": expected["page_payload_bytes"],
        "independently_verified_pages": len(records),
        "independently_verified_documents": documents,
        "independently_verified_bytes": total,
        "ground_truth_sha256": expected["ground_truth_sha256"],
        "scope_sha256": expected["scope_sha256"],
        "page_hash_index_sha256": digest(index),
        "page_hash_index": ".b/g05-data/marathon-independent-page-index.json",
        "metadata_sha256": digest(ROOT / ".b/benchmark-audit/marathon-meta.json"),
        "producer_sha256": digest(Path(__file__)),
        "initial_acquisition_producer_sha256": (
            digest(DATA / "acquirer-initial.py")
            if (DATA / "acquirer-initial.py").exists()
            else None
        ),
        "eager_batch_attempt_producer_sha256": (
            digest(DATA / "acquirer-batch.py") if (DATA / "acquirer-batch.py").exists() else None
        ),
        "streaming_acquisition_producer_sha256": (
            digest(DATA / "acquirer-batch-streaming.py")
            if (DATA / "acquirer-batch-streaming.py").exists()
            else None
        ),
        "verification": (
            "Separate sequential reread against LFS SHA256 and sizes; "
            "ordered LFS hash lists bound to lock-authenticated frozen document scope"
        ),
        "license_metadata": "Apache-2.0",
        "redistribution": "No page images, weights or GT transcriptions in Git",
        "failures": failures,
    }
    write_json(ROOT / "eval/benchmarks/g05/datasets/MarathonDocBench.json", receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != "failures"}), flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} verification failures")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["acquire", "verify"])
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--transport", choices=["resolve", "lfs-batch"], default="resolve")
    args = parser.parse_args()
    if not 1 <= args.workers <= 48:
        parser.error("workers must be between 1 and 48")
    if args.command == "acquire":
        acquire(args.workers, args.transport)
    else:
        verify()


if __name__ == "__main__":
    main()
