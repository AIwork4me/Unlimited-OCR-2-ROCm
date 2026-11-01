"""Metadata-only manifests. Ground truth, never filesystem globbing, defines scope."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

DENOMINATORS = {"MPDocBench": (420, 3135), "MarathonDocBench": (100, 18272)}


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def file_hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def safe_id(value: str) -> str:
    if (
        not re.fullmatch(r"[\w.-]+", value)
        or value in {".", ".."}
        or value.endswith((".", " "))
        or value.split(".")[0].upper()
        in {
            "CON",
            "PRN",
            "AUX",
            "NUL",
            *(f"COM{i}" for i in range(10)),
            *(f"LPT{i}" for i in range(10)),
        }
    ):
        raise ValueError(f"Unsafe identifier: {value!r}")
    return value


def local_path(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if (
        not relative
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in relative
        or ":" in relative
        or path.as_posix() != relative
    ):
        raise ValueError(f"Unsafe relative path: {relative!r}")
    for part in path.parts:
        safe_id(part)
    result = (root / relative).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError(f"Path escape: {relative}")
    return result


def inventory(
    ground_truth: Path,
    root: Path,
    dataset: str,
    revision: str,
    expected_documents: int,
    expected_pages: int,
) -> dict[str, Any]:
    if dataset in DENOMINATORS and DENOMINATORS[dataset] != (expected_documents, expected_pages):
        raise ValueError("Official denominator cannot be changed")
    rows = json.loads(ground_truth.read_text(encoding="utf-8"))
    documents = []
    for row in rows:
        info = row["page_info"]
        name = info["image_path"]
        if PurePosixPath(name).name != name or not name.endswith(".pdf"):
            raise ValueError("Expected a document basename ending in .pdf")
        doc_id = safe_id(name[:-4])
        pages = []
        for index, relative in enumerate(info["images_list"], 1):
            path = local_path(root, relative)
            pages.append(
                {
                    "id": index,
                    "path": relative,
                    "sha256": file_hash(path) if path.is_file() else None,
                }
            )
        documents.append({"id": doc_id, "scorer_name": name[:-4] + ".md", "pages": pages})
    manifest = {
        "schema": 1,
        "dataset": safe_id(dataset),
        "revision": revision,
        "ground_truth_sha256": file_hash(ground_truth),
        "expected_documents": expected_documents,
        "expected_pages": expected_pages,
        "documents": sorted(documents, key=lambda d: d["id"]),
    }
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    """Reject malformed scope; return BLOCKED for absent/unverified/changed payloads."""
    if manifest["schema"] != 1:
        raise ValueError("Unknown manifest schema")
    dataset = safe_id(manifest["dataset"])
    if not re.fullmatch(r"[0-9a-f]{40}", manifest["revision"]):
        raise ValueError("Immutable 40-character source revision required")
    if not re.fullmatch(r"[0-9a-f]{64}", manifest["ground_truth_sha256"]):
        raise ValueError("Ground-truth SHA256 required")
    expected = (manifest["expected_documents"], manifest["expected_pages"])
    if dataset in DENOMINATORS and expected != DENOMINATORS[dataset]:
        raise ValueError("Official denominator cannot be changed")
    docs = manifest["documents"]
    ids = [safe_id(doc["id"]) for doc in docs]
    if ids != sorted(ids) or len({name.casefold() for name in ids}) != len(ids):
        raise ValueError("Document IDs must be unique and sorted (including Windows case)")
    if expected != (len(docs), sum(len(doc["pages"]) for doc in docs)) or not docs:
        raise ValueError("Document/page denominator mismatch")
    seen: set[str] = set()
    errors = []
    for doc in docs:
        if not doc["pages"] or doc["scorer_name"] != doc["id"] + ".md":
            raise ValueError("Empty document or scorer filename mismatch")
        for index, page in enumerate(doc["pages"], 1):
            relative = page["path"]
            path = local_path(root or Path.cwd(), relative)
            if (
                page["id"] != index
                or not re.fullmatch(
                    rf"page_{index}\.(?:png|jpg|jpeg)", PurePosixPath(relative).name
                )
                or PurePosixPath(relative).parent.as_posix() != "images/" + doc["id"]
                or relative.casefold() in seen
            ):
                raise ValueError(
                    "Page IDs/paths must be unique, contiguous and in ground-truth order"
                )
            seen.add(relative.casefold())
            sha = page["sha256"]
            if sha is None:
                errors.append({"path": relative, "reason": "UNVERIFIED"})
            elif not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
                raise ValueError("Invalid page SHA256")
            elif root is not None and (not path.is_file() or file_hash(path) != sha):
                errors.append({"path": relative, "reason": "MISSING_OR_CHANGED"})
    return {
        "status": "BLOCKED" if errors else "PASS",
        "expected_documents": expected[0],
        "expected_pages": expected[1],
        "manifest_sha256": digest(manifest),
        "errors": errors,
    }
