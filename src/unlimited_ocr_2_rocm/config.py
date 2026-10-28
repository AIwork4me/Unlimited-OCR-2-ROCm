"""Immutable official source identity and streaming checksum validation."""

from __future__ import annotations

import hashlib
import json
import re
from importlib.resources import files
from pathlib import Path
from typing import Any


def check_revision(revision: str) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("exact immutable 40-character commit required")


def read_lock() -> dict[str, Any]:
    lock: dict[str, Any] = json.loads(
        files("unlimited_ocr_2_rocm").joinpath("upstream-lock.json").read_text(encoding="utf-8-sig")
    )
    check_revision(lock["model_revision"])
    check_revision(lock["code_revision"])
    return lock


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_files(snapshot: Path, expected: dict[str, str]) -> dict[str, str]:
    observed = {}
    for name, sha256 in expected.items():
        if Path(name).name != name or "/" in name or "\\" in name:
            raise ValueError("snapshot manifest permits only direct child files")
        path = snapshot / name
        if not path.is_file():
            raise RuntimeError(f"reviewed file missing: {name}")
        observed[name] = file_sha256(path)
        if observed[name] != sha256:
            raise RuntimeError(f"reviewed file checksum mismatch: {name}")
    return observed


def download_snapshot(cache_dir: Path | None, offline: bool = False) -> Path:
    """Download data only. No trust_remote_code call occurs here."""
    from huggingface_hub import snapshot_download

    lock = read_lock()
    return Path(
        snapshot_download(
            lock["model_id"],
            revision=lock["model_revision"],
            cache_dir=str(cache_dir) if cache_dir else None,
            local_files_only=offline,
            allow_patterns=list(lock["files"]) + list(lock["weights"]),
            max_workers=2,
        )
    )
