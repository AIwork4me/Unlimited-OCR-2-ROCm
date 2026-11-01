"""Append-only, atomic document attempts with honest full-denominator accounting."""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .._upstream.process_unlimitedOCR2 import _parse_blocks, convert
from .manifest import digest, file_hash, safe_id, validate_manifest

STATES = ("PASS", "FAIL", "TIMEOUT", "OOM", "TRUNCATED")


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def write_bytes(path: Path, value: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class Run:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data = json.loads((path / "run.json").read_text(encoding="utf-8"))
        if digest(self.data) != path.name:
            raise ValueError("Run identity hash mismatch")
        validate_manifest(self.data["manifest"])
        self.documents = {d["id"]: d for d in self.data["manifest"]["documents"]}

    @classmethod
    def create(
        cls,
        root: Path,
        manifest: dict[str, Any],
        config: dict[str, Any],
        environment: dict[str, Any],
    ) -> Run:
        if validate_manifest(manifest)["status"] != "PASS":
            raise ValueError("Cannot initialize a run with unverified page hashes")
        if not config or not environment:
            raise ValueError("Frozen protocol and environment identity required")
        data = {"schema": 1, "manifest": manifest, "config": config, "environment": environment}
        parent = root / safe_id(manifest["dataset"])
        parent.mkdir(parents=True, exist_ok=True)
        path = parent / digest(data)
        if not path.exists():
            staging = parent / (".init-" + uuid.uuid4().hex)
            staging.mkdir()
            write_bytes(staging / "run.json", json_bytes(data))
            try:
                staging.rename(path)
            except OSError:
                if not path.exists():
                    raise
                (staging / "run.json").unlink()
                staging.rmdir()
        return cls(path)

    def _document(self, doc_id: str) -> Path:
        safe_id(doc_id)
        if doc_id not in self.documents:
            raise ValueError("Document is outside frozen scope")
        return self.path / "documents" / doc_id

    def begin(self, doc_id: str) -> str:
        """Reserve a monotonically numbered attempt; a crash remains visible as FAIL."""
        directory = self._document(doc_id)
        directory.mkdir(parents=True, exist_ok=True)
        for number in range(1, 1_000_000):
            name = f"{number:06d}"
            if (directory / name).exists():
                continue
            pending = directory / (name + ".pending")
            try:
                pending.mkdir()
            except FileExistsError:
                continue
            write_bytes(pending / "started.json", json_bytes({"started_utc": timestamp()}))
            return name
        raise ValueError("Attempt counter exhausted")

    def finish(
        self, doc_id: str, attempt: str, raw: str, state: str, metadata: dict[str, Any]
    ) -> None:
        if state not in STATES or len(attempt) != 6 or not attempt.isdecimal():
            raise ValueError("Invalid state/attempt")
        directory = self._document(doc_id)
        pending = directory / (attempt + ".pending")
        if not (pending / "started.json").is_file() or (directory / attempt).exists():
            raise ValueError("Unknown/already completed attempt")
        write_bytes(pending / "raw.txt", raw.encode("utf-8"))
        reason = metadata.get("error", "")
        if (
            state == "PASS"
            and not {"finish_reason", "output_tokens", "max_tokens", "memory"} <= metadata.keys()
        ):
            state, reason = "FAIL", "MISSING_COMPLETION_METADATA"
        if state == "PASS":
            if (
                metadata["finish_reason"] == "length"
                or metadata["output_tokens"] >= metadata["max_tokens"]
            ):
                state, reason = "TRUNCATED", "TOKEN_LIMIT"
            else:
                pages = raw.split("<PAGE>")
                if pages[-1:] == [""]:
                    pages = pages[:-1]
                if (
                    metadata["finish_reason"] != "stop"
                    or not raw.strip()
                    or not pages
                    or pages[0].strip()
                    or len(pages) - 1 != len(self.documents[doc_id]["pages"])
                    or any(
                        "<BLOCK>" not in page or "<FORK>" not in page or not _parse_blocks(page)
                        for page in pages[1:]
                    )
                ):
                    state, reason = "FAIL", "EMPTY_MALFORMED_OR_PAGE_MISMATCH"
        # Raw is persisted even if conversion fails. No normalization of model bytes.
        if state == "PASS":
            try:
                for level in re.findall(
                    r"<\s*/?\s*LEVEL\s*(\d*)\s*[>}]?", raw, flags=re.IGNORECASE
                ):
                    if len(level) > 2 or (level and int(level) > 16):
                        raise ValueError("Generated heading level exceeds the existing safe bound")
                markdown = str(convert(raw))
                if not markdown.strip():
                    raise ValueError("Empty converted Markdown")
                write_bytes(pending / "prediction.md", markdown.encode("utf-8"))
            except Exception as error:
                state, reason = "FAIL", f"POSTPROCESS: {type(error).__name__}: {error}"
        row = {
            "doc_id": doc_id,
            "attempt": attempt,
            "state": state,
            "reason": reason,
            "completed_utc": timestamp(),
            "metadata": metadata,
            "expected_pages": len(self.documents[doc_id]["pages"]),
            "raw_page_markers": raw.count("<PAGE>"),
            "terminal_empty_marker": raw.endswith("<PAGE>"),
        }
        write_bytes(pending / "metadata.json", json_bytes(row))
        seal = {p.name: file_hash(p) for p in sorted(pending.iterdir()) if p.is_file()}
        write_bytes(pending / "COMMIT.json", json_bytes(seal))
        pending.rename(directory / attempt)

    def attempts(self, doc_id: str) -> list[dict[str, Any]]:
        directory = self._document(doc_id)
        rows: list[dict[str, Any]] = []
        if not directory.exists():
            return rows
        for path in sorted(directory.iterdir()):
            if path.name.endswith(".pending"):
                rows.append({"state": "FAIL", "reason": "INTERRUPTED", "attempt": path.name})
                continue
            seal = json.loads((path / "COMMIT.json").read_text(encoding="utf-8"))
            if not {"raw.txt", "metadata.json", "started.json"} <= seal.keys():
                raise ValueError("Incomplete attempt hash seal")
            for name, sha in seal.items():
                safe_id(name)
                if file_hash(path / name) != sha:
                    raise ValueError(f"Artifact hash mismatch: {doc_id}/{path.name}/{name}")
            row = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
            if row["state"] not in STATES or row["doc_id"] != doc_id or row["attempt"] != path.name:
                raise ValueError("Invalid committed attempt identity/state")
            if row["state"] == "PASS" and "prediction.md" not in seal:
                raise ValueError("Missing PASS prediction hash")
            rows.append(row)
        return rows

    def summary(self) -> dict[str, Any]:
        counts = dict.fromkeys((*STATES, "NOT_RUN"), 0)
        documents = []
        for doc_id, doc in self.documents.items():
            history = self.attempts(doc_id)
            row = history[-1] if history else {"state": "NOT_RUN", "reason": "NO_ATTEMPT"}
            counts[row["state"]] += 1
            documents.append(
                {**row, "doc_id": doc_id, "pages": len(doc["pages"]), "attempts": history}
            )
        manifest = self.data["manifest"]
        return {
            "run_id": self.path.name,
            "dataset": manifest["dataset"],
            "status": "COMPLETE" if counts["PASS"] == len(documents) else "PARTIAL",
            "expected_documents": manifest["expected_documents"],
            "expected_pages": manifest["expected_pages"],
            "counts": counts,
            "success_rate": counts["PASS"] / len(documents),
            "documents": documents,
        }

    def todo(self) -> list[str]:
        return [row["doc_id"] for row in self.summary()["documents"] if row["state"] != "PASS"]

    def export(self, output: Path) -> None:
        """Official per-document Markdown format; failures are explicit empty files."""
        summary = self.summary()
        staging = output.with_name("." + output.name + "-" + uuid.uuid4().hex)
        staging.mkdir(parents=True)
        for row in summary["documents"]:
            data = b""
            if row["state"] == "PASS":
                source = self._document(row["doc_id"]) / row["attempt"] / "prediction.md"
                data = source.read_bytes()
            write_bytes(staging / self.documents[row["doc_id"]]["scorer_name"], data)
        write_bytes(staging / "accounting.json", json_bytes(summary))
        if output.exists():
            raise FileExistsError(output)
        staging.rename(output)
