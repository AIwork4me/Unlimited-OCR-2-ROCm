"""Package sanitized G1 evidence with explicit raw/member/Git byte bindings."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import subprocess
import tarfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sanitize(data: bytes, root: Path = ROOT, home: Path | None = None) -> bytes:
    text = data.decode("utf-8")
    replacements = [(str(root), "<REPO>"), (str(home or Path.home()), "<USERPROFILE>")]
    replacements += [("C:\\uocr", "<REPO>"), ("C:\\b", "<REPO>/.b")]
    for original, label in replacements:
        for variant in sorted(
            {original, original.replace("\\", "/"), original.replace("\\", "\\\\")},
            key=len,
            reverse=True,
        ):
            text = re.sub(
                re.escape(variant), lambda _, label=label: label, text, flags=re.IGNORECASE
            )
    return text.encode("utf-8")


def sensitive(data: bytes) -> list[str]:
    patterns = {
        "user_directory": rb"(?:[A-Za-z]:[\\/]+Users[\\/]+|/mnt/[a-z]/Users/)",
        "token_value": rb"hf_[A-Za-z0-9]{20,}",
        "signed_query": rb"X-Amz-(?:Signature|Credential)=[A-Za-z0-9%/]{20,}",
    }
    return [name for name, pattern in patterns.items() if re.search(pattern, data, re.I)]


def archive_bytes(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for name, data in sorted(members.items()):
                path = Path(name)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError("Archive member must be relative")
                entry = tarfile.TarInfo(name)
                entry.size, entry.mode, entry.mtime = len(data), 0o644, 0
                entry.uid = entry.gid = 0
                entry.uname = entry.gname = ""
                archive.addfile(entry, io.BytesIO(data))
    return buffer.getvalue()


def write_json(path: Path, data: Any) -> None:
    path.write_bytes((json.dumps(data, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def evidence_files(directory: Path, *, top_level: bool = False) -> list[Path]:
    paths = directory.iterdir() if top_level else directory.rglob("*")
    return sorted(
        path
        for path in paths
        if path.is_file() and path.suffix in {".json", ".jsonl", ".txt", ".log", ".md", ".png"}
    )


def publish(output: Path) -> dict[str, Any]:
    if not output.resolve().is_relative_to(ROOT) or output.exists():
        raise ValueError("Use a fresh publication path inside the repository")
    roots = {
        "before": ROOT / ".b/g1/before",
        "after": ROOT / ".b/g1/after",
        "comparison": ROOT / ".b/g1/comparison",
        "math-dispatch": ROOT / ".b/g1/math-dispatch",
        "ledger-replay": ROOT / ".b/g1/failure-replay-final",
        "analysis": ROOT / ".b/g1-physical/analysis/final",
        "preflight": ROOT / ".b/g1-physical/preflight",
        "sparse-probe": ROOT / ".b/g1-physical/sparse-original",
        "supervisor-before": ROOT / ".b/g1-physical/before-process",
        "supervisor-after": ROOT / ".b/g1-physical/after-process",
        "math-process": ROOT / ".b/g1-physical/math-dispatch-process",
        "review": ROOT / ".b/g1-physical/review",
        "integration": ROOT / ".b/g1-physical/integration",
        "resource-append": ROOT / ".b/resource",
    }
    sources = {
        "producers/preflight.py.txt": ROOT / ".b/g1-physical/preflight.py",
        "producers/supervise.py.txt": ROOT / ".b/g1-physical/supervise.py",
        "producers/profile_math.py.txt": ROOT / ".b/g1-physical/profile_math.py",
        "producers/before-inference.py.txt": ROOT / ".b/g1/before-inference.py",
        "producers/baseline-source-hashes.json": ROOT / ".b/g1/baseline-source-hashes.json",
        "producers/final-preservation.py.txt": ROOT / ".b/g1-physical/final_preservation.py",
        "producers/final-ci.py.txt": ROOT / ".b/g1-physical/final_ci.py",
    }
    for name in (
        "checks.json",
        "pytest.stdout.log",
        "ruff.stdout.log",
        "format.stdout.log",
        "mypy.stdout.log",
        "pip-check.stdout.log",
    ):
        sources["local-checks/" + name] = ROOT / ".b/g1-physical/checks" / name
    for prefix, directory in roots.items():
        if not directory.is_dir():
            raise ValueError("Missing evidence root: " + prefix)
        # Review receipts are flat; nested pytest caches are not publication evidence.
        for path in evidence_files(directory, top_level=prefix == "review"):
            sources[prefix + "/" + path.relative_to(directory).as_posix()] = path
    members: dict[str, bytes] = {}
    entries = []
    for name, source in sorted(sources.items()):
        raw = source.read_bytes()
        row: dict[str, Any] = {
            "raw_local_path": source.relative_to(ROOT).as_posix(),
            "raw_sha256": digest(raw),
            "raw_bytes": len(raw),
        }
        if source.suffix == ".png":
            row.update(publication="EXCLUDED_DUPLICATE_PUBLIC_SYNTHETIC_IMAGE", member=None)
        else:
            # Native model raw bytes never receive normalization/redaction.
            copied = raw if name.endswith("raw.txt") else sanitize(raw)
            findings = sensitive(copied)
            if findings:
                raise ValueError("Sensitive publication content: " + name + ": " + str(findings))
            members[name] = copied
            row.update(
                member=name,
                member_sha256=digest(copied),
                member_bytes=len(copied),
                publication="EXACT" if copied == raw else "SANITIZED_TEXT",
            )
        entries.append(row)
    payload = archive_bytes(members)
    output.mkdir(parents=True)
    (output / "evidence.tar.gz").write_bytes(payload)
    analysis = json.loads((ROOT / ".b/g1-physical/analysis/final/analysis.json").read_text("utf-8"))
    failures = json.loads(
        (ROOT / ".b/g1-physical/analysis/final/failure-contract.json").read_text("utf-8")
    )
    dispatch = json.loads((ROOT / ".b/g1/math-dispatch/dispatch.json").read_text("utf-8"))
    if analysis["status"] != "PASS" or failures["status"] != "PASS" or dispatch["status"] != "PASS":
        raise ValueError("Physical or replay gate did not pass")
    if any(dispatch[key]["status"] != "PASS" for key in ("hip_before", "hip_after")):
        raise ValueError("Profiler HIP evidence did not pass")
    index = {
        "schema": 1,
        "scope": "Original-wheel Windows gfx1151 MATH synthetic regression",
        "archive": "evidence.tar.gz",
        "archive_sha256": digest(payload),
        "archive_bytes": len(payload),
        "members": entries,
        "producer_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT)
        .decode()
        .strip(),
        "publisher_sha256": digest(Path(__file__).read_bytes()),
        "source_raw_unchanged": True,
        "restricted_payload_in_archive": False,
        "normal_pairs_passed": analysis["case_counts"]["PASS"],
        "normal_raw_bytes_per_arm": analysis["byte_comparison_counts"]["before_total_bytes"],
        "failure_contract": failures["status"],
        "math_dispatch": dispatch["operators"],
        "memory_limit": (
            "Parent RSS samples identify venv launcher; use native worker snapshots only"
        ),
        "hardware_unvalidated": ["Linux", "W7900/gfx1100", "other devices"],
        "full_benchmark": "NOT RUN",
        "above_five_pages": "NOT RUN; current caps preserved",
    }
    write_json(output / "index.json", index)
    (output / "comparison.json").write_bytes(members["comparison/comparison.json"])
    (output / "README.md").write_text(
        "# G1 sealed physical evidence\n\n"
        "The archive contains sanitized metadata and exact native raw text "
        "from public synthetic fixtures. "
        "It contains no model/wheel/SDK/data image payload or credentials.\n\n"
        "index.json binds each private raw artifact SHA256 to its public archive member SHA256; "
        "unchanged raw hash manifests describe private raw counterparts, "
        "not sanitized byte copies. "
        "Verify member hashes through the index. "
        "The archive uses fixed metadata and gzip mtime=0.\n\n"
        "Native raw.txt bytes remain exact. Duplicate fixture PNGs are excluded and hash-recorded. "
        "Windows launcher RSS samples do not measure model RSS; "
        "native worker snapshots are separate. "
        "Controlled lower-limit/cancellation tests and synthetic ledger corruptions are labeled. "
        "No full benchmark, long-document capacity or performance claim follows.\n",
        encoding="utf-8",
        newline="\n",
    )
    write_json(
        output / "SHA256SUMS.json",
        {p.name: digest(p.read_bytes()) for p in sorted(output.iterdir())},
    )
    return {
        "status": "PASS",
        "archive_bytes": len(payload),
        "members": len(members),
        "excluded_images": len(entries) - len(members),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.output)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
