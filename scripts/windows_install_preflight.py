"""CPU-only checks for an unused Windows environment and original AMD archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


def read_lock(path: Path) -> list[dict[str, str]]:
    rows = []
    pending: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# wheel: "):
            pending = {"filename": line.removeprefix("# wheel: ")}
        elif line.startswith("# source: "):
            pending["url"] = line.removeprefix("# source: ")
        elif line and not line.startswith("#"):
            match = re.fullmatch(r"([\w-]+)==(\S+) --hash=sha256:([a-f0-9]{64})", line)
            if not match or set(pending) != {"filename", "url"}:
                raise ValueError("Malformed original-wheel lock")
            pending.update(name=match[1], version=match[2], sha256=match[3])
            if not pending["url"].startswith("https://repo.amd.com/rocm/whl-multi-arch/"):
                raise ValueError("Original wheel must originate at the AMD index")
            rows.append(pending)
            pending = {}
    if pending or len(rows) != 9 or len({row["filename"] for row in rows}) != 9:
        raise ValueError("Exactly nine distinct original AMD archives required")
    return rows


def contained(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError("Use a dedicated path inside the development root")
    return resolved


def member_target(environment: Path, name: str) -> Path:
    parts = PurePosixPath(name).parts
    if not parts or "\\" in name or ":" in name or name.startswith("/") or ".." in parts:
        raise ValueError("Unsafe wheel member")
    if parts[0].endswith(".data"):
        if len(parts) < 3:
            raise ValueError("Malformed wheel data member")
        schemes = {
            "purelib": environment / "Lib/site-packages",
            "platlib": environment / "Lib/site-packages",
            "scripts": environment / "Scripts",
            "data": environment,
            "headers": environment / "Include/site/python3.10" / parts[0].split("-")[0],
        }
        if parts[1] not in schemes:
            raise ValueError("Unknown wheel member install scheme")
        return schemes[parts[1]].joinpath(*parts[2:])
    return environment.joinpath("Lib", "site-packages", *parts)


def preflight(
    root: Path,
    environment: Path,
    wheelhouse: Path | None,
    rows: list[dict[str, str]],
    *,
    offline: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    if environment.exists() or environment.is_symlink():
        raise ValueError("Fresh installation requires an unused VenvPath; refusing overwrite")
    environment = contained(root, environment)
    if offline and wheelhouse is None:
        raise ValueError("Offline installation requires a complete prepared wheelhouse")
    if wheelhouse is not None:
        wheelhouse = contained(root, wheelhouse)
        if not wheelhouse.is_dir():
            raise ValueError("Prepared wheelhouse directory missing")
    artifacts = []
    for row in rows:
        entry: dict[str, Any] = dict(row)
        name = row["filename"]
        if Path(name).name != name or not name.endswith(
            ("-cp310-cp310-win_amd64.whl", "-py3-none-win_amd64.whl")
        ):
            raise ValueError("Original archive tag must support CPython 3.10 Windows x64")
        if wheelhouse is not None:
            archive = contained(root, wheelhouse / name)
            digest = hashlib.sha256()
            with archive.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != row["sha256"]:
                raise ValueError(f"Original archive checksum mismatch: {name}")
            with zipfile.ZipFile(archive) as wheel:
                metadata = [item for item in wheel.namelist() if item.endswith(".dist-info/WHEEL")]
                if len(metadata) != 1:
                    raise ValueError("Exactly one wheel tag metadata member required")
                tags = re.findall(r"^Tag: (.+)$", wheel.read(metadata[0]).decode(), re.MULTILINE)
                if not tags or any(
                    tag.strip() not in {"cp310-cp310-win_amd64", "py3-none-win_amd64"}
                    for tag in tags
                ):
                    raise ValueError("Wheel metadata tag is not CPython 3.10 Windows x64")
                targets = [
                    member_target(environment, item)
                    for item in wheel.namelist()
                    if not item.endswith("/")
                ]
                maximum = max(len(str(target)) for target in targets)
                if maximum >= 260:
                    raise ValueError(f"Wheel member reaches 260-character path boundary: {name}")
                entry.update(verified=True, max_target_length=maximum)
        else:
            entry["verified"] = False
        artifacts.append(entry)
    return {"status": "PASS", "offline": offline, "artifacts": artifacts}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--wheelhouse", type=Path)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    try:
        if sys.platform != "win32" or sys.version_info[:2] != (3, 10) or struct.calcsize("P") != 8:
            raise ValueError("Use 64-bit native Windows CPython 3.10")
        rows = read_lock(
            Path(__file__).resolve().parents[1] / "requirements/windows-rocm7141.lock.txt"
        )
        result = preflight(args.root, args.venv, args.wheelhouse, rows, offline=args.offline)
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
