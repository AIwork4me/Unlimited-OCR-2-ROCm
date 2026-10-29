"""Sanitize a selected evidence tree and bind published bytes to original digests."""

import argparse
import json
from pathlib import Path

from reporting import sanitize, sha256, write_json


def export_file(source, target, home):
    source, target = Path(source), Path(target)
    original = source.read_bytes()
    if source.suffix == ".json":

        def clean(value):
            if isinstance(value, str):
                return sanitize(value, home)
            if isinstance(value, list):
                return [clean(item) for item in value]
            if isinstance(value, dict):
                return {sanitize(key, home): clean(item) for key, item in value.items()}
            return value

        value = clean(json.loads(original.decode("utf-8-sig")))
        write_json(target, value)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(sanitize(original.decode("utf-8-sig"), home).encode("utf-8"))
    return {
        "original_sha256": sha256(source),
        "published_sha256": sha256(target),
        "original_size": len(original),
        "published_size": target.stat().st_size,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a fresh public evidence directory")
    records = []
    for source in sorted(args.source.rglob("*")):
        if not source.is_file() or source.suffix not in (".json", ".log", ".txt"):
            continue
        relative = source.relative_to(args.source)
        record = export_file(source, args.output / relative, str(Path.home()))
        records.append({"path": relative.as_posix(), **record})
    write_json(args.output / "export-manifest.json", records)
    paths = sorted(p for p in args.output.rglob("*") if p.is_file())
    (args.output / "SHA256SUMS.txt").write_text(
        "".join(f"{sha256(p)}  {p.relative_to(args.output).as_posix()}\n" for p in paths),
        encoding="utf-8",
    )
    print("Exported", len(records), "sanitized artifacts; original digests retained")


if __name__ == "__main__":
    main()
