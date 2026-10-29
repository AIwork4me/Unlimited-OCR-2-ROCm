"""Publish selected text files with profile redaction and two independent digests."""

import argparse
import hashlib
import json
import re
from pathlib import Path


def sanitize_text(text: str, home: str) -> str:
    normal = home.replace("\\", "/").rstrip("/")
    variants = {normal, normal.replace("/", "\\"), normal.replace("/", "\\\\")}
    for prefix in sorted(variants, key=len, reverse=True):
        text = re.sub(re.escape(prefix), lambda _: "<USERPROFILE>", text, flags=re.IGNORECASE)

    # CMake's verbose HTTP traces can contain temporary signed asset redirects.
    def redact_url(match: re.Match[str]) -> str:
        url = match.group(0)
        if "?" in url and re.search(r"(?:sig|jwt|signature|x-amz-credential)=", url, re.I):
            return url.split("?", 1)[0] + "?<SIGNED_QUERY_REDACTED>"
        return url

    text = re.sub(r'https?://[^\s<>"\x27]+', redact_url, text)
    text = re.sub(r'(?<![^\s<>"\x27])/[^\s<>"\x27]+\?[^\s<>"\x27]+', redact_url, text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for src in sorted(args.source.rglob("*")):
        if not src.is_file() or src.suffix not in {".json", ".txt", ".log"}:
            continue
        relative = src.relative_to(args.source)
        raw = src.read_bytes()
        published = sanitize_text(raw.decode("utf-8-sig"), str(Path.home())).encode("utf-8")
        # JSON is validated before publication; no malformed sanitization accepted.
        if src.suffix == ".json":
            json.loads(published)
        dst = args.destination / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(published)
        records.append(
            {
                "file": relative.as_posix(),
                "original_sha256": hashlib.sha256(raw).hexdigest(),
                "published_sha256": hashlib.sha256(published).hexdigest(),
                "original_bytes": len(raw),
                "published_bytes": len(published),
            }
        )
    args.destination.mkdir(parents=True, exist_ok=True)
    (args.destination / "export-manifest.json").write_text(
        json.dumps({"files": records}, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
