"""Publication gate CLI: refuse dataset annotation content in published trees.

Modes:
  --staged        scan files staged in the git index (pre-commit hook entrypoint)
  --scan ROOT...  walk each published tree and report every violation
  --paths FILE... scan exactly the given files

Any finding exits non-zero, so this is safe to wire into CI and hooks. The
detector is structural (annotation rows, images, private host paths), not a
file-name blocklist; whitelisted synthetic fixtures are admitted only at
their pinned SHA-256 (unlimited_ocr_2_rocm.publication.WHITELIST).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from unlimited_ocr_2_rocm.publication import (  # noqa: E402
    WHITELIST,
    publication_findings,
    scan_tree,
)


def _staged_files() -> list[Path]:
    out = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [_REPO_ROOT / line for line in out.splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--staged", action="store_true")
    modes.add_argument("--scan", nargs="+", type=Path)
    modes.add_argument("--paths", nargs="+", type=Path)
    args = parser.parse_args(argv)

    violations: list[tuple[str, list[str]]] = []
    if args.staged:
        for path in _staged_files():
            rel = path.relative_to(_REPO_ROOT).as_posix()
            expected_hash = WHITELIST.get(rel)
            if expected_hash is not None:
                continue  # hash-verified by the tree scan in CI
            findings = publication_findings(path) if path.is_file() else []
            if findings:
                violations.append((rel, findings))
    elif args.scan:
        for root in args.scan:
            prefix = ""
            try:
                prefix = root.resolve().relative_to(_REPO_ROOT).as_posix()
            except ValueError:
                prefix = ""  # scanning a tree outside the repository
            violations.extend(scan_tree(root, prefix=prefix))
    else:
        for path in args.paths:
            if path.is_file():
                findings = publication_findings(path)
                if findings:
                    violations.append((str(path), findings))

    if violations:
        print(json.dumps({"publication_gate": "FAIL", "violations": violations}, indent=1))
        return 1
    print(json.dumps({"publication_gate": "PASS"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
