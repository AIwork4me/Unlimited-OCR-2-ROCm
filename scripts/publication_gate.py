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
    parse_staged_name_status,
    publication_findings,
    scan_staged_files,
    scan_tree,
)


def _git(*args: str) -> bytes:
    return subprocess.run(  # noqa: S603 - fixed argv from this code base
        ["git", "-C", str(_REPO_ROOT), *args],
        check=True,
        capture_output=True,
    ).stdout


def _staged_entries() -> list[tuple[str, bytes | None]]:
    """Index-side staged content: (path, blob bytes or None when unreadable).

    Reads the INDEX, not the worktree (TOCTOU-proof), with -z output so
    quoted/non-ASCII paths cannot slip through unscanned, and with rename/
    typechange statuses included (deletions commit no content). Unmerged
    index entries fail closed.
    """
    if _git("ls-files", "-u", "-z"):
        return [("<unmerged index entries present>", None)]
    entries: list[tuple[str, bytes | None]] = []
    for path, must_exist in parse_staged_name_status(
        _git("diff", "--cached", "--name-status", "-z", "--diff-filter=ACMRT")
    ):
        try:
            blob = _git("cat-file", "blob", f":{path}")
        except subprocess.CalledProcessError:
            if not must_exist:
                continue  # rename source removed from the index: nothing commits
            blob = None  # exists in the index but unreadable: fail closed
        entries.append((path, blob))
    return entries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--staged", action="store_true")
    modes.add_argument("--scan", nargs="+", type=Path)
    modes.add_argument("--paths", nargs="+", type=Path)
    args = parser.parse_args(argv)

    violations: list[tuple[str, list[str]]] = []
    if args.staged:
        violations.extend(scan_staged_files(_staged_entries()))
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
