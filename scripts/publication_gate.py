"""Publication gate CLI: refuse dataset annotation content in published trees.

Modes:
  --staged        scan files staged in the git index (pre-commit hook entrypoint)
  --scan ROOT...  walk each published tree and report every violation
  --paths FILE... scan exactly the given files

Any finding exits non-zero, so this is safe to wire into CI and hooks. The
detector is structural (annotation rows, images, opaque binaries, private host
paths, host-identity fields, credential material), not a file-name blocklist;
whitelisted files are admitted only at their pinned SHA-256
(unlimited_ocr_2_rocm.publication.WHITELIST).

Scan roots are anchored to the repository root derived from THIS script's
location, so ``--scan .`` and ``--scan $PWD`` anchor whitelist lookups the
same way (the PR #24 gate resolved ``.`` to a ``./``-prefixed prefix and
missed every whitelist entry; this CLI fixes that anchoring).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from unlimited_ocr_2_rocm.publication import (  # noqa: E402
    scan_staged,
    scan_tree,
)


def _anchored_prefix(root: Path) -> str:
    try:
        rel = root.resolve().relative_to(_REPO_ROOT)
    except ValueError:
        return ""
    posix = rel.as_posix()
    return "" if posix in ("", ".") else posix


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--staged", action="store_true")
    modes.add_argument("--scan", nargs="+", type=Path)
    modes.add_argument("--paths", nargs="+", type=Path)
    args = parser.parse_args(argv)

    violations: list[tuple[str, list[str]]] = []
    if args.staged:
        violations.extend(scan_staged(_REPO_ROOT))
    elif args.scan:
        for root in args.scan:
            violations.extend(scan_tree(root, prefix=_anchored_prefix(root)))
    else:
        from unlimited_ocr_2_rocm.publication import publication_findings

        for path in args.paths:
            if not path.is_file():
                # Refuse silently-passing inputs: a directory or missing path
                # must never look like a clean scan.
                violations.append((str(path), ["not a readable file"]))
                continue
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
