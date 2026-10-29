"""Capture declared upstream source pins without asserting a build occurred."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

OVERRIDES = (
    "PYTORCH_AOTRITON_COMMIT",
    "PYTORCH_AOTRITON_BASE_URL",
    "AOTRITON_INSTALLED_PREFIX",
    "AOTRITON_INSTALL_FROM_SOURCE",
)


def git(path: Path, *argv: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *argv], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pytorch", type=Path, required=True)
    parser.add_argument("--therock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    recipe = args.pytorch / "cmake/External/aotriton.cmake"
    text = recipe.read_text(encoding="utf-8")
    version = re.search(r'set\(__AOTRITON_VER "([^"]+)"', text)
    pin = re.search(r'set\(__AOTRITON_CI_COMMIT "([0-9a-f]{40})"', text)
    image = re.search(r'"([0-9a-f]{64})"\s*# amd-gfx115x', text)
    if not version or not pin or not image:
        raise ValueError("Cannot parse required AOTriton pins; do not guess")
    sources = {}
    for name, path, branch in (
        ("pytorch", args.pytorch, "release/2.12"),
        ("therock", args.therock, "main"),
    ):
        sources[name] = {
            "head": git(path, "rev-parse", "HEAD"),
            "remote": git(path, "remote", "get-url", "origin"),
            "branch_observed": branch,
            "status": git(path, "status", "--porcelain"),
            "remote_head_at_collection": git(path, "ls-remote", "origin", "refs/heads/" + branch),
        }
    report = {
        "run_id": str(uuid.uuid4()),
        "utc": datetime.now(timezone.utc).isoformat(),
        "sources": sources,
        "torch_source_version": (args.pytorch / "version.txt").read_text().strip(),
        "aotriton_version": version[1],
        "aotriton_source_pin": pin[1],
        "gfx115x_image_sha256_declared": image[1],
        "gfx115x_image_url": f"https://github.com/ROCm/aotriton/releases/download/{version[1]}/aotriton-{version[1]}-images-amd-gfx115x.tar.gz",
        "aotriton_recipe_sha256": hashlib.sha256(recipe.read_bytes()).hexdigest(),
        "source_overrides": {k: os.environ[k] for k in OVERRIDES if k in os.environ},
        "claim_limit": (
            "Source declarations only; downloaded/compiled/installed identities "
            "require separate evidence."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
