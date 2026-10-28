"""Candidate-only identity adapter around the unchanged baseline GPU experiment."""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

from candidate_identity import adapt_identity_only, validate_acceptance_arguments, validate_identity


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def runtime_facts(profile: dict[str, Any], environment: dict[str, Any]) -> dict[str, Any]:
    import torch

    prefix = Path(sys.prefix).resolve()
    site = prefix / "Lib/site-packages"
    wheel = Path(profile["wheel_path"])
    install = json.loads(Path(profile["install_report"]).read_text(encoding="utf-8-sig"))
    entries = [e for e in install["install"] if e["metadata"]["name"].lower() == "torch"]
    if len(entries) != 1:
        raise ValueError("Install report must identify exactly one explicit torch artifact")
    origin = entries[0]["download_info"]
    if not origin["url"].startswith("file:"):
        raise ValueError("Candidate torch was not installed from an explicit local artifact")
    installed_hash = origin["archive_info"]["hashes"]["sha256"]
    # Keep the legacy helper byte-identical; do not type-check it as new code.
    modules = [Path(p) for p in importlib.import_module("collect_provenance").loaded_modules()]
    wanted = {Path(n).name.lower() for n in profile["binaries"]}
    selected = [p.resolve() for p in modules if p.name.lower() in wanted]
    if len(selected) != len(wanted) or len({p.name.lower() for p in selected}) != len(wanted):
        raise ValueError("Missing or duplicate candidate runtime modules")
    inside = all(p.is_relative_to(site) for p in selected)
    binaries = {p.relative_to(site).as_posix(): digest(p) for p in selected} if inside else {}
    image_root = site / "torch/lib/aotriton.images"
    images = {
        p.relative_to(site).as_posix(): digest(p) for p in image_root.rglob("*") if p.is_file()
    }
    signatures = [json.loads(p.read_text()) for p in sorted(image_root.rglob("__signature__"))]
    check_command = [sys.executable, "-m", "pip", "check"]
    check = subprocess.run(check_command, capture_output=True, text=True, check=False)
    facts = {
        "schema": profile["schema"],
        "prefix": sys.prefix,
        "isolated_venv": sys.prefix != sys.base_prefix,
        "source_sha": profile["source_sha"],
        "therock_sha": profile["therock_sha"],
        "aotriton_recipe_pin": profile["aotriton_recipe_pin"],
        "wheel_sha256": digest(wheel),
        "installed_wheel_sha256": installed_hash,
        "wheel_hash_recomputed": True,
        "torch_version": str(torch.__version__),
        "imported_git_version": torch.version.git_version,
        "hip": torch.version.hip,
        "architecture": environment["architecture"],
        "python_abi": "cp310" if sys.version_info[:2] == (3, 10) else "WRONG_ABI",
        "platform": "win_amd64"
        if sys.platform == "win32" and struct.calcsize("P") == 8
        else "WRONG_PLATFORM",
        "import_within_venv": Path(torch.__file__).resolve().is_relative_to(site),
        "loaded_binaries_within_venv": inside,
        "binaries": binaries,
        "images": images,
        "runtime_header_sha256": digest(site / "torch/include/aotriton/config.h"),
        "image_signatures": signatures,
        "pip_check_exit": check.returncode,
        "pip_check_command": check_command,
        "pip_check_prefix": sys.prefix,
        "pip_check_stdout": check.stdout,
        "pip_check_stderr": check.stderr,
        "packages": {
            d.metadata["Name"].lower(): d.version for d in importlib.metadata.distributions()
        },
        "source_overrides": {
            k: v
            for k, v in os.environ.items()
            if "AOTRITON" in k or k == "HSA_OVERRIDE_GFX_VERSION"
        },
        "torch_file": str(torch.__file__),
        "loaded_paths": [str(p) for p in selected],
    }
    return facts


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--identity-profile", type=Path, required=True)
    options, remaining = parser.parse_known_args()
    original_command = list(sys.argv)
    if "--clear-error-nonacceptance" in remaining:
        raise ValueError("Candidate acceptance refuses error clearing")
    profile = json.loads(options.identity_profile.read_text(encoding="utf-8-sig"))
    frozen = Path(__file__).with_name("repro_sdpa.py")
    source, changes = adapt_identity_only(frozen.read_bytes())
    evidence: dict[str, Any] = {}

    def check(environment: dict[str, Any]) -> None:
        observed = runtime_facts(profile, environment)
        evidence.update(observed)
        validate_identity(profile, observed)

    namespace: dict[str, Any] = {
        "__name__": "frozen_candidate_experiment",
        "__file__": str(frozen),
        "_candidate_identity_check": check,
    }
    exec(compile(source, str(frozen), "exec"), namespace)
    execute = namespace["execute"]

    def candidate_execute(args: Any, report: dict[str, Any]) -> None:
        validate_acceptance_arguments(args)
        report.update(
            identity_profile_sha256=digest(options.identity_profile),
            candidate_adapter_sha256=digest(Path(__file__)),
            candidate_identity_gate_sha256=digest(
                Path(__file__).with_name("candidate_identity.py")
            ),
            identity_only_ast_changes=changes,
            candidate_command=original_command,
            candidate_identity=evidence,
        )
        execute(args, report)

    namespace["execute"] = candidate_execute
    sys.argv = [sys.argv[0], *remaining]
    return int(namespace["main"]())


if __name__ == "__main__":
    raise SystemExit(main())
