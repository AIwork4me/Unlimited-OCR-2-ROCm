"""Derive a candidate identity profile from the actual torch and SDK wheel bytes."""

import argparse
import ast
import hashlib
import json
import sys
import zipfile
from email.parser import Parser
from pathlib import Path

from candidate_identity import AOTRITON_PIN, SOURCE_SHA, THEROCK_SHA, validate_identity
from candidate_repro import digest, runtime_facts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--sdk-core-wheel", type=Path, required=True)
    parser.add_argument("--install-report", type=Path, required=True)
    parser.add_argument("--pip-check-record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with zipfile.ZipFile(args.wheel) as wheel:
        tags = wheel.read(
            next(n for n in wheel.namelist() if n.endswith(".dist-info/WHEEL"))
        ).decode()
        if "Tag: cp310-cp310-win_amd64" not in tags:
            raise ValueError("Candidate is not a Windows AMD64 cp310 wheel")
        metadata = Parser().parsestr(
            wheel.read(
                next(n for n in wheel.namelist() if n.endswith(".dist-info/METADATA"))
            ).decode()
        )
        if metadata["Name"].lower() != "torch":
            raise ValueError("Expected the built torch wheel")
        tree = ast.parse(wheel.read("torch/version.py"))
        values = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in ("git_version", "hip"):
                        values[target.id] = ast.literal_eval(node.value)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.target.id in ("git_version", "hip"):
                    if node.value is None:
                        raise ValueError("Candidate version.py has no identity value")
                    values[node.target.id] = ast.literal_eval(node.value)
        if values.get("git_version") != SOURCE_SHA or values.get("hip") != "7.14.60850":
            raise ValueError("Built wheel version.py does not identify the frozen source and SDK")
        binaries = {
            n: hashlib.sha256(wheel.read(n)).hexdigest()
            for n in wheel.namelist()
            if n.startswith("torch/") and n.lower().endswith((".dll", ".pyd"))
        }
        images = {
            n: hashlib.sha256(wheel.read(n)).hexdigest()
            for n in wheel.namelist()
            if n.startswith("torch/lib/aotriton.images/") and not n.endswith("/")
        }
        signatures = [
            json.loads(wheel.read(n)) for n in sorted(images) if n.endswith("/__signature__")
        ]
        header = wheel.read("torch/include/aotriton/config.h")
    if (
        digest(args.sdk_core_wheel)
        != "49eb83cb89d50562944a1f0eaf65f4e21f4557813de53e78d022fee81bbfb6e7"
    ):
        raise ValueError("SDK core wheel differs from the verified 7.14.1 artifact")
    with zipfile.ZipFile(args.sdk_core_wheel) as sdk:
        name = "_rocm_sdk_core/bin/amdhip64_7.dll"
        binaries[name] = hashlib.sha256(sdk.read(name)).hexdigest()
    profile = {
        "schema": 1,
        "source_sha": SOURCE_SHA,
        "therock_sha": THEROCK_SHA,
        "aotriton_recipe_pin": AOTRITON_PIN,
        "wheel_path": str(args.wheel.resolve()),
        "wheel_sha256": digest(args.wheel),
        "wheel_size": args.wheel.stat().st_size,
        "torch_version": metadata["Version"],
        "hip": values["hip"],
        "architecture": "gfx1151",
        "python_abi": "cp310",
        "platform": "win_amd64",
        "prefix": sys.prefix,
        "binaries": binaries,
        "images": images,
        "image_signatures": signatures,
        "runtime_header_sha256": hashlib.sha256(header).hexdigest(),
        "runtime_header_text": header.decode(),
        "install_report": str(args.install_report.resolve()),
        "pip_check_record": str(args.pip_check_record.resolve()),
        "claim_limit": (
            "Binary/image identities are matched to artifact bytes; "
            "compatibility requires GPU acceptance."
        ),
    }
    import torch

    props = torch.cuda.get_device_properties(0)
    observed = runtime_facts(profile, {"architecture": props.gcnArchName})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".observed.json").write_text(json.dumps(observed, indent=2) + "\n")
    validate_identity(profile, observed)
    args.output.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    print("Candidate identity established:", profile["wheel_sha256"])


if __name__ == "__main__":
    main()
