"""Strict torch-only candidate profile and a frozen identity-only AST adapter."""

import ast
import hashlib
import re
from typing import Any

SOURCE_SHA = "60dd50f143fd9b10ac23cde8cdd34be61232b139"
THEROCK_SHA = "6d34c9b6f3e752b8f718b0c81f3070df77ce3283"
AOTRITON_PIN = "7a6963a7265d2cf0fc48c6a63464386134d32483"
REPRO_SHA256 = "9673a18c38e48b1a5969d04b50ef691c13a8a6099092150ae5eda2d0380baf13"
REQUIRED_LOADED_BINARIES = {
    "aotriton_v2.dll",
    "torch_hip.dll",
    "torch_cpu.dll",
    "torch_python.dll",
    "c10.dll",
    "c10_hip.dll",
    "_c.cp310-win_amd64.pyd",
    "dl.dll",
    "liblzma.dll",
    "amdhip64_7.dll",
}


def validate_acceptance_arguments(args: Any) -> None:
    if getattr(args, "clear_error_nonacceptance", False):
        raise ValueError("Candidate acceptance refuses parsed error-clearing requests")


def validate_identity(profile: dict[str, Any], runtime: dict[str, Any]) -> None:
    """Reject mixed installations; audio/vision are intentionally not required."""
    required = {
        "schema": 1,
        "source_sha": SOURCE_SHA,
        "therock_sha": THEROCK_SHA,
        "aotriton_recipe_pin": AOTRITON_PIN,
        "hip": "7.14.60850",
        "architecture": "gfx1151",
        "python_abi": "cp310",
        "platform": "win_amd64",
    }
    for key, expected in required.items():
        if profile.get(key) != expected or runtime.get(key) != expected:
            raise ValueError(f"Candidate identity mismatch: {key}")
    if runtime.get("imported_git_version") != SOURCE_SHA:
        raise ValueError("Imported torch is not the frozen source artifact")
    for field in (
        "import_within_venv",
        "loaded_binaries_within_venv",
        "wheel_hash_recomputed",
        "isolated_venv",
    ):
        if runtime.get(field) is not True:
            raise ValueError(f"Candidate isolation not established: {field}")
    if (
        runtime.get("prefix", "").replace("\\", "/").lower()
        != profile.get("prefix", "").replace("\\", "/").lower()
    ):
        raise ValueError("Candidate imported from an unexpected environment")
    if runtime.get("pip_check_exit") != 0 or runtime.get("source_overrides") != {}:
        raise ValueError("Candidate dependency/override gate failed")
    normal = str(profile.get("prefix", "")).replace("\\", "/").lower()
    if str(runtime.get("pip_check_prefix", "")).replace("\\", "/").lower() != normal:
        raise ValueError("pip check was not run in the candidate environment")
    command = runtime.get("pip_check_command", [])
    if (
        len(command) != 4
        or command[1:] != ["-m", "pip", "check"]
        or (command[0].replace("\\", "/").lower() != normal.rstrip("/") + "/scripts/python.exe")
    ):
        raise ValueError("pip check interpreter does not match candidate prefix")
    digest = profile.get("wheel_sha256", "")
    if not re.fullmatch(r"[0-9a-f]{64}", digest) or any(
        runtime.get(k) != digest for k in ("wheel_sha256", "installed_wheel_sha256")
    ):
        raise ValueError("Candidate built/installed/recomputed wheel hashes disagree")
    if not profile.get("torch_version") or (
        runtime.get("torch_version") != profile["torch_version"]
        or runtime.get("packages", {}).get("torch") != profile["torch_version"]
    ):
        raise ValueError("Imported and installed torch versions disagree")
    for field in ("binaries", "images"):
        values = profile.get(field, {})
        if not values or values != runtime.get(field):
            raise ValueError(f"Candidate {field} differ from artifact inventory")
        if any(not re.fullmatch(r"[0-9a-f]{64}", h) for h in values.values()):
            raise ValueError(f"Invalid {field} digest")
    artifact = {k: v for k, v in profile["binaries"].items() if k.startswith("torch/")}
    if not artifact or artifact != runtime.get("artifact_binaries"):
        raise ValueError("Native profile does not cover the actual wheel inventory")
    loaded = runtime.get("loaded_binaries", {})
    if not loaded or any(profile["binaries"].get(k) != v for k, v in loaded.items()):
        raise ValueError("Loaded native binaries differ from their artifact paths or hashes")
    names = {p.replace("\\", "/").split("/")[-1].lower() for p in loaded}
    if len(names) != len(loaded) or not REQUIRED_LOADED_BINARIES <= names:
        raise ValueError("Required loaded torch/ROCm/AOTriton core binaries absent or duplicated")
    for field in ("image_signatures", "runtime_header_sha256"):
        if not profile.get(field) or profile[field] != runtime.get(field):
            raise ValueError(f"Candidate {field} differs from packaged identity")


def adapt_identity_only(source: bytes) -> tuple[str, list[str]]:
    """Replace two known baseline identity nodes; preserve every other AST node."""
    if hashlib.sha256(source).hexdigest() != REPRO_SHA256:
        raise ValueError("Frozen baseline producer digest changed; adaptation refused")
    tree = ast.parse(source)
    execute = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "execute")
    changed = []
    for index, node in enumerate(execute.body):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "expected" for t in node.targets
        ):
            expected = ast.literal_eval(node.value)
            if expected != {
                "torch": "2.12.0+rocm7.14.1",
                "torchvision": "0.27.0+rocm7.14.1",
                "torchaudio": "2.11.0+rocm7.14.1",
            }:
                raise ValueError("Baseline identity node no longer matches")
            execute.body[index] = ast.parse(
                '_candidate_identity_check(report["environment"])'
            ).body[0]
            changed.append("baseline_expected_versions")
        elif isinstance(node, ast.If) and any(
            isinstance(n, ast.Constant)
            and n.value == "Mandated baseline identity mismatch; experiment refused"
            for n in ast.walk(node)
        ):
            execute.body[index] = ast.Pass()
            changed.append("baseline_identity_predicate")
    if changed != ["baseline_expected_versions", "baseline_identity_predicate"]:
        raise ValueError("Identity-only adaptation did not identify exactly two nodes")
    ast.fix_missing_locations(tree)
    return ast.unparse(tree), changed
