"""A source-built torch-only candidate must independently prove its identity."""

import argparse
import ast
import copy
import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).parents[2] / "experiments/pytorch212_wheel/candidate_identity.py"
SPEC = importlib.util.spec_from_file_location("candidate_identity", PATH)
assert SPEC and SPEC.loader
identity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(identity)


def facts():
    p = {
        "schema": 1,
        "source_sha": "60dd50f143fd9b10ac23cde8cdd34be61232b139",
        "therock_sha": "6d34c9b6f3e752b8f718b0c81f3070df77ce3283",
        "aotriton_recipe_pin": "7a6963a7265d2cf0fc48c6a63464386134d32483",
        "wheel_sha256": "a" * 64,
        "torch_version": "2.12.0+rocm7.14.1",
        "hip": "7.14.60850",
        "architecture": "gfx1151",
        "python_abi": "cp310",
        "platform": "win_amd64",
        "prefix": "C:/b/new",
        "binaries": {
            "torch/lib/aotriton_v2.dll": "b" * 64,
            "torch/lib/torch_hip.dll": "c" * 64,
            "_rocm_sdk_core/bin/amdhip64_7.dll": "d" * 64,
            **{
                "torch/lib/" + name: "b" * 64
                for name in (
                    "torch_cpu.dll",
                    "torch_python.dll",
                    "c10.dll",
                    "c10_hip.dll",
                    "dl.dll",
                    "liblzma.dll",
                    "nnapi_backend.dll",
                )
            },
            "torch/_C.cp310-win_amd64.pyd": "c" * 64,
        },
        "images": {"torch/lib/aotriton.images/amd-gfx115x/attn_fwd.zip": "e" * 64},
        "image_signatures": [{"TRITON_VERSION": "aotriton0.14"}],
        "runtime_header_sha256": "f" * 64,
    }
    observed = copy.deepcopy(p)
    observed.update(
        artifact_binaries={k: v for k, v in p["binaries"].items() if k.startswith("torch/")},
        loaded_binaries={k: v for k, v in p["binaries"].items() if "nnapi_backend" not in k},
        import_within_venv=True,
        loaded_binaries_within_venv=True,
        imported_git_version=p["source_sha"],
        source_overrides={},
        packages={"torch": p["torch_version"]},
        isolated_venv=True,
        pip_check_exit=0,
        pip_check_prefix="C:/b/new",
        pip_check_command=["C:/b/new/Scripts/python.exe", "-m", "pip", "check"],
        installed_wheel_sha256=p["wheel_sha256"],
        wheel_hash_recomputed=True,
    )
    return p, observed


def test_torch_only_candidate_does_not_need_old_audio_or_vision():
    profile, runtime = facts()
    identity.validate_identity(profile, runtime)
    assert "torchvision" not in runtime["packages"]
    assert "torchaudio" not in runtime["packages"]


@pytest.mark.parametrize(
    "member",
    [
        "torch/lib/torch_cpu.dll",
        "torch/lib/torch_python.dll",
        "torch/lib/c10.dll",
        "torch/lib/c10_hip.dll",
        "torch/_C.cp310-win_amd64.pyd",
    ],
)
def test_foreign_loaded_core_binary_is_rejected(member):
    p, r = facts()
    value = r["loaded_binaries"].pop(member)
    r["loaded_binaries"]["foreign/" + member.split("/")[-1]] = value
    with pytest.raises(ValueError):
        identity.validate_identity(p, r)


def test_required_cpu_reference_binary_cannot_be_omitted():
    p, r = facts()
    for inventory in (p["binaries"], r["binaries"], r["loaded_binaries"]):
        inventory.pop("torch/lib/torch_cpu.dll")
    with pytest.raises(ValueError):
        identity.validate_identity(p, r)


def test_core_binary_must_actually_be_loaded():
    p, r = facts()
    r["loaded_binaries"].pop("torch/lib/torch_cpu.dll")
    with pytest.raises(ValueError):
        identity.validate_identity(p, r)


def test_unloaded_optional_binary_still_has_its_installed_hash_checked():
    p, r = facts()
    identity.validate_identity(p, r)
    r["binaries"]["torch/lib/nnapi_backend.dll"] = "0" * 64
    with pytest.raises(ValueError):
        identity.validate_identity(p, r)


def test_profile_cannot_omit_an_optional_artifact_binary():
    p, r = facts()
    p["binaries"].pop("torch/lib/nnapi_backend.dll")
    r["binaries"].pop("torch/lib/nnapi_backend.dll")
    with pytest.raises(ValueError):
        identity.validate_identity(p, r)


@pytest.mark.parametrize(
    "field,bad",
    [
        ("imported_git_version", "0" * 40),
        ("wheel_sha256", "0" * 64),
        ("installed_wheel_sha256", "0" * 64),
        ("hip", "7.15.0"),
        ("architecture", "gfx1100"),
        ("python_abi", "cp313"),
        ("platform", "linux_x86_64"),
        ("import_within_venv", False),
        ("loaded_binaries_within_venv", False),
        ("wheel_hash_recomputed", False),
        ("source_overrides", {"AOTRITON_INSTALLED_PREFIX": "custom"}),
        ("pip_check_exit", 1),
        ("runtime_header_sha256", "0" * 64),
        ("image_signatures", []),
        ("prefix", "C:/other"),
        ("isolated_venv", False),
        ("pip_check_prefix", "C:/b/qa"),
        ("pip_check_command", ["C:/b/qa/Scripts/python.exe", "-m", "pip", "check"]),
    ],
)
def test_candidate_rejects_drift(field, bad):
    p, r = facts()
    r[field] = bad
    with pytest.raises(ValueError):
        identity.validate_identity(p, r)


@pytest.mark.parametrize("field", ["binaries", "images"])
def test_candidate_rejects_changed_or_missing_payload(field):
    p, r = facts()
    first = next(iter(r[field]))
    r[field][first] = "0" * 64
    with pytest.raises(ValueError):
        identity.validate_identity(p, r)
    r[field] = {}
    with pytest.raises(ValueError):
        identity.validate_identity(p, r)


def test_profile_cannot_redefine_frozen_source():
    p, r = facts()
    p["source_sha"] = r["source_sha"] = r["imported_git_version"] = "0" * 40
    with pytest.raises(ValueError):
        identity.validate_identity(p, r)


def test_candidate_adapter_changes_only_baseline_identity_nodes():
    frozen = PATH.with_name("repro_sdpa.py").read_bytes()
    adapted, changes = identity.adapt_identity_only(frozen)
    assert changes == ["baseline_expected_versions", "baseline_identity_predicate"]
    assert "make_fixture" in adapted
    assert "reference_attention" in adapted
    assert "numerical_stats" in adapted
    before, after = ast.parse(frozen), ast.parse(adapted)
    for old, new in zip(before.body, after.body, strict=True):
        if isinstance(old, ast.FunctionDef) and old.name == "execute":
            differences = [
                (left, right)
                for left, right in zip(old.body, new.body, strict=True)
                if ast.dump(left) != ast.dump(right)
            ]
            assert len(differences) == 2
            assert isinstance(differences[0][0], ast.Assign)
            assert isinstance(differences[0][1], ast.Expr)
            assert isinstance(differences[1][0], ast.If)
            assert isinstance(differences[1][1], ast.Pass)
        else:
            assert ast.dump(old) == ast.dump(new)
    with pytest.raises(ValueError):
        identity.adapt_identity_only(frozen + b"\n# changed\n")


@pytest.mark.parametrize(
    "flag", ["--clear-error-nonacceptance", "--clear-error", "--clear-error-nonacceptanc"]
)
def test_abbreviated_error_clearing_is_rejected_after_parsing(flag):
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear-error-nonacceptance", action="store_true")
    with pytest.raises(ValueError):
        identity.validate_acceptance_arguments(parser.parse_args([flag]))
