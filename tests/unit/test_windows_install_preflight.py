"""CPU-only archive and PowerShell install safety contracts."""

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import uuid
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/windows_install_preflight.py"


def verifier():
    spec = importlib.util.spec_from_file_location("windows_install_preflight", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def archive(tmp_path, member="torch/lib/test.dll", tag="cp310-cp310-win_amd64"):
    house = tmp_path / "wheels"
    house.mkdir()
    name = f"torch-2.12.0+rocm7.14.1-{tag}.whl"
    path = house / name
    with zipfile.ZipFile(path, "w") as wheel:
        wheel.writestr(member, b"test")
        wheel.writestr("torch-2.12.0+rocm7.14.1.dist-info/WHEEL", f"Tag: {tag}\n")
    row = {
        "filename": name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "url": "https://repo.amd.com/rocm/whl-multi-arch/" + name,
    }
    return house, row


def test_original_lock_contains_nine_known_archives():
    rows = verifier().read_lock(REPO / "requirements/windows-rocm7141.lock.txt")
    assert len(rows) == 9
    torch = next(row for row in rows if row["filename"].startswith("torch-"))
    assert torch["sha256"] == "415d2dbd4b6ab002cff1c54795fdd66322abd9a46bf982cb64eebb4493ac2e7c"
    assert all(row["url"].startswith("https://repo.amd.com/rocm/whl-multi-arch/") for row in rows)


def test_verified_allowlist_and_data_member_mapping(tmp_path):
    house, row = archive(tmp_path, "torch.data/scripts/test.exe")
    env = tmp_path / "new"
    result = verifier().preflight(tmp_path, env, house, [row])
    assert result["status"] == "PASS"
    assert result["artifacts"][0]["sha256"] == row["sha256"]
    assert result["artifacts"][0]["max_target_length"] == max(
        len(str(env / "Scripts/test.exe")),
        len(str(env / "Lib/site-packages/torch-2.12.0+rocm7.14.1.dist-info/WHEEL")),
    )
    assert not env.exists()


@pytest.mark.parametrize("failure", ["missing", "tampered", "existing", "escape", "long"])
def test_preflight_refuses_before_environment_creation(tmp_path, failure):
    module = verifier()
    house, row = archive(tmp_path)
    env = tmp_path / "new"
    if failure == "missing":
        (house / row["filename"]).unlink()
    elif failure == "tampered":
        with (house / row["filename"]).open("ab") as stream:
            stream.write(b"changed")
    elif failure == "existing":
        env.mkdir()
    elif failure == "escape":
        env = tmp_path.parent / "outside"
    else:
        env = tmp_path / ("x" * 160)
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        module.preflight(tmp_path, env, house, [row])
    assert not (env / "pyvenv.cfg").exists()


@pytest.mark.parametrize("member", ["../escape.dll", "torch.data/data/../../escape.dll"])
def test_archive_traversal_refused(tmp_path, member):
    house, row = archive(tmp_path, member)
    with pytest.raises(ValueError, match="member"):
        verifier().preflight(tmp_path, tmp_path / "new", house, [row])


def test_wrong_platform_tag_refused_even_with_matching_hash(tmp_path):
    house, row = archive(tmp_path, tag="cp311-cp311-win_amd64")
    with pytest.raises(ValueError, match="tag"):
        verifier().preflight(tmp_path, tmp_path / "new", house, [row])


def test_offline_requires_wheelhouse_without_creating_venv(tmp_path):
    with pytest.raises(ValueError, match="wheelhouse"):
        verifier().preflight(tmp_path, tmp_path / "new", None, [], offline=True)
    assert not (tmp_path / "new").exists()


@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell Windows setup")
@pytest.mark.parametrize("mode", ["offline-missing", "existing"])
def test_powershell_rejects_before_install_and_restores_on_failure(tmp_path, mode):
    env = Path.cwd() / ".b" / ("pf" + uuid.uuid4().hex[:8])
    if mode == "existing":
        # The repository's short cache root exists; preflight must not alter it.
        env = Path.cwd() / ".b"
        assert env.is_dir()
    flag = "-Offline" if mode == "offline-missing" else ""
    command = (
        "$ErrorActionPreference='Stop'; $env:PYTHONIOENCODING='caller-encoding'; "
        "try { "
        f"& '{REPO / 'scripts/setup_windows_rocm7141.ps1'}' "
        f"-Python '{sys.executable}' -VenvPath '{env}' -PreflightOnly {flag}; "
        "throw 'Unexpected success' } catch { "
        "if ($_.Exception.Message -eq 'Unexpected success') { throw }; "
        "if (-not $_.Exception.Message.Contains('Python command failed')) { throw }; "
        "if ($env:PYTHONIOENCODING -ne 'caller-encoding') {throw 'environment leaked'} }; exit 0"
    )
    proc = subprocess.run(
        [shutil.which("powershell.exe"), "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not (env / "pyvenv.cfg").exists()


@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell script parser")
def test_both_scripts_parse_in_windows_powershell():
    for name in ("setup_windows_rocm7141.ps1", "reproduce_windows.ps1"):
        command = (
            "$tokens=$null; $errors=$null; "
            "[System.Management.Automation.Language.Parser]::ParseFile("
            f"'{REPO / 'scripts' / name}',[ref]$tokens,[ref]$errors) | Out-Null; "
            "if ($errors.Count) {$errors | Out-String | Write-Error; exit 1}"
        )
        proc = subprocess.run(
            [shutil.which("powershell.exe"), "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell Windows setup")
def test_powershell_preflight_only_is_cpu_only_and_restores_environment(tmp_path):
    powershell = shutil.which("powershell.exe")
    env = Path.cwd() / ".b" / ("pf" + uuid.uuid4().hex[:8])
    command = (
        "$ErrorActionPreference='Stop'; $env:TEMP='caller-temp'; "
        f"& '{REPO / 'scripts/setup_windows_rocm7141.ps1'}' "
        f"-Python '{sys.executable}' -VenvPath '{env}' -PreflightOnly; "
        "if ($env:TEMP -ne 'caller-temp') {throw 'environment leaked'}"
    )
    proc = subprocess.run(
        [powershell, "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    start = proc.stdout.index("{")
    assert json.loads(proc.stdout[start:])["status"] == "PASS"
    assert not env.exists()
