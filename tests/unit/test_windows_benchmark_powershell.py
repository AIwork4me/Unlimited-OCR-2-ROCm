"""Exercise the Windows entry point without torch or a model."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/run_windows_benchmark.ps1"
POWERSHELL = shutil.which("powershell.exe")
pytestmark = pytest.mark.skipif(os.name != "nt" or not POWERSHELL, reason="native PowerShell")


def invoke(*args):
    return subprocess.run(
        [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SCRIPT), *args],
        capture_output=True,
        timeout=20,
    )


def test_gpu_execution_requires_explicit_switch():
    result = invoke("-Python", sys.executable, "-Action", "run")
    assert result.returncode != 0
    assert b"ExecuteGpu" in result.stderr


def test_report_help_delegates_to_shared_runner():
    result = invoke("-Python", sys.executable, "-Action", "report", "-RunnerArguments", "--help")
    assert result.returncode == 0, result.stderr
    assert b"--selection" in result.stdout and b"--output" in result.stdout


def test_runtime_directory_cannot_escape_repository():
    outside = Path.cwd().parent / "not-an-approved-runtime-directory"
    assert not outside.exists()
    result = invoke("-Python", sys.executable, "-Action", "report", "-RuntimeRoot", str(outside))
    assert result.returncode != 0 and b"inside" in result.stderr
    assert not outside.exists()
