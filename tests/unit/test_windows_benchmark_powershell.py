"""Exercise the Windows entry point without torch or a model."""

import json
import os
import shutil
import subprocess
import sys
import uuid
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


def test_huggingface_cache_is_scoped_and_caller_environment_restored(tmp_path):
    source = tmp_path / "source"
    scripts = source / "scripts"
    scripts.mkdir(parents=True)
    wrapper = scripts / SCRIPT.name
    shutil.copyfile(SCRIPT, wrapper)
    (scripts / "run_mpdoc_sample.py").write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(sys.argv[2]).write_text(json.dumps({k: os.environ.get(k) "
        "for k in ('HF_HOME', 'HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE')}))\n",
        encoding="utf-8",
    )
    observed = tmp_path / "observed.json"
    runtime = Path.cwd() / ".b" / ("ps-test-" + uuid.uuid4().hex[:8])
    command = tmp_path / "invoke.ps1"

    def literal(path):
        return "'" + str(path).replace("'", "''") + "'"

    command.write_text(
        "$env:HF_HUB_CACHE = 'inherited-external-cache'\n"
        "$env:HUGGINGFACE_HUB_CACHE = 'inherited-legacy-cache'\n"
        f"& {literal(wrapper)} -Python {literal(sys.executable)} -Action report "
        f"-RuntimeRoot {literal(runtime)} -RunnerArguments {literal(observed)}\n"
        "if ($env:HF_HUB_CACHE -ne 'inherited-external-cache' -or "
        "$env:HUGGINGFACE_HUB_CACHE -ne 'inherited-legacy-cache') "
        "{ throw 'Caller cache environment was not restored' }\n",
        encoding="utf-8",
    )
    try:
        result = subprocess.run(
            [POWERSHELL, "-NoProfile", "-File", str(command)], capture_output=True, timeout=20
        )
    finally:
        assert runtime.resolve().is_relative_to((Path.cwd() / ".b").resolve())
        shutil.rmtree(runtime, ignore_errors=True)
    assert result.returncode == 0, result.stderr
    actual = json.loads(observed.read_text())
    assert Path(actual["HF_HOME"]) == runtime / "huggingface"
    assert Path(actual["HF_HUB_CACHE"]) == runtime / "huggingface/hub"
    assert Path(actual["HUGGINGFACE_HUB_CACHE"]) == runtime / "huggingface/hub"
