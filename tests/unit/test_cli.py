import os
import subprocess
import sys
from pathlib import Path

from unlimited_ocr_2_rocm.cli import main


def test_help_does_not_import_or_download_model():
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / "src"))
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import unlimited_ocr_2_rocm.cli; assert 'torch' not in sys.modules; "
            "assert 'transformers' not in sys.modules",
        ],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_bad_path_is_actionable_and_does_not_load_model(tmp_path, capsys):
    assert main(["single", str(tmp_path / "不存在.png"), "--output", str(tmp_path / "result")]) == 1
    assert "不存在.png" in capsys.readouterr().err


def test_error_writes_utf8_manifest(tmp_path, capsys):
    output = tmp_path / "中文 output"
    result = main(["single", str(tmp_path / "missing.png"), "--output", str(output)])
    assert result == 1
    assert '"status": "FAIL"' in (output / "error.json").read_text(encoding="utf-8")


def test_version_requires_no_gpu(capsys):
    assert main(["--version"]) == 0
    assert "0.0.1.dev0" in capsys.readouterr().out


def test_doctor_failure_replaces_requested_manifest(tmp_path, monkeypatch):
    import json

    import unlimited_ocr_2_rocm.cli as cli

    output = tmp_path / "doctor.json"
    output.write_text('{"status": "PASS"}', encoding="utf-8")

    def fail():
        raise RuntimeError("missing GPU stack")

    monkeypatch.setattr(cli, "collect_environment", fail)
    assert cli.main(["doctor", "--output", str(output)]) == 1
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "FAIL"
