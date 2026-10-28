from pathlib import Path

import pytest

from unlimited_ocr_2_rocm.inference import validate_inputs


def test_missing_or_empty_input_fails_before_loading_model(tmp_path: Path):
    with pytest.raises(ValueError):
        validate_inputs([], multi=False)
    with pytest.raises(FileNotFoundError):
        validate_inputs([tmp_path / "missing.png"], multi=False)


def test_single_cannot_silently_turn_into_multiple_calls(tmp_path: Path):
    image = tmp_path / "page.png"
    image.write_bytes(b"not actually decoded at this boundary")
    with pytest.raises(ValueError):
        validate_inputs([image, image], multi=False)
    assert validate_inputs([image, image], multi=True) == [image.resolve(), image.resolve()]


def test_unicode_and_space_paths_are_preserved(tmp_path: Path):
    image = tmp_path / "中文 page.png"
    image.write_bytes(b"input")
    assert validate_inputs([image], multi=False) == [image.resolve()]


def test_phase1_page_limit_is_enforced(tmp_path: Path):
    image = tmp_path / "page.png"
    image.touch()
    with pytest.raises(ValueError):
        validate_inputs([image] * 6, multi=True)


def test_failed_gpu_telemetry_does_not_hide_original_error(tmp_path, monkeypatch):
    import contextlib
    import json
    import sys
    from types import SimpleNamespace

    import unlimited_ocr_2_rocm.inference as inference

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            inference_mode=contextlib.nullcontext,
            cuda=SimpleNamespace(reset_peak_memory_stats=lambda: None, synchronize=lambda: None),
        ),
    )
    samples = 0

    def memory():
        nonlocal samples
        samples += 1
        if samples > 1:
            raise RuntimeError("diagnostic query failed")
        return {"process_rss_bytes": 123}

    monkeypatch.setattr(inference, "memory_snapshot", memory)

    def fail_kernel(*args, **kwargs):
        raise RuntimeError("original kernel error")

    native = SimpleNamespace(
        model=SimpleNamespace(infer=fail_kernel),
        tokenizer=None,
        kernels=[],
        evidence={"unit_test_integration_double": True},
    )
    image = Path(__file__).resolve().parents[1] / "fixtures" / "english.png"
    output = tmp_path / "failure"
    report = inference.run_native(native, [image], output)
    saved = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert report["status"] == "FAIL"
    assert "original kernel error" in saved["traceback"]
    assert "diagnostic query failed" in saved["memory_after_error"]
