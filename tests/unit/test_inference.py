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
