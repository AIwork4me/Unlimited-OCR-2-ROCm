from pathlib import Path

import pytest

from unlimited_ocr_2_rocm.config import check_revision, verify_files


def test_refuses_mutable_or_malformed_revision():
    for revision in ("main", "v2", "../escape", "a" * 39, "g" * 40):
        with pytest.raises(ValueError):
            check_revision(revision)
    check_revision("9cf077c4b206e0e2a31879df937b0b0bcba3661c")


def test_refuses_missing_and_tampered_reviewed_code(tmp_path: Path):
    hashes = {"reviewed.py": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"}
    with pytest.raises(RuntimeError, match="missing"):
        verify_files(tmp_path, hashes)
    (tmp_path / "reviewed.py").write_bytes(b"abc")
    assert verify_files(tmp_path, hashes) == hashes
    (tmp_path / "reviewed.py").write_bytes(b"malicious replacement")
    with pytest.raises(RuntimeError, match="checksum"):
        verify_files(tmp_path, hashes)


def test_refuses_paths_outside_snapshot(tmp_path: Path):
    with pytest.raises(ValueError):
        verify_files(tmp_path, {"../outside.py": "0" * 64})
