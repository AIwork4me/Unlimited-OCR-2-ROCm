import io
import tarfile
from pathlib import Path

import pytest

from experiments.native_regression.publish import archive_bytes, sanitize, sensitive


def test_sanitization_handles_windows_json_paths_without_changing_native_text():
    root = Path("C:/Users/operator/project")
    data = b'{"path":"C:\\\\Users\\\\operator\\\\project/file","text":"x+y=z"}'
    result = sanitize(data, root=root, home=Path("C:/Users/operator"))
    assert b"x+y=z" in result
    assert sensitive(result) == []
    assert b"<REPO>" in result


def test_archive_is_deterministic_and_preserves_raw_bytes():
    data = {"after/raw.txt": "<PAGE>原始文本".encode(), "index.json": b"{}\n"}
    first = archive_bytes(data)
    assert first == archive_bytes(dict(reversed(list(data.items()))))
    with tarfile.open(fileobj=io.BytesIO(first), mode="r:gz") as archive:
        assert archive.extractfile("after/raw.txt").read() == data["after/raw.txt"]
        assert all(member.mtime == 0 for member in archive.getmembers())


def test_archive_rejects_path_escape_and_flags_credential_like_values():
    with pytest.raises(ValueError, match="relative"):
        archive_bytes({"../outside": b"data"})
    assert sensitive(b"hf_" + b"X" * 30) == ["token_value"]
