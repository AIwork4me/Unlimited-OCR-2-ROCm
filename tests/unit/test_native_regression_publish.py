import io
import tarfile
from pathlib import Path, PurePosixPath

import pytest

from experiments.native_regression.publish import archive_bytes, evidence_files, sanitize, sensitive


@pytest.mark.parametrize("path_type", [Path, PurePosixPath])
def test_sanitization_handles_windows_json_paths_without_changing_native_text(path_type):
    root = path_type("C:/Users/operator/project")
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


def test_review_publication_excludes_nested_test_workspaces(tmp_path):
    (tmp_path / "review.json").write_text("{}", encoding="utf-8")
    workspace = tmp_path / "physical-analysis-pytest"
    workspace.mkdir()
    (workspace / "private-test-result.json").write_text("{}", encoding="utf-8")
    assert [p.name for p in evidence_files(tmp_path, top_level=True)] == ["review.json"]
