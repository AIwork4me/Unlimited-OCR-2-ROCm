"""Optional local Parquet tests for the official SlideVQA image serialization."""

import hashlib
import importlib.util
from io import BytesIO
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
pytest.importorskip("datasets")
Image = pytest.importorskip("PIL.Image")


def implementation():
    path = Path(__file__).parents[1] / "scripts/g05_acquire_slidevqa.py"
    assert path.exists(), "authorized SlideVQA extractor has not been implemented"
    spec = importlib.util.spec_from_file_location("g05_slidevqa", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def image_bytes(color):
    stream = BytesIO()
    Image.new("RGB", (8, 8), color).save(stream, format="PNG")
    return stream.getvalue()


def test_last_matching_row_uses_same_default_jpeg_serialization_as_official(tmp_path):
    module = implementation()
    first, last = image_bytes("red"), image_bytes("blue")
    shard = tmp_path / "test.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "deck_url": "https://example.test/exact",
                    "page_1": {"bytes": first, "path": None},
                },
                {
                    "deck_url": "https://example.test/unrelated",
                    "page_1": {"bytes": first, "path": None},
                },
                {"deck_url": "https://example.test/exact", "page_1": {"bytes": last, "path": None}},
            ]
        ),
        shard,
    )
    result = module.extract_frozen_decks(
        [shard], {"doc": "https://example.test/exact"}, {"doc": 1}, tmp_path / "output"
    )
    expected = tmp_path / "official.jpg"
    Image.open(BytesIO(last)).save(expected)
    actual = tmp_path / "output/images/doc/page_1.jpg"
    assert actual.read_bytes() == expected.read_bytes()
    assert result["matching_rows"] == 2
    assert result["pages"][0]["output_sha256"] == hashlib.sha256(expected.read_bytes()).hexdigest()
    assert result["pages"][0]["input_image_bytes_sha256"] == hashlib.sha256(last).hexdigest()


def test_null_expected_page_is_a_failure_without_substitution(tmp_path):
    module = implementation()
    shard = tmp_path / "test.parquet"
    pq.write_table(pa.Table.from_pylist([{"deck_url": "url", "page_1": None}]), shard)
    with pytest.raises(ValueError, match="missing expected page"):
        module.extract_frozen_decks([shard], {"doc": "url"}, {"doc": 1}, tmp_path / "output")
