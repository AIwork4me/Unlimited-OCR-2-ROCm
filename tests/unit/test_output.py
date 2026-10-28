import pytest

from unlimited_ocr_2_rocm.output import assess, character_error_rate, normalize, split_pages


def test_ignores_model_layout_markup_but_preserves_text_and_numbers():
    raw = "<PAGE><|ref|>text<|/ref|><|det|>[[0, 0, 100, 100]]<|/det|> Hello 42!"
    assert normalize(raw) == "Hello42!"
    assert normalize("中文，１２３") == "中文,123"


def test_character_error_rate_uses_edits_and_not_substring_presence():
    assert character_error_rate("abc", "axc") == pytest.approx(1 / 3)
    assert character_error_rate("abc", "abcc") == pytest.approx(1 / 3)
    assert character_error_rate("abc", "") == 1
    assert character_error_rate("", "") == 0
    assert character_error_rate("", "x") == 1


def test_preserves_empty_pages_to_detect_missing_content():
    assert split_pages("<PAGE>one<PAGE><PAGE>three") == ["one", "", "three"]


def test_assessment_requires_markers_order_and_exact_numbers():
    truth = [{"text": "Alpha 123", "numbers": ["123"]}, {"text": "Beta 456", "numbers": ["456"]}]
    assert assess("<PAGE>Alpha 123<PAGE>Beta 456", truth)["status"] == "PASS"
    for bad in (
        "<PAGE>Beta 456<PAGE>Alpha 123",
        "<PAGE>Alpha 123",
        "<PAGE>Alpha 123<PAGE>Alpha 123",
        "<PAGE>Alpha 123<PAGE>Beta 457",
        "Alpha 123 Beta 456",
        "<PAGE>Alpha 123<PAGE>Beta 456<PAGE><PAGE>",
    ):
        assert assess(bad, truth)["status"] == "FAIL"


def test_refuses_empty_truth_and_text():
    with pytest.raises(ValueError):
        assess("<PAGE>anything", [])
    with pytest.raises(ValueError):
        assess("<PAGE>anything", [{"text": ""}])


def test_untrusted_markup_is_never_evaluated():
    raw = "<|det|>__import__('os').system('bad')<|/det|>Safe 42"
    assert normalize(raw) == "Safe42"


def test_correct_table_keeps_adjacent_numeric_cell_boundaries():
    raw = (
        "<PAGE><table><tr><td>Item</td><td>Quantity</td><td>Price</td></tr>"
        "<tr><td>Apples</td><td>3</td><td>12.50</td></tr>"
        "<tr><td>Oranges</td><td>5</td><td>8.00</td></tr></table>"
    )
    truth = [
        {
            "text": "Item Quantity Price Apples 3 12.50 Oranges 5 8.00",
            "numbers": ["3", "12.50", "5", "8.00"],
            "requires_table": True,
        }
    ]
    assert assess(raw, truth)["status"] == "PASS"


def test_raw_output_bytes_preserve_mixed_line_endings(tmp_path):
    from unlimited_ocr_2_rocm.output import preserve_raw

    raw = "<PAGE>中文\nsecond\r\nthird\r"
    path = tmp_path / "raw.txt"
    preserve_raw(path, raw)
    assert path.read_bytes() == b"<PAGE>\xe4\xb8\xad\xe6\x96\x87\nsecond\r\nthird\r"


def test_official_v2_block_metadata_is_not_ocr_text():
    raw = (
        "<PAGE><BLOCK>title<LEVEL1>[10,20,900,90]<FORK>ROCm OCR Test\n"
        "<BLOCK>text[10,100,900,180]<FORK>Hello 42"
    )
    assert normalize(raw) == "ROCmOCRTestHello42"
    assert assess(raw, [{"text": "ROCm OCR Test Hello 42", "numbers": ["42"]}])["status"] == "PASS"


def test_one_empty_terminal_page_marker_is_reported_separately():
    raw = "<PAGE>Alpha 123<PAGE>Beta 456<PAGE>\n"
    truth = [{"text": "Alpha 123", "numbers": ["123"]}, {"text": "Beta 456", "numbers": ["456"]}]
    result = assess(raw, truth)
    assert result["status"] == "PASS"
    assert result["raw_page_markers"] == 3
    assert result["content_page_count"] == 2
    assert result["terminal_empty_marker"] is True
    assert result["raw_marker_count_matches_input"] is False


def test_readable_conversion_bounds_generated_heading_levels():
    from unlimited_ocr_2_rocm.output import convert_readable

    assert convert_readable("<PAGE><BLOCK>title<LEVEL1>[1,2,3,4]<FORK>Hi") == "# Hi"
    for level in ("1000000", "9" * 5000):
        with pytest.raises(ValueError, match="heading"):
            convert_readable(f"<PAGE><BLOCK>title<LEVEL{level}>[1,2,3,4]<FORK>Hi")
