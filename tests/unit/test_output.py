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
        "<PAGE>Alpha 123<PAGE>Beta 456<PAGE>",
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
