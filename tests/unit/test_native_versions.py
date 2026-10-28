import pytest

from unlimited_ocr_2_rocm.model import require_model_dependencies


def test_model_refuses_incompatible_triton_or_transformers():
    correct = {
        "transformers": "4.52.1",
        "triton-windows": "3.7.1.post27",
        "huggingface-hub": "0.36.0",
        "tokenizers": "0.21.4",
    }
    require_model_dependencies(correct)
    for package, bad in (
        ("transformers", "5.0.0"),
        ("triton-windows", "3.8.0.post1"),
        ("huggingface-hub", "1.0.0"),
        ("tokenizers", "0.22.0"),
    ):
        changed = dict(correct, **{package: bad})
        with pytest.raises(RuntimeError):
            require_model_dependencies(changed)
