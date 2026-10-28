"""These tests execute real GPU work only with explicit local opt-in."""

import os
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.environ.get("UOCR_RUN_GPU") != "1", reason="physical GPU opt-in required"
    ),
]


def test_exact_windows_stack_and_real_bf16():
    from unlimited_ocr_2_rocm.doctor import bf16_probe, collect_environment, require_stack

    require_stack(collect_environment())
    assert bf16_probe()["status"] == "PASS"


def test_real_single_page_ocr(tmp_path: Path):
    import json

    from unlimited_ocr_2_rocm.inference import run_native
    from unlimited_ocr_2_rocm.model import load_model

    cache = Path(os.environ["UOCR_MODEL_CACHE"])
    fixtures = Path(__file__).resolve().parents[1] / "fixtures"
    truth = json.loads((fixtures / "ground_truth.json").read_text(encoding="utf-8"))["cases"][
        "english"
    ]
    native = load_model(cache, offline=True)
    report = run_native(native, [fixtures / "english.png"], tmp_path / "ocr", truth=[truth])
    assert report["status"] == "PASS", report.get("traceback", report.get("correctness"))
