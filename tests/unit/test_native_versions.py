import pytest

from unlimited_ocr_2_rocm.model import native_dependency_pins, require_model_dependencies


def test_model_refuses_incompatible_triton_or_transformers():
    correct = dict(native_dependency_pins())
    assert "triton-windows" in correct or "pytorch-triton-rocm" in correct
    require_model_dependencies(correct)
    for package in correct:
        changed = dict(correct, **{package: "0.0.0"})
        with pytest.raises(RuntimeError):
            require_model_dependencies(changed)


def test_environment_cannot_silently_change_native_routing():
    from unlimited_ocr_2_rocm.model import require_native_environment

    require_native_environment({})
    require_native_environment({"DASH_STAGE0": "head_cls", "DASH_LOG_PAGE_EVERY": "1"})
    for environment in (
        {"DASH_STAGE0": "indexer"},
        {"DASH_SPARSITY_SCHED": "1"},
        {"TRITON_INTERPRET": "1"},
        {"HSA_OVERRIDE_GFX_VERSION": "11.5.1"},
    ):
        with pytest.raises(RuntimeError):
            require_native_environment(environment)
