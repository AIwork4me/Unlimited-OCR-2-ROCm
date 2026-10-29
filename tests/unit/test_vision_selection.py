"""CPU checks for the scoped vision selector; no torch download required."""

import inspect
import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import pytest

from unlimited_ocr_2_rocm.compat import wrap_vision_forward
from unlimited_ocr_2_rocm.model import load_model


@pytest.fixture
def fake_torch(monkeypatch):
    state = {"enabled": ["caller"], "synchronizations": 0}

    class Tensor:
        device = SimpleNamespace(type="cuda")
        dtype = "bfloat16"
        shape = (1, 2, 3)

    @contextmanager
    def kernel(backends):
        previous = state["enabled"]
        state["enabled"] = backends if isinstance(backends, list) else [backends]
        try:
            yield
        finally:
            state["enabled"] = previous

    def synchronize():
        state["synchronizations"] += 1

    torch = ModuleType("torch")
    torch.Tensor = Tensor
    torch.cuda = SimpleNamespace(synchronize=synchronize)
    attention = ModuleType("torch.nn.attention")
    attention.SDPBackend = SimpleNamespace(
        MATH="math", FLASH_ATTENTION="flash", EFFICIENT_ATTENTION="efficient"
    )
    attention.sdpa_kernel = kernel
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.nn.attention", attention)
    return state, Tensor()


@pytest.mark.parametrize(
    ("selection", "expected"),
    [(None, ["math"]), ("math", ["math"]), ("auto", ["flash", "efficient", "math"])],
)
def test_selection_is_scoped_and_restores_caller(fake_torch, selection, expected):
    state, tensor = fake_torch
    events = []

    def forward(value):
        assert state["enabled"] == expected
        return value

    kwargs = {} if selection is None else {"backend": selection}
    wrapped = wrap_vision_forward(forward, "sam_model", events, **kwargs)
    assert wrapped(tensor) is tensor
    assert state["enabled"] == ["caller"]
    assert state["synchronizations"] == 1
    assert events[0]["status"] == "PASS"


def test_auto_restores_caller_on_failure(fake_torch):
    state, tensor = fake_torch

    def fail(value):
        raise RuntimeError("attention failed")

    with pytest.raises(RuntimeError, match="attention failed"):
        wrap_vision_forward(fail, "vision_model", [], backend="auto")(tensor)
    assert state["enabled"] == ["caller"]


def test_invalid_selection_fails_before_torch_import():
    with pytest.raises(ValueError, match="math.*auto"):
        wrap_vision_forward(lambda value: value, "sam_model", [], backend="flash")


def test_model_default_remains_math():
    assert inspect.signature(load_model).parameters["vision_sdpa"].default == "math"
