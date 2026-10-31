"""Binding/lifecycle contracts; no torch, weights, network or GPU required."""

import importlib.util
import math
import types
from pathlib import Path

import pytest

ADAPTER = Path(__file__).resolve().parents[2] / "experiments/scoped_eager/adapter.py"
SOURCE = """@torch.jit.script
def quick_gelu(x):
    return x * torch.sigmoid(1.702 * x)

class NoTPFeedForward:
    def forward(self, x):
        output = self.fc2(quick_gelu(self.fc1(x)))
        return output
"""


def adapter():
    assert ADAPTER.is_file(), "Experimental lifecycle adapter is missing"
    spec = importlib.util.spec_from_file_location("scoped_eager_test", ADAPTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Linear:
    def __init__(self, incoming, outgoing, multiplier):
        self.in_features, self.out_features = incoming, outgoing
        self.multiplier = multiplier

    def __call__(self, x):
        return self.multiplier * x


class ScriptFunction:
    schema = "quick_gelu(Tensor x) -> Tensor"

    def __init__(self, function):
        self.function = function

    def __call__(self, x):
        return self.function(x) + 100


def fixture(tmp_path, monkeypatch):
    import hashlib

    a = adapter()
    source = tmp_path / "deepencoder.py"
    source.write_text(SOURCE, encoding="utf-8")
    # Only the source hash is replaced for a CPU-only synthetic module. The
    # arithmetic, forward structure and bytecode checks are exercised unchanged.
    monkeypatch.setattr(a, "SOURCE_SHA256", hashlib.sha256(source.read_bytes()).hexdigest())
    deep = types.ModuleType("synthetic_deepencoder")
    deep.__file__ = str(source)
    deep.torch = types.SimpleNamespace(
        sigmoid=lambda x: 1 / (1 + math.exp(-x)),
        jit=types.SimpleNamespace(script=ScriptFunction),
        nn=types.SimpleNamespace(Linear=Linear),
    )
    exec(compile(SOURCE, str(source), "exec"), deep.__dict__)

    def model():
        layers = []
        for _ in range(24):
            mlp = deep.NoTPFeedForward()
            mlp.fc1, mlp.fc2 = Linear(1024, 4096, 2), Linear(4096, 1024, 3)
            layers.append(types.SimpleNamespace(mlp=mlp))
        return types.SimpleNamespace(
            model=types.SimpleNamespace(
                vision_model=types.SimpleNamespace(transformer=types.SimpleNamespace(layers=layers))
            )
        )

    return a, deep, model


def mlps(model):
    return [b.mlp for b in model.model.vision_model.transformer.layers]


def test_eager_binding_is_instance_local_and_restores_attribute_absence(tmp_path, monkeypatch):
    a, deep, make = fixture(tmp_path, monkeypatch)
    first, second = make(), make()
    script, class_forward = deep.quick_gelu, deep.NoTPFeedForward.forward
    original = mlps(first)[0].forward(1)
    handle = a.install(first, deep)
    eager = mlps(first)[0].forward(1)
    assert eager == 6 / (1 + math.exp(-3.404))
    assert mlps(second)[0].forward(1) == original
    assert original == eager + 300
    assert deep.quick_gelu is script
    assert deep.NoTPFeedForward.forward is class_forward
    handle.restore()
    assert "forward" not in vars(mlps(first)[0])
    assert mlps(first)[0].forward(1) == original
    handle.restore()  # A closed handle has a predictable no-op cleanup.
    assert handle.closed


def test_two_scopes_restore_independently_and_reinstall(tmp_path, monkeypatch):
    a, deep, make = fixture(tmp_path, monkeypatch)
    first, second = make(), make()
    h1, h2 = a.install(first, deep), a.install(second, deep)
    h1.restore()
    assert mlps(first)[0].forward(1) != mlps(second)[0].forward(1)
    h2.restore()
    a.install(first, deep).restore()


def test_double_installation_is_rejected_without_losing_ownership(tmp_path, monkeypatch):
    a, deep, make = fixture(tmp_path, monkeypatch)
    model = make()
    handle = a.install(model, deep)
    with pytest.raises(a.AdmissionError):
        a.install(model, deep)
    handle.restore()
    assert all("forward" not in vars(m) for m in mlps(model))


@pytest.mark.parametrize("mutation", ["source", "bytecode", "globals", "layers", "shape", "alias"])
def test_unknown_contract_is_rejected_before_any_mutation(tmp_path, monkeypatch, mutation):
    a, deep, make = fixture(tmp_path, monkeypatch)
    model = make()
    if mutation == "source":
        Path(deep.__file__).write_text(SOURCE + "\n# changed", encoding="utf-8")
    elif mutation == "bytecode":
        forward = deep.NoTPFeedForward.forward
        forward.__code__ = forward.__code__.replace(co_consts=(None, 99))
    elif mutation == "globals":
        deep.quick_gelu = lambda x: x
    elif mutation == "layers":
        model.model.vision_model.transformer.layers.pop()
    elif mutation == "shape":
        mlps(model)[-1].fc2.out_features = 99
    else:
        model.model.vision_model.transformer.layers[-1].mlp = mlps(model)[0]
    with pytest.raises(a.AdmissionError):
        a.install(model, deep)
    assert all("forward" not in vars(m) for m in mlps(model))


def test_partial_assignment_failure_rolls_back_all_prior_layers(tmp_path, monkeypatch):
    a, deep, make = fixture(tmp_path, monkeypatch)
    model = make()
    original_setattr = deep.NoTPFeedForward.__setattr__
    victim = mlps(model)[7]

    def failing(self, name, value):
        if self is victim and name == "forward":
            raise RuntimeError("injected assignment failure")
        original_setattr(self, name, value)

    monkeypatch.setattr(deep.NoTPFeedForward, "__setattr__", failing)
    with pytest.raises(RuntimeError, match="injected"):
        a.install(model, deep)
    assert all("forward" not in vars(m) for m in mlps(model))
    monkeypatch.setattr(deep.NoTPFeedForward, "__setattr__", original_setattr)
    a.install(model, deep).restore()


def test_restore_refuses_foreign_mutation_without_partial_cleanup(tmp_path, monkeypatch):
    a, deep, make = fixture(tmp_path, monkeypatch)
    model = make()
    handle = a.install(model, deep)
    owned = mlps(model)[-1].forward
    mlps(model)[-1].forward = lambda x: x
    with pytest.raises(a.AdmissionError):
        handle.restore()
    assert not handle.closed
    assert all("forward" in vars(m) for m in mlps(model))
    mlps(model)[-1].forward = owned
    handle.restore()


def test_existing_instance_override_is_rejected(tmp_path, monkeypatch):
    a, deep, make = fixture(tmp_path, monkeypatch)
    model = make()
    mlps(model)[-1].forward = mlps(model)[-1].forward
    with pytest.raises(a.AdmissionError):
        a.install(model, deep)
    assert "forward" not in vars(mlps(model)[0])


def test_teardown_exception_keeps_handle_and_installed_state(tmp_path, monkeypatch):
    a, deep, make = fixture(tmp_path, monkeypatch)
    model = make()
    handle = a.install(model, deep)
    victim = mlps(model)[7]
    original_delete = deep.NoTPFeedForward.__delattr__

    def failing(self, name):
        if self is victim and name == "forward":
            raise RuntimeError("injected teardown failure")
        original_delete(self, name)

    monkeypatch.setattr(deep.NoTPFeedForward, "__delattr__", failing)
    with pytest.raises(RuntimeError, match="teardown"):
        handle.restore()
    assert not handle.closed
    assert all("forward" in vars(m) for m in mlps(model))
    monkeypatch.setattr(deep.NoTPFeedForward, "__delattr__", original_delete)
    handle.restore()


def test_concurrent_installers_get_one_owner(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    a, deep, make = fixture(tmp_path, monkeypatch)
    model = make()

    def attempt():
        try:
            return a.install(model, deep)
        except a.AdmissionError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: attempt(), range(2)))
    owners = [h for h in results if h is not None]
    assert len(owners) == 1
    owners[0].restore()


def test_teardown_mutates_then_raises_and_can_be_retried(tmp_path, monkeypatch):
    a, deep, make = fixture(tmp_path, monkeypatch)
    model = make()
    handle = a.install(model, deep)
    original_delete = deep.NoTPFeedForward.__delattr__
    victim = mlps(model)[7]

    def failing(self, name):
        original_delete(self, name)
        if self is victim and name == "forward":
            raise RuntimeError("mutation before teardown exception")

    monkeypatch.setattr(deep.NoTPFeedForward, "__delattr__", failing)
    with pytest.raises(RuntimeError, match="mutation before"):
        handle.restore()
    assert all("forward" in vars(m) for m in mlps(model))
    assert not handle.closed
    monkeypatch.setattr(deep.NoTPFeedForward, "__delattr__", original_delete)
    handle.restore()
    assert all("forward" not in vars(m) for m in mlps(model))


def test_install_rollback_survives_failing_deletion_hooks(tmp_path, monkeypatch):
    a, deep, make = fixture(tmp_path, monkeypatch)
    model = make()
    original_set = deep.NoTPFeedForward.__setattr__
    original_delete = deep.NoTPFeedForward.__delattr__
    victim = mlps(model)[7]

    def failing_set(self, name, value):
        original_set(self, name, value)
        if self is victim and name == "forward":
            raise RuntimeError("original install failure")

    def failing_delete(self, name):
        raise RuntimeError("rollback deletion hook rejected")

    monkeypatch.setattr(deep.NoTPFeedForward, "__setattr__", failing_set)
    monkeypatch.setattr(deep.NoTPFeedForward, "__delattr__", failing_delete)
    with pytest.raises(RuntimeError, match="original install failure"):
        a.install(model, deep)
    assert all("forward" not in vars(m) for m in mlps(model))
    monkeypatch.setattr(deep.NoTPFeedForward, "__setattr__", original_set)
    monkeypatch.setattr(deep.NoTPFeedForward, "__delattr__", original_delete)
    a.install(model, deep).restore()


def test_foreign_hooks_parameters_buffers_and_jit_objects_are_untouched(tmp_path, monkeypatch):
    a, deep, make = fixture(tmp_path, monkeypatch)
    first, second = make(), make()
    hooks = {"foreign": object()}
    for m in mlps(first):
        m._forward_hooks = hooks
        m._parameters = {"weight": object()}
        m._buffers = {"state": object()}
    before = [dict(vars(m)) for m in mlps(first)]
    other = [dict(vars(m)) for m in mlps(second)]
    jit = deep.torch.jit
    script = jit.script
    handle = a.install(first, deep)
    handle.restore()
    handle.restore()
    assert [vars(m) for m in mlps(first)] == before
    assert [vars(m) for m in mlps(second)] == other
    assert deep.torch.jit is jit and jit.script is script
    assert hooks == {"foreign": hooks["foreign"]}


def test_owner_marker_mutates_then_raises_without_residue(tmp_path, monkeypatch):
    a, deep, make = fixture(tmp_path, monkeypatch)
    first, second = make(), make()
    before = [dict(vars(m)) for m in mlps(first)]
    other = [dict(vars(m)) for m in mlps(second)]
    original = deep.NoTPFeedForward.__setattr__
    victim = mlps(first)[7]
    failure = RuntimeError("owner setter mutated then raised")

    def failing(self, name, value):
        original(self, name, value)
        if self is victim and name == a._OWNER:
            raise failure

    monkeypatch.setattr(deep.NoTPFeedForward, "__setattr__", failing)
    with pytest.raises(RuntimeError) as caught:
        a.install(first, deep)
    assert caught.value is failure
    assert [vars(m) for m in mlps(first)] == before
    assert [vars(m) for m in mlps(second)] == other
