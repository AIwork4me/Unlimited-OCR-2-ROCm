"""Version-locked experimental CLIP binding, never imported by production.

Install/restore only while exclusively owning an idle model. This adapter does
not serialize inference or qualify the upstream shared DASH state. It preserves
the official eager BF16 expression; it does not touch JIT flags or frozen files.
"""

from __future__ import annotations

import ast
import hashlib
import sys
import threading
import types
from pathlib import Path

SOURCE_SHA256 = "419b63388c1645b9b72c716070cdca36cd0be6047049424dc75941d3a00aaf1f"
_OWNER = "_uocr_experimental_eager_owner"
_LOCK = threading.RLock()


class AdmissionError(RuntimeError):
    """Unknown implementation or conflicting lifecycle ownership."""


def _code_identity(code):
    """Exclude file/line metadata, retain executable and calling conventions."""
    return (
        code.co_code,
        tuple(_code_identity(c) if isinstance(c, types.CodeType) else c for c in code.co_consts),
        code.co_names,
        code.co_varnames,
        code.co_freevars,
        code.co_cellvars,
        code.co_argcount,
        code.co_posonlyargcount,
        code.co_kwonlyargcount,
        code.co_flags,
    )


def _recover(deep):
    if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 10):
        raise AdmissionError("Only the reviewed CPython 3.10 implementation is admitted")
    data = Path(deep.__file__).read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise AdmissionError("Unknown deepencoder source identity")
    tree = ast.parse(data.decode("utf-8"))
    eager_node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "quick_gelu"
    )
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NoTPFeedForward")
    forward_node = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward"
    )
    eager_node.decorator_list = []
    namespace = {"torch": deep.torch}
    # Execute two verified function definitions only, never the module body.
    exec(
        compile(
            ast.Module(body=[eager_node, forward_node], type_ignores=[]),
            "<verified-clip-eager>",
            "exec",
            dont_inherit=True,
        ),
        namespace,
    )
    original = deep.NoTPFeedForward.forward
    if (
        not isinstance(original, types.FunctionType)
        or original.__globals__ is not vars(deep)
        or original.__defaults__ is not None
        or original.__kwdefaults__ is not None
        or original.__closure__ is not None
        or _code_identity(original.__code__) != _code_identity(namespace["forward"].__code__)
        or original.__code__.co_names != ("fc2", "quick_gelu", "fc1")
        or str(getattr(deep.quick_gelu, "schema", "")) != "quick_gelu(Tensor x) -> Tensor"
    ):
        raise AdmissionError("Unknown live forward bytecode, globals or scripted activation")
    return namespace["quick_gelu"], original


class Scope:
    """Explicit cleanup handle; no finalizer can run during an active request."""

    def __init__(self, records, token):
        self._records, self._token = records, token
        self.closed = False

    def restore(self):
        with _LOCK:
            if self.closed:
                return
            # Validate every layer before modifying any. Never overwrite a
            # foreign binding introduced after installation.
            if any(
                vars(m).get(_OWNER) is not self._token or vars(m).get("forward") is not bound
                for m, bound in self._records
            ):
                raise AdmissionError("Scope ownership changed; refuse partial restoration")
            restored = []
            try:
                for m, bound in reversed(self._records):
                    restored.append((m, bound))
                    delattr(m, "forward")
                    delattr(m, _OWNER)
            except BaseException:
                # A model subclass may reject teardown. Restore the installed
                # state and retain the handle for an explicit later recovery.
                for m, bound in reversed(restored):
                    # Both are known plain instance attributes. Rollback must
                    # not re-enter hooks that may have mutated then raised.
                    vars(m)[_OWNER] = self._token
                    vars(m)["forward"] = bound
                raise
            self.closed = True
            self._records = []
            self._token = None


def install(model, deep):
    """Bind only 24 known CLIP MLP instances; reject duplicates/unknown models.

    The caller verifies wheel/model/weights and owns the complete model process.
    An existing per-instance forward override is incompatible and is rejected.
    Concurrent install/restore operations are serialized; inference is not.
    """
    with _LOCK:
        eager, original = _recover(deep)
        try:
            layers = list(model.model.vision_model.transformer.layers)
            mlps = [b.mlp for b in layers]
        except (AttributeError, TypeError) as e:
            raise AdmissionError("Incompatible CLIP model structure") from e
        if len(mlps) != 24 or len({id(m) for m in mlps}) != 24:
            raise AdmissionError("Expected 24 distinct CLIP MLP instances")
        for m in mlps:
            if (
                type(m) is not deep.NoTPFeedForward
                or "forward" in vars(m)
                or _OWNER in vars(m)
                or m.forward.__func__ is not original
                or type(m.fc1) is not deep.torch.nn.Linear
                or type(m.fc2) is not deep.torch.nn.Linear
                or (m.fc1.in_features, m.fc1.out_features, m.fc2.in_features, m.fc2.out_features)
                != (1024, 4096, 4096, 1024)
            ):
                raise AdmissionError("Unknown MLP shape/type or already-installed instance")
        token, records = object(), []
        try:
            for m in mlps:
                clone = types.FunctionType(
                    original.__code__,
                    dict(original.__globals__, quick_gelu=eager),
                    original.__name__,
                    original.__defaults__,
                    original.__closure__,
                )
                clone.__qualname__ = original.__qualname__
                clone.__module__ = original.__module__
                clone.__doc__ = original.__doc__
                clone.__annotations__ = dict(original.__annotations__)
                clone.__dict__.update(original.__dict__)
                bound = types.MethodType(clone, m)
                # Record before mutation so a setter that mutates then raises
                # can also be rolled back without leaving a marker behind.
                records.append((m, bound))
                setattr(m, _OWNER, token)
                m.forward = bound
        except BaseException:
            for m, _ in reversed(records):
                vars(m).pop("forward", None)
                vars(m).pop(_OWNER, None)
            raise
        return Scope(records, token)
