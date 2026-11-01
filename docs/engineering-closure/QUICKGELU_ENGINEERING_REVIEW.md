# Scoped QuickGELU engineering review

KEEP EXPERIMENTAL. This helper is opt-in and never imported by production.
The 175-line adapter is retained byte-for-byte from the reviewed v2 source;
unnecessary refactoring would invalidate its physical source identity.

`install(model, deepencoder)` accepts the raw, exclusively owned idle model and
its authenticated imported module. It returns a `Scope`; retain the handle and
call `restore()` only after requests stop. Closed cleanup is idempotent; second
installation and pre-existing per-instance forwards are rejected. One lock
serializes installation/cleanup; it does not serialize inference or DASH state.

Compatibility requires CPython 3.10, exact deepencoder SHA256
`419b63388c1645b9b72c716070cdca36cd0be6047049424dc75941d3a00aaf1f`, 24 distinct
CLIP MLPs and original Linear dimensions 1024→4096→1024. Caller authenticates
the complete model revision `9cf077c4b206e0e2a31879df937b0b0bcba3661c`, weights,
torch 2.12.0+rocm7.14.1, torchvision 0.27.0+rocm7.14.1 and torchaudio
2.11.0+rocm7.14.1. Unknown source/live forward versions fail closed. This helper
does not authenticate a live ScriptFunction graph beyond its schema.

Only two hash-reviewed AST function definitions are compiled. The activation
decorator is removed; the eager expression is exactly
`x * torch.sigmoid(1.702 * x)`, with no dtype casts, reassociation or substitute
activation. Each original forward code object receives its own globals copy
with eager QuickGELU and is bound to one instance with MethodType. Original
module/class bindings and fc1/fc2 objects remain untouched. No global JIT flag,
frozen implementation file, parameter or buffer is changed.

The two-attribute transaction records each attempted instance before mutation.
Failed install restores attribute absence through instance dictionaries, retaining
the original exception. Cleanup first checks every owner/method, records before
deletion and, on failure, restores the installed state through dictionaries for
explicit retry. This avoids re-entering mutating/failing attribute hooks.
The adapter registers/removes no torch hooks. Arbitrary side effects from foreign
hooks are outside this narrowly documented transaction; do not claim general
rollback of arbitrary subclass behavior. A foreign method change rejects cleanup
before any partial teardown; discard the model process if ownership is unknown.

Regression coverage retains the original 16 cases, including mutate-then-raise
teardown and failed rollback deletion hooks. Two added cases cover owner-marker
mutate-then-raise and preservation of foreign hook/parameter/buffer/JIT objects.
These scalar stand-ins do not qualify physical BF16 or shared-process serving.
Both historical defects are replayed against sealed v1 before v2 verification;
no new arithmetic/lifecycle implementation change is needed.

Run from a clean checkout with Python 3.10 and the existing development tools:

```powershell
python -m pytest tests/unit/test_scoped_eager.py -q
python -m ruff check experiments/scoped_eager tests/unit/test_scoped_eager.py
python -m ruff format --check experiments/scoped_eager tests/unit/test_scoped_eager.py
```

No torch, model download or private experiment directory is needed for these
tests. Original model/license notices remain at the baseline; no upstream model
implementation was copied. Runtime dependencies do not grow. Rollback means
explicit restore or discard the entire owned process on hard faults.
