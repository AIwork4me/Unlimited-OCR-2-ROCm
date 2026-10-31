# Experimental scoped eager QuickGELU

KEEP EXPERIMENTAL. `adapter.py` is an explicitly called experiment helper, never
imported by production. It recovers `x * torch.sigmoid(1.702 * x)` from the exact
reviewed `deepencoder.py` hash and clones only the24 CLIP MLP instance forwards.
The SAM path, parameters, class forward and module QuickGELU binding stay owned
by the original model. No global JIT flag or frozen model file is changed.

The caller first authenticates the pinned wheel, loaded native stack, complete
model/weights and reviewed module, and exclusively owns an idle model in one
fresh process. Pass the actual raw model and its imported `deepencoder` module
to `install(model, deepencoder)`, retain the returned handle, and call
`handle.restore()` only after requests stop. Double installation and existing
instance forward overrides are rejected. Successful cleanup removes the added
instance attributes; repeated cleanup on a closed handle is a no-op. A foreign
method mutation raises `AdmissionError` before any restoration.

Only reviewed CPython3.10 source/live **MLP forward** bytecode and the1024/4096
architecture are admitted. The original ScriptFunction is schema-checked;
its graph is not independently authenticated by this helper. The eager function
comes from hashed source and does not call that ScriptFunction. Dependencies
are the original model's verified torch module and existing fc1/fc2 instances.

Installation/restoration are serialized against other adapter operations, not
against inference. The owner must enforce the full request lifecycle. Upstream
DASH/module/config state makes same-process parallel requests and multiple real
model loads unsupported by this qualification. Separate processes are tested
explicitly; a Python lock in this adapter is not a serving scheduler.

Exception rollback restores the two known plain instance attributes through
their dictionaries so failing attribute hooks cannot interrupt that rollback.
It does not undo arbitrary extra side effects introduced by a foreign hook or
recover a process killed during execution. Hard faults discard the complete
owned process; Windows Job receipts and a fresh model establish recovery.
No destructor or garbage-collection callback performs implicit restoration.

CPU regression tests use scalar stand-ins and need no torch/model download.
Physical BF16/inference evidence and remaining gates are described in
[overnight qualification](../../docs/overnight/SCOPED_EAGER_QUALIFICATION.md).
