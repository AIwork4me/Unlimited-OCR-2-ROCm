# Native Windows SGLang feasibility

Verdict: **BLOCKED_WITH_ACTIONABLE_FIX**. Windows native execution: **NOT TESTED**. This is a static immutable-artifact assessment, not a serving benchmark or an installation endorsement.

The official customized wheel is `sglang-0.5.20.dev10+g8d62f5fb9c-py3-none-any.whl`, 15281032 bytes, SHA256 `a3f3e2c94e09de90b61b43d01475fd5d1aa818d16eec05204b8079c09c2ba70d`. The common benchmark lock pins wheel METADATA and selected Unlimited-OCR implementation files. Local wheel rehash matches. See [common source audit](../BENCHMARK_PROTOCOL.md) and [Baidu pinned client](https://github.com/baidu/Unlimited-OCR/tree/48ff432fb68a18ed91f2f70021f29028b1dde354).

Its metadata requires torch==2.13.0, cuda-python>=13.0, cuda-tile, flash-attn-4, flashinfer_python[cu13], nvidia-cutlass-dsl[cu13] and nvidia-mathdx. These requirements do not resolve into the preserved original Windows torch2.12.0+rocm7.14.1 stack. The universal Python wheel tag does not certify Windows/HIP native dependency support. Installing with --no-deps hides conflicts; it does not fix kernels or prove parity. Generic SGLang is not the customized Baidu route.

DASH backend imports custom Triton operations and uses torch.cuda; that API name alone is compatible with ROCm and does not establish NVIDIA exclusivity. Conversely, its presence does not establish tested HIP kernels. Native Windows compiler/library bindings, scheduler/graphs, process lifecycle, long-context/window continuation, KV-tail reuse and custom ngram parity remain unvalidated.

An actionable future port must first review the existing Linux ROCm work for reuse, freeze a separate dependency/kernel identity compatible with the required stack, isolate all experimentation from the working Native installation, then pass minimal physical kernel/model/serving/window parity gates. Reconcile unsupported native packages explicitly; do not alter global SDK/driver/security or silently change torch. This is a separate port and authorization workstream, not a small dependency flag fix. Linux results cannot be promoted to Windows PASS.

No SGLang package was installed or executed in this mission. Stable Native scoped-MATH delivery remains independent of this blocked route.
