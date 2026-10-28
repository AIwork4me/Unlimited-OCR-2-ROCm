# Limitations

- Physical target is only Windows 11 build 26200, Ryzen AI MAX+ 395,
  Radeon 8060S / gfx1151 and the exact CPython 3.10 ROCm stack. Hosted CPU CI
  and upstream vendor statements do not validate another platform/device.
- See VALIDATION.md for actually executed model, OCR and reproduction gates.
  Passing a GPU microkernel does not establish native model correctness.
- Fixtures are small and synthetic. No broad benchmark, production quality
  score, maximum throughput, speedup or unlimited-length guarantee is claimed.
- Native Transformers decode grows KV/router caches and computes full-cache
  matmuls with sparse masks. This wrapper preserves that official algorithm.
  It does not implement SGLang's serving scheduler or Phase 2 optimization.
- Requests are serialized in a dedicated process, at most five images,
  at most 16 million pixels per page. The model source mutates process-global
  attention classes/state. Load from a clean process before importing Transformers.
- Raw page markup is retained. The safe normalization is an inspection helper,
  not a new full long-document parser. Formula reconstruction, image cropping,
  active HTML rendering and upstream eval-based visualization are not included.
- Successful forward sparse-kernel tests do not validate training/backward or
  the checkpoint's unused alternative base-stage kernels.
- The model card MIT label coexists with Apache-2.0 source headers and other
  attribution. This package does not redistribute model code/weights or fonts.
- Setup requires network access or a separately prepared wheelhouse. Cached
  model reruns can be offline in an existing prepared environment.
- Long-path policy and antivirus edge conditions are not automatically changed
  or certified. Preserve failures before making any system-setting decision.
- No persistent self-hosted runner, PyPI publication, GitHub Release, stable
  tag or automatic feature-PR merge is part of Phase 1.
