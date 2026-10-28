# Contributing

Please include a minimal reproduction, exact environment versions, input and
expected output, actual raw OCR, traceback and sanitized validation manifest.
Use public synthetic inputs instead of private documents. Explain whether an
error occurs in PyTorch, Triton compilation, sparse attention, model loading,
generation or output assessment.

If you report an actionable issue and are interested and able to fix it, we
encourage you to submit your own patch. Maintainers should provide reproduction
guidance, test expectations, review support and clear next steps. Do not take
over a contributor's issue with an unsolicited competing implementation.

Keep patches small. Add a failing regression before fixing a demonstrated bug.
Preserve the official algorithms and NVIDIA behavior when proposing an upstream
compatibility change. Label diagnostic fallbacks NON-NATIVE / NON-PARITY.
Keep pristine upstream source and identify patches separately.

Install developer tools in a separate venv with
`python -m pip install -r requirements/dev.txt`. Run the checks in AGENTS.md.
CPU tests require no GPU/model. Run physical tests only on the supported native
Windows device, following docs/REPRODUCIBILITY.md. Do not describe mocked
integration boundaries as hardware evidence.
