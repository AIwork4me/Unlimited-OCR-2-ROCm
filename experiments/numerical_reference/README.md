# Experimental numerical references

KEEP EXPERIMENTAL. These tools do not change production inference or define an
OCR acceptance tolerance.

- `reference.py`: independent NumPy FP32/FP64 attention equations and descriptive
  error statistics. No torch, model, weights or network is used by this module.
- `retained.py`: verifies retained tensor-file/logical hashes, validates captured
  SDPA arguments, and compares GPU outputs with CPU references. It requires
  torch only to read the local frozen tensor archives.
- `diagnostic.py`: explicit diagnostic teacher-forcing clone. It records actual
  selection before supplying a common prefix and preserves the original
  decorator/guard semantics. It is never called by production.
- `analyze_diagnostics.py`: audits completed captures, matched operator inputs,
  fixed-hidden-state lm_head projections, and cache append contracts. It reads
  the immutable local weight file for the projection reference; it runs no OCR.

In an existing compatible environment, run the nine local CPU tests with:

```powershell
python -m unittest discover -s experiments/numerical_reference -p 'test_*.py'
```

The attention-equation tests need NumPy only; decorator/cache tests also use the
already installed torch/Transformers runtime. No model or GPU is needed by the
tests. They are outside the repository's default pytest collection.

The retained-input CLI deliberately distinguishes older CLIP own-input checks
from matched backend comparisons. Existing private captures are required; no
data download is performed. Full experimental results, scope and limitations
are in `docs/rca/numerical-reference-20261031/README.md`. Executed instrumentation
snapshots there are for review and must not be run from the documentation folder.
