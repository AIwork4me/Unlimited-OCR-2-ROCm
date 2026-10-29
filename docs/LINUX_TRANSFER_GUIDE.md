# Reuse verified downloads on Linux

Copy these repository-relative directories to the same relative locations in
the Linux checkout. They are ordinary files, not Windows symbolic-link trees.
No GPU/runtime portability is inferred from copying model and dataset bytes.

| Directory | Verified contents | Bytes |
| --- | --- | ---: |
| `.b/h/models--baidu--Unlimited-OCR-2` | Model, pinned code/config/tokenizer and two weight shards | 6,699,236,093 |
| `.b/g05-data/MPDocBench` | 420 documents / 3,135 page images, including the 84 authorized supplemental JPEGs | 688,382,455 |
| `.b/g05-data/MarathonDocBench` | 100 documents / 18,272 page images | 11,805,490,064 |

Preserve `snapshots/9cf077c4b206e0e2a31879df937b0b0bcba3661c/` inside the model
cache. Python bytecode caches can be omitted. Copy these GT files separately;
the image directories do not contain them:

- `.b/benchmark-audit/qwen/MPDocBench/MPDocBench.json`
- `.b/benchmark-audit/marathon/MarathonDocBench.json`

Optional reusable CPU-scoring downloads:

- `.b/g05-scorer/source/`: both clean, individually hash-locked official profiles.
- `.b/g05-scorer/downloads/`: Linux x86_64 Node, fonts and Ubuntu 22.04 packages;
  roughly 99 MB. Distribution-specific packages need a compatible target.
- `.b/g05-scorer/cache/pip/`: Linux CPython 3.10 scorer dependency cache,
  roughly 255 MB; rebuild the venv rather than moving its absolute paths.
- `.b/g05-data/SlideVQA/data/`: all twelve pinned Parquet shards,
  1,393,939,599 bytes; needed only for source/serialization reconciliation.

For the existing complete source-reconciliation scripts, also retain the small
inputs below at their original repository-relative paths:

- `.b/benchmark-audit/marathon-meta.json`
- `.b/benchmark-audit/slidevqa-meta.json`
- `.b/benchmark-audit/mpdoc-archive-inventory-final.json`
- `.b/benchmark-audit/qwen/MPDocBench/tools/download_data_from_slidevqa.py`
- `.b/benchmark-data/MPDocBench_data.zip` (600,152,009 bytes), if rechecking the
  original archive as well as the extracted pages.

Pull the tracked source and `eval/benchmarks/` receipts from PR #9's branch;
they contain all expected source, dataset and page-index hashes. Rehash after
transfer before execution. Use fresh Linux evidence paths and retain the copied
Windows receipts as historical source evidence; commands that write to fixed
receipt paths must not silently replace them. Cached restricted payloads remain
outside Git and shared/public archives; do not transfer authentication files.

Do not reuse Windows venvs, Windows wheel/SDK directories, the custom candidate
wheel, Windows Triton compiled cache, or the WSL venv as a ready Linux runtime.
Build the Linux environment separately with reviewed Linux pins and physically
validate its GPU. Copying these files saves downloads; it does not establish
Windows/Linux or native/SGLang equivalence.
