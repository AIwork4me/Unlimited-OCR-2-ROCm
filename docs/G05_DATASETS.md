# G05 dataset acquisition and byte identity

The frozen benchmark lock and manifests remain unchanged. Dataset closure uses
the exact GT-defined ordered pages, with no substituted pages or inferred scope.
All payloads, partial downloads, raw failures and independent page indexes stay
under ignored `.b/g05-data/`. The original MPDocBench ZIP remains in its recorded
location under `.b/benchmark-data/`.

## MarathonDocBench

The required scope is 100 documents, 18,272 pages and 11,805,490,064 image bytes at
revision `fa05fccbe82ee6ec628006a1d58bb153954c3ee3`. Acquisition and the separate
complete-payload verification both passed: 100 documents, 18,272 pages and
11,805,490,064 bytes. Every page was reread against its frozen LFS SHA256 and
size, with ordered document hash lists bound to the locked scope. The receipt is
`eval/benchmarks/g05/datasets/MarathonDocBench.json`; the complete metadata-only
public index and its export binding are beside it. This establishes payload
readiness, not inference or benchmark quality.

`scripts/g05_acquire_datasets.py` downloads from the official pinned Hugging Face
resolve paths or the official Git LFS batch endpoint with bounded workers. Batch
responses must contain the exact frozen OID/size pairs and official HTTPS download
hosts; signed download actions remain in memory. It supports validated HTTP Range continuation,
restarts if Range is ignored, and accepts a final page only after matching its
LFS size and SHA256. A subsequent independent sequential reread verifies all local
page bytes. The local metadata's per-document ordered LFS hashes, page paths and
annotation hashes must match the frozen manifest authenticated by the benchmark
lock's scope SHA256. Metadata cannot silently redefine a page's expected hash.

Run from the repository root:

```powershell
.b/benchmark-cpu/Scripts/python.exe scripts/g05_acquire_datasets.py acquire --workers 48 --transport lfs-batch
.b/benchmark-cpu/Scripts/python.exe scripts/g05_acquire_datasets.py verify
```

The official pinned metadata declares Apache-2.0. This repository records metadata
and hashes, and does not redistribute page images or GT transcriptions.

## MPDocBench and SlideVQA

The original 600,152,009-byte ZIP SHA256 is
`f9a4c05d586b11a749e60ce13565090fdfbf8f4387fd35e5511a4a1a3d682d22`.
An independent full archive reread, original page SHA256 comparison against the
historical inventory and a reread of extracted files verified 409 documents and
3,051 original pages. The historical inventory itself matches the hash in the
published acquisition receipt. Authorized SlideVQA acquisition supplied the
remaining 11 decks and 84 pages. A separate full-scope verification reread the
original ZIP, all 12 input shards and all 3,135 GT-defined output pages: the
complete locked scope is now verified at **420 documents and 3,135 pages**, with
all 3,051 original page hashes unchanged and no missing paths.

The complete gate receipt is
`eval/benchmarks/g05/datasets/MPDocBench-full-scope.json`; its public metadata-only
`MPDocBench-page-index.json` records every page's hash and byte count. The earlier
`MPDocBench.json` describes the original archive's partial scope and is retained
as provenance, alongside the original published acquisition receipt.

Reproduce the complete independent reconciliation from the repository root after
acquisition:

```powershell
.b/benchmark-cpu/Scripts/python.exe scripts/g05_reconcile_mpdoc.py
```

This tracked producer rehashes the original ZIP, all 12 pinned SlideVQA shards and
every GT-defined output page. Its data checks match the retained original local
producer; the new receipt additionally links the serialization runtime receipt.

```powershell
.b/benchmark-cpu/Scripts/python.exe scripts/g05_verify_mpdoc.py
```

The pinned SlideVQA revision is `e780ea258d734958befbfaed19fa059584642ca8`.
A direct official test-shard probe initially returned HTTP 401 with `GatedRepo`
while no active local credential was available. After the user confirmed account
access and made a standard login available, an authenticated pinned-shard probe
returned HTTP 206. All 12 pinned test shards, totaling 1,393,939,599 bytes, passed
their frozen SHA256 checks and a separate reread. No alternate images, gate bypass
or Baidu contact was used. Credentials were read in memory and were not copied or
included in evidence.

The authorized acquisition uses the existing scorer environment without executing
remote dataset code:

```powershell
.b/benchmark-scorer/Scripts/python.exe scripts/g05_acquire_slidevqa.py
```

The reviewed serialization versions are datasets 5.1.0, pyarrow 25.0.1 and Pillow
10.4.0; the command refuses version drift. It independently rehashes all 12 frozen
test shards and rereads all output JPEGs. Local synthetic Parquet tests verify
exact default JPEG output, last-matching-row behavior, and rejection of null
expected pages. Bearer headers are removed from CDN redirects. The initial 401
receipt is preserved as `SlideVQA-attempt-1.json`; a subsequent authorized attempt
gets a separate receipt.
The completed authorized attempt is recorded in `SlideVQA-attempt-2.json` and
`SlideVQA.json`. Its 60 matching test rows produced exactly 84 final JPEGs in the
11 frozen decks, preserving the official last-matching-row overwrite semantics.
`SlideVQA-runtime.json` supplements the successful serialization evidence with
the actual interpreter's `sys.version`, executable and base-runtime hashes,
Pillow imaging-binary hash and JPEG codec identity: CPython 3.10.20, JPEG codec
8.0 and libjpeg-turbo 3.0.3. The full-scope receipt links this runtime file by hash.

The reviewed Qwen downloader at `4e5aee32280febb7b0844237ba087f837e3c8905`
has SHA256 `8cb011d1350db47ad42a8dfc50939b1bd1093890e84834ce4cbf92d3f9f03382`.
It selects exact `deck_url` matches from the SlideVQA test split, decodes each
`page_i` through `datasets.Image`, and saves the image with Pillow's default JPEG
settings as `images/<document>/page_i.jpg`. Consequently, an authorized completion
must record input shard hashes and the Python/datasets/Pillow versions and output
hashes for that official serialization. Original ZIP pages remain byte-for-byte.

MPDocBench image metadata declares CC-BY-NC-SA-4.0. The SlideVQA gate separately
provides an NTT evaluation agreement for internal testing, analysis and evaluation,
with restrictions on transfer/disclosure and modification. Neither unrestricted
redistribution nor a general commercial-use grant is inferred from these sources.
See the pinned upstream metadata and agreement for the complete terms.
The receipt links the immutable API metadata and README revision URLs and hashes
the retained metadata/agreement bytes. The README request was also gated; no
external unpinned agreement was substituted. These are recorded access constraints,
not a claim of legal permission.

Sanitized public receipts are in `eval/benchmarks/g05/datasets/`; raw access
failure bytes are local under `.b/g05-data/`. Credentials and signed URLs are
excluded from receipts. Dataset acquisition and byte identity do not constitute a
GPU inference or quality-result claim.
