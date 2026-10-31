"""Publication gates: keep dataset-derived annotation content out of the repository.

MPDocBench/MarathonDocBench ground truth is licensed evaluation data. Scorer
inputs built from it (GT slices, prediction exports) live under the ignored
``.b/`` workspace and are preserved outside Git; what may be published is a
sanitized identity receipt — hashes, counts and approved metadata only. The
detector here is structural, so it fires on any annotation-shaped export
(a list of rows carrying ``page_info`` + ``layout_dets``), not just on the
file names known today.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PosixPath
from typing import Any

from .config import file_sha256

# Structural signature of one dataset annotation row (MPDocBench family).
ANNOTATION_ROW_KEYS = frozenset({"page_info", "layout_dets"})

# Published evidence may never contain private host filesystem paths.
PRIVATE_PATH_PATTERN = re.compile(r"/home/[A-Za-z0-9_.-]+|/Users/[A-Za-z0-9_.-]+")
# Windows equivalents: escaped or raw backslashes both match (JSON text and
# plain-text evidence carry different escaping of the same leak).
_WINDOWS_PATH_PATTERN = re.compile(r"[A-Za-z]:[/\\]+Users[/\\]+[A-Za-z0-9_.\- ]+", re.IGNORECASE)

# Binary image files are never publishable evidence (page renders, crops).
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"})

# Exactly-known files admitted despite detector matches. Any edit changes the
# hash and re-trips the gate. Synthetic fixtures reuse the annotation shape;
# the historical receipts predate the gate (PR #17 review: history left
# intact, future receipts sanitized) — do not extend this list casually.
WHITELIST: dict[str, dict[str, str]] = {
    "eval/benchmarks/g05/scorer/final/pipelines/marathon-pipeline/ground-truth.json": {
        "sha256": "89ba7ccb586faa58ba86b2b49029a795f25704fe6d4191e7b850811c6fb29f7e",
        "reason": "synthetic scorer self-test fixture",
    },
    "eval/benchmarks/g05/scorer/final/pipelines/qwen-pipeline/ground-truth.json": {
        "sha256": "89ba7ccb586faa58ba86b2b49029a795f25704fe6d4191e7b850811c6fb29f7e",
        "reason": "synthetic scorer self-test fixture",
    },
    "eval/benchmarks/g05/scorer/preliminary/marathon-pipeline/ground-truth.json": {
        "sha256": "89ba7ccb586faa58ba86b2b49029a795f25704fe6d4191e7b850811c6fb29f7e",
        "reason": "synthetic scorer self-test fixture",
    },
    "eval/benchmarks/g05/scorer/preliminary/qwen-pipeline/ground-truth.json": {
        "sha256": "89ba7ccb586faa58ba86b2b49029a795f25704fe6d4191e7b850811c6fb29f7e",
        "reason": "synthetic scorer self-test fixture",
    },
    "eval/rca/gfx1151-rocm7141/provenance-baseline.json": {
        "sha256": "6655cf525383997a7a4ca6b475b028a719f37a261dcd2d40a1367b44702fbd82",
        "reason": "historical GitHub Actions runner-home infra path",
    },
    "eval/rca/gfx1151-rocm7141/provenance-primary-after.json": {
        "sha256": "d63e2fabe1f599fba6f2c0b3af1ad740faaace91971617bc7d6aa4ca92351e6c",
        "reason": "historical GitHub Actions runner-home infra path",
    },
    "eval/rca/gfx1151-rocm7141/provenance-torch-only-A-restored.json": {
        "sha256": "f721430e0ed384f89ec9935929a1874f6c2f7b1da7c9c3990be52c7ffe75bfaa",
        "reason": "historical GitHub Actions runner-home infra path",
    },
    "eval/rca/gfx1151-rocm7141/provenance-torch-only-A.json": {
        "sha256": "75299be3ab9cfe441e1dfd347517d2b0f435c5b53615d53405d4725ceb23e4a5",
        "reason": "historical GitHub Actions runner-home infra path",
    },
    "eval/rca/gfx1151-rocm7141/provenance-torch-only-B.json": {
        "sha256": "25c3e94fc2c20e7cecc6f5bea0e52f2dfc1cdab8d69afe1196d42aebf088df58",
        "reason": "historical GitHub Actions runner-home infra path",
    },
    "eval/results/Linux gfx1151 verified/datasets/dataset-verification-attempt-1-blocked.json": {
        "sha256": "3b375270cb42512be8bb60beec54d4f1b5c55e4960c23e0ffe3f1d4052d10a67",
        "reason": "historical validation-host home-path receipt; PR #17 reviewed",
    },
    "eval/results/Linux gfx1151 verified/datasets/"
    "dataset-verification-attempt-2-repaired-pass.json": {
        "sha256": "81ca87bd5b9fed70982c36d0690cb6ed9d68cce125e0d572817844d38a151ec3",
        "reason": "historical validation-host home-path receipt; PR #17 reviewed",
    },
    "eval/results/Linux gfx1151 verified/datasets/dataset-verification.json": {
        "sha256": "81ca87bd5b9fed70982c36d0690cb6ed9d68cce125e0d572817844d38a151ec3",
        "reason": "historical validation-host home-path receipt; PR #17 reviewed",
    },
    "eval/results/Linux gfx1151 verified/datasets/page-repair-mp3lan_0165-43.json": {
        "sha256": "b69e2a2ec9f55a95b9a1841ea1b612bcc92f0cd2c8dfb176a4f82e0cf969a4af",
        "reason": "historical validation-host home-path receipt; PR #17 reviewed",
    },
    "eval/results/Linux gfx1151 verified/mpdoc-sample/run-receipt.json": {
        "sha256": "d1aa4efcc9bd05439b3d5fd86ad19394d38b9e753ac3c733cfe3339c7c21fbc1",
        "reason": "historical validation-host home-path receipt; PR #17 reviewed",
    },
    "eval/wheel-validation/windows-gfx1151/continuation/build-tool-preinstall.json": {
        "sha256": "fc8e46a42cb45ebfd0d65bec0a8b962192838b024288f9ba8d8519a1cc258305",
        "reason": "historical build-container default-user path",
    },
    "eval/wheel-validation/windows-gfx1151/continuation/logs/relative-url-redaction-red.txt": {
        "sha256": "24092a7fee5462f3c83035542a0111e591a0e6767389a67e4358285a9c4389c5",
        "reason": "synthetic redaction-test macOS user-path string",
    },
    "eval/wheel-validation/windows-gfx1151/continuation/logs/signed-delimiter-redaction-red.txt": {
        "sha256": "e0579849ab7c75948b478347cc4e780d8ce5d7ffcf2644042b28ab11da2df5a5",
        "reason": "synthetic redaction-test macOS user-path string",
    },
    "eval/wheel-validation/windows-gfx1151/continuation/logs/signed-url-redaction-red.txt": {
        "sha256": "49128a185243eb8f9c28545b06a544b4ea528b00e01ef6d4fbce8fcc620dc9e8",
        "reason": "synthetic redaction-test macOS user-path string",
    },
    "eval/wheel-validation/windows-gfx1151/provenance/baseline-provenance.json": {
        "sha256": "bb0990ab06de182d1825b8425f33990c4e771e8e241cd7bc63ac9ffa61f990f3",
        "reason": "historical GitHub Actions runner-home infra path",
    },
    # Historical binary artifacts, hash-pinned when binary-like content became
    # fail-closed (Task A independent review F8): any byte drift re-trips the
    # gate, and no NEW binary-like file can pass anywhere.
    "eval/rca/gfx1151-rocm7141/pdf-followup/input/Unlimited-OCR-2.pdf": {
        "sha256": "1eb5885f4ddfa4ec6fbff375092377f30b5d6a5e034ec4936213f125bf2f9444",
        "reason": "historical committed RCA test input (F8: binary fail-closed)",
    },
    "eval/wheel-validation/windows-gfx1151/continuation/logs/build-after-kineto.stdout.log.gz": {
        "sha256": "c0304ec60156dd9c01f787d8083c18bfa72ada22c338cbc3db114fa30af55492",
        "reason": "historical compressed wheel-validation log (F8: binary fail-closed)",
    },
    "eval/wheel-validation/windows-gfx1151/continuation/logs/build-after-psimd.stderr.log.gz": {
        "sha256": "a82fc9f5faaf2b38fabb0f9240740bc21302d9d4f166dca9fbfb0d5339526fbe",
        "reason": "historical compressed wheel-validation log (F8: binary fail-closed)",
    },
    "eval/wheel-validation/windows-gfx1151/continuation/logs/build-after-psimd.stdout.log.gz": {
        "sha256": "4d9ec479b631509abec7c6a6381aca0a2132d47a1ce6296451dd4130f91c16fd",
        "reason": "historical compressed wheel-validation log (F8: binary fail-closed)",
    },
    "eval/wheel-validation/windows-gfx1151/continuation/logs/build-cached.stdout.log.gz": {
        "sha256": "d8567fbffefda10235d2c4d67ac8389f16923f00efb6471888f0f3c8b272901e",
        "reason": "historical compressed wheel-validation log (F8: binary fail-closed)",
    },
    "eval/wheel-validation/windows-gfx1151/continuation/logs/build-deps-ready.stdout.log.gz": {
        "sha256": "ab88e787df87100b7db7bed63dcc86843cdf2f243f895779fe42b1858a8d6b17",
        "reason": "historical compressed wheel-validation log (F8: binary fail-closed)",
    },
    "eval/wheel-validation/windows-gfx1151/continuation/logs/build-prepared.stdout.log.gz": {
        "sha256": "e7bc5326cab50a25c1766ab74994b87e4c81dd5a532b37e5eeb5d2d487f73f47",
        "reason": "historical compressed wheel-validation log (F8: binary fail-closed)",
    },
    "eval/wheel-validation/windows-gfx1151/continuation/logs/cutlass-tree-api.json.gz": {
        "sha256": "e3682233193f7bfbdcd15eb004ff5c675dee20957d79f069cefa605a1f51b6c4",
        "reason": "historical compressed wheel-validation log (F8: binary fail-closed)",
    },
    "eval/wheel-validation/windows-gfx1151/continuation/logs/torch-cpu-component.stdout.log.gz": {
        "sha256": "fe01285b42dcb7f6fd980723317957a233f6fb83b64e09dda70b400a8143f405",
        "reason": "historical compressed wheel-validation log (F8: binary fail-closed)",
    },
}

RECEIPT_SCHEMA = 2
RECEIPT_LABEL = "SANITIZED_GT_IDENTITY_RECEIPT"
_RECEIPT_TOP_KEYS = frozenset(
    {
        "schema",
        "label",
        "dataset",
        "experiment",
        "run_id",
        "policy_sha256",
        "ground_truth_sha256",
        "slice_sha256",
        "claim_limit",
        "note",
        "documents",
    }
)
_RECEIPT_DOC_KEYS = frozenset({"pages", "layout_dets", "relation_edges"})
_HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")
_CJK = re.compile(r"[\u4e00-\u9fff]")


def looks_like_dataset_annotations(value: Any) -> bool:
    """True for annotation-shaped exports (full GT or any slice of it)."""
    if not isinstance(value, list) or not value:
        return False
    for row in value:
        if not isinstance(row, dict) or not ANNOTATION_ROW_KEYS <= row.keys():
            return False
        page_info = row["page_info"]
        if not isinstance(page_info, dict) or "image_path" not in page_info:
            return False
    return True


#: Bounds for the nested-row search. A value whose scan exceeds either bound
#: cannot be proven clean, so it fails closed (treated as GT content). The
#: node budget sits far above the largest committed evidence file (the
#: Marathon page index, ~73k nodes) while the full 420-document GT alone
#: measures ~1.09M nodes — budget refusals land on GT-scale exports.
_ANNOTATION_SEARCH_DEPTH = 8
_ANNOTATION_SEARCH_NODES = 500_000


def _contains_annotation_rows(value: Any) -> bool:
    """True when annotation-shaped rows appear anywhere inside a JSON value.

    Bounded iterative search: nested exports (rows wrapped in arbitrary
    containers) must not escape the detector. Oversized structures fail
    closed instead of being trusted.
    """
    stack: list[tuple[Any, int]] = [(value, 0)]
    nodes = 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > _ANNOTATION_SEARCH_NODES or depth > _ANNOTATION_SEARCH_DEPTH:
            return True  # fail closed: oversized/deep candidate is not provably clean
        if isinstance(item, dict) and ANNOTATION_ROW_KEYS <= item.keys():
            # Key presence is the criterion (matching the top-level detector):
            # type-mangled rows (page_info as str, image_path as int) are GT
            # content just the same and must not escape when wrapped.
            return True
        if isinstance(item, dict):
            stack.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            stack.extend((child, depth + 1) for child in item)
    return False


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """object_pairs_hook that refuses duplicate keys ( smuggling vector: a
    later duplicate can hide annotation rows from the parsed value while the
    raw bytes carry them)."""
    keys = [key for key, _ in pairs]
    duplicates = {key for key in keys if keys.count(key) > 1}
    if duplicates:
        raise ValueError(f"duplicate JSON keys: {sorted(duplicates)[:4]}")
    return dict(pairs)


def _parse_json_strict(text: str) -> Any:
    return json.loads(text, object_pairs_hook=_reject_duplicate_keys)


def _findings_for_bytes(suffix: str, data: bytes) -> list[str]:
    """Shared detection core over raw bytes; used for files and index blobs."""
    if suffix in IMAGE_SUFFIXES:
        return ["binary image content is not publishable evidence"]
    if suffix == ".json":
        try:
            value = _parse_json_strict(data.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as error:
            return [f"unreadable JSON evidence file: {error}"]
        if looks_like_dataset_annotations(value):
            return ["dataset annotation rows (page_info/layout_dets) — GT content"]
        if _contains_annotation_rows(value):
            return ["dataset annotation rows nested inside a JSON export — GT content"]
        text = json.dumps(value, ensure_ascii=False)
    elif suffix in {".jsonl", ".ndjson"}:
        try:
            full_text = data.decode("utf-8")
        except UnicodeDecodeError as error:
            return [f"unreadable JSONL evidence file: {error}"]
        lines = full_text.splitlines()
        text_parts: list[str] = []
        for number, line in enumerate(lines, start=1):
            text_parts.append(line)
            if not line.strip():
                continue
            try:
                value = _parse_json_strict(line)
            except ValueError as error:
                return [f"unreadable JSONL evidence at line {number}: {error}"]
            if looks_like_dataset_annotations(value) or _contains_annotation_rows(value):
                return [
                    f"dataset annotation rows (page_info/layout_dets) — GT content (line {number})"
                ]
        text = "\n".join(text_parts)
    else:
        if b"\x00" in data[:4096]:
            return ["binary-like evidence file is not publishable (fail-closed)"]
        text = data.decode("utf-8", errors="replace")
    return _private_path_findings(text)


def _private_path_findings(text: str) -> list[str]:
    for pattern in (PRIVATE_PATH_PATTERN, _WINDOWS_PATH_PATTERN):
        match = pattern.search(text)
        if match:
            return [f"private host path {match.group(0)!r}"]
    return []


def publication_findings(path: Path) -> list[str]:
    """Classify one file; empty list means publishable."""
    try:
        data = path.read_bytes()
    except OSError as error:
        return [f"unreadable evidence file: {error}"]
    return _findings_for_bytes(path.suffix.lower(), data)


def scan_tree(root: Path, *, prefix: str = "") -> list[tuple[str, list[str]]]:
    """Walk a published tree; return (relative path, findings) violations.

    ``prefix`` anchors reported paths (and whitelist lookups) to the
    repository root, e.g. ``scan_tree(repo / "eval", prefix="eval")``.
    Whitelisted fixtures are admitted only at their pinned SHA-256.
    """
    violations: list[tuple[str, list[str]]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part in {".git", ".b", "__pycache__"} for part in path.parts):
            continue
        rel = (f"{prefix}/" if prefix else "") + path.relative_to(root).as_posix()
        entry = WHITELIST.get(rel)
        if entry is not None:
            if file_sha256(path) != entry["sha256"]:
                violations.append((rel, ["whitelisted file hash changed — re-review required"]))
            continue
        findings = publication_findings(path)
        if findings:
            violations.append((rel, findings))
    return violations


def scan_staged_files(
    entries: list[tuple[str, bytes | None]], *, whitelist: dict[str, dict[str, str]] | None = None
) -> list[tuple[str, list[str]]]:
    """Scan staged index blobs; whitelisted paths are hash-verified here.

    ``entries`` carry what the INDEX holds (``git cat-file blob :path``), not
    the worktree: a stage-tamper-restore sequence cannot defeat verification,
    and quoted/unreadable worktree paths cannot silently pass. A ``None``
    blob for a path that exists in the index is itself a violation.
    """
    table = WHITELIST if whitelist is None else whitelist
    violations: list[tuple[str, list[str]]] = []
    for rel, data in entries:
        entry = table.get(rel)
        if entry is not None:
            if data is None or hashlib.sha256(data).hexdigest() != entry["sha256"]:
                violations.append((rel, ["whitelisted file hash changed — re-review required"]))
            continue
        if data is None:
            violations.append((rel, ["unreadable staged blob — fail-closed"]))
            continue
        findings = _findings_for_bytes(PosixPath(rel).suffix.lower(), data)
        if findings:
            violations.append((rel, findings))
    return violations


def parse_staged_name_status(raw: bytes) -> list[tuple[str, bool]]:
    """Parse ``git diff --cached --name-status -z`` output.

    Returns (path, must_exist_in_index) pairs; deletions are skipped (a
    deletion commits no content) and rename/copy entries report BOTH
    endpoints — the destination must exist in the index, the source may have
    been removed by the rename (its bytes are the destination's bytes).
    """
    tokens = raw.split(b"\0")
    parsed: list[tuple[str, bool]] = []
    index = 0
    while index < len(tokens):
        status = tokens[index].decode("utf-8", errors="replace")
        if not status:
            index += 1
            continue
        kind = status[:1]
        if kind == "D":
            index += 2
            continue
        if kind in {"R", "C"}:
            old = tokens[index + 1].decode("utf-8", errors="surrogateescape")
            new = tokens[index + 2].decode("utf-8", errors="surrogateescape")
            parsed.append((old, kind == "C"))
            parsed.append((new, True))
            index += 3
            continue
        path = tokens[index + 1].decode("utf-8", errors="surrogateescape")
        parsed.append((path, True))
        index += 2
    return parsed


def build_gt_slice_receipt(
    slice_path: Path,
    *,
    dataset: str,
    experiment: str,
    run_id: str,
    policy_sha256: str,
    ground_truth_sha256: str,
    claim_limit: str,
) -> dict[str, Any]:
    """Derive the publishable identity receipt from a real GT slice.

    The receipt carries hashes and counts only; the annotation bytes stay
    under ``.b/`` (never committed) and in the external evidence archive.
    """
    value = json.loads(slice_path.read_text(encoding="utf-8"))
    if not looks_like_dataset_annotations(value):
        raise ValueError("Input is not an annotation-shaped GT slice")
    documents: dict[str, dict[str, int]] = {}
    for row in value:
        page_info = row["page_info"]
        doc_id = page_info["image_path"].removesuffix(".pdf")
        if doc_id in documents:
            raise ValueError(f"Duplicate document in slice: {doc_id}")
        pages = len(page_info.get("images_list", ())) or len(page_info.get("annotations_list", ()))
        if pages <= 0:
            raise ValueError(f"No page inventory for {doc_id}")
        relation = row.get("extra") or {}
        documents[doc_id] = {
            "pages": pages,
            "layout_dets": len(row["layout_dets"]),
            "relation_edges": len(relation.get("relation", ())),
        }
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "label": RECEIPT_LABEL,
        "dataset": dataset,
        "experiment": experiment,
        "run_id": run_id,
        "policy_sha256": policy_sha256,
        "ground_truth_sha256": ground_truth_sha256,
        "slice_sha256": file_sha256(slice_path),
        "claim_limit": claim_limit,
        "note": (
            "Identity receipt only. The raw annotation slice for these documents was "
            "removed from the published tree (P0 publication remediation); it is "
            "verifiable by rebuilding the slice from the licensed dataset and "
            "comparing against slice_sha256."
        ),
        "documents": documents,
    }
    validate_receipt(receipt)
    return receipt


def validate_receipt(receipt: dict[str, Any]) -> None:
    """Fail closed unless the receipt carries hashes/counts and nothing else."""
    if set(receipt) != _RECEIPT_TOP_KEYS:
        raise ValueError(f"Receipt keys must be exactly {sorted(_RECEIPT_TOP_KEYS)}")
    if receipt["schema"] != RECEIPT_SCHEMA or receipt["label"] != RECEIPT_LABEL:
        raise ValueError("Unknown receipt schema or label")
    for key in ("dataset", "experiment", "claim_limit", "note"):
        value = receipt[key]
        if not isinstance(value, str) or not value or len(value) > 400:
            raise ValueError(f"Receipt field {key} must be a bounded non-empty string")
    for key in ("run_id", "policy_sha256", "ground_truth_sha256", "slice_sha256"):
        value = receipt[key]
        if not isinstance(value, str) or not _HEX64.fullmatch(value):
            raise ValueError(f"Receipt field {key} must be a lowercase sha-256 hex digest")
    prose = [receipt["dataset"], receipt["experiment"], receipt["claim_limit"], receipt["note"]]
    for value in prose:
        if _CJK.search(value):
            raise ValueError("Receipt prose must stay ASCII to exclude dataset text")
    documents = receipt["documents"]
    if not isinstance(documents, dict) or not documents:
        raise ValueError("Receipt needs a non-empty documents map")
    for doc_id, counts in documents.items():
        if not isinstance(doc_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", doc_id):
            raise ValueError(f"Unsanitized document id: {doc_id!r}")
        if not isinstance(counts, dict) or set(counts) != _RECEIPT_DOC_KEYS:
            raise ValueError(f"Document counts keys must be exactly {sorted(_RECEIPT_DOC_KEYS)}")
        for name in _RECEIPT_DOC_KEYS:
            if type(counts[name]) is not int or counts[name] < 0:
                raise ValueError(f"Document count {name} must be a non-negative integer")
