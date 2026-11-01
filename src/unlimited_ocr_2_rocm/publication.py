"""Publication gates: keep dataset-derived annotation content out of the repository.

MPDocBench/MarathonDocBench ground truth is licensed evaluation data. What may
be published are sanitized identity receipts (hashes and counts only); detector
hits are admitted only through an exact-content-hash whitelist whose entries
carry a reviewed reason. The detector is structural, so it fires on any
annotation-shaped export (rows carrying ``page_info`` + ``layout_dets``),
nested or line-delimited, not just on file names known today.

Extraction provenance: independently re-implemented on a clean baseline from
the PR #24 gate design (P0 remediation), with these deliberate changes:
recursive/JSONL annotation detection, Windows home-path and host-identity and
credential detection, binary files fail closed, ``--scan .`` prefix
normalization (the PR #24 whitelist-anchoring defect), and ``--staged``
verifying whitelisted files by hash instead of skipping them.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from .config import file_sha256

# Structural signature of one dataset annotation row (MPDocBench family).
ANNOTATION_ROW_KEYS = frozenset({"page_info", "layout_dets"})

# Published evidence may never contain private host filesystem paths
# (POSIX or Windows forms) or host-identity keys.
PRIVATE_PATH_PATTERN = re.compile(
    r"/home/[A-Za-z0-9_.-]+"
    r"|/Users/[A-Za-z0-9_.-]+"
    r"|C:[/\\]Users[/\\][A-Za-z0-9_.-]+"
)
HOST_IDENTITY_PATTERN = re.compile(r'"(?:hostname|host_name)"\s*:\s*"[^"]+"')
CREDENTIAL_PATTERN = re.compile(
    r"ghp_[A-Za-z0-9]{20,}"
    r"|gho_[A-Za-z0-9]{20,}"
    r"|github_pat_[A-Za-z0-9_]{20,}"
    r"|hf_[A-Za-z0-9]{20,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"
)

# Raw page renders and other opaque binaries are never publishable evidence.
IMAGE_SUFFIXES = frozenset(
    {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif", ".ico"}
)
BINARY_SUFFIXES = frozenset(
    {
        ".pdf",
        ".zip",
        ".gz",
        ".tgz",
        ".whl",
        ".bin",
        ".so",
        ".dll",
        ".exe",
        ".pt",
        ".pth",
        ".onnx",
        ".parquet",
        ".db",
        ".sqlite",
        ".ttf",
        ".woff",
        ".woff2",
    }
)
TEXT_SUFFIXES = frozenset(
    {
        ".json",
        ".jsonl",
        ".md",
        ".txt",
        ".py",
        ".toml",
        ".cfg",
        ".ini",
        ".yml",
        ".yaml",
        ".sh",
        ".ps1",
        ".log",
        ".csv",
        ".svg",
        ".html",
        ".css",
        ".js",
    }
)


# Never-published working-tree directories: local tooling caches and build
# output. They are not part of the committed tree (gitignore covers them); a
# full-tree audit of the PUBLISHED content should use scan_tracked().
_SKIP_PARTS = frozenset(
    {
        ".git",
        ".b",
        "__pycache__",
        ".venv",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "node_modules",
        "dist",
        "build",
        ".upstream",
    }
)
# Exactly-known files admitted despite detector matches. Any edit changes the
# hash and re-trips the gate. Do not extend this list casually: every entry
# needs an independent publication-audit reason (see docs/integration/).
WHITELIST: dict[str, dict[str, str]] = {
    "docs/reviews/PR16_INDEPENDENT_REVIEW.md": {
        "sha256": "57b8e5d047e7105ab4f550546c2437e86085f9b104945555f11fd4c9929946f2",
        "reason": "historical prose host-layout documentation",
    },
    "docs/validation/LINUX_DATASET_READINESS.md": {
        "sha256": "f0c76f2d95aace81aed1cb699e119e96fcd335c28f5aef3278d8f99e7b1b94ab",
        "reason": "historical prose host-layout documentation",
    },
    "eval/benchmarks/g05/scorer/final/pipelines/marathon-pipeline/ground-truth.json": {
        "sha256": "89ba7ccb586faa58ba86b2b49029a795f25704fe6d4191e7b850811c6fb29f7e",
        "reason": "authored synthetic annotation fixture",
    },
    "eval/benchmarks/g05/scorer/final/pipelines/qwen-pipeline/ground-truth.json": {
        "sha256": "89ba7ccb586faa58ba86b2b49029a795f25704fe6d4191e7b850811c6fb29f7e",
        "reason": "authored synthetic annotation fixture",
    },
    "eval/benchmarks/g05/scorer/preliminary/marathon-pipeline/ground-truth.json": {
        "sha256": "89ba7ccb586faa58ba86b2b49029a795f25704fe6d4191e7b850811c6fb29f7e",
        "reason": "authored synthetic annotation fixture",
    },
    "eval/benchmarks/g05/scorer/preliminary/qwen-pipeline/ground-truth.json": {
        "sha256": "89ba7ccb586faa58ba86b2b49029a795f25704fe6d4191e7b850811c6fb29f7e",
        "reason": "authored synthetic annotation fixture",
    },
    "eval/g1/windows-gfx1151/evidence.tar.gz": {
        "sha256": "491dc83a3d65528672060ef2001a2133ffb7842e13a553482693a5ec51dac9d5",
        "reason": "sealed G1 synthetic-fixture evidence; independent review PASS",
    },
    "eval/rca/gfx1151-rocm7141/pdf-followup/input/Unlimited-OCR-2.pdf": {
        "sha256": "1eb5885f4ddfa4ec6fbff375092377f30b5d6a5e034ec4936213f125bf2f9444",
        "reason": "user-supplied third-party paper (see pdf-followup/README.md)",
    },
    "eval/rca/gfx1151-rocm7141/provenance-baseline.json": {
        "sha256": "6655cf525383997a7a4ca6b475b028a719f37a261dcd2d40a1367b44702fbd82",
        "reason": "historical CI runner-home infra path",
    },
    "eval/rca/gfx1151-rocm7141/provenance-primary-after.json": {
        "sha256": "d63e2fabe1f599fba6f2c0b3af1ad740faaace91971617bc7d6aa4ca92351e6c",
        "reason": "historical CI runner-home infra path",
    },
    "eval/rca/gfx1151-rocm7141/provenance-torch-only-A-restored.json": {
        "sha256": "f721430e0ed384f89ec9935929a1874f6c2f7b1da7c9c3990be52c7ffe75bfaa",
        "reason": "historical CI runner-home infra path",
    },
    "eval/rca/gfx1151-rocm7141/provenance-torch-only-A.json": {
        "sha256": "75299be3ab9cfe441e1dfd347517d2b0f435c5b53615d53405d4725ceb23e4a5",
        "reason": "historical CI runner-home infra path",
    },
    "eval/rca/gfx1151-rocm7141/provenance-torch-only-B.json": {
        "sha256": "25c3e94fc2c20e7cecc6f5bea0e52f2dfc1cdab8d69afe1196d42aebf088df58",
        "reason": "historical CI runner-home infra path",
    },
    "eval/results/Linux gfx1151 verified/datasets/dataset-verification-attempt-1-blocked.json": {
        "sha256": "3b375270cb42512be8bb60beec54d4f1b5c55e4960c23e0ffe3f1d4052d10a67",
        "reason": "historical validation-host receipt (PR #17 review)",
    },
    "eval/results/Linux gfx1151 verified/datasets/"
    "dataset-verification-attempt-2-repaired-pass.json": {
        "sha256": "81ca87bd5b9fed70982c36d0690cb6ed9d68cce125e0d572817844d38a151ec3",
        "reason": "historical validation-host receipt (PR #17 review)",
    },
    "eval/results/Linux gfx1151 verified/datasets/dataset-verification.json": {
        "sha256": "81ca87bd5b9fed70982c36d0690cb6ed9d68cce125e0d572817844d38a151ec3",
        "reason": "historical validation-host receipt (PR #17 review)",
    },
    "eval/results/Linux gfx1151 verified/datasets/page-repair-mp3lan_0165-43.json": {
        "sha256": "b69e2a2ec9f55a95b9a1841ea1b612bcc92f0cd2c8dfb176a4f82e0cf969a4af",
        "reason": "historical validation-host receipt (PR #17 review)",
    },
    "eval/results/Linux gfx1151 verified/mpdoc-sample/run-receipt.json": {
        "sha256": "d1aa4efcc9bd05439b3d5fd86ad19394d38b9e753ac3c733cfe3339c7c21fbc1",
        "reason": "historical validation-host receipt (PR #17 review)",
    },
    "eval/wheel-validation/windows-gfx1151/continuation/build-tool-preinstall.json": {
        "sha256": "fc8e46a42cb45ebfd0d65bec0a8b962192838b024288f9ba8d8519a1cc258305",
        "reason": "historical build-container infra path",
    },
    "eval/wheel-validation/windows-gfx1151/continuation/logs/build-after-kineto.stdout.log.gz": {
        "sha256": "c0304ec60156dd9c01f787d8083c18bfa72ada22c338cbc3db114fa30af55492",
        "reason": "historical wheel-build log (compressed)",
    },
    "eval/wheel-validation/windows-gfx1151/continuation/logs/build-after-psimd.stderr.log.gz": {
        "sha256": "a82fc9f5faaf2b38fabb0f9240740bc21302d9d4f166dca9fbfb0d5339526fbe",
        "reason": "historical wheel-build log (compressed)",
    },
    "eval/wheel-validation/windows-gfx1151/continuation/logs/build-after-psimd.stdout.log.gz": {
        "sha256": "4d9ec479b631509abec7c6a6381aca0a2132d47a1ce6296451dd4130f91c16fd",
        "reason": "historical wheel-build log (compressed)",
    },
    "eval/wheel-validation/windows-gfx1151/continuation/logs/build-cached.stdout.log.gz": {
        "sha256": "d8567fbffefda10235d2c4d67ac8389f16923f00efb6471888f0f3c8b272901e",
        "reason": "historical wheel-build log (compressed)",
    },
    "eval/wheel-validation/windows-gfx1151/continuation/logs/build-deps-ready.stdout.log.gz": {
        "sha256": "ab88e787df87100b7db7bed63dcc86843cdf2f243f895779fe42b1858a8d6b17",
        "reason": "historical wheel-build log (compressed)",
    },
    "eval/wheel-validation/windows-gfx1151/continuation/logs/build-prepared.stdout.log.gz": {
        "sha256": "e7bc5326cab50a25c1766ab74994b87e4c81dd5a532b37e5eeb5d2d487f73f47",
        "reason": "historical wheel-build log (compressed)",
    },
    "eval/wheel-validation/windows-gfx1151/continuation/logs/cutlass-tree-api.json.gz": {
        "sha256": "e3682233193f7bfbdcd15eb004ff5c675dee20957d79f069cefa605a1f51b6c4",
        "reason": "historical wheel-build log (compressed)",
    },
    "eval/wheel-validation/windows-gfx1151/continuation/logs/relative-url-redaction-red.txt": {
        "sha256": "24092a7fee5462f3c83035542a0111e591a0e6767389a67e4358285a9c4389c5",
        "reason": "synthetic redaction self-test content",
    },
    "eval/wheel-validation/windows-gfx1151/continuation/logs/signed-delimiter-redaction-red.txt": {
        "sha256": "e0579849ab7c75948b478347cc4e780d8ce5d7ffcf2644042b28ab11da2df5a5",
        "reason": "synthetic redaction self-test content",
    },
    "eval/wheel-validation/windows-gfx1151/continuation/logs/signed-url-redaction-red.txt": {
        "sha256": "49128a185243eb8f9c28545b06a544b4ea528b00e01ef6d4fbce8fcc620dc9e8",
        "reason": "synthetic redaction self-test content",
    },
    "eval/wheel-validation/windows-gfx1151/continuation/logs/torch-cpu-component.stdout.log.gz": {
        "sha256": "fe01285b42dcb7f6fd980723317957a233f6fb83b64e09dda70b400a8143f405",
        "reason": "historical wheel-build log (compressed)",
    },
    "eval/wheel-validation/windows-gfx1151/provenance/baseline-provenance.json": {
        "sha256": "bb0990ab06de182d1825b8425f33990c4e771e8e241cd7bc63ac9ffa61f990f3",
        "reason": "historical build-container infra path",
    },
    "requirements/linux-cp310.lock.txt": {
        "sha256": "549072addd8b84fcc65e7422b22fec7c304566302de96f7d6a86309a80bc9447",
        "reason": (
            "clean portable dependency lock introduced by PR #14 (no machine-local "
            "wheel URL; torchvision acquired via the separately verified wheelhouse)"
        ),
    },
    "tests/fixtures/benchmark/synthetic.json": {
        "sha256": "a0974da4a956c0a8f65840ffd35cd2d1251035de433dca9cf710b32cb4d18997",
        "reason": "authored synthetic annotation fixture",
    },
    "tests/fixtures/chinese.png": {
        "sha256": "a6b464b0f4e6065d952a031b6f09e09efcc1a81a400c7195f1935e1b844e54f5",
        "reason": "repo-owned synthetic OCR test-fixture image",
    },
    "tests/fixtures/english.png": {
        "sha256": "0415e96e99e66b079e12bb7e40006a966cd5c9b3d0e18f8fb070e61bcfde4cd0",
        "reason": "repo-owned synthetic OCR test-fixture image",
    },
    "tests/fixtures/math.png": {
        "sha256": "ddf58c93e7d5a15c5110747b94b7fe1e9a7579eae3aecd1f80d4e2aca905cb90",
        "reason": "repo-owned synthetic OCR test-fixture image",
    },
    "tests/fixtures/mixed.png": {
        "sha256": "09714bd58f4042dde37d4c1152b273585f7beb86562b046e02d5b2b58205ef03",
        "reason": "repo-owned synthetic OCR test-fixture image",
    },
    "tests/fixtures/page_1.png": {
        "sha256": "25e1f9283d1444b82813ce8b9f55c34944e50dd2b90f1ad0df2dc0a329fbcb4e",
        "reason": "repo-owned synthetic OCR test-fixture image",
    },
    "tests/fixtures/page_2.png": {
        "sha256": "1bc65824178d3caef566d1f052457bf98325f2b24da1e6aab0b4f5f97d2d5625",
        "reason": "repo-owned synthetic OCR test-fixture image",
    },
    "tests/fixtures/page_3.png": {
        "sha256": "4aab8ecfb402dac49daf070704768e0ceeaab14bec572e07ee6c76310ed538d8",
        "reason": "repo-owned synthetic OCR test-fixture image",
    },
    "tests/fixtures/page_4.png": {
        "sha256": "fbbedcae51819976300ae30459c9b6e3dffc3c882b7aa2fb41d93e4b8a44d74e",
        "reason": "repo-owned synthetic OCR test-fixture image",
    },
    "tests/fixtures/page_5.png": {
        "sha256": "530773d30a2907a3644069bf80e3b7aec04f9f20f8c0cd8ccb1ce83ff96bed51",
        "reason": "repo-owned synthetic OCR test-fixture image",
    },
    "tests/fixtures/table.png": {
        "sha256": "bda4a05724a65eb16d3bc1a5685037ff9ddf2e34e80ae2fd25741fe1294563ae",
        "reason": "repo-owned synthetic OCR test-fixture image",
    },
    "tests/fixtures/table_page_1.png": {
        "sha256": "e98e7723145341e5410783a6624b73d2880fd133eefe6a3db6aa5a861a82c9c7",
        "reason": "repo-owned synthetic OCR test-fixture image",
    },
    "tests/fixtures/table_page_2.png": {
        "sha256": "99af64fe1a270a94089a2a7125a57453d315e122f4ee396a2e4bb5dacf3f0b25",
        "reason": "repo-owned synthetic OCR test-fixture image",
    },
    "tests/unit/test_native_regression_publish.py": {
        "sha256": "fd11a3e99f37e1727b192d246022955be82152b16182ce87c8e072eb4f68d9fb",
        "reason": "sanitizer test with a synthetic path literal",
    },
    "tests/unit/test_p15_runner_hardening.py": {
        "sha256": "4abd6d64abc4ab764894b89f23a1d857e0706309e9f2968c8cd03189702f6017",
        "reason": "sanitizer fixture; reviewed non-destructive Windows liveness probe",
    },
    "tests/unit/test_sdpa_rca_reporting.py": {
        "sha256": "32161608a588c21f9f00fa838d1826848d46d5ce45eb618ad98361b4de0e15bc",
        "reason": "sanitizer test with a synthetic path literal",
    },
    "tests/unit/test_wheel_validation.py": {
        "sha256": "a753e7d01c08861213a201dfc1a8635a1540c1eed510157c4af2633d2524833e",
        "reason": "sanitizer test with a synthetic path literal",
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


def _is_annotation_row(value: Any) -> bool:
    if not isinstance(value, dict) or not ANNOTATION_ROW_KEYS <= value.keys():
        return False
    page_info = value["page_info"]
    return isinstance(page_info, dict) and "image_path" in page_info


def looks_like_dataset_annotations(value: Any, *, _depth: int = 0) -> bool:
    """True for annotation-shaped exports (full GT or any slice of it).

    Matches a top-level list of annotation rows, a homogenous list of such
    rows nested anywhere inside the JSON value, and JSONL documents where any
    parsed line is such a list. Depth-bounded so pathological inputs cannot
    recurse unbounded.
    """
    if _depth > 8:
        return False
    if isinstance(value, list):
        if value and all(_is_annotation_row(row) for row in value):
            return True
        return any(looks_like_dataset_annotations(item, _depth=_depth + 1) for item in value)
    if isinstance(value, dict):
        return any(
            looks_like_dataset_annotations(item, _depth=_depth + 1) for item in value.values()
        )
    return False


def _jsonl_annotation_hit(text: str) -> bool:
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if looks_like_dataset_annotations(value):
            return True
    return False


def _text_findings(text: str) -> list[str]:
    findings: list[str] = []
    for pattern, label in (
        (PRIVATE_PATH_PATTERN, "private host path"),
        (HOST_IDENTITY_PATTERN, "host identity field"),
        (CREDENTIAL_PATTERN, "credential or secret material"),
    ):
        match = pattern.search(text)
        if match is not None:
            findings.append(f"{label} {match.group(0)[:32]!r}")
    return findings


def publication_findings(path: Path) -> list[str]:
    """Classify one file; empty list means publishable."""
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return ["binary image content is not publishable evidence"]
    if suffix in BINARY_SUFFIXES:
        return [f"binary content ({suffix}) is not publishable evidence"]
    try:
        raw = path.read_bytes()
    except OSError as error:
        return [f"unreadable evidence file: {error}"]
    if suffix not in TEXT_SUFFIXES and b"\x00" in raw[:4096]:
        return ["binary content is not publishable evidence"]
    text = raw.decode("utf-8", errors="replace")
    if suffix == ".json":
        try:
            value = json.loads(text)
        except json.JSONDecodeError as error:
            return [f"unreadable JSON evidence file: {error}"]
        if looks_like_dataset_annotations(value):
            return ["dataset annotation rows (page_info/layout_dets) — GT content"]
        text = json.dumps(value, ensure_ascii=False)
    elif suffix == ".jsonl":
        if _jsonl_annotation_hit(text):
            return ["dataset annotation rows (jsonl) — GT content"]
    return _text_findings(text)


def _normalized_prefix(root: Path, repo_root: Path) -> str:
    try:
        rel = root.resolve().relative_to(repo_root)
    except ValueError:
        return ""
    posix = rel.as_posix()
    return "" if posix in ("", ".") else posix


def scan_tree(root: Path, *, prefix: str | None = None) -> list[tuple[str, list[str]]]:
    """Walk a published tree; return (relative path, findings) violations.

    ``prefix`` anchors reported paths (and whitelist lookups) to the
    repository root. When omitted it is derived from ``root`` relative to the
    repository root, so scanning ``.`` (the repository root itself) anchors
    whitelist lookups correctly instead of under a ``./`` prefix.
    Whitelisted files are admitted only at their pinned SHA-256.
    """
    repo_root = Path(__file__).resolve().parents[2]
    if prefix is None:
        prefix = _normalized_prefix(root, repo_root)
    violations: list[tuple[str, list[str]]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part in _SKIP_PARTS for part in path.relative_to(root).parts):
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


def scan_staged(repo_root: Path) -> list[tuple[str, list[str]]]:
    """Scan files staged in the git index of ``repo_root``.

    Everything is read from the INDEX (``git cat-file blob :path``), never
    from the worktree, so the check cannot be defeated by modifying the
    worktree after staging, and rename entries (R) are scanned at their new
    path — PR #24 filtered renames out with ``--diff-filter=ACM`` and hashed
    the mutable worktree copy. Deleted paths resolve no index blob and are
    skipped (a removal cannot leak content). Whitelisted staged files are
    verified against their pinned hash (an edit re-trips the gate at commit
    time, not only at tree-scan time).
    """
    import subprocess

    out = subprocess.run(
        ["git", "-C", str(repo_root), "diff", "--cached", "--name-only", "-z"],
        check=True,
        capture_output=True,
    ).stdout
    violations: list[tuple[str, list[str]]] = []
    for raw in out.split(b"\x00"):
        # fsdecode/surrogateescape round-trips non-UTF-8 paths back through
        # os.fsencode in the subprocess call below; errors="replace" here would
        # corrupt the path and silently skip the entry.
        rel = os.fsdecode(raw)
        if not rel:
            continue
        staged = subprocess.run(
            ["git", "-C", str(repo_root), "cat-file", "blob", f":{rel}"],
            check=False,
            capture_output=True,
        )
        if staged.returncode != 0 or not staged.stdout:
            continue  # deleted from the index, or unreadable: nothing to scan
        blob = staged.stdout
        entry = WHITELIST.get(rel)
        if entry is not None:
            if hashlib.sha256(blob).hexdigest() != entry["sha256"]:
                violations.append((rel, ["whitelisted file hash changed — re-review required"]))
            continue
        probe = _index_blob_probe(rel, blob)
        try:
            findings = publication_findings(probe)
        finally:
            probe.unlink(missing_ok=True)
        if findings:
            violations.append((rel, findings))
    return violations


def _index_blob_probe(rel: str, blob: bytes) -> Path:
    """Materialize an index blob under the original suffix for classification."""
    descriptor, name = tempfile.mkstemp(prefix="pubgate-index-", suffix=Path(rel).suffix)
    probe = Path(name)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(blob)
    return probe


def scan_tracked(repo_root: Path) -> list[tuple[str, list[str]]]:
    """Scan exactly the tracked (committed) tree of ``repo_root``.

    This is the authoritative full-publication-tree scan: untracked working
    files never ship, so they are out of scope here by design.
    """
    import subprocess

    out = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "-z"],
        check=True,
        capture_output=True,
    ).stdout
    violations: list[tuple[str, list[str]]] = []
    for raw in out.split(b"\x00"):
        rel = os.fsdecode(raw)
        if not rel:
            continue
        path = repo_root / rel
        entry = WHITELIST.get(rel)
        if entry is not None:
            if not path.is_file() or file_sha256(path) != entry["sha256"]:
                violations.append((rel, ["whitelisted file hash changed — re-review required"]))
            continue
        findings = publication_findings(path) if path.is_file() else []
        if findings:
            violations.append((rel, findings))
    return violations


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
            "Identity receipt only. The raw annotation slice for these documents is "
            "not part of the published tree; it is verifiable by rebuilding the slice "
            "from the licensed dataset and comparing against slice_sha256."
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
    prose_values = (
        receipt["dataset"],
        receipt["experiment"],
        receipt["claim_limit"],
        receipt["note"],
    )
    for value in prose_values:
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
