"""Native infer/infer_multi calls with raw output and evidence preservation."""

from __future__ import annotations

import json
import re
import time
import traceback
from pathlib import Path
from typing import Any

from .completion import (
    completion_metadata,
    diagnostic_evidence,
    infer_document,
    require_complete_pages,
)
from .config import file_sha256
from .doctor import memory_snapshot, write_json
from .model import NativeModel
from .output import assess, convert_readable, normalize, preserve_raw

PRODUCTION_MAX_NEW_TOKENS = 8192
#: Pinned digest of the frozen P2-B policy (eval/experiments/p2b-ngram-suppression/
#: policy.json). The gate compares the presented file against this digest, so a
#: locally forged schema-1 policy cannot authorize suppression
#: (docs/reviews/POLICY_IDENTITY_AUDIT.md hardening requirement).
EXPECTED_NGRAM_POLICY_SHA256 = "0095b0c47da9a693d6bef70305a4385b986ea79aac129d628a500dac0ef8e1f5"
#: Pinned digest of the frozen P2-C combined policy (eval/experiments/p2c-combined/
#: policy.json) — same suppression identity (35/1024, generated ids only) at the
#: raised budgets it lists. Same fail-closed rule: only these two digests
#: authorize suppression, ever.
EXPECTED_P2C_POLICY_SHA256 = "2068efad7e7e2e037202107027f35c483ae51fbc007099580ecae0fd7bc12397"
#: Pinned digest of the frozen L1 policy (eval/experiments/l1-fifteen-page/
#: policy.json) — the THIRD authorized digest: one supervised 15-page request at
#: budget 39321 with suppression 35/1024, a predeclared single-document
#: selection, the guard-collision observer contract and runtime resource
#: enforcement. Authorizes the L1_FIFTEEN_PAGE_NATIVE experiment only.
EXPECTED_L1_POLICY_SHA256 = "dd23a5462c28f3bd93a97c17fe5890e4ad01764de6fb39e9ba0c61a22904db6e"
#: digest -> required experiment name; the gate enforces both together.
#: the FOURTH authorized digest: the L1.1 case-B experiment (ONE supervised
#: 15-page request on paper_eng_64b5e1f4, the predeclared complex-layout
#: candidate), finalized from the admission-packet draft 64bae56a... at
#: authorization time. The only delta vs the draft is the selection_receipt
#: field (mission-workspace annotation -> committed repo path); the selection
#: receipt itself is byte-identical to the reviewed packet (8d80ae7b...).
#: Delta vs the draft = exactly the two receipt-reference fields (path +
#: inline digest), both pre-declared by the admission packet.
EXPECTED_L11_B_POLICY_SHA256 = "d95e5bd03904da635fd8b6c82b9ab0ee586c299b872bd9c5cc5a13d3592b4aa9"
_AUTHORIZED_NGRAM_POLICIES: dict[str, str] = {
    EXPECTED_NGRAM_POLICY_SHA256: "P2B_NGRAM_SUPPRESSION_RCA",
    EXPECTED_P2C_POLICY_SHA256: "P2C_COMBINED_NGRAM_BUDGET",
    EXPECTED_L1_POLICY_SHA256: "L1_FIFTEEN_PAGE_NATIVE",
    EXPECTED_L11_B_POLICY_SHA256: "L11_FIFTEEN_PAGE_NATIVE_B",
}


def validate_ngram(ngram_size: int, ngram_window: int, policy_path: Path | None) -> None:
    """No-repeat n-gram suppression is opt-in via the frozen policies only.

    The window must be passed explicitly and positively: the frozen model
    silently coerces a non-positive window to 100 (modeling_unlimitedocr.py
    infer()), which is NOT the official 1024 (docs/rca/NGRAM35_SOURCE_AUDIT.md).
    """
    if ngram_size == 0 and ngram_window == 0:
        return
    if policy_path is None:
        raise ValueError(
            "no_repeat_ngram suppression requires the frozen experimental "
            "n-gram policy (production default is suppression off)"
        )
    if ngram_size <= 0 or ngram_window <= 0:
        raise ValueError(
            "Repetition suppression requires an explicit positive ngram_size and "
            "ngram_window from the frozen policy (a non-positive window would be "
            "silently coerced to 100 by the frozen model, not the official 1024)"
        )
    try:
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Unreadable experimental n-gram policy: {error}") from error
    experiment = _AUTHORIZED_NGRAM_POLICIES.get(file_sha256(policy_path))
    if experiment is None:
        raise ValueError(
            "Experimental n-gram policy digest is not one of the pinned frozen identities"
        )
    sizes = policy.get("ngram_sizes") if isinstance(policy, dict) else None
    if (
        policy.get("schema") != 1
        or policy.get("experiment") != experiment
        or not isinstance(sizes, list)
        or ngram_size not in sizes
        or policy.get("ngram_window") != ngram_window
    ):
        raise ValueError(
            f"Requested n-gram configuration {ngram_size}/{ngram_window} is not "
            "authorized by the frozen policy"
        )


def validate_budget(max_new_tokens: int, policy_path: Path | None = None) -> None:
    """Production cap is 8192; larger budgets require the frozen P2-A policy."""
    if 1 <= max_new_tokens <= PRODUCTION_MAX_NEW_TOKENS:
        return
    if policy_path is None:
        raise ValueError(
            "max_new_tokens beyond the production cap requires the frozen "
            "experimental budget policy (production defaults are unchanged)"
        )
    try:
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Unreadable experimental budget policy: {error}") from error
    budgets = policy.get("budgets_new_tokens")
    if policy.get("schema") != 1 or not isinstance(budgets, list) or max_new_tokens not in budgets:
        raise ValueError(
            f"Requested budget {max_new_tokens} is not authorized by the frozen policy"
        )


def validate_inputs(images: list[Path], multi: bool, *, limit: int = 5) -> list[Path]:
    if not images or (not multi and len(images) != 1):
        raise ValueError("single requires exactly one image; multi requires ordered pages")
    if len(images) > limit:
        raise ValueError(
            f"Requests support at most {limit} pages; a higher page count needs the "
            "pinned-digest experimental policy that authorizes it (production default is 5)"
        )
    resolved = [path.resolve() for path in images]
    for path in resolved:
        if not path.is_file():
            raise FileNotFoundError(path)
    return resolved


_L1_PAGE_HASH_HEX = re.compile(r"\A[0-9a-f]{64}\Z")
_L1_REQUIRED_SAFETY_KEYS = frozenset(
    {"observer", "qualification", "qualification_sha256", "contract", "cpu_test_file"}
)
_L1_REQUIRED_RESOURCE_KEYS = frozenset(
    {
        "gpu_allocated_hard_limit_gib",
        "gpu_enforcement",
        "timeout_seconds",
        "timeout_rationale",
    }
)
_L1_REQUIRED_SUPERVISION_KEYS = frozenset(
    {
        "timeout_seconds_per_document",
        "process_group_sigkill",
        "grace_seconds",
        "heartbeat_seconds",
        "terminate_incomplete_is_failure",
    }
)


def validate_l1_policy(
    policy_path: Path,
    *,
    ngram_size: int,
    ngram_window: int,
    max_new_tokens: int,
    expected_pages: int,
    repo_root: Path,
    expected_digest: str | None = None,
) -> dict[str, Any]:
    """Third pinned-digest admission path: the frozen L1 15-page policy.

    Fail-closed on every axis: digest, experiment name, schema, page count,
    budget, n-gram identity, single-document identity, the 15 pinned page
    hashes (cross-checked against the pinned selection receipt), the
    guard-collision safety contract (qualification doc digest) and the
    resource/supervision admission block. ``expected_digest`` exists for
    tests; production always pins the frozen constant.
    """
    digest = expected_digest if expected_digest is not None else file_sha256(policy_path)
    experiment = _AUTHORIZED_NGRAM_POLICIES.get(digest)
    if experiment is None:
        raise ValueError("L1 policy digest is not the pinned frozen identity")
    try:
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Unreadable L1 policy: {error}") from error
    if not isinstance(policy, dict) or policy.get("experiment") != experiment:
        raise ValueError(f"L1 policy experiment name is not {experiment!r}")
    if policy.get("schema") != 1:
        raise ValueError("L1 policy schema must be 1")
    if policy.get("pages") != expected_pages:
        raise ValueError(f"L1 policy pages must be {expected_pages}")
    budgets = policy.get("budgets_new_tokens")
    if not isinstance(budgets, list) or max_new_tokens not in budgets:
        raise ValueError(f"Budget {max_new_tokens} is not authorized by the frozen L1 policy")
    sizes = policy.get("ngram_sizes")
    if (
        not isinstance(sizes, list)
        or ngram_size not in sizes
        or policy.get("ngram_window") != ngram_window
    ):
        raise ValueError(
            f"N-gram {ngram_size}/{ngram_window} is not authorized by the frozen L1 policy"
        )
    documents = policy.get("documents")
    if (
        not isinstance(documents, list)
        or len(documents) != 1
        or not all(isinstance(doc, str) and doc for doc in documents)
    ):
        raise ValueError("L1 policy needs exactly one frozen document identity")
    page_hashes = policy.get("page_hashes")
    if (
        not isinstance(page_hashes, list)
        or len(page_hashes) != expected_pages
        or any(
            not isinstance(value, str) or _L1_PAGE_HASH_HEX.fullmatch(value) is None
            for value in page_hashes
        )
    ):
        raise ValueError(f"L1 policy needs {expected_pages} pinned page SHA-256 hashes")
    _validate_l1_selection_receipt(policy, page_hashes, repo_root)
    _validate_l1_safety_and_resources(policy, repo_root)
    return policy


def _validate_l1_selection_receipt(
    policy: dict[str, Any], page_hashes: list[str], repo_root: Path
) -> None:
    receipt_rel = policy.get("selection_receipt")
    receipt_sha = policy.get("selection_receipt_sha256")
    if not isinstance(receipt_rel, str) or not receipt_rel:
        raise ValueError("L1 policy needs a frozen selection receipt reference")
    if not isinstance(receipt_sha, str) or _L1_PAGE_HASH_HEX.fullmatch(receipt_sha) is None:
        raise ValueError("L1 policy selection receipt digest must be a sha-256 hex string")
    receipt_path = repo_root / receipt_rel
    try:
        receipt_bytes = receipt_path.read_bytes()
    except OSError as error:
        raise ValueError(f"L1 selection receipt unreadable: {error}") from error
    if file_sha256(receipt_path) != receipt_sha:
        raise ValueError("L1 selection receipt digest does not match the pinned reference")
    try:
        receipt = json.loads(receipt_bytes.decode("utf-8"))
    except ValueError as error:
        raise ValueError(f"L1 selection receipt is not valid JSON: {error}") from error
    document = receipt.get("document") if isinstance(receipt, dict) else None
    if not isinstance(document, dict) or document.get("doc_id") != policy["documents"][0]:
        raise ValueError("L1 selection receipt document identity disagrees with the policy")
    receipt_pages = receipt.get("pages") if isinstance(receipt, dict) else None
    if not isinstance(receipt_pages, list) or len(receipt_pages) != len(page_hashes):
        raise ValueError("L1 selection receipt page count disagrees with the policy")
    receipt_hashes = [row.get("sha256") for row in receipt_pages if isinstance(row, dict)]
    if receipt_hashes != page_hashes:
        raise ValueError("L1 selection receipt page hashes disagree with the policy")


def _validate_l1_safety_and_resources(policy: dict[str, Any], repo_root: Path) -> None:
    safety = policy.get("guard_collision_safety")
    if not isinstance(safety, dict) or not _L1_REQUIRED_SAFETY_KEYS <= safety.keys():
        raise ValueError("L1 policy is missing the guard-collision safety qualification")
    qualification = repo_root / str(safety["qualification"])
    try:
        if file_sha256(qualification) != safety["qualification_sha256"]:
            raise ValueError("L1 guard-collision qualification digest does not match")
    except OSError as error:
        raise ValueError(f"L1 guard-collision qualification unreadable: {error}") from error
    admission = policy.get("admission")
    if not isinstance(admission, dict) or admission.get("context_cap_tokens") != 131072:
        raise ValueError("L1 policy needs the frozen 131072 context admission rule")
    total = (
        policy["pages"] * admission.get("prompt_upper_bound_tokens_per_page", 0)
        + max(policy["budgets_new_tokens"])
        + admission.get("required_margin_tokens", 0)
    )
    if total > admission["context_cap_tokens"]:
        raise ValueError("L1 policy fails its own context admission arithmetic")
    resource = policy.get("resource_admission")
    if not isinstance(resource, dict) or not _L1_REQUIRED_RESOURCE_KEYS <= resource.keys():
        raise ValueError("L1 policy is missing the resource admission block")
    if resource.get("gpu_allocated_hard_limit_gib") != 16:
        raise ValueError("The L1 GPU allocated hard limit is frozen at 16 GiB")
    supervision = policy.get("supervision")
    if not isinstance(supervision, dict) or not _L1_REQUIRED_SUPERVISION_KEYS <= supervision.keys():
        raise ValueError("L1 policy is missing the supervision block")
    timeout = supervision.get("timeout_seconds_per_document")
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
        raise ValueError("L1 supervision timeout must be a positive number")
    if timeout != resource.get("timeout_seconds"):
        raise ValueError("L1 supervision and resource blocks disagree on the timeout")


def run_native(
    native: NativeModel,
    images: list[Path],
    output: Path,
    *,
    multi: bool = False,
    truth: list[dict[str, Any]] | None = None,
    max_new_tokens: int = 1024,
    budget_policy: Path | None = None,
    no_repeat_ngram_size: int = 0,
    ngram_window: int = 0,
    ngram_policy: Path | None = None,
    page_limit: int = 5,
) -> dict[str, Any]:
    import torch
    from PIL import Image

    images = validate_inputs(images, multi, limit=page_limit)
    validate_budget(max_new_tokens, budget_policy)
    validate_ngram(no_repeat_ngram_size, ngram_window, ngram_policy)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh output directory to preserve earlier evidence")
    output.mkdir(parents=True, exist_ok=True)
    for path in images:
        with Image.open(path) as image:
            if image.width * image.height > 16_000_000:
                raise ValueError("Phase 1 fixture limit: 16 million pixels per page")
            image.verify()
    report: dict[str, Any] = {
        "native_entrypoint": "infer",
        "native_mode": "infer_multi_equivalent_base" if multi else "single_auto_crop",
        "completion_interpretation": (
            "Native EOS/length only; stop may be forced by the official page guard. "
            "Valid page structure does not prove semantic content completeness."
        ),
        "inputs": [{"path": str(path), "sha256": file_sha256(path)} for path in images],
        "memory_before": memory_snapshot(),
        "max_new_tokens": max_new_tokens,
        "model": native.evidence,
    }
    if budget_policy is not None:
        report["experimental_budget_policy_sha256"] = file_sha256(budget_policy)
    if no_repeat_ngram_size > 0 and ngram_policy is not None:
        # validate_ngram already refused suppression without the policy.
        report["no_repeat_ngram_size"] = no_repeat_ngram_size
        report["ngram_window"] = ngram_window
        report["experimental_ngram_policy_sha256"] = file_sha256(ngram_policy)
    previous_kernels = len(native.kernels)
    previous_vision = len(getattr(native, "vision", []))
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        with torch.inference_mode():
            result = infer_document(
                native.model,
                native.tokenizer,
                [str(path) for path in images],
                multi=multi,
                max_new_tokens=max_new_tokens,
                no_repeat_ngram_size=no_repeat_ngram_size,
                ngram_window=ngram_window,
            )
        # Persist all available native evidence before metadata or Markdown
        # validation. Failure must not erase returned model text/finish fields.
        report["native_generation"], malformed_evidence = diagnostic_evidence(result)
        if malformed_evidence:
            report["native_generation_diagnostic"] = True
        if isinstance(result, dict):
            report["finish_reason"] = diagnostic_evidence(result.get("finish_reason"))[0]
            raw = result.get("text")
            if isinstance(raw, str):
                preserve_raw(output / "raw.txt", raw)
                report["raw_output"] = raw
                report["raw_sha256"] = file_sha256(output / "raw.txt")
        torch.cuda.synchronize()
        # Convert only after retaining raw; metadata validation below remains
        # independent of any best-effort readable representation.
        if isinstance(result, dict) and isinstance(result.get("text"), str):
            raw = result["text"]
            preserve_raw(output / "readable.md", convert_readable(raw))
        metadata = completion_metadata(
            result, expected_pages=len(images), max_new_tokens=max_new_tokens
        )
        raw, tokens, finish = result["text"], metadata["output_tokens"], metadata["finish_reason"]
        report["upstream_timing"] = {
            "generation_seconds": result["time"],
            "prefill_seconds": result["prefill_time"],
        }
        (output / "normalized.txt").write_text(normalize(raw), encoding="utf-8")
        kernels = native.kernels[previous_kernels:]
        report.update(
            output_token_count=tokens,
            finish_reason=finish,
            raw_sha256=file_sha256(output / "raw.txt"),
            raw_output=raw,
        )
        if len(kernels) < 12:
            raise RuntimeError("No successful native DASH kernel recorded for every decoder layer")
        if not raw.strip() or finish == "length":
            raise RuntimeError("Empty or native length-truncated OCR output")
        require_complete_pages(raw, len(images), max_chars=1_000_000)
        if truth is not None:
            report["correctness"] = assess(raw, truth)
            report["status"] = report["correctness"]["status"]
        else:
            report["status"] = "INFERENCE_ONLY"
    except KeyboardInterrupt:
        report["status"] = "FAIL"
        report["error"] = "INTERRUPTED"
        report["traceback"] = traceback.format_exc()
        raise
    except Exception:
        report["status"] = "FAIL"
        report["traceback"] = traceback.format_exc()
    finally:
        kernels = native.kernels[previous_kernels:]
        report["native_kernels"] = kernels
        report["native_kernel_count"] = len(kernels)
        report["vision_inputs"] = getattr(native, "vision", [])[previous_vision:]
        report["synchronized_wall_seconds"] = time.perf_counter() - start
        try:
            report["memory_after"] = memory_snapshot()
        except Exception:
            report["memory_after_error"] = traceback.format_exc()
        # Malformed native/diagnostic values must never break the FAIL manifest.
        write_json(output / "manifest.json", diagnostic_evidence(report)[0])
    return report
