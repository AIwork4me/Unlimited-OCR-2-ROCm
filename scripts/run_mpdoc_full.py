"""Full MPDocBench campaign runner: deterministic full-scope dry-run, frozen
run inputs, and ledger-integrated execution behind explicit authorization.

Commands:
  dryrun   — read-only classification of EVERY manifest document (page counts,
             per-page hash verification against the frozen D0 page index,
             context/prompt admission arithmetic, execution identity). No GPU,
             no ledger, no exclusions: the report must account for 420/420.
  prepare  — frozen run inputs (full-dataset manifest via the official
             denominators, config with the authorization block, environment).
  run      — build the execution plan (authorization gate) and execute it
             through benchmark.runner.execute_selection (resume, timeout,
             OOM/traceback classification, sanitized receipts).
  worker   — internal: one supervised attempt (watchdog + pinned-policy
             admission inside run_native; the pinned budget gate refuses any
             budget beyond 8192 until a reviewed campaign policy exists).
  report / export / gt-slice — accounting and scorer hand-off.

The frozen L1/L1.1-B policies authorize single predeclared documents only.
A 420-document campaign requires a NEW reviewed campaign policy plus an
explicit owner authorization in the run config; without them the runner
executes exactly the production-class documents and reports the rest as
explicitly blocked. Nothing is silently excluded.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
try:
    from unlimited_ocr_2_rocm.benchmark import campaign
    from unlimited_ocr_2_rocm.benchmark import runner as benchmark_runner
    from unlimited_ocr_2_rocm.benchmark.ledger import Run
    from unlimited_ocr_2_rocm.benchmark.manifest import (
        digest,
        inventory,
        validate_manifest,
    )
    from unlimited_ocr_2_rocm.config import file_sha256
except ImportError:  # pragma: no cover - direct script execution without install
    sys.path.insert(0, str(_REPO_ROOT / "src"))
    from unlimited_ocr_2_rocm.benchmark import campaign
    from unlimited_ocr_2_rocm.benchmark import runner as benchmark_runner
    from unlimited_ocr_2_rocm.benchmark.ledger import Run
    from unlimited_ocr_2_rocm.benchmark.manifest import (
        digest,
        inventory,
        validate_manifest,
    )
    from unlimited_ocr_2_rocm.config import file_sha256

PAGE_INDEX_REPO_PATH = "eval/benchmarks/g05/datasets/MPDocBench-page-index.json"
SELF_TEST_MODES = ("good", "length", "oom", "crash", "hang", "malformed")


def _document_id(row: dict[str, Any]) -> str:
    return row["page_info"]["image_path"].removesuffix(".pdf")


def _verify_document_pages(
    pages: list[str], data_root: Path, index: dict[str, dict[str, Any]]
) -> tuple[str, int]:
    """Per-page hash verdict; the document inherits the worst verdict.

    Returns (verdict, mismatch_count). Verdicts: VERIFIED, MISMATCH,
    MISSING_PAGE_FILE, UNINDEXED — a document is never dropped for failing
    verification, it is reported.
    """
    mismatches = 0
    verdict = "VERIFIED"
    for rel in pages:
        entry = index.get(rel)
        if entry is None:
            return "UNINDEXED", mismatches
        path = data_root / rel
        if not path.is_file():
            return "MISSING_PAGE_FILE", mismatches
        if file_sha256(path) != entry["sha256"]:
            verdict = "MISMATCH"
            mismatches += 1
    return verdict, mismatches


def dryrun_command(args: argparse.Namespace) -> int:
    """Full-scope, zero-exclusion, deterministic classification report."""
    lock = json.loads(args.lock.read_text("utf-8"))
    pin = lock["datasets"][args.dataset]
    index_rows = json.loads(args.page_index.read_text("utf-8"))["pages"]
    index = {page["path"]: page for page in index_rows}
    rows = json.loads(args.ground_truth.read_text("utf-8"))

    documents: list[dict[str, Any]] = []
    counts = {category: 0 for category in campaign.CATEGORIES}
    hash_failures = 0
    expected_pages = 0
    for row in sorted(rows, key=_document_id):
        doc_id = _document_id(row)
        pages = list(row["page_info"]["images_list"])
        expected_pages += len(pages)
        verdict, mismatches = _verify_document_pages(pages, args.dataset_root, index)
        hash_failures += 1 if verdict != "VERIFIED" else 0
        classification = campaign.classify_document(len(pages))
        counts[classification["category"]] += 1
        documents.append(
            {
                "doc_id": doc_id,
                "pages": len(pages),
                "page_hashes": verdict,
                "page_hash_mismatches": mismatches,
                **classification,
            }
        )

    report = {
        "schema": 1,
        "tool": "scripts/run_mpdoc_full.py dryrun",
        "dataset": args.dataset,
        "dataset_revision": pin["revision"],
        "ground_truth_sha256": file_sha256(args.ground_truth),
        "page_index_sha256": file_sha256(args.page_index),
        "expected_documents": len(rows),
        "expected_pages": expected_pages,
        "category_counts": counts,
        "classified_documents": len(documents),
        "zero_exclusion": len(documents) == len(rows),
        "page_hash_failures": hash_failures,
        "denominator": {"documents": len(rows), "pages": expected_pages},
        "arithmetic_bounds": {
            "prompt_upper_bound_tokens_per_page": campaign.PROMPT_UPPER_BOUND_TOKENS_PER_PAGE,
            "required_margin_tokens": campaign.REQUIRED_MARGIN_TOKENS,
            "context_cap_tokens": campaign.CONTEXT_CAP_TOKENS,
            "production_budget": campaign.PRODUCTION_BUDGET,
            "campaign_budget": campaign.CAMPAIGN_BUDGET,
            "campaign_policy_note": (
                "classify/admit arithmetic ONLY. No campaign policy exists yet; "
                "budget sufficiency is a runtime outcome (TRUNCATED accounting)."
            ),
        },
        "full_scope_manifest_sha256": digest(documents),
        "documents": documents,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=1) + "\n", "utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "classified": report["classified_documents"],
                "zero_exclusion": report["zero_exclusion"],
                "category_counts": counts,
                "full_scope_manifest_sha256": report["full_scope_manifest_sha256"],
            }
        )
    )
    return 0


def prepare_command(args: argparse.Namespace) -> int:
    """Frozen run inputs for the FULL dataset (no selection, no subsetting)."""
    if args.max_new_tokens > campaign.PRODUCTION_BUDGET:
        raise ValueError(
            "prepare pins the production budget; campaign-class budgets enter "
            "only through the per-document classification and a frozen policy"
        )
    lock = json.loads(args.lock.read_text("utf-8"))
    pin = lock["datasets"]["MPDocBench"]
    manifest = inventory(
        args.ground_truth,
        args.dataset_root,
        "MPDocBench",
        pin["revision"],
        420,
        3135,
    )
    report = validate_manifest(manifest, args.dataset_root)
    if report["status"] != "PASS":
        raise ValueError(f"Local dataset does not satisfy the frozen manifest: {report}")
    config: dict[str, Any] = {
        "schema": 1,
        "label": "FULL_SCOPE_CAMPAIGN",
        "runner": "scripts/run_mpdoc_full.py",
        "ground_truth_sha256": file_sha256(args.ground_truth),
        "page_index_sha256": file_sha256(_REPO_ROOT / PAGE_INDEX_REPO_PATH),
        "budget_policy_sha256": None,
        "ngram_policy_sha256": None,
        "authorization": None,
        "max_pages_per_request_class": {
            "PRODUCTION_ADMITTED": campaign.PRODUCTION_MAX_PAGES,
            "CAMPAIGN_POLICY_REQUIRED": campaign.CAMPAIGN_MAX_PAGES,
        },
        "production_budget": campaign.PRODUCTION_BUDGET,
        "campaign_budget": campaign.CAMPAIGN_BUDGET,
        "timeout_seconds_per_document": args.timeout_seconds,
        "page_hash_gate": PAGE_INDEX_REPO_PATH,
        "claim_limit": (
            "Full-scope campaign manifest. Executable set is gated by "
            "authorization + a reviewed campaign policy; blocked documents "
            "are reported, never dropped."
        ),
    }
    if args.authorization:
        config["authorization"] = json.loads(args.authorization.read_text("utf-8"))
    if args.budget_policy:
        config["budget_policy_sha256"] = file_sha256(args.budget_policy)
        config["budget_policy_repo_path"] = str(
            args.budget_policy.resolve().relative_to(_REPO_ROOT)
        )
    if args.ngram_policy:
        config["ngram_policy_sha256"] = file_sha256(args.ngram_policy)
        config["ngram_policy_repo_path"] = str(args.ngram_policy.resolve().relative_to(_REPO_ROOT))
    environment = benchmark_runner.sanitized_environment()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", "utf-8")
    (args.output / "config.json").write_text(json.dumps(config, indent=1) + "\n", "utf-8")
    (args.output / "environment.json").write_text(
        json.dumps(environment, indent=1, sort_keys=True) + "\n", "utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "manifest_sha256": report["manifest_sha256"],
                "authorized": config["authorization"] is not None,
            }
        )
    )
    return 0


def run_command(args: argparse.Namespace) -> int:
    run = Run(args.run)
    config = run.data["config"]
    # Fail fast BEFORE any attempt is reserved: a worker python without the
    # GPU stack must not burn 24 ledger attempts on ModuleNotFoundError.
    probe = subprocess.run(
        [sys.executable, "-c", "import torch"],
        capture_output=True,
    )
    if probe.returncode != 0:
        raise ValueError(
            "worker python cannot import torch — run this command with the "
            "GPU venv python (envs/linux-gfx1151-rocm/bin/python)"
        )
    plan = campaign.execution_plan(list(run.documents.values()), config)
    if args.include_blocked:
        raise ValueError(
            "Blocked documents cannot be forced into execution; their status "
            "is the honest accounting (NOT_RUN/BLOCKED in the ledger)"
        )
    print(
        json.dumps(
            {
                "plan": {
                    "executable": len(plan["executable"]),
                    "blocked": len(plan["blocked"]),
                    "blocked_reasons": sorted({reason for reason in plan["blocked"].values()}),
                }
            }
        )
    )
    result = benchmark_runner.execute_selection(
        run,
        {doc_id: {} for doc_id in plan["executable"]},
        worker_command=[
            sys.executable,
            str(Path(__file__).resolve()),
            "worker",
            "--repo-root",
            str(args.repo_root),
            *(["--model-cache", str(args.model_cache)] if args.model_cache else []),
        ],
        data_root=args.data_root,
        max_new_tokens=campaign.production_budget(),
        timeout_seconds=float(config.get("timeout_seconds_per_document", 3600.0)),
        restrict=set(args.documents.split(",")) if args.documents else None,
    )
    print(json.dumps({"counts": result["counts"], "todo_remaining": result["todo_remaining"]}))
    return 0


def worker_command(args: argparse.Namespace) -> int:
    """Execute exactly one document attempt; the parent owns the ledger."""
    run = Run(args.run)
    doc = run.documents[args.document]
    pages = benchmark_runner.resolve_pages(run, args.document, args.data_root)
    rel_pages = [page["path"] for page in doc["pages"]]
    index = {
        page["path"]: page
        for page in json.loads((args.repo_root / PAGE_INDEX_REPO_PATH).read_text("utf-8"))["pages"]
    }
    from unlimited_ocr_2_rocm.benchmark.manifest import file_hash

    for rel in rel_pages:
        entry = index.get(rel)
        if entry is None or file_hash(args.data_root / rel) != entry["sha256"]:
            raise ValueError(f"page hash gate failed before inference: {rel}")

    classification = campaign.classify_document(len(pages))
    result_path = Path(args.result)
    attempt_directory = run.path / "documents" / args.document / f"{args.attempt}.pending"

    if args.self_test_failure:
        try:
            outcome = _self_test_outcome(args, pages, attempt_directory)
        except BaseException as error:  # noqa: BLE001 - publish like a real worker failure
            import traceback

            outcome = {
                "kind": "error",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        if outcome is not None:
            benchmark_runner.write_outcome(result_path, outcome)
        return 0

    config = run.data["config"]
    budget_policy = None
    ngram_policy = None
    if classification["category"] == "CAMPAIGN_POLICY_REQUIRED":
        if not config.get("budget_policy_sha256"):
            raise ValueError(
                "campaign-class document requires a frozen budget policy in "
                "the run config; the pinned gate will refuse the budget anyway"
            )
        budget_policy = _REPO_ROOT / config["budget_policy_repo_path"]
        if config.get("ngram_policy_sha256"):
            ngram_policy = _REPO_ROOT / config["ngram_policy_repo_path"]

    from unlimited_ocr_2_rocm.benchmark.watchdog import ResourceWatchdog
    from unlimited_ocr_2_rocm.doctor import memory_snapshot
    from unlimited_ocr_2_rocm.inference import run_native
    from unlimited_ocr_2_rocm.model import load_model

    def _gpu_allocated_bytes() -> int:
        import torch

        return int(torch.cuda.memory_allocated())

    started = time.time()
    scratch = Path(tempfile.mkdtemp(prefix="uocr2-campaign-"))
    watchdog = None
    try:
        model = load_model(args.model_cache, offline=True)
        # L1.1-B measured envelope: 16 GiB GPU-allocated hard gate with a
        # heartbeat in the attempt directory. A breach triggers the watchdog's
        # safe abort; the parent records the attempt as FAIL, never as a
        # partial PASS.
        watchdog = ResourceWatchdog(
            sampler=_gpu_allocated_bytes,
            limit_bytes=int(16 * 1024**3),
            heartbeat_path=attempt_directory / "heartbeat.json",
            heartbeat_seconds=20.0,
        )
        watchdog.start()
        report = run_native(
            model,
            pages,
            scratch,
            multi=True,
            max_new_tokens=classification["budget_new_tokens"],
            budget_policy=budget_policy,
            ngram_policy=ngram_policy,
            page_limit=len(pages),
        )
        report["worker_load_seconds"] = round(time.time() - started, 3)
        watchdog.stop()
        report["watchdog"] = watchdog.receipt()["watchdog"]
        sanitized = json.loads(
            json.dumps(
                {
                    key: value
                    for key, value in report.items()
                    if key not in {"raw_output", "native_generation", "model"}
                }
            )
        )
        (attempt_directory / "native-report.json").write_text(
            json.dumps(sanitized, indent=1, sort_keys=True) + "\n", "utf-8"
        )
        (attempt_directory / "model-evidence.json").write_text(
            json.dumps(model.evidence, indent=1, sort_keys=True) + "\n", "utf-8"
        )
        benchmark_runner.write_outcome(
            result_path,
            {"kind": "ok", "report": report, "memory_final": memory_snapshot()},
        )
    except BaseException as error:  # noqa: BLE001 - outcome must reach the parent
        import traceback

        if watchdog is not None:
            watchdog.stop()
        benchmark_runner.write_outcome(
            result_path,
            {
                "kind": "error",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            },
        )
    finally:
        import shutil

        shutil.rmtree(scratch, ignore_errors=True)
    return 0


def _self_test_outcome(
    args: argparse.Namespace, pages: list[Path], attempt_directory: Path
) -> dict | None:
    """CPU failure injection executed by the real supervised worker path."""
    count = len(pages)
    good_raw = "<PAGE><BLOCK>text[0,0,1,1]<FORK>hello" * count
    mode = args.self_test_failure
    if mode == "hang":
        time.sleep(args.timeout_seconds + 60)
        return None
    if mode == "crash":
        import os
        import signal

        os.kill(os.getpid(), 137 if sys.platform == "win32" else signal.SIGKILL)
        return None
    if mode == "oom":
        raise RuntimeError("torch.OutOfMemoryError: HIP out of memory on allocation of 8192 MiB")
    if mode == "length":
        raw = good_raw + "<PAGE><BLOCK>text[0,0,1,1]<FORK>truncat"
        return {
            "kind": "selftest",
            "mode": mode,
            "raw": raw,
            "report": {
                "status": "FAIL",
                "finish_reason": "length",
                "output_token_count": args.max_new_tokens,
                "max_new_tokens": args.max_new_tokens,
                "memory_after": {"process_rss_bytes": 1},
                "upstream_timing": {"generation_seconds": 0.1, "prefill_seconds": 0.1},
            },
        }
    report = {
        "status": "INFERENCE_ONLY",
        "finish_reason": "stop",
        "output_token_count": 6 * count,
        "max_new_tokens": args.max_new_tokens,
        "memory_after": {"process_rss_bytes": 1},
        "upstream_timing": {"generation_seconds": 0.1, "prefill_seconds": 0.1},
    }
    if mode == "malformed":
        report["raw_output"] = "no page structure at all"
    else:
        report["raw_output"] = good_raw
    return {"kind": "selftest", "mode": mode, "raw": report["raw_output"], "report": report}


def report_command(args: argparse.Namespace) -> int:
    run = Run(args.run)
    plan = campaign.execution_plan(list(run.documents.values()), run.data["config"])
    receipt = benchmark_runner.sanitized_receipt(
        run,
        selection={doc_id: {} for doc_id in plan["executable"]},
        selection_sha256=digest(sorted(plan["executable"])),
        worker_identity={"script": "scripts/run_mpdoc_full.py", "supervision": "subprocess"},
    )
    receipt["execution_plan"] = plan
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n", "utf-8")
    print(json.dumps({"receipt": str(args.output), "counts": receipt["counts"]}))
    return 0


def export_command(args: argparse.Namespace) -> int:
    run = Run(args.run)
    plan = campaign.execution_plan(list(run.documents.values()), run.data["config"])
    run.export(args.output, include=set(plan["executable"]), include_truncated=args.diagnostic)
    print(json.dumps({"export": str(args.output), "diagnostic": args.diagnostic}))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    dryrun = commands.add_parser("dryrun", help="full-scope zero-exclusion classification")
    for flag, kwargs in (
        ("--ground-truth", {"required": True, "type": Path}),
        ("--dataset-root", {"required": True, "type": Path}),
        ("--lock", {"required": True, "type": Path}),
        ("--page-index", {"required": True, "type": Path}),
        ("--dataset", {"default": "MPDocBench"}),
        ("--output", {"required": True, "type": Path}),
    ):
        dryrun.add_argument(flag, **kwargs)

    prepare = commands.add_parser("prepare", help="write frozen full-dataset run inputs")
    for flag, kwargs in (
        ("--ground-truth", {"required": True, "type": Path}),
        ("--dataset-root", {"required": True, "type": Path}),
        ("--lock", {"required": True, "type": Path}),
        ("--max-new-tokens", {"type": int, "default": campaign.PRODUCTION_BUDGET}),
        ("--timeout-seconds", {"type": float, "default": 3600.0}),
        ("--authorization", {"type": Path, "default": None}),
        ("--budget-policy", {"type": Path, "default": None}),
        ("--ngram-policy", {"type": Path, "default": None}),
        ("--output", {"required": True, "type": Path}),
    ):
        prepare.add_argument(flag, **kwargs)

    run = commands.add_parser("run", help="execute the authorized plan through the ledger")
    run.add_argument("--run", required=True, type=Path)
    run.add_argument("--data-root", required=True, type=Path)
    run.add_argument("--repo-root", required=True, type=Path)
    run.add_argument("--model-cache", type=Path, default=None)
    run.add_argument("--documents", default=None, help="comma-separated subset")
    run.add_argument(
        "--include-blocked",
        action="store_true",
        help="accepted for CLI symmetry but ALWAYS refused: blocked documents "
        "are honest accounting (NOT_RUN/BLOCKED), never executable",
    )

    worker = commands.add_parser("worker", help="internal: one supervised attempt")
    worker.add_argument("--run", required=True, type=Path)
    worker.add_argument("--document", required=True)
    worker.add_argument("--attempt", required=True)
    worker.add_argument("--data-root", required=True, type=Path)
    worker.add_argument("--repo-root", required=True, type=Path)
    worker.add_argument("--model-cache", type=Path, default=None)
    worker.add_argument("--max-new-tokens", type=int, default=campaign.PRODUCTION_BUDGET)
    worker.add_argument("--timeout-seconds", type=float, default=3600.0)
    worker.add_argument("--result", required=True, type=Path)
    worker.add_argument("--self-test-failure", choices=SELF_TEST_MODES, default=None)

    report = commands.add_parser("report", help="sanitized accounting receipt with the plan")
    report.add_argument("--run", required=True, type=Path)
    report.add_argument("--output", required=True, type=Path)

    export = commands.add_parser("export", help="export scorer predictions for the plan")
    export.add_argument("--run", required=True, type=Path)
    export.add_argument("--output", required=True, type=Path)
    export.add_argument("--diagnostic", action="store_true")

    args = parser.parse_args(argv)
    handlers = {
        "dryrun": dryrun_command,
        "prepare": prepare_command,
        "run": run_command,
        "worker": worker_command,
        "report": report_command,
        "export": export_command,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
