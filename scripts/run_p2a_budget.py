"""P2-A controlled output-token-budget experiments (explicit opt-in policy).

Re-runs ONLY the four frozen P1 documents that ended with native
finish_reason=length at 8192, at the budgets authorized by the committed
policy (eval/experiments/p2a-token-budget/policy.json). Everything else —
images and hashes, model revision, prompt, crop policy, vision MATH/DASH,
decoder, page guard, max_pages=5, 16 Mpx/page — stays identical to the P1.5
ledger runs. Production default caps are unchanged: run_native still refuses
budgets above 8192 unless the frozen policy is presented.

Each (document, budget) attempt runs in an isolated supervised subprocess
and is recorded in a per-budget ledger run whose identity freezes the policy
hash, the budget and the four-document selection.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
try:
    from unlimited_ocr_2_rocm.benchmark import runner as benchmark_runner
    from unlimited_ocr_2_rocm.benchmark.ledger import Run
    from unlimited_ocr_2_rocm.benchmark.manifest import inventory, validate_manifest
    from unlimited_ocr_2_rocm.config import file_sha256
except ImportError:  # pragma: no cover - direct script execution without install
    sys.path.insert(0, str(_REPO_ROOT / "src"))
    from unlimited_ocr_2_rocm.benchmark import runner as benchmark_runner
    from unlimited_ocr_2_rocm.benchmark.ledger import Run
    from unlimited_ocr_2_rocm.benchmark.manifest import inventory, validate_manifest
    from unlimited_ocr_2_rocm.config import file_sha256

PAGE_INDEX_REPO_PATH = "eval/benchmarks/g05/datasets/MPDocBench-page-index.json"


def load_policy(path: Path) -> dict:
    policy = json.loads(path.read_text(encoding="utf-8"))
    if policy.get("schema") != 1 or policy.get("experiment") != "P2A_OUTPUT_TOKEN_BUDGET_RCA":
        raise ValueError("Not the frozen P2-A budget policy")
    return policy


def admission_report(policy: dict, pages: int, budget: int) -> dict:
    """Pre-flight resource admission: context arithmetic before any stage."""
    rule = policy["admission"]
    prompt_bound = pages * rule["prompt_upper_bound_tokens_per_page"]
    total = prompt_bound + budget + rule["required_margin_tokens"]
    return {
        "pages": pages,
        "prompt_upper_bound_tokens": prompt_bound,
        "budget": budget,
        "required_margin_tokens": rule["required_margin_tokens"],
        "total_upper_bound_tokens": total,
        "context_cap_tokens": rule["context_cap_tokens"],
        "admitted": total <= rule["context_cap_tokens"],
    }


def prepare_command(args: argparse.Namespace) -> int:
    policy = load_policy(args.policy)
    if args.budget not in policy["budgets_new_tokens"]:
        raise ValueError(f"Budget {args.budget} is not authorized by the frozen policy")
    lock = json.loads(args.lock.read_text("utf-8"))
    pin = lock["datasets"]["MPDocBench"]
    manifest = inventory(
        args.ground_truth, args.dataset_root, "MPDocBench", pin["revision"], 420, 3135
    )
    if validate_manifest(manifest, args.dataset_root)["status"] != "PASS":
        raise ValueError("Local dataset does not satisfy the frozen manifest")
    config = {
        "schema": 1,
        "label": "EXPERIMENTAL_BUDGET_ONLY",
        "experiment": policy["experiment"],
        "policy_sha256": file_sha256(args.policy),
        "policy_repo_path": str(args.policy.resolve().relative_to(_REPO_ROOT)),
        "documents": sorted(policy["documents"]),
        "budget_new_tokens": args.budget,
        "baseline_new_tokens": policy["baseline_new_tokens"],
        "unchanged": policy["unchanged"],
        "timeout_seconds_per_document": policy["supervision"]["timeout_seconds_per_document"],
        "claim_limit": (
            "Four frozen length-hit documents only; NOT full MPDocBench accuracy, "
            "NOT leaderboard parity; production default caps unchanged"
        ),
    }
    environment = benchmark_runner.sanitized_environment()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", "utf-8")
    (args.output / "config.json").write_text(json.dumps(config, indent=1) + "\n", "utf-8")
    (args.output / "environment.json").write_text(
        json.dumps(environment, indent=1, sort_keys=True) + "\n", "utf-8"
    )
    print(json.dumps({"output": str(args.output), "budget": args.budget}))
    return 0


def run_command(args: argparse.Namespace) -> int:
    run = Run(args.run)
    config = run.data["config"]
    if config.get("policy_sha256") != file_sha256(args.policy):
        raise ValueError("Policy file does not match the frozen experiment identity")
    policy = load_policy(args.policy)
    budget = int(config["budget_new_tokens"])
    selection = {
        doc_id: {"reason": "P2-A frozen length-hit document"} for doc_id in policy["documents"]
    }
    restrict = set(args.documents.split(",")) if args.documents else None
    result = benchmark_runner.execute_selection(
        run,
        selection,
        worker_command=[
            sys.executable,
            str(Path(__file__).resolve()),
            "worker",
            "--policy",
            str(args.policy.resolve()),
            "--repo-root",
            str(args.repo_root),
            "--model-cache",
            str(args.model_cache),
        ],
        data_root=args.data_root,
        max_new_tokens=budget,
        timeout_seconds=float(policy["supervision"]["timeout_seconds_per_document"]),
        restrict=restrict,
    )
    print(json.dumps({"counts": result["counts"], "todo_remaining": result["todo_remaining"]}))
    return 0


def verify_page_gate(repo_root: Path, data_root: Path, rel_pages: list[str]) -> None:
    index = json.loads((repo_root / PAGE_INDEX_REPO_PATH).read_text("utf-8"))
    page_index = {page["path"]: page for page in index["pages"]}
    for rel in rel_pages:
        expected = page_index.get(rel)
        if expected is None:
            raise ValueError(f"page not present in frozen index: {rel}")
        if file_sha256(data_root / rel) != expected["sha256"]:
            raise ValueError(f"page hash mismatch against frozen index: {rel}")


def worker_command(args: argparse.Namespace) -> int:
    policy = load_policy(args.policy)
    run = Run(args.run)
    if run.data["config"].get("policy_sha256") != file_sha256(args.policy):
        raise ValueError("Worker policy does not match the run identity")
    if args.document not in policy["documents"]:
        raise ValueError(f"Document is not authorized by the P2-A policy: {args.document}")
    pages = benchmark_runner.resolve_pages(run, args.document, args.data_root)
    admission = admission_report(policy, len(pages), args.max_new_tokens)
    if not admission["admitted"]:
        benchmark_runner.write_outcome(
            Path(args.result),
            {"kind": "error", "error": "ADMISSION_REFUSED", "admission": admission},
        )
        return 0
    rel_pages = [page["path"] for page in run.documents[args.document]["pages"]]
    verify_page_gate(args.repo_root, args.data_root, rel_pages)
    attempt_directory = run.path / "documents" / args.document / f"{args.attempt}.pending"

    from unlimited_ocr_2_rocm.inference import run_native
    from unlimited_ocr_2_rocm.model import load_model

    started = time.time()
    scratch = Path(tempfile.mkdtemp(prefix="uocr2-p2a-"))
    try:
        model = load_model(args.model_cache, offline=True)
        report = run_native(
            model,
            pages,
            scratch,
            multi=True,
            max_new_tokens=args.max_new_tokens,
            budget_policy=args.policy,
        )
        native = report.get("native_generation") or {}
        prompt_ids = native.get("prompt_token_ids") or []
        report["experiment"] = {
            "admission": admission,
            "prompt_token_count_actual": len(prompt_ids),
            "context_usage_upper_bound": len(prompt_ids) + args.max_new_tokens,
            "page_guard": {
                "expected_pages": native.get("expected_pages"),
                "num_pages_out": native.get("num_pages_out"),
                "raw_page_markers": (report.get("raw_output") or "").count("<PAGE>"),
            },
            "worker_load_seconds": round(time.time() - started, 3),
            "policy_sha256": file_sha256(args.policy),
            "budget": args.max_new_tokens,
        }
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
        benchmark_runner.write_outcome(Path(args.result), {"kind": "ok", "report": report})
    except BaseException as error:  # noqa: BLE001 - outcome must reach the parent
        import traceback

        benchmark_runner.write_outcome(
            Path(args.result),
            {
                "kind": "error",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "admission": admission,
            },
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return 0


def report_command(args: argparse.Namespace) -> int:
    run = Run(args.run)
    selection = {
        doc_id: {"reason": "P2-A frozen length-hit document"}
        for doc_id in load_policy(args.policy)["documents"]
    }
    receipt = benchmark_runner.sanitized_receipt(
        run,
        selection=selection,
        selection_sha256=file_sha256(args.policy),
        worker_identity={"script": "scripts/run_p2a_budget.py", "supervision": "subprocess"},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n", "utf-8")
    print(json.dumps({"receipt": str(args.output), "counts": receipt["counts"]}))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--policy", required=True, type=Path)
    prepare.add_argument("--ground-truth", required=True, type=Path)
    prepare.add_argument("--dataset-root", required=True, type=Path)
    prepare.add_argument("--lock", required=True, type=Path)
    prepare.add_argument("--budget", type=int, required=True)
    prepare.add_argument("--output", required=True, type=Path)

    run = commands.add_parser("run")
    run.add_argument("--run", required=True, type=Path)
    run.add_argument("--policy", required=True, type=Path)
    run.add_argument("--data-root", required=True, type=Path)
    run.add_argument("--repo-root", required=True, type=Path)
    run.add_argument("--model-cache", required=True, type=Path)
    run.add_argument("--documents", default=None)

    worker = commands.add_parser("worker")
    worker.add_argument("--policy", required=True, type=Path)
    worker.add_argument("--run", required=True, type=Path)
    worker.add_argument("--document", required=True)
    worker.add_argument("--attempt", required=True)
    worker.add_argument("--data-root", required=True, type=Path)
    worker.add_argument("--repo-root", required=True, type=Path)
    worker.add_argument("--model-cache", required=True, type=Path)
    worker.add_argument("--max-new-tokens", type=int, required=True)
    worker.add_argument("--result", required=True, type=Path)
    worker.add_argument("--timeout-seconds", type=float, default=7200.0)

    report = commands.add_parser("report")
    report.add_argument("--run", required=True, type=Path)
    report.add_argument("--policy", required=True, type=Path)
    report.add_argument("--output", required=True, type=Path)

    export = commands.add_parser("export")
    export.add_argument("--run", required=True, type=Path)
    export.add_argument("--policy", required=True, type=Path)
    export.add_argument("--output", required=True, type=Path)
    export.add_argument("--diagnostic", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "prepare":
        return prepare_command(args)
    if args.command == "run":
        return run_command(args)
    if args.command == "worker":
        return worker_command(args)
    if args.command == "report":
        return report_command(args)
    selection = {
        doc_id: {"reason": "P2-A frozen length-hit document"}
        for doc_id in load_policy(args.policy)["documents"]
    }
    Run(args.run).export(args.output, include=set(selection), include_truncated=args.diagnostic)
    print(json.dumps({"export": str(args.output), "diagnostic": args.diagnostic}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
