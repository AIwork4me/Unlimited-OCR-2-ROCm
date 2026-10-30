"""P2-C controlled combined experiments: suppression × raised budgets.

Re-runs ONLY the four frozen P2-A documents with the SAME official repetition
suppression as P2-B (no_repeat_ngram_size=35, trailing window 1024 over
generated ids only) at RAISED output budgets 12288 and then 16384, per the
frozen policy (eval/experiments/p2c-combined/policy.json). The budget source
of truth is the frozen P2-A policy, presented to validate_budget at run time.
Controls are the immutable ngram=0 P1.5/P2-A evidence and the P2-B@8192 run
6e1ae22e — never re-run, never rewritten. No 39321 escalation, no page-count
escalation. Everything else — images and hashes, model revision, prompt, crop
policy, vision MATH/DASH, decoder, native page guard, sampling — stays
identical. Production defaults are unchanged; the gate authorizes suppression
only for the two pinned policy digests.

Stage admission is explicit: budget 16384 runs only after the ``gates``
command admits the finished 12288 stage (all four documents completed, zero
forced-marker mismatches, zero supervision failures, memory within the frozen
envelope). Each attempt runs in an isolated supervised subprocess and is
recorded in a ledger run; the worker persists token ids for replay by
scripts/analyze_ngram_evidence.py.
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
    from unlimited_ocr_2_rocm.inference import validate_ngram
except ImportError:  # pragma: no cover - direct script execution without install
    sys.path.insert(0, str(_REPO_ROOT / "src"))
    from unlimited_ocr_2_rocm.benchmark import runner as benchmark_runner
    from unlimited_ocr_2_rocm.benchmark.ledger import Run
    from unlimited_ocr_2_rocm.benchmark.manifest import inventory, validate_manifest
    from unlimited_ocr_2_rocm.config import file_sha256
    from unlimited_ocr_2_rocm.inference import validate_ngram

PAGE_INDEX_REPO_PATH = "eval/benchmarks/g05/datasets/MPDocBench-page-index.json"


def load_policy(path: Path) -> dict:
    policy = json.loads(path.read_text(encoding="utf-8"))
    if policy.get("schema") != 1 or policy.get("experiment") != "P2C_COMBINED_NGRAM_BUDGET":
        raise ValueError("Not the frozen P2-C combined policy")
    return policy


def load_budget_policy(path: Path) -> dict:
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


def _attempt_reports(run: Run, selection: dict[str, dict]) -> list[dict]:
    reports = []
    for doc_id in sorted(selection):
        directory = run.path / "documents" / doc_id
        for attempt in sorted(p.name for p in directory.glob("*") if p.is_dir()):
            report_path = directory / attempt / "native-report.json"
            if report_path.is_file():
                reports.append(
                    {
                        "document": doc_id,
                        "attempt": attempt,
                        **json.loads(report_path.read_text(encoding="utf-8")),
                    }
                )
    return reports


def evaluate_stage_gates(policy: dict, summary_documents: list[dict], reports: list[dict]) -> dict:
    """Pure stage-admission evaluation over a finished budget stage.

    ``summary_documents`` are ledger summary rows (doc_id, state); ``reports``
    are the per-attempt native reports (memory_after, experiment.page_guard).
    """
    rule = policy["stage_admission"]
    max_gpu = float(rule["max_gpu_peak_allocated_gib"])
    max_rss = float(rule["max_host_peak_rss_gib"])
    selection = set(policy["documents"])
    failures: list[str] = []

    completed = {
        row["doc_id"] for row in summary_documents if row["state"] in {"PASS", "TRUNCATED"}
    }
    missing = sorted(selection - completed)
    if missing:
        failures.append(f"documents not completed: {missing}")
    supervision_bad = {
        row["doc_id"]: row["state"]
        for row in summary_documents
        if row["state"] in {"TIMEOUT", "OOM", "FAIL"}
    }
    if supervision_bad:
        failures.append(f"supervision failures: {supervision_bad}")

    peaks = {"gpu_peak_allocated_gib": 0.0, "host_peak_rss_gib": 0.0}
    forced_mismatches = 0
    for report in reports:
        memory = report.get("memory_after") or {}
        gpu = (memory.get("gpu_peak_allocated_bytes") or 0) / (1024**3)
        rss = (report.get("experiment", {}).get("host_peak_rss_gib")) or (
            (memory.get("process_peak_rss_bytes") or 0) / (1024**3)
        )
        peaks["gpu_peak_allocated_gib"] = max(peaks["gpu_peak_allocated_gib"], round(gpu, 3))
        peaks["host_peak_rss_gib"] = max(peaks["host_peak_rss_gib"], round(rss, 3))
        guard = (report.get("experiment") or {}).get("page_guard") or {}
        if guard.get("forced_mismatches"):
            forced_mismatches += len(guard["forced_mismatches"])
    if peaks["gpu_peak_allocated_gib"] > max_gpu:
        failures.append(f"GPU peak {peaks['gpu_peak_allocated_gib']} GiB exceeds {max_gpu}")
    if peaks["host_peak_rss_gib"] > max_rss:
        failures.append(f"host RSS peak {peaks['host_peak_rss_gib']} GiB exceeds {max_rss}")
    if forced_mismatches:
        failures.append(f"forced-marker mismatches: {forced_mismatches}")

    return {
        "schema": 1,
        "experiment": policy["experiment"],
        "documents_completed": sorted(completed & selection),
        "resource_peaks": peaks,
        "forced_marker_mismatches": forced_mismatches,
        "failures": failures,
        "next_budget_admitted": not failures,
        "rule": rule["rule"],
    }


def gates_command(args: argparse.Namespace) -> int:
    """Evaluate the frozen stage-admission gates over a finished stage."""
    run = Run(args.run)
    policy = load_policy(args.policy)
    config = run.data["config"]
    if config.get("policy_sha256") != file_sha256(args.policy):
        raise ValueError("Policy file does not match the frozen experiment identity")
    selection = {doc_id: {} for doc_id in policy["documents"]}
    summary = run.summary()
    verdict = evaluate_stage_gates(policy, summary["documents"], _attempt_reports(run, selection))
    verdict["run_id"] = run.data.get("run_id")
    verdict["budget"] = config.get("budget_new_tokens")
    verdict["counts"] = summary["counts"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(verdict, indent=1, sort_keys=True) + "\n", "utf-8")
    print(
        json.dumps(
            {
                "gates": "PASS" if not verdict["failures"] else "FAIL",
                "failures": verdict["failures"],
            }
        )
    )
    return 0


def prepare_command(args: argparse.Namespace) -> int:
    policy = load_policy(args.policy)
    budget_policy = load_budget_policy(args.budget_policy)
    validate_ngram(args.ngram_size, policy["ngram_window"], args.policy)
    if args.budget not in policy["budgets_new_tokens"]:
        raise ValueError(f"Budget {args.budget} is not authorized by the frozen P2-C policy")
    if args.budget not in budget_policy["budgets_new_tokens"]:
        raise ValueError(f"Budget {args.budget} is not authorized by the frozen P2-A budget policy")
    lock = json.loads(args.lock.read_text("utf-8"))
    pin = lock["datasets"]["MPDocBench"]
    manifest = inventory(
        args.ground_truth, args.dataset_root, "MPDocBench", pin["revision"], 420, 3135
    )
    if validate_manifest(manifest, args.dataset_root)["status"] != "PASS":
        raise ValueError("Local dataset does not satisfy the frozen manifest")
    config = {
        "schema": 1,
        "label": "EXPERIMENTAL_NGRAM_BUDGET_COMBINED",
        "experiment": policy["experiment"],
        "policy_sha256": file_sha256(args.policy),
        "policy_repo_path": str(args.policy.resolve().relative_to(_REPO_ROOT)),
        "budget_policy_sha256": file_sha256(args.budget_policy),
        "budget_policy_repo_path": str(args.budget_policy.resolve().relative_to(_REPO_ROOT)),
        "documents": sorted(policy["documents"]),
        "budget_new_tokens": args.budget,
        "no_repeat_ngram_size": args.ngram_size,
        "ngram_window": policy["ngram_window"],
        "control_evidence": policy["baseline_evidence"],
        "unchanged": policy["unchanged"],
        "timeout_seconds_per_document": policy["supervision"]["timeout_seconds_per_document"],
        "claim_limit": (
            "Four frozen P2-A documents only; NOT full MPDocBench accuracy, "
            "NOT leaderboard parity; production defaults unchanged"
        ),
    }
    environment = benchmark_runner.sanitized_environment()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", "utf-8")
    (args.output / "config.json").write_text(json.dumps(config, indent=1) + "\n", "utf-8")
    (args.output / "environment.json").write_text(
        json.dumps(environment, indent=1, sort_keys=True) + "\n", "utf-8"
    )
    print(json.dumps({"output": str(args.output), "ngram": args.ngram_size, "budget": args.budget}))
    return 0


def run_command(args: argparse.Namespace) -> int:
    run = Run(args.run)
    config = run.data["config"]
    if config.get("policy_sha256") != file_sha256(args.policy):
        raise ValueError("Policy file does not match the frozen experiment identity")
    if config.get("budget_policy_sha256") != file_sha256(args.budget_policy):
        raise ValueError("Budget policy file does not match the frozen experiment identity")
    policy = load_policy(args.policy)
    budget = int(config["budget_new_tokens"])
    selection = {
        doc_id: {"reason": "P2-C frozen combined document"} for doc_id in policy["documents"]
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
            "--budget-policy",
            str(args.budget_policy.resolve()),
            "--repo-root",
            str(args.repo_root),
            "--model-cache",
            str(args.model_cache),
            "--ngram-size",
            str(int(config["no_repeat_ngram_size"])),
            "--ngram-window",
            str(int(config["ngram_window"])),
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
    load_budget_policy(args.budget_policy)
    validate_ngram(args.ngram_size, args.ngram_window, args.policy)
    run = Run(args.run)
    if run.data["config"].get("policy_sha256") != file_sha256(args.policy):
        raise ValueError("Worker policy does not match the run identity")
    if args.document not in policy["documents"]:
        raise ValueError(f"Document is not authorized by the P2-C policy: {args.document}")
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
    scratch = Path(tempfile.mkdtemp(prefix="uocr2-p2c-"))
    try:
        model = load_model(args.model_cache, offline=True)
        report = run_native(
            model,
            pages,
            scratch,
            multi=True,
            max_new_tokens=args.max_new_tokens,
            no_repeat_ngram_size=args.ngram_size,
            ngram_window=args.ngram_window,
            ngram_policy=args.policy,
            budget_policy=args.budget_policy,
        )
        native = report.get("native_generation") or {}
        prompt_ids = native.get("prompt_token_ids") or []
        token_ids = native.get("token_ids") or []
        (attempt_directory / "token_ids.json").write_text(json.dumps(token_ids) + "\n", "utf-8")
        memory = report.get("memory_after") or {}
        report["experiment"] = {
            "admission": admission,
            "prompt_token_count_actual": len(prompt_ids),
            "context_usage_upper_bound": len(prompt_ids) + args.max_new_tokens,
            "page_guard": {
                "expected_pages": native.get("expected_pages"),
                "num_pages_out": native.get("num_pages_out"),
                "raw_page_markers": (report.get("raw_output") or "").count("<PAGE>"),
            },
            "host_peak_rss_gib": round((memory.get("process_peak_rss_bytes") or 0) / (1024**3), 3),
            "worker_load_seconds": round(time.time() - started, 3),
            "policy_sha256": file_sha256(args.policy),
            "budget_policy_sha256": file_sha256(args.budget_policy),
            "budget": args.max_new_tokens,
            "no_repeat_ngram_size": args.ngram_size,
            "ngram_window": args.ngram_window,
            "token_ids_persisted": len(token_ids),
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
        doc_id: {"reason": "P2-C frozen combined document"}
        for doc_id in load_policy(args.policy)["documents"]
    }
    receipt = benchmark_runner.sanitized_receipt(
        run,
        selection=selection,
        selection_sha256=file_sha256(args.policy),
        worker_identity={"script": "scripts/run_p2c_combined.py", "supervision": "subprocess"},
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
    prepare.add_argument("--budget-policy", required=True, type=Path)
    prepare.add_argument("--ground-truth", required=True, type=Path)
    prepare.add_argument("--dataset-root", required=True, type=Path)
    prepare.add_argument("--lock", required=True, type=Path)
    prepare.add_argument("--ngram-size", type=int, required=True)
    prepare.add_argument("--budget", type=int, required=True)
    prepare.add_argument("--output", required=True, type=Path)

    gates = commands.add_parser("gates")
    gates.add_argument("--run", required=True, type=Path)
    gates.add_argument("--policy", required=True, type=Path)
    gates.add_argument("--output", required=True, type=Path)

    run = commands.add_parser("run")
    run.add_argument("--run", required=True, type=Path)
    run.add_argument("--policy", required=True, type=Path)
    run.add_argument("--budget-policy", required=True, type=Path)
    run.add_argument("--data-root", required=True, type=Path)
    run.add_argument("--repo-root", required=True, type=Path)
    run.add_argument("--model-cache", required=True, type=Path)
    run.add_argument("--documents", default=None)

    worker = commands.add_parser("worker")
    worker.add_argument("--policy", required=True, type=Path)
    worker.add_argument("--budget-policy", required=True, type=Path)
    worker.add_argument("--run", required=True, type=Path)
    worker.add_argument("--document", required=True)
    worker.add_argument("--attempt", required=True)
    worker.add_argument("--data-root", required=True, type=Path)
    worker.add_argument("--repo-root", required=True, type=Path)
    worker.add_argument("--model-cache", required=True, type=Path)
    worker.add_argument("--ngram-size", type=int, required=True)
    worker.add_argument("--ngram-window", type=int, required=True)
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
    if args.command == "gates":
        return gates_command(args)
    if args.command == "run":
        return run_command(args)
    if args.command == "worker":
        return worker_command(args)
    if args.command == "report":
        return report_command(args)
    selection = {
        doc_id: {"reason": "P2-C frozen combined document"}
        for doc_id in load_policy(args.policy)["documents"]
    }
    Run(args.run).export(args.output, include=set(selection), include_truncated=args.diagnostic)
    print(json.dumps({"export": str(args.output), "diagnostic": args.diagnostic}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
