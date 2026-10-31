"""L1_FIFTEEN_PAGE_NATIVE: ONE supervised 15-page native request at the
official deployment budget class.

The frozen policy (eval/experiments/l1-fifteen-page/policy.json, third pinned
digest) authorizes exactly one document (book_eng_77552588, 15/15 pages),
budget 39321 new tokens, suppression 35/1024 over generated ids only, the
predeclared selection receipt, the guard-collision observer contract
(docs/validation/L1_GUARD_COLLISION_QUALIFICATION.md) and runtime resource
enforcement (16 GiB allocated hard limit via the in-worker watchdog, 14400 s
supervised timeout). Production defaults are unchanged; the ledger keeps the
full 420-document denominator with 419 documents marked NOT_RUN.

Stage layout mirrors scripts/run_p2c_combined.py: prepare → run → gates →
report/export. Failures, interrupts and partial attempts are preserved in
the ledger and never relabeled.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
try:
    from unlimited_ocr_2_rocm.benchmark import runner as benchmark_runner
    from unlimited_ocr_2_rocm.benchmark.ledger import Run
    from unlimited_ocr_2_rocm.benchmark.manifest import inventory, validate_manifest
    from unlimited_ocr_2_rocm.benchmark.watchdog import ResourceWatchdog
    from unlimited_ocr_2_rocm.config import file_sha256
    from unlimited_ocr_2_rocm.inference import validate_l1_policy
except ImportError:  # pragma: no cover - direct script execution without install
    sys.path.insert(0, str(_REPO_ROOT / "src"))
    from unlimited_ocr_2_rocm.benchmark import runner as benchmark_runner
    from unlimited_ocr_2_rocm.benchmark.ledger import Run
    from unlimited_ocr_2_rocm.benchmark.manifest import inventory, validate_manifest
    from unlimited_ocr_2_rocm.benchmark.watchdog import ResourceWatchdog
    from unlimited_ocr_2_rocm.config import file_sha256
    from unlimited_ocr_2_rocm.inference import validate_l1_policy

PAGE_INDEX_REPO_PATH = "eval/benchmarks/g05/datasets/MPDocBench-page-index.json"
GIB = 1024**3


def load_policy(path: Path) -> dict[str, Any]:
    """Validate against the pinned digest and return the frozen contents."""
    return validate_l1_policy(
        path,
        ngram_size=35,
        ngram_window=1024,
        max_new_tokens=39321,
        expected_pages=15,
        repo_root=_REPO_ROOT,
    )


def admission_report(pages: int, budget: int, policy: dict[str, Any]) -> dict[str, Any]:
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


def verify_policy_page_hashes(
    data_root: Path, rel_pages: list[str], page_hashes: list[str]
) -> None:
    if len(rel_pages) != len(page_hashes):
        raise ValueError("Policy page-hash count disagrees with the document pages")
    for rel, expected in zip(rel_pages, page_hashes, strict=True):
        if file_sha256(data_root / rel) != expected:
            raise ValueError(f"Page bytes disagree with the frozen policy hash: {rel}")


def prepare_command(args: argparse.Namespace) -> int:
    policy = load_policy(args.policy)
    lock = json.loads(args.lock.read_text("utf-8"))
    pin = lock["datasets"]["MPDocBench"]
    manifest = inventory(
        args.ground_truth, args.dataset_root, "MPDocBench", pin["revision"], 420, 3135
    )
    if validate_manifest(manifest, args.dataset_root)["status"] != "PASS":
        raise ValueError("Local dataset does not satisfy the frozen manifest")
    admission = admission_report(policy["pages"], max(policy["budgets_new_tokens"]), policy)
    if not admission["admitted"]:
        raise ValueError("L1 context admission refused — see the policy admission rule")
    config = {
        "schema": 1,
        "label": "EXPERIMENTAL_L1_FIFTEEN_PAGE_NATIVE",
        "experiment": policy["experiment"],
        "policy_sha256": file_sha256(args.policy),
        "policy_repo_path": str(args.policy.resolve().relative_to(_REPO_ROOT)),
        "selection_receipt_sha256": policy["selection_receipt_sha256"],
        "documents": sorted(policy["documents"]),
        "expected_pages": policy["pages"],
        "budget_new_tokens": max(policy["budgets_new_tokens"]),
        "no_repeat_ngram_size": max(policy["ngram_sizes"]),
        "ngram_window": policy["ngram_window"],
        "unchanged": policy["unchanged"],
        "admission": admission,
        "timeout_seconds_per_document": policy["supervision"]["timeout_seconds_per_document"],
        "gpu_allocated_hard_limit_gib": policy["resource_admission"][
            "gpu_allocated_hard_limit_gib"
        ],
        "claim_limit": policy["claim_limit"],
    }
    environment = benchmark_runner.sanitized_environment()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", "utf-8")
    (args.output / "config.json").write_text(json.dumps(config, indent=1) + "\n", "utf-8")
    (args.output / "environment.json").write_text(
        json.dumps(environment, indent=1, sort_keys=True) + "\n", "utf-8"
    )
    print(json.dumps({"output": str(args.output), "admission": admission}))
    return 0


def run_command(args: argparse.Namespace) -> int:
    run = Run(args.run)
    config = run.data["config"]
    if config.get("policy_sha256") != file_sha256(args.policy):
        raise ValueError("Policy file does not match the frozen experiment identity")
    policy = load_policy(args.policy)
    selection = {
        doc_id: {"reason": "L1 predeclared 15-page selection"} for doc_id in policy["documents"]
    }
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
        max_new_tokens=int(config["budget_new_tokens"]),
        timeout_seconds=float(policy["supervision"]["timeout_seconds_per_document"]),
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
        raise ValueError(f"Document is not authorized by the L1 policy: {args.document}")
    pages = benchmark_runner.resolve_pages(run, args.document, args.data_root)
    admission = admission_report(len(pages), args.max_new_tokens, policy)
    if not admission["admitted"]:
        benchmark_runner.write_outcome(
            Path(args.result),
            {"kind": "error", "error": "ADMISSION_REFUSED", "admission": admission},
        )
        return 0
    rel_pages = [page["path"] for page in run.documents[args.document]["pages"]]
    verify_page_gate(args.repo_root, args.data_root, rel_pages)
    verify_policy_page_hashes(args.data_root, rel_pages, policy["page_hashes"])
    attempt_directory = run.path / "documents" / args.document / f"{args.attempt}.pending"

    from unlimited_ocr_2_rocm.collision_observer import (
        GuardCollisionObserver,
        locate_decode_module,
    )
    from unlimited_ocr_2_rocm.inference import run_native
    from unlimited_ocr_2_rocm.model import load_model

    started = time.time()
    scratch = Path(tempfile.mkdtemp(prefix="uocr2-l1-"))
    observer = GuardCollisionObserver()
    watchdog: ResourceWatchdog | None = None
    try:
        model = load_model(args.model_cache, offline=True)
        decode_module = locate_decode_module(model.model)
        observer.install(decode_module)
        limit_bytes = int(float(policy["resource_admission"]["gpu_allocated_hard_limit_gib"]) * GIB)
        watchdog = ResourceWatchdog(
            sampler=lambda: _allocated_bytes(),
            limit_bytes=limit_bytes,
            heartbeat_path=attempt_directory / "heartbeat.json",
            heartbeat_seconds=float(policy["supervision"]["heartbeat_seconds"]),
        )
        watchdog.start()
        report = run_native(
            model,
            pages,
            scratch,
            multi=True,
            max_new_tokens=args.max_new_tokens,
            no_repeat_ngram_size=int(policy["ngram_sizes"][0]),
            ngram_window=int(policy["ngram_window"]),
            ngram_policy=args.policy,
            budget_policy=args.policy,
            page_limit=policy["pages"],
        )
    except BaseException as error:  # noqa: BLE001 - outcome must reach the parent
        import traceback

        benchmark_runner.write_outcome(
            Path(args.result),
            {
                "kind": "error",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "admission": admission,
                "observer_collision": observer.export_events()["collision"],
            },
        )
        return 0
    finally:
        try:
            observer.uninstall()
        except RuntimeError:
            pass
        if watchdog is not None:
            watchdog.stop()
    try:
        native = report.get("native_generation") or {}
        prompt_ids = native.get("prompt_token_ids") or []
        token_ids = native.get("token_ids") or []
        attempt_directory.mkdir(parents=True, exist_ok=True)
        (attempt_directory / "token_ids.json").write_text(json.dumps(token_ids) + "\n", "utf-8")
        observer.write_events(attempt_directory / "collision-observer-events.json")
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
            "observer": observer.export_events(),
            "watchdog": watchdog.receipt()["watchdog"] if watchdog else None,
            "host_peak_rss_gib": round((memory.get("process_peak_rss_bytes") or 0) / GIB, 3),
            "worker_load_seconds": round(time.time() - started, 3),
            "policy_sha256": file_sha256(args.policy),
            "budget": args.max_new_tokens,
            "no_repeat_ngram_size": int(policy["ngram_sizes"][0]),
            "ngram_window": int(policy["ngram_window"]),
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


def _allocated_bytes() -> int:
    import torch

    return int(torch.cuda.memory_allocated())


def evaluate_l1_gates(run: Run, policy: dict[str, Any], reports: list[dict[str, Any]]) -> dict:
    """Pure L1 gate evaluation over the finished single-document attempt."""
    failures: list[str] = []
    resource = policy["resource_admission"]
    max_gpu = float(resource["gpu_allocated_hard_limit_gib"])
    max_rss = float(resource["host_rss_soft_limit_gib"])
    selection = set(policy["documents"])
    summary = run.summary()
    completed = {
        row["doc_id"]: row
        for row in summary["documents"]
        if row["doc_id"] in selection and row["state"] in {"PASS", "TRUNCATED"}
    }
    if len(completed) != len(selection):
        failures.append("selected document did not complete")
    peaks = {"gpu_peak_allocated_gib": 0.0, "host_peak_rss_gib": 0.0}
    forced_mismatches = 0
    observer_aborted = False
    watchdog_breach = False
    for report in reports:
        memory = report.get("memory_after") or {}
        gpu = (memory.get("gpu_peak_allocated_bytes") or 0) / GIB
        rss = (report.get("experiment", {}).get("host_peak_rss_gib")) or (
            (memory.get("process_peak_rss_bytes") or 0) / GIB
        )
        peaks["gpu_peak_allocated_gib"] = max(peaks["gpu_peak_allocated_gib"], round(gpu, 3))
        peaks["host_peak_rss_gib"] = max(peaks["host_peak_rss_gib"], round(rss, 3))
        experiment = report.get("experiment") or {}
        guard = experiment.get("page_guard") or {}
        if guard.get("forced_mismatches"):
            forced_mismatches += 1
        observer = experiment.get("observer") or {}
        if observer.get("aborted"):
            observer_aborted = True
        watchdog_receipt = experiment.get("watchdog") or {}
        if (watchdog_receipt or {}).get("breach"):
            watchdog_breach = True
    if peaks["gpu_peak_allocated_gib"] > max_gpu:
        failures.append(f"GPU peak {peaks['gpu_peak_allocated_gib']} GiB exceeds {max_gpu}")
    if peaks["host_peak_rss_gib"] > max_rss:
        failures.append(f"host RSS peak {peaks['host_peak_rss_gib']} GiB exceeds {max_rss}")
    if forced_mismatches:
        failures.append("forced-marker mismatches present (replay hard gate)")
    if observer_aborted:
        failures.append("guard-collision observer aborted the attempt")
    if watchdog_breach:
        failures.append("GPU allocated hard limit breached")
    promotion = list(failures) == []
    if completed:
        row = next(iter(completed.values()))
        if row["state"] != "PASS":
            promotion = False
            failures.append(f"15-page promotion requires natural completion, got {row['state']}")
        metadata = row.get("metadata") or {}
        if metadata.get("finish_reason") != "stop":
            promotion = False
        if (metadata.get("experiment") or {}).get("page_guard", {}).get("num_pages_out") != policy[
            "pages"
        ]:
            promotion = False
            failures.append("page completion gate: expected 15/15 pages")
    return {
        "schema": 1,
        "experiment": policy["experiment"],
        "resource_peaks": peaks,
        "forced_marker_mismatches": forced_mismatches,
        "observer_aborted": observer_aborted,
        "watchdog_breach": watchdog_breach,
        "failures": failures,
        "promotion_admitted": promotion,
        "claim_limit": policy["claim_limit"],
    }


def _attempt_reports(run: Run, doc_ids: list[str]) -> list[dict[str, Any]]:
    reports = []
    for doc_id in doc_ids:
        directory = run.path / "documents" / doc_id
        if not directory.is_dir():
            continue
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


def gates_command(args: argparse.Namespace) -> int:
    run = Run(args.run)
    policy = load_policy(args.policy)
    config = run.data["config"]
    if config.get("policy_sha256") != file_sha256(args.policy):
        raise ValueError("Policy file does not match the frozen experiment identity")
    verdict = evaluate_l1_gates(run, policy, _attempt_reports(run, sorted(policy["documents"])))
    summary = run.summary()
    verdict["run_id"] = run.data.get("run_id") if hasattr(run.data, "get") else None
    verdict["counts"] = summary["counts"]
    verdict["ledger_status"] = summary["status"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(verdict, indent=1, sort_keys=True) + "\n", "utf-8")
    print(
        json.dumps(
            {
                "l1_gates": "PASS" if not verdict["failures"] else "FAIL",
                "promotion_admitted": verdict["promotion_admitted"],
                "failures": verdict["failures"],
            }
        )
    )
    return 0


def report_command(args: argparse.Namespace) -> int:
    run = Run(args.run)
    policy = load_policy(args.policy)
    selection = {
        doc_id: {"reason": "L1 predeclared 15-page selection"} for doc_id in policy["documents"]
    }
    receipt = benchmark_runner.sanitized_receipt(
        run,
        selection=selection,
        selection_sha256=file_sha256(args.policy),
        worker_identity={"script": "scripts/run_l1_fifteen_page.py", "supervision": "subprocess"},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n", "utf-8")
    print(json.dumps({"receipt": str(args.output), "counts": receipt["counts"]}))
    return 0


def export_command(args: argparse.Namespace) -> int:
    run = Run(args.run)
    policy = load_policy(args.policy)
    selection = {doc_id for doc_id in policy["documents"]}
    Run.export(run, args.output, include=selection, include_truncated=args.diagnostic)
    print(json.dumps({"export": str(args.output), "diagnostic": args.diagnostic}))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--policy", required=True, type=Path)
    prepare.add_argument("--ground-truth", required=True, type=Path)
    prepare.add_argument("--dataset-root", required=True, type=Path)
    prepare.add_argument("--lock", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)

    run = commands.add_parser("run")
    run.add_argument("--run", required=True, type=Path)
    run.add_argument("--policy", required=True, type=Path)
    run.add_argument("--data-root", required=True, type=Path)
    run.add_argument("--repo-root", required=True, type=Path)
    run.add_argument("--model-cache", required=True, type=Path)

    gates = commands.add_parser("gates")
    gates.add_argument("--run", required=True, type=Path)
    gates.add_argument("--policy", required=True, type=Path)
    gates.add_argument("--output", required=True, type=Path)

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
    worker.add_argument("--timeout-seconds", type=float, default=14400.0)

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
    if args.command == "gates":
        return gates_command(args)
    if args.command == "report":
        return report_command(args)
    return export_command(args)


if __name__ == "__main__":
    raise SystemExit(main())
