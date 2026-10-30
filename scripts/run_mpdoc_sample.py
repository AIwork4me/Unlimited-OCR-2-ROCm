"""Ledger-integrated MPDocBench real-document runner (P1.5 hardening).

Drives the append-only ``benchmark.ledger.Run`` interface. The parent owns
begin/finish; each document attempt executes in an isolated worker
subprocess supervised with an external timeout, so PASS / FAIL / TRUNCATED /
TIMEOUT / OOM / INTERRUPTED / NOT_RUN are all explicit, resumable states.
Raw model bytes and worker logs stay inside the ledger run directory
(untracked); only sanitized accounting receipts are exported for Git.

Production limits are unchanged: at most five pages and 8192 new tokens per
request. Results remain REAL_DATA_INTEGRATION_ONLY — never full MPDocBench
accuracy or leaderboard parity.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
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
SELF_TEST_MODES = ("good", "length", "oom", "crash", "hang", "malformed")


def load_ground_truth(gt_path: Path) -> dict[str, dict]:
    gt = json.loads(gt_path.read_text("utf-8"))
    by_id = {}
    for document in gt:
        info = document["page_info"]
        by_id[info["image_path"].removesuffix(".pdf")] = document
    return by_id


def document_pages(document: dict) -> list[str]:
    return list(document["page_info"]["images_list"])


def verify_page_hashes(data_root: Path, pages: list[str], page_index: dict[str, dict]) -> None:
    for rel in pages:
        expected = page_index.get(rel)
        if expected is None:
            raise ValueError(f"page not present in frozen index: {rel}")
        actual = file_sha256(data_root / rel)
        if actual != expected["sha256"]:
            raise ValueError(f"page hash mismatch against frozen index: {rel}")


def load_page_index(repo_root: Path) -> dict[str, dict]:
    index = json.loads((repo_root / PAGE_INDEX_REPO_PATH).read_text("utf-8"))
    return {page["path"]: page for page in index["pages"]}


def prepare_command(args: argparse.Namespace) -> int:
    """Write the frozen run inputs: full-dataset manifest, config, environment."""
    selection = benchmark_runner.selection_documents(args.selection)
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
    config = {
        "schema": 1,
        "label": "REAL_DATA_INTEGRATION_ONLY",
        "runner": "scripts/run_mpdoc_sample.py (ledger-integrated P1.5)",
        "selection_sha256": file_sha256(args.selection),
        "selection_documents": sorted(selection),
        "max_new_tokens": args.max_new_tokens,
        "max_pages_per_request": 5,
        "max_pixels_per_page": 16_000_000,
        "timeout_seconds_per_document": args.timeout_seconds,
        "page_hash_gate": PAGE_INDEX_REPO_PATH,
        "claim_limit": (
            "Small predeclared MPDocBench subset; NOT full MPDocBench accuracy, "
            "NOT leaderboard parity, NOT full dataset completion"
        ),
    }
    environment = benchmark_runner.sanitized_environment()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", "utf-8")
    (args.output / "config.json").write_text(json.dumps(config, indent=1) + "\n", "utf-8")
    (args.output / "environment.json").write_text(
        json.dumps(environment, indent=1, sort_keys=True) + "\n", "utf-8"
    )
    print(json.dumps({"output": str(args.output), "manifest_sha256": report["manifest_sha256"]}))
    print(
        "Initialize the ledger run with:\n"
        f"  python -m unlimited_ocr_2_rocm.benchmark.cli init "
        f"--manifest {args.output / 'manifest.json'} --dataset-root {args.dataset_root} "
        f"--config {args.output / 'config.json'} --environment {args.output / 'environment.json'} "
        f"--output <runs-root>"
    )
    return 0


def run_command(args: argparse.Namespace) -> int:
    run = Run(args.run)
    selection = benchmark_runner.selection_documents(args.selection)
    config = run.data["config"]
    if config.get("selection_sha256") != file_sha256(args.selection):
        raise ValueError("Selection file does not match the frozen run identity")
    max_new_tokens = int(config.get("max_new_tokens", args.max_new_tokens))
    timeout_seconds = float(config.get("timeout_seconds_per_document", args.timeout_seconds))
    restrict = set(args.documents.split(",")) if args.documents else None
    result = benchmark_runner.execute_selection(
        run,
        selection,
        worker_command=[
            sys.executable,
            str(Path(__file__).resolve()),
            "worker",
            "--repo-root",
            str(args.repo_root),
            *(["--model-cache", str(args.model_cache)] if args.model_cache else []),
        ],
        data_root=args.data_root,
        max_new_tokens=max_new_tokens,
        timeout_seconds=timeout_seconds,
        restrict=restrict,
    )
    print(json.dumps({"counts": result["counts"], "todo_remaining": result["todo_remaining"]}))
    return 0


def worker_command(args: argparse.Namespace) -> int:
    """Execute exactly one document attempt; the parent owns the ledger."""
    run = Run(args.run)
    pages = benchmark_runner.resolve_pages(run, args.document, args.data_root)
    rel_pages = [page["path"] for page in run.documents[args.document]["pages"]]
    page_index = load_page_index(args.repo_root)
    verify_page_hashes(args.data_root, rel_pages, page_index)
    result_path = Path(args.result)
    attempt_directory = run.path / "documents" / args.document / f"{args.attempt}.pending"

    if args.self_test_failure:
        try:
            outcome = self_test_outcome(args, run, pages, attempt_directory)
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

    from unlimited_ocr_2_rocm.doctor import memory_snapshot
    from unlimited_ocr_2_rocm.inference import run_native
    from unlimited_ocr_2_rocm.model import load_model

    started = time.time()
    scratch = Path(tempfile.mkdtemp(prefix="uocr2-native-"))
    try:
        model = load_model(args.model_cache, offline=True)
        report = run_native(
            model,
            pages,
            scratch,
            multi=True,
            max_new_tokens=args.max_new_tokens,
        )
        report["worker_load_seconds"] = round(time.time() - started, 3)
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


def self_test_outcome(
    args: argparse.Namespace, run: Run, pages: list[Path], attempt_directory: Path
) -> dict | None:
    """CPU failure injection executed by the real supervised worker path."""
    count = len(pages)
    good_raw = "<PAGE><BLOCK>text[0,0,1,1]<FORK>hello" * count
    mode = args.self_test_failure
    if mode == "hang":
        time.sleep(args.timeout_seconds + 60)
        return None
    if mode == "crash":
        # Die like an externally killed worker: SIGKILL on POSIX. On Windows
        # os.kill maps values other than CTRL_C_EVENT(0)/CTRL_BREAK_EVENT(1)
        # to TerminateProcess; 1 would broadcast a console break event to
        # every process sharing the console, including the supervisor.
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
        "native_kernel_count": 12,
    }
    if mode == "malformed":
        report["raw_output"] = "no page structure at all"
    else:
        report["raw_output"] = good_raw
        report["raw_sha256"] = None
    (attempt_directory / "selftest.json").write_text(
        json.dumps({"mode": mode, "pages": count}) + "\n", "utf-8"
    )
    return {"kind": "selftest", "mode": mode, "raw": report["raw_output"], "report": report}


def report_command(args: argparse.Namespace) -> int:
    run = Run(args.run)
    selection = benchmark_runner.selection_documents(args.selection)
    receipt = benchmark_runner.sanitized_receipt(
        run,
        selection=selection,
        selection_sha256=file_sha256(args.selection),
        worker_identity={"script": "scripts/run_mpdoc_sample.py", "supervision": "subprocess"},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    (args.output).write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n", "utf-8")
    print(json.dumps({"receipt": str(args.output), "counts": receipt["counts"]}))
    return 0


def gt_slice_command(args: argparse.Namespace) -> int:
    run = Run(args.run)
    selection = benchmark_runner.selection_documents(args.selection)
    evidence = benchmark_runner.write_gt_slice(
        run, args.ground_truth, args.selection, selection, args.output
    )
    print(json.dumps(evidence))
    return 0


def export_command(args: argparse.Namespace) -> int:
    run = Run(args.run)
    selection = benchmark_runner.selection_documents(args.selection)
    run.export(
        args.output,
        include=set(selection),
        include_truncated=args.diagnostic,
    )
    print(json.dumps({"export": str(args.output), "diagnostic": args.diagnostic}))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="write frozen manifest/config/environment")
    for flag, kwargs in (
        ("--ground-truth", {"required": True, "type": Path}),
        ("--dataset-root", {"required": True, "type": Path}),
        ("--lock", {"required": True, "type": Path}),
        ("--selection", {"required": True, "type": Path}),
        ("--max-new-tokens", {"type": int, "default": 8192}),
        ("--timeout-seconds", {"type": float, "default": 3600.0}),
        ("--output", {"required": True, "type": Path}),
    ):
        prepare.add_argument(flag, **kwargs)

    run = commands.add_parser("run", help="execute selected documents through the ledger")
    run.add_argument("--run", required=True, type=Path)
    run.add_argument("--selection", required=True, type=Path)
    run.add_argument("--data-root", required=True, type=Path)
    run.add_argument("--repo-root", required=True, type=Path)
    run.add_argument("--model-cache", type=Path, default=None)
    run.add_argument("--max-new-tokens", type=int, default=8192)
    run.add_argument("--timeout-seconds", type=float, default=3600.0)
    run.add_argument("--documents", default=None, help="comma-separated subset")

    worker = commands.add_parser("worker", help="internal: one supervised attempt")
    worker.add_argument("--run", required=True, type=Path)
    worker.add_argument("--document", required=True)
    worker.add_argument("--attempt", required=True)
    worker.add_argument("--data-root", required=True, type=Path)
    worker.add_argument("--repo-root", required=True, type=Path)
    worker.add_argument("--model-cache", type=Path, default=None)
    worker.add_argument("--max-new-tokens", type=int, default=8192)
    worker.add_argument("--timeout-seconds", type=float, default=3600.0)
    worker.add_argument("--result", required=True, type=Path)
    worker.add_argument("--self-test-failure", choices=SELF_TEST_MODES, default=None)

    report = commands.add_parser("report", help="write the sanitized accounting receipt")
    report.add_argument("--run", required=True, type=Path)
    report.add_argument("--selection", required=True, type=Path)
    report.add_argument("--output", required=True, type=Path)

    gt = commands.add_parser("gt-slice", help="write the scorer ground-truth slice")
    gt.add_argument("--run", required=True, type=Path)
    gt.add_argument("--ground-truth", required=True, type=Path)
    gt.add_argument("--selection", required=True, type=Path)
    gt.add_argument("--output", required=True, type=Path)

    export = commands.add_parser("export", help="export scorer predictions from the ledger")
    export.add_argument("--run", required=True, type=Path)
    export.add_argument("--selection", required=True, type=Path)
    export.add_argument("--output", required=True, type=Path)
    export.add_argument("--diagnostic", action="store_true")

    args = parser.parse_args(argv)
    handlers = {
        "prepare": prepare_command,
        "run": run_command,
        "worker": worker_command,
        "report": report_command,
        "gt-slice": gt_slice_command,
        "export": export_command,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
