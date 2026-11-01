"""Ledger-integrated, supervised execution of real-document benchmark selections.

The parent process owns the append-only ledger; every document attempt runs
in an isolated worker subprocess so that hangs, crashes, OOM and interruption
are observable and recoverable states instead of lost evidence.
"""

from __future__ import annotations

import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import IO, Any

from .ledger import Run, json_bytes, write_bytes
from .manifest import file_hash, local_path
from .telemetry import ResourceTelemetryPolicy, TelemetryMonitor
from .watchdog import WATCHDOG_EXIT_CODE

WORKER_TIMEOUT_STATE = "TIMEOUT"
_OOM_MARKERS = (
    "OutOfMemoryError",
    "out of memory",
    "HIP out of memory",
    "hipErrorOutOfMemory",
    "CUDA out of memory",
    "HSA_STATUS_ERROR_OUT_OF_RESOURCES",
    "amdmemoryallocator",
)
# Receipt fields that disclose the private host layout or stable host IDs.
_PRIVATE_KEYS = {
    "executable",
    "machine_id_sha256",
    "render_node",
    "node",
    "uuid",
    "pci_bus_id",
    "pci_device_id",
    "pci_domain_id",
}


def sanitize_tree(value: Any) -> Any:
    """Drop private keys recursively and refuse local absolute paths."""

    def scrub(item: Any) -> Any:
        if isinstance(item, dict):
            return {key: scrub(child) for key, child in item.items() if key not in _PRIVATE_KEYS}
        if isinstance(item, list):
            return [scrub(child) for child in item]
        if isinstance(item, tuple):
            return [scrub(child) for child in item]
        if isinstance(item, float) and not math.isfinite(item):
            return str(item)
        if isinstance(item, str) and re.search(
            r"/home/|/Users/|[a-z]:[/\\]users[/\\]", item, flags=re.IGNORECASE
        ):
            raise ValueError("Private local path in sanitized evidence")
        return item

    return scrub(value)


def sanitized_environment(model_evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    """Deterministic, host-anonymous environment identity for run/receipts."""
    environment: dict[str, Any] = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": sys.platform,
        "packages": {},
        "gpu": {},
    }
    evidence = model_evidence or {}
    for name in (
        "torch",
        "transformers",
        "triton-rocm",
        "triton-windows",
        "huggingface-hub",
        "tokenizers",
    ):
        version = (evidence.get("native_dependencies") or {}).get(name)
        if version is None:
            try:
                import importlib.metadata as importlib_metadata

                version = importlib_metadata.version(name)
            except Exception:  # noqa: BLE001 - optional dependency identity
                continue
        environment["packages"][name] = version
    gpu = (evidence.get("environment") or {}).get("gpu") or {}
    if isinstance(gpu, dict):
        environment["gpu"] = {"name": gpu.get("name"), "arch": gpu.get("arch")}
    for name in ("model_id", "model_revision", "code_revision", "attention_dispatch"):
        if name in evidence:
            environment[name] = evidence[name]
    return sanitize_tree(environment)


def oom_in_traceback(traceback_text: str) -> bool:
    lowered = traceback_text.lower()
    return any(marker.lower() in lowered for marker in _OOM_MARKERS)


def last_error_line(traceback_text: str) -> str:
    lines = [line.strip() for line in traceback_text.splitlines() if line.strip()]
    return lines[-1] if lines else "UNKNOWN"


def classify_report(report: dict[str, Any]) -> tuple[str, str]:
    """Map a run_native report to an explicit ledger state and reason."""
    if report.get("status") in {"PASS", "INFERENCE_ONLY"} and report.get("finish_reason") == "stop":
        return "PASS", ""
    finish = report.get("finish_reason")
    if finish == "length" or report.get("output_token_count") == report.get("max_new_tokens"):
        return "TRUNCATED", "TOKEN_LIMIT_NATIVE_LENGTH"
    if report.get("traceback") and oom_in_traceback(str(report["traceback"])):
        return "OOM", "WORKER_OUT_OF_MEMORY"
    detail = last_error_line(str(report.get("traceback") or report.get("error") or "NATIVE_FAIL"))
    return "FAIL", f"NATIVE: {detail}"[:200]


#: Wall-clock bound for confirming a terminated worker is actually gone.
TERMINATION_GRACE_SECONDS = 10.0


def terminate_process_tree(
    process: subprocess.Popen[bytes],
    *,
    grace_seconds: float = TERMINATION_GRACE_SECONDS,
    taskkill_log: IO[bytes] | None = None,
) -> str:
    """Force-terminate a supervised worker together with its descendants.

    POSIX: the worker was started in its own session, so one signal to its
    process group reaches the worker and every descendant that did not detach.
    Windows: ``taskkill /T /F`` walks and force-terminates the whole child
    tree. Returns ``"TERMINATED"`` once the worker itself is confirmed reaped,
    or ``"TERMINATION_INCOMPLETE"`` when it outlives the grace window —
    callers must treat that as failed cleanup, never as success. Descendants
    that detached themselves from the group or tree are outside this reach.
    """
    try:
        if sys.platform == "win32":
            killed = subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=taskkill_log,
                stderr=taskkill_log,
                check=False,
                capture_output=taskkill_log is None,
            )
            if killed.returncode != 0 and process.poll() is None:
                # taskkill refused while the worker still lives: direct
                # termination so the attempt cannot outlive supervision.
                process.kill()
        else:
            # The worker leads its own process group (start_new_session=True).
            os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass  # The worker exited between the timeout and the kill signal.
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        return "TERMINATION_INCOMPLETE"
    return "TERMINATED"


def supervise_worker(
    command: list[str],
    *,
    attempt_directory: Path,
    timeout_seconds: float,
    resource_policy: ResourceTelemetryPolicy | None = None,
) -> tuple[dict[str, Any] | None, str]:
    """Run one worker subprocess; returns (outcome JSON or None, failure kind).

    ``resource_policy`` (Rev-2 prerequisite P5) additionally runs the 1 Hz
    telemetry monitor with the staleness rule and the host available-memory
    floor; a policy stop terminates the attempt with an explicit
    ``WORKER_RESOURCE_STOP_<REASON>`` failure kind — never silently.
    """
    if sys.platform == "win32":
        return _supervise_windows_worker(
            command,
            attempt_directory=attempt_directory,
            timeout_seconds=timeout_seconds,
            resource_policy=resource_policy,
        )
    staging = Path(tempfile.mkdtemp(prefix="uocr2-worker-"))
    result_path = staging / "outcome.json"
    command = [*command, "--result", str(result_path)]
    log_path = attempt_directory / "worker.log"
    try:
        with log_path.open("xb") as log:
            process = subprocess.Popen(  # noqa: S603 - fixed argv from this code base
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=os.name != "nt",
            )
            try:
                code = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                cleanup = terminate_process_tree(process)
                if cleanup != "TERMINATED":
                    return None, f"WORKER_TIMEOUT_{cleanup}"
                return None, "WORKER_TIMEOUT"
        if result_path.is_file():
            try:
                outcome = json.loads(result_path.read_text(encoding="utf-8"))
            except ValueError:
                return None, f"WORKER_CRASH_EXIT_{code}_UNREADABLE_OUTCOME"
            if code != 0 and outcome.get("kind") != "error":
                return outcome, f"WORKER_EXIT_{code}"
            return outcome, ""
        return None, f"WORKER_CRASH_EXIT_{code}"
    finally:
        for path in staging.iterdir():
            path.unlink(missing_ok=True)
        staging.rmdir()


class WorkerCleanupError(RuntimeError):
    """Ownership could not be drained; stop the campaign with evidence retained."""


def _supervise_windows_worker(
    command: list[str],
    *,
    attempt_directory: Path,
    timeout_seconds: float,
    resource_policy: ResourceTelemetryPolicy | None = None,
) -> tuple[dict[str, Any] | None, str]:
    from .windows_job import NativeJob

    # The worker publishes its own atomic outcome here. Retain both complete
    # and partial bytes on every failure; no system temp or shared scratch.
    result_path = attempt_directory / "worker-outcome.json"
    accounting: dict[str, Any] = {"ownership": "windows-job-suspended-before-assignment"}
    process: subprocess.Popen[bytes] | None = None
    job: NativeJob | None = None
    code: int | None = None
    timed_out = False
    resource_stop: str | None = None
    cleanup_error: BaseException | None = None
    monitor: TelemetryMonitor | None = None
    if resource_policy is not None:
        monitor = TelemetryMonitor(attempt_directory, resource_policy)
    try:
        job = NativeJob()
        with (attempt_directory / "worker.log").open("xb") as log:
            process = job.spawn(
                [*command, "--result", str(result_path)], stdout=log, stderr=subprocess.STDOUT
            )
            accounting["worker_pid"] = process.pid
            if monitor is not None:
                monitor.start()
                deadline = time.monotonic() + timeout_seconds
                while True:
                    try:
                        code = process.wait(timeout=min(1.0, max(deadline - time.monotonic(), 0.1)))
                        break
                    except subprocess.TimeoutExpired:
                        if time.monotonic() >= deadline:
                            timed_out = True
                            break
                        if monitor.stop_reason is not None:
                            resource_stop = monitor.stop_reason
                            break
            else:
                try:
                    code = process.wait(timeout=timeout_seconds)
                except subprocess.TimeoutExpired:
                    timed_out = True
    except BaseException as error:
        accounting["exception"] = type(error).__name__
        accounting["exception_detail"] = str(error)
        raise
    finally:
        # This must also run after a normal worker exit: descendants can detach
        # from its console while remaining owned by the Job. No process runs
        # before assignment; KILL_ON_JOB_CLOSE protects assigned processes on
        # supervisor death (the creation/assignment hard-exit gap is documented
        # in windows_job.py).
        try:
            if job is None:
                accounting["cleanup"] = {"status": "NOT_STARTED"}
            else:
                cleanup = job.terminate(grace_seconds=TERMINATION_GRACE_SECONDS)
                if process is not None:
                    code = process.wait(timeout=TERMINATION_GRACE_SECONDS)
                if cleanup["active_processes"] != 0:
                    raise WorkerCleanupError("Windows Job still has active processes")
                accounting["cleanup"] = {"status": "TERMINATED", **cleanup}
        except BaseException as error:
            cleanup_error = error
            accounting["cleanup"] = {
                "status": "UNVERIFIED",
                "error": type(error).__name__,
                "detail": str(error),
            }
        finally:
            try:
                if job is not None:
                    job.close()
            except BaseException as error:
                cleanup_error = error
                accounting["close_error"] = type(error).__name__
                accounting["close_error_detail"] = str(error)
            if monitor is not None:
                accounting.update(monitor.stop())
            accounting.update(exit_code=code, timed_out=timed_out)
            write_bytes(attempt_directory / "supervision.json", json_bytes(accounting))
        if cleanup_error is not None:
            raise WorkerCleanupError(
                "Windows Job cleanup unverified; campaign stopped"
            ) from cleanup_error
    if timed_out:
        return None, "WORKER_TIMEOUT"
    if resource_stop is not None:
        return None, f"WORKER_RESOURCE_STOP_{resource_stop}"
    if code == WATCHDOG_EXIT_CODE:
        # The worker-side ResourceWatchdog aborted at a frozen limit (its
        # breach receipt lives next to the heartbeat); os._exit may leave the
        # outcome file missing or truncated, so classify before parsing it.
        return None, "WORKER_RESOURCE_LIMIT_BREACH"
    if not result_path.is_file():
        return None, f"WORKER_CRASH_EXIT_{code}"
    try:
        outcome = json.loads(result_path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError):
        return None, f"WORKER_CRASH_EXIT_{code}_UNREADABLE_OUTCOME"
    kind = outcome.get("kind") if isinstance(outcome, dict) else None
    if not isinstance(kind, str) or kind not in {"ok", "error", "selftest"}:
        return None, "WORKER_INVALID_OUTCOME"
    if outcome["kind"] != "error" and not isinstance(outcome.get("report"), dict):
        return None, "WORKER_INVALID_OUTCOME"
    if code != 0 and outcome["kind"] != "error":
        return outcome, f"WORKER_EXIT_{code}"
    return outcome, ""


def worker_outcome_to_finish(
    outcome: dict[str, Any] | None,
    failure_kind: str,
    *,
    attempt: str,
    max_new_tokens: int,
    started: float,
) -> tuple[str, str, dict[str, Any]]:
    """Build (state, reason, metadata) for ledger.finish from worker supervision."""
    wall = round(time.time() - started, 3)
    base: dict[str, Any] = {"attempt_wall_seconds": wall, "max_tokens": max_new_tokens}
    if failure_kind == "WORKER_TIMEOUT":
        return WORKER_TIMEOUT_STATE, "WORKER_TIMEOUT", {**base, "error": "WORKER_TIMEOUT"}
    if failure_kind == "WORKER_RESOURCE_LIMIT_BREACH":
        # Allocator/reserved hard limit enforced in-worker by ResourceWatchdog;
        # OOM-class, evidence retained next to the attempt's heartbeat.
        return (
            "OOM",
            "WORKER_RESOURCE_LIMIT_BREACH",
            {**base, "error": "WORKER_RESOURCE_LIMIT_BREACH"},
        )
    if outcome is None:
        return "FAIL", failure_kind, {**base, "error": failure_kind}
    if failure_kind:
        return "FAIL", failure_kind, {**base, "error": failure_kind}
    if outcome.get("kind") == "error":
        traceback_text = str(outcome.get("traceback") or "")
        if oom_in_traceback(traceback_text):
            return (
                "OOM",
                "WORKER_OUT_OF_MEMORY",
                {
                    **base,
                    "error": "WORKER_OUT_OF_MEMORY",
                    "detail": last_error_line(traceback_text),
                },
            )
        return (
            "FAIL",
            f"WORKER: {last_error_line(traceback_text)}"[:200],
            {
                **base,
                "error": f"WORKER: {last_error_line(traceback_text)}"[:200],
            },
        )
    report = outcome.get("report") or {}
    state, reason = classify_report(report)
    metadata = {
        **base,
        "error": reason,
        "finish_reason": report.get("finish_reason"),
        "output_tokens": report.get("output_token_count"),
        "memory": report.get("memory_after"),
        "upstream_timing": report.get("upstream_timing"),
        "native_kernel_count": report.get("native_kernel_count"),
        "synchronized_wall_seconds": report.get("synchronized_wall_seconds"),
        "raw_sha256": report.get("raw_sha256"),
    }
    if isinstance(report.get("experiment"), dict):
        metadata["experiment"] = report["experiment"]
    return state, reason, metadata


def execute_selection(
    run: Run,
    selection: dict[str, dict[str, Any]],
    *,
    worker_command: list[str],
    data_root: Path,
    max_new_tokens: int,
    timeout_seconds: float,
    restrict: set[str] | None = None,
    resource_policy: ResourceTelemetryPolicy | None = None,
) -> dict[str, Any]:
    """Attempt every selected, not-yet-PASS document exactly once per call."""
    unknown = sorted(set(selection) - set(run.documents))
    if unknown:
        raise ValueError(f"Selection documents outside frozen manifest: {unknown}")
    todo = [doc_id for doc_id in selection if doc_id in set(run.todo())]
    if restrict is not None:
        todo = [doc_id for doc_id in todo if doc_id in restrict]
    executed: list[dict[str, Any]] = []
    for doc_id in todo:
        attempt = run.begin(doc_id)
        attempt_directory = run.path / "documents" / doc_id / f"{attempt}.pending"
        started = time.time()
        command = [
            *worker_command,
            "--run",
            str(run.path),
            "--document",
            doc_id,
            "--attempt",
            attempt,
            "--data-root",
            str(data_root),
            "--max-new-tokens",
            str(max_new_tokens),
            "--timeout-seconds",
            str(timeout_seconds),
        ]
        outcome, failure = supervise_worker(
            command,
            attempt_directory=attempt_directory,
            timeout_seconds=timeout_seconds,
            resource_policy=resource_policy,
        )
        state, reason, metadata = worker_outcome_to_finish(
            outcome,
            failure,
            attempt=attempt,
            max_new_tokens=max_new_tokens,
            started=started,
        )
        raw = ""
        if outcome is not None and outcome.get("kind") == "ok":
            raw = str(outcome["report"].get("raw_output") or "")
        elif outcome is not None and outcome.get("kind") == "selftest":
            raw = str(outcome.get("raw") or "")
        try:
            run.finish(doc_id, attempt, raw, state, metadata)
        except Exception as error:  # noqa: BLE001 - ledger refusal is itself a failure state
            run.finish(
                doc_id,
                attempt,
                raw,
                "FAIL",
                {**metadata, "error": f"LEDGER_REFUSED: {type(error).__name__}: {error}"[:200]},
            )
            state, reason = "FAIL", f"LEDGER_REFUSED: {error}"[:200]
        executed.append({"doc_id": doc_id, "attempt": attempt, "state": state, "reason": reason})
        print(f"{doc_id}/{attempt}: {state} {reason}", flush=True)
    summary = run.summary()
    return {
        "executed": executed,
        "counts": summary["counts"],
        "todo_remaining": run.todo(),
    }


def selection_documents(selection_path: Path) -> dict[str, dict[str, Any]]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not isinstance(selection, dict) or not selection:
        raise ValueError("Selection must be a non-empty object of document id -> reason record")
    for doc_id, entry in selection.items():
        if not isinstance(entry, dict) or "reason" not in entry:
            raise ValueError(f"Selection entry needs a frozen reason: {doc_id}")
    return selection


def write_gt_slice(
    run: Run,
    ground_truth: Path,
    selection_path: Path,
    selection: dict[str, dict[str, Any]],
    output: Path,
) -> dict[str, Any]:
    """Scorer ground truth restricted to the selection; annotation bytes unmodified."""
    rows = json.loads(ground_truth.read_text(encoding="utf-8"))
    by_id = {row["page_info"]["image_path"].removesuffix(".pdf"): row for row in rows}
    missing = sorted(set(selection) - set(by_id))
    if missing:
        raise ValueError(f"Selection documents absent from ground truth: {missing}")
    slice_rows = [by_id[doc_id] for doc_id in selection]
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_bytes(output, json.dumps(slice_rows, indent=1, ensure_ascii=False).encode("utf-8"))
    return {
        "ground_truth_sha256": file_hash(ground_truth),
        "selection_sha256": file_hash(selection_path),
        "documents": sorted(selection),
        "slice_sha256": file_hash(output),
    }


def sanitized_receipt(
    run: Run,
    *,
    selection: dict[str, dict[str, Any]],
    selection_sha256: str,
    worker_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Public accounting receipt: no raw text, no private paths, no host IDs."""
    summary = run.summary()
    documents = []
    for row in summary["documents"]:
        if row["doc_id"] not in selection:
            continue
        documents.append(
            {
                "doc_id": row["doc_id"],
                "pages": row["pages"],
                "state": row["state"],
                "reason": row.get("reason", ""),
                "attempts": len(row["attempts"]),
                "finish_reason": (row.get("metadata") or {}).get("finish_reason"),
                "output_tokens": (row.get("metadata") or {}).get("output_tokens"),
                "attempt_wall_seconds": (row.get("metadata") or {}).get("attempt_wall_seconds"),
            }
        )
    receipt = {
        "schema": 1,
        "label": "REAL_DATA_INTEGRATION_ONLY",
        "claim_limit": (
            "Small predeclared MPDocBench subset; NOT full MPDocBench accuracy, "
            "NOT leaderboard parity, NOT full dataset completion"
        ),
        "run_id": run.path.name,
        "dataset": summary["dataset"],
        "expected_documents": summary["expected_documents"],
        "selection_documents": sorted(selection),
        "selection_sha256": selection_sha256,
        "config": run.data["config"],
        "environment": run.data["environment"],
        "counts": summary["counts"],
        "status": summary["status"],
        "documents": documents,
        "worker_identity": worker_identity or {},
    }
    return sanitize_tree(receipt)


def resolve_pages(run: Run, doc_id: str, data_root: Path) -> list[Path]:
    """Manifest-ordered page paths, contained inside the dataset root."""
    document = run.documents.get(doc_id)
    if document is None:
        raise ValueError(f"Document is outside frozen manifest: {doc_id}")
    return [local_path(data_root, page["path"]) for page in document["pages"]]


def write_outcome(path: Path, payload: dict[str, Any]) -> None:
    """Atomic worker outcome publication for the supervising parent."""
    staging = path.with_name(path.name + ".partial")
    write_bytes(staging, json_bytes(payload))
    staging.rename(path)
