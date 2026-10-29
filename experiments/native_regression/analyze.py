"""CPU-only exact capture comparison and real append-only ledger failure replay."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.native_regression.runner import (  # noqa: E402
    compare_case,
    comparison_arm_gate,
    fresh_output,
    hash_manifest,
    normal_cases,
    read_case,
    sha256,
    write_json,
)
from unlimited_ocr_2_rocm.benchmark.ledger import Run  # noqa: E402
from unlimited_ocr_2_rocm.benchmark.manifest import inventory  # noqa: E402


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def source_identity(sources: dict[str, Any]) -> dict[str, Any]:
    return {
        name: {key: value for key, value in record.items() if key in {"sha256", "line", "function"}}
        for name, record in sources.items()
    }


def memory_report(summary: dict[str, Any]) -> dict[str, Any]:
    memory = summary.get("memory_after", {})
    return {
        "native_process": memory,
        "launcher_rss_used": False,
        "source": "memory_snapshot executed inside the actual model Python process",
        "scope": (
            "process RSS/peak working set and ROCm-compatible torch.cuda allocator "
            "allocated/reserved peaks; APU allocation is not dedicated VRAM or total system use"
        ),
        "launcher_note": (
            "parent host-memory.jsonl may track the Windows venv launcher/redirector; "
            "its approximately 4 MB RSS is not model-process memory"
        ),
    }


def guard_events(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (path / "diagnostics/guard.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]


def analyze_pair(before: Path, after: Path) -> dict[str, Any]:
    left, right = read_case(before), read_case(after)
    result = compare_case(left, right)
    left_raw = (before / "diagnostics/native-001.raw.txt").read_bytes()
    right_raw = (after / "diagnostics/native-001.raw.txt").read_bytes()
    guards = [guard_events(path) for path in (before, after)]
    checks = {
        "raw_bytes": left_raw == right_raw,
        "native_source_hashes": source_identity(left["native_source_map"])
        == source_identity(right["native_source_map"]),
        "vision_MATH_scope_events": left["vision"] == right["vision"],
        "guard_trace": guards[0] == guards[1],
        "guard_counts": left["guard_branch_counts"] == right["guard_branch_counts"],
        "ground_truth_assessment": left.get("correctness") == right.get("correctness"),
        "guard_trace_present": bool(guards[0]) and bool(guards[1]),
    }
    differences = result["differences"] + [name for name, equal in checks.items() if not equal]
    native = left["native"]
    return {
        **result,
        "status": "PASS" if not differences else "FAIL",
        "differences": differences,
        "exact_checks": checks,
        "raw_bytes_before": len(left_raw),
        "raw_bytes_after": len(right_raw),
        "output_token_count": len(native["token_ids"]),
        "prompt_token_count": len(native["prompt_token_ids"]),
        "native_finish": native["finish_reason"],
        "input_pages": native["expected_pages"],
        "raw_PAGE_markers": native["num_pages_out"],
        "terminal_empty_PAGE": native["text"].endswith("<PAGE>"),
        "kernel_count": len(left["kernels"]),
        "kernel_query_shapes": [event["query_shape"] for event in left["kernels"]],
        "guard_branch_counts": left["guard_branch_counts"],
        "final_guard_branch": [events[-1]["branch"] for events in guards],
        "wrapper_metadata_interpretation": (
            "Old multi tuple caller discarded native finish; new caller retains authoritative "
            "native finish. "
            "This expected metadata improvement is distinct from native output equality."
        ),
        "memory": {"before_arm": memory_report(left), "after_arm": memory_report(right)},
        "synchronized_wall_seconds": {
            "before": left["synchronized_wall_seconds"],
            "after": right["synchronized_wall_seconds"],
        },
        "source_artifacts": {
            "before_summary_sha256": sha256(before / "summary.json"),
            "after_summary_sha256": sha256(after / "summary.json"),
        },
    }


def analyze_arms(before: Path, after: Path) -> dict[str, Any]:
    cases = {}
    for case in normal_cases():
        try:
            cases[case["name"]] = analyze_pair(before / case["name"], after / case["name"])
        except FileNotFoundError as error:
            cases[case["name"]] = {"status": "NOTRUN", "error": str(error)}
        except (ValueError, KeyError) as error:
            cases[case["name"]] = {"status": "FAIL", "error": str(error)}
    gate = (
        comparison_arm_gate(before, after)
        if all((arm / "suite.json").is_file() for arm in (before, after))
        else {"status": "NOTRUN"}
    )
    source_equality = {}
    for filename, key in (
        ("loaded-native-sources.json", None),
        ("load.json", "source_and_weight_hashes"),
    ):
        if all((arm / filename).is_file() for arm in (before, after)):
            left, right = [read_json(arm / filename) for arm in (before, after)]
            source_equality[filename] = (
                (source_identity(left) == source_identity(right))
                if key is None
                else left[key] == right[key]
            )
        else:
            source_equality[filename] = False
    passed = sum(case["status"] == "PASS" for case in cases.values())
    failed = sum(case["status"] == "FAIL" for case in cases.values())
    notrun = sum(case["status"] == "NOTRUN" for case in cases.values())
    status = (
        "PASS"
        if passed == 13 and gate["status"] == "PASS" and all(source_equality.values())
        else "NOTRUN"
        if notrun and not failed
        else "FAIL"
    )
    physical = physical_failure_report(before, after)
    if physical["status"] == "FAIL":
        status = "FAIL"
    elif physical["status"] == "NOTRUN" and status == "PASS":
        status = "NOTRUN"
    return {
        "status": status,
        "scope": "Original AMD wheel, physical Windows gfx1151 synthetic fixtures only",
        "arm_gate": gate,
        "source_and_weight_equality": source_equality,
        "case_counts": {"expected": 13, "PASS": passed, "FAIL": failed, "NOTRUN": notrun},
        "byte_comparison_counts": {
            "cases_compared": passed + failed,
            "raw_bytes_equal": sum(
                case.get("exact_checks", {}).get("raw_bytes", False) for case in cases.values()
            ),
            "before_total_bytes": sum(case.get("raw_bytes_before", 0) for case in cases.values()),
            "after_total_bytes": sum(case.get("raw_bytes_after", 0) for case in cases.values()),
        },
        "MATH_evidence_limit": (
            "Scoped MATH API, immutable compatibility source and completed "
            "vision events; no aten operator profiler claim"
        ),
        "timing_limit": (
            "No speed claim: before/after JIT cache warmth differs; synchronized "
            "timings are descriptive only"
        ),
        "cases": cases,
        "physical_failure_contracts": physical,
        "native_memory_peaks": {
            arm: {
                key: max(
                    (
                        case.get("memory", {})
                        .get(arm + "_arm", {})
                        .get("native_process", {})
                        .get(key, 0)
                        for case in cases.values()
                    ),
                    default=0,
                )
                for key in (
                    "process_rss_bytes",
                    "process_peak_rss_bytes",
                    "gpu_peak_allocated_bytes",
                    "gpu_peak_reserved_bytes",
                )
            }
            for arm in ("before", "after")
        },
    }


def physical_failure_report(before: Path, after: Path) -> dict[str, Any]:
    """Summarize captured GPU controls separately from synthetic CPU ledger replay."""
    controls = {}
    for arm_name, arm in (("before", before), ("after", after)):
        controls[arm_name] = {}
        for name in (
            "length_2_pages_8_tokens",
            "control_max_length_prompt_plus_8",
            "control_page_cap_8",
        ):
            if not (arm / name / "summary.json").is_file():
                controls[arm_name][name] = {"status": "NOTRUN"}
                continue
            row = read_case(arm / name)
            native = row["native"]
            count = len(native["token_ids"])
            if name == "length_2_pages_8_tokens":
                okay = (
                    row["status"] == "FAIL" and native["finish_reason"] == "length" and count == 8
                )
            elif name == "control_max_length_prompt_plus_8":
                okay = (
                    row["status"] == "DIAGNOSTIC_ONLY"
                    and native["finish_reason"] == "length"
                    and count == 8 < row["case"]["max_new_tokens"]
                    and row["direct_diagnostic_options"]["max_length"]
                    == len(native["prompt_token_ids"]) + 8
                )
            else:
                branches = row["guard_branch_counts"]
                okay = (
                    row["status"] == "DIAGNOSTIC_ONLY"
                    and native["finish_reason"] == "stop"
                    and (
                        branches.get("force_page_at_page_cap", 0)
                        + branches.get("force_eos_at_page_cap", 0)
                    )
                    > 0
                    and row.get("correctness", {}).get("status") != "PASS"
                )
            okay = (
                okay
                and row["physical_contract"] == "PASS"
                and row["runtime_state_restored"] is True
            )
            controls[arm_name][name] = {
                "status": "PASS" if okay else "FAIL",
                "wrapper_status": row["status"],
                "native_finish": native["finish_reason"],
                "output_tokens": count,
                "requested_max_new_tokens": row["case"]["max_new_tokens"],
                "raw_sha256": row["raw_sha256"],
                "raw_PAGE_markers": native["num_pages_out"],
                "guard_branch_counts": row["guard_branch_counts"],
                "semantic_acceptance": row.get("correctness", {}).get(
                    "status", "FAIL_OR_UNAVAILABLE"
                ),
                "source_summary_sha256": sha256(arm / name / "summary.json"),
            }
    interrupted_path = after / "controlled_interrupt_2_pages"
    interruption = {"status": "NOTRUN"}
    if (interrupted_path / "summary.json").is_file():
        row = read_json(interrupted_path / "summary.json")
        injection = row.get("fault_injection", {})
        native_present = (
            "native" in row or (interrupted_path / "diagnostics/native-001.json").exists()
        )
        okay = (
            row["status"] == "FAIL"
            and row.get("error") == "INTERRUPTED"
            and row["runtime_state_restored"] is True
            and row["physical_contract"] == "PASS"
            and injection.get("injected") is True
            and injection.get("completed_forward_calls") == 3
            and not native_present
        )
        interruption = {
            "status": "PASS" if okay else "FAIL",
            "wrapper_status": row["status"],
            "error": row.get("error"),
            "fault_injection": injection,
            "native_return_present": native_present,
            "partial_output": (
                "Unavailable: official loop interrupted before returning; "
                "no fabricated partial text/IDs"
            ),
            "runtime_state_restored": row["runtime_state_restored"],
            "hip_after": row["hip_after"],
            "memory": memory_report(row),
        }
    retry = {"status": "NOTRUN"}
    retry_path = after / "retry_after_controlled_interrupt_2_pages"
    if (retry_path / "summary.json").is_file() and (after / "pages_2_run_1/summary.json").is_file():
        retry = compare_case(read_case(after / "pages_2_run_1"), read_case(retry_path))
    states = [row["status"] for arm in controls.values() for row in arm.values()] + [
        interruption["status"],
        retry["status"],
    ]
    return {
        "status": "FAIL" if "FAIL" in states else "NOTRUN" if "NOTRUN" in states else "PASS",
        "scope": (
            "Read-only analysis of actual GPU controls; CPU ledger replays reported separately"
        ),
        "controls": controls,
        "interruption": interruption,
        "retry_exact_comparison": retry,
        "OOM": "NOTRUN: neither physical nor CPU OOM injected",
        "TIMEOUT": "NOTRUN: neither physical nor CPU timeout injected",
    }


def replay_failures(source_arm: Path, fixtures: Path, output: Path) -> dict[str, Any]:
    """Replay captured text through the real ledger; never execute native inference."""
    if os.name == "nt" and not str(output).startswith("\\\\?\\"):
        output = Path("\\\\?\\" + str(output.resolve()))
    fresh_output(output)
    accepted = read_case(source_arm / "pages_2_run_1")
    length = read_case(source_arm / "length_2_pages_8_tokens")
    raw = accepted["native"]["text"]
    names = [
        "accepted",
        "interrupted_retry",
        "length",
        "malformed",
        "missing_page",
        "not_run",
        "rename_failure",
    ]
    rows = []
    for name in names:
        paths = []
        for page in (1, 2):
            relative = f"images/{name}/page_{page}.png"
            path = output / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(fixtures / f"page_{page}.png", path)
            paths.append(relative)
        rows.append({"page_info": {"image_path": name + ".pdf", "images_list": paths}})
    write_json(output / "ground-truth-scope.json", rows)
    manifest = inventory(
        output / "ground-truth-scope.json",
        output,
        "G1-public-synthetic-ledger-replay",
        "9cf077c4b206e0e2a31879df937b0b0bcba3661c",
        7,
        14,
    )
    run = Run.create(
        output / "runs",
        manifest,
        {"scope": "CPU_LEDGER_REPLAY_OF_CAPTURED_OUTPUTS_NOT_NEW_GPU_INFERENCE"},
        {
            "source_arm": str(source_arm),
            "accepted_summary_sha256": sha256(source_arm / "pages_2_run_1/summary.json"),
            "length_summary_sha256": sha256(source_arm / "length_2_pages_8_tokens/summary.json"),
        },
    )

    def metadata(summary: dict[str, Any], provenance: str) -> dict[str, Any]:
        native = summary["native"]
        return {
            "finish_reason": native["finish_reason"],
            "output_tokens": len(native["token_ids"]),
            "max_tokens": summary["case"]["max_new_tokens"],
            "memory": summary["memory_after"],
            "captured_native_result": native,
            "provenance": provenance,
            "token_ids_scope": (
                "Original capture IDs retained; derivative/authored raw is not "
                "claimed to have generated these IDs"
            ),
        }

    labels = {
        "accepted": "CAPTURED_NATIVE_GPU_OUTPUT_CPU_LEDGER_REPLAY",
        "length": "CAPTURED_NATIVE_GPU_LENGTH_OUTPUT_CPU_LEDGER_REPLAY",
        "missing_page": "SYNTHETIC_REPLAY_NOT_GPU_CORRUPTION: DERIVATIVE",
        "malformed": "SYNTHETIC_REPLAY_NOT_GPU_CORRUPTION: AUTHORED",
        "interrupted_retry": "SYNTHETIC_CPU_PENDING_ATTEMPT_THEN_CAPTURED_OUTPUT_RETRY",
        "rename_failure": "SYNTHETIC_CPU_PATH_RENAME_FAILURE_WITH_CAPTURED_OUTPUT",
        "not_run": "INTENTIONALLY_NOT_RUN_FULL_DENOMINATOR_CONTROL",
    }

    def finish(name: str, text: str, summary: dict[str, Any] = accepted) -> str:
        attempt = run.begin(name)
        run.finish(name, attempt, text, "PASS", metadata(summary, labels[name]))
        return attempt

    finish("accepted", raw)
    finish("length", length["native"]["text"], length)
    # Keep only the first actual content page; never call this GPU corruption.
    missing = "<PAGE>" + raw.split("<PAGE>")[1]
    finish("missing_page", missing)
    malformed = "<PAGE>AUTHORED malformed page without canonical blocks" * 2
    finish("malformed", malformed)
    pending = run.begin("interrupted_retry")
    retry = finish("interrupted_retry", raw)
    committed = finish("rename_failure", raw)
    committed_path = run.path / "documents/rename_failure" / committed
    old_raw_hash, old_commit_hash = (
        sha256(committed_path / "raw.txt"),
        sha256(committed_path / "COMMIT.json"),
    )
    renamed_attempt = run.begin("rename_failure")
    original_rename = Path.rename

    def fail_rename(path: Path, target: Path) -> Path:
        if path.name == renamed_attempt + ".pending" and path.parent.name == "rename_failure":
            raise OSError("CONTROLLED_CPU_PATH_RENAME_FAILURE")
        return original_rename(path, target)

    rename_error = None
    with patch.object(Path, "rename", fail_rename):
        try:
            run.finish(
                "rename_failure",
                renamed_attempt,
                raw,
                "PASS",
                metadata(accepted, labels["rename_failure"]),
            )
        except OSError as error:
            rename_error = str(error)
    pending_path = run.path / "documents/rename_failure" / (renamed_attempt + ".pending")
    pending_seal = read_json(pending_path / "COMMIT.json")
    accounting = run.summary()
    histories = {name: run.attempts(name) for name in names}
    run.export(output / "predictions")
    checks = {
        "accepted_raw_exact": (run.path / "documents/accepted/000001/raw.txt").read_bytes()
        == raw.encode(),
        "length_raw_exact": (run.path / "documents/length/000001/raw.txt").read_bytes()
        == length["native"]["text"].encode(),
        "length_is_TRUNCATED": histories["length"][0]["state"] == "TRUNCATED",
        "missing_page_is_FAIL": histories["missing_page"][0]["state"] == "FAIL",
        "malformed_is_FAIL": histories["malformed"][0]["state"] == "FAIL",
        "pending_retained": (
            run.path / "documents/interrupted_retry" / (pending + ".pending/started.json")
        ).is_file(),
        "retry_appended": retry != pending and len(histories["interrupted_retry"]) == 2,
        "rename_injected": rename_error == "CONTROLLED_CPU_PATH_RENAME_FAILURE",
        "rename_prior_commit_unchanged": (
            old_raw_hash == sha256(committed_path / "raw.txt")
            and old_commit_hash == sha256(committed_path / "COMMIT.json")
        ),
        "rename_new_pending_raw_preserved": (
            (pending_path / "raw.txt").is_file()
            and sha256(pending_path / "raw.txt") == old_raw_hash
            and (pending_path / "COMMIT.json").is_file()
        ),
        "rename_new_pending_hash_seal_valid": all(
            sha256(pending_path / name) == expected for name, expected in pending_seal.items()
        ),
        "full_document_denominator": accounting["expected_documents"] == 7,
        "full_page_denominator": accounting["expected_pages"] == 14,
        "full_attempt_denominator": sum(len(history) for history in histories.values()) == 8,
        "NOT_RUN_retained": accounting["counts"]["NOT_RUN"] == 1,
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": "Real production Run ledger CPU replay; no new GPU or native algorithm execution",
        "labels": labels,
        "checks": checks,
        "total_attempts": sum(map(len, histories.values())),
        "accounting": accounting,
        "rename_error": rename_error,
        "source_artifacts": {
            "normal_native_sha256": sha256(
                source_arm / "pages_2_run_1/diagnostics/native-001.json"
            ),
            "length_native_sha256": sha256(
                source_arm / "length_2_pages_8_tokens/diagnostics/native-001.json"
            ),
        },
        "ledger_path": str(run.path),
        "captured_memory": memory_report(accepted),
    }
    write_json(output / "failure-contract.json", result)
    hash_manifest(output)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, default=Path(".b/g1/before"))
    parser.add_argument("--after", type=Path, default=Path(".b/g1/after"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-output", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, default=Path("tests/fixtures"))
    args = parser.parse_args(argv)
    if Path.cwd().resolve() != ROOT:
        raise ValueError("Run from primary repository root")
    for path in (args.output, args.replay_output):
        if not path.resolve().is_relative_to(ROOT):
            raise ValueError("All generated analysis/replay evidence must stay inside root")
    fresh_output(args.output)
    analysis = analyze_arms(args.before, args.after)
    write_json(args.output / "analysis.json", analysis)
    failure = replay_failures(args.before, args.fixtures, args.replay_output)
    failure["actual_GPU_failure_controls"] = analysis["physical_failure_contracts"]
    write_json(args.output / "failure-contract.json", failure)
    hash_manifest(args.output)
    print(
        json.dumps(
            {
                "analysis": analysis["status"],
                "ledger_replay": failure["status"],
                "case_counts": analysis["case_counts"],
            }
        ),
        flush=True,
    )
    return 0 if analysis["status"] == failure["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
