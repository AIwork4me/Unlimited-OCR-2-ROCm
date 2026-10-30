"""Publish new sanitized metrics, retaining local raw data and historical failures."""
# ruff: noqa: E501

from __future__ import annotations

import argparse
import os
from pathlib import Path

from analyze import compare, read
from worker import ROOT, sha, write


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items() if k != "token_context"}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def process_gate(folder):
    finished = folder / "finished.json"
    complete = folder / "worker/complete.json"
    if not finished.exists() or not complete.exists() or (folder / "worker/failure.json").exists():
        return False
    receipt = read(finished)
    count = len(list(folder.glob("worker/request-*/summary.json")))
    return (
        receipt.get("exit_code") == 0
        and receipt.get("accounting_after_cleanup", {}).get("active_processes") == 0
        and read(complete).get("requests") == count
        and count > 0
    )


def requests(folder):
    if not process_gate(folder):
        raise ValueError("Incomplete, failed, or unclean process cannot enter accepted pairs")
    return [read(p)["native"] for p in sorted(folder.glob("worker/request-*/summary.json"))]


def stages(left, right):
    for i, (a, b) in enumerate(zip(left, right, strict=False)):
        if a["stage"] != b["stage"] or a.get("value") != b.get("value"):
            return dict(
                event_index=i,
                left=a,
                right=b,
                caveat="Earliest OBSERVED boundary only, not proof of a kernel cause",
            )
    return None if len(left) == len(right) else dict(event_count_mismatch=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    local = ROOT / ".b/cold-start-rca"
    candidate_profile = read(ROOT / ".b/p230/candidate-profile.json")
    original_manifest = read(ROOT / ".b/raw/baseline-wheel-manifest.json")
    original_identity = read(local / "A-attempt2/A1/worker/identity.json")
    write(
        args.output / "source-artifact-identities.json",
        dict(
            integration_base=__import__("worker").BASE,
            original=dict(
                wheel_sha256=__import__("worker").ORIGINAL,
                wheel_size=114472348,
                torch_git_version=original_manifest["git_version"],
                manifest_sha256=sha(ROOT / ".b/raw/baseline-wheel-manifest.json"),
            ),
            candidate={
                k: candidate_profile[k]
                for k in (
                    "source_sha",
                    "therock_sha",
                    "aotriton_recipe_pin",
                    "wheel_sha256",
                    "wheel_size",
                    "torch_version",
                    "architecture",
                    "runtime_header_sha256",
                )
            },
            candidate_profile_sha256=sha(ROOT / ".b/p230/candidate-profile.json"),
            model_revision=original_identity["model"]["model_revision"],
            model_source_and_weight_hashes=original_identity["model"]["source_and_weight_hashes"],
            mandatory_versions=original_identity["model"]["environment"]["packages"],
            physical_device={
                k: original_identity["model"]["environment"]["gpu"][k]
                for k in ("name", "arch", "memory_note")
            },
            backend_scope="Joint SAM/CLIP vision backend selection; decoder DASH/SR-SWA unchanged",
        ),
    )
    a = local / "A-attempt2"
    names = ("A1", "A2", "A3", "A4")
    result = {name: requests(a / name) for name in names}
    pairs = {
        "A1_cold_vs_warm": compare(*result["A1"]),
        "A2_cold_vs_warm": compare(*result["A2"]),
        "A1_vs_A2_cold": compare(result["A1"][0], result["A2"][0]),
        "A1_vs_A2_warm": compare(result["A1"][1], result["A2"][1]),
        "A1_vs_A3_first_only": compare(result["A1"][0], result["A3"][0]),
        "A1_vs_A4_explicit_init": compare(result["A1"][0], result["A4"][0]),
    }
    write(args.output / "original-wheel.json", clean(dict(pairs=pairs)))
    historical = read(local / "historical-bbox-analysis.json")
    for case, row in historical["pairs"].items():
        raw_root = ROOT / ".b/p240/ab-attempt1/diagnostic"
        recomputed = compare(
            *(
                read(raw_root / f"diagnostic_{case}_0_{mode}/private-native.json")
                for mode in ("math", "auto")
            )
        )
        if __import__("json").loads(__import__("json").dumps(recomputed)) != row:
            raise ValueError(
                "New strict analysis differs from retained initial descriptive metrics"
            )
    write(args.output / "bbox-correctness.json", clean(historical))
    lines = [
        "# Bbox correctness analysis",
        "",
        "PR #15 exact-parity FAIL is preserved. These are descriptive measurements, with no relaxed acceptance.",
        "",
        "| Case | Changed coordinates | Max absolute delta | Minimum valid IoU | Invalid boxes |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for case, row in historical["pairs"].items():
        distribution = row["signed_delta_distribution"]
        lines.append(
            f"| {case} | {sum(v for k, v in distribution.items() if int(k))} | {max(abs(int(k)) for k in distribution)} | {min(b['iou'] for b in row['boxes']):.9f} | {len(row['invalid_boxes'])} |"
        )
    lines += [
        "",
        "All ten pairs preserve exact non-bbox bytes and canonical page/block/label/content order. Full token equality fails in eight pairs. Per-box labels, signed deltas, coordinates and IoU are retained in bbox-correctness.json. No box is reversed, zero-area or outside 0..999. Boundary contact alone cannot establish clipping against the source image; annotated text geometry is unavailable.",
        "",
        "The pinned model renderer maps coordinates using int(coordinate / 999 * image_axis_pixels). The Markdown converter separately clamps image crop references to 0..1000. A delta of 3 therefore means 3/999 of the corresponding page dimension, before integer rounding. This can move a reconstructed edge, change overlap/containment or a crop boundary. High IoU does not establish text containment or downstream safety.",
        "",
        "No semantic acceptance threshold is proposed from these observations. A new protocol would need independently annotated synthetic geometry, predeclared containment/crop/order criteria and boundary adversarial cases, followed by a new experiment. BF16 operator atol/rtol 0.03/0.03 provides no bbox acceptance rule.",
    ]
    (args.output / "bbox-correctness.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if (local / "ink-geometry.json").exists():
        write(args.output / "synthetic-ink-geometry.json", read(local / "ink-geometry.json"))
    traces = local / "trace-attempt1"
    if traces.exists() and (traces / "T2/worker/complete.json").exists():
        observations = [
            [read(traces / name / f"worker/observation-{i}.json") for i in (0, 1)]
            for name in ("T1", "T2")
        ]
        trace_report = dict(
            cold_warm=[stages(rows[0]["events"], rows[1]["events"]) for rows in observations],
            cold_processes=stages(observations[0][0]["events"], observations[1][0]["events"]),
            observed_vs_A1=[
                compare(result["A1"][i], requests(traces / name)[i])
                for name in ("T1", "T2")
                for i in (0, 1)
            ],
        )
        write(args.output / "stage-localization.json", clean(trace_report))
    for condition in ("deep-attempt1", "jit-attempt2"):
        folder = local / condition
        if not (folder / "T2/worker/complete.json").exists():
            continue
        obs = [
            [read(folder / name / f"worker/observation-{i}.json") for i in (0, 1)]
            for name in ("T1", "T2")
        ]
        details = dict(
            cold_warm=[stages(v[0]["events"], v[1]["events"]) for v in obs],
            cold_processes=stages(obs[0][0]["events"], obs[1][0]["events"]),
            output_comparisons=[compare(*requests(folder / name)) for name in ("T1", "T2")],
        )
        if condition.startswith("deep"):
            first = compare(*requests(folder / "T1"))["first_differing_token"]
            details["matched_prefix_pre_guard_logits"] = [
                {
                    k: v
                    for k, v in [e for e in arm["events"] if e["stage"] == "lm_head"][first].items()
                    if k not in ("top_ids", "value")
                }
                for arm in obs[0]
            ]
        if condition.startswith("jit"):
            normal = read(local / "deep-attempt1/T1/worker/observation-0.json")
            details["normal_cold_vs_unoptimized_cold_first_difference"] = stages(
                normal["events"], obs[0][0]["events"]
            )
        write(args.output / (condition + ".json"), clean(details))
    b = local / "B-attempt1"
    if b.exists():
        matrix = {}
        for case in ("chinese", "pages_2", "mixed"):
            arms = {
                pair: [requests(b / f"{case}-{pair}-{r}") for r in (1, 2)]
                for pair in ("AA", "BB", "AB", "BA")
            }
            if not all(len(rows) == 2 for repeats in arms.values() for rows in repeats):
                raise ValueError("Incomplete backend matrix")
            matrix[case] = dict(
                within_backend={
                    pair: [compare(*rows) for rows in arms[pair]] for pair in ("AA", "BB")
                },
                cold_process_repeatability={
                    pair: compare(arms[pair][0][0], arms[pair][1][0]) for pair in arms
                },
                warm_process_repeatability={
                    pair: compare(arms[pair][0][1], arms[pair][1][1]) for pair in arms
                },
                cold_backend=[compare(arms["AA"][r][0], arms["BB"][r][0]) for r in (0, 1)],
                cold_same_backend_different_future_schedule={
                    "MATH": [compare(arms["AA"][r][0], arms["AB"][r][0]) for r in (0, 1)],
                    "AUTO": [compare(arms["BB"][r][0], arms["BA"][r][0]) for r in (0, 1)],
                },
                warm_backend_same_prior_math=[
                    compare(arms["AA"][r][1], arms["AB"][r][1]) for r in (0, 1)
                ],
                warm_backend_same_prior_auto=[
                    compare(arms["BA"][r][1], arms["BB"][r][1]) for r in (0, 1)
                ],
                prior_backend_math_target=[
                    compare(arms["AA"][r][1], arms["BA"][r][1]) for r in (0, 1)
                ],
                prior_backend_auto_target=[
                    compare(arms["AB"][r][1], arms["BB"][r][1]) for r in (0, 1)
                ],
            )
        write(args.output / "candidate-backend-matrix.json", clean(matrix))
    d = local / "D-attempt1"
    if (d / "recovery/worker/complete.json").exists():
        baseline, recovery = requests(d / "no_kill"), requests(d / "recovery")
        fault = read(d / "fault/finished.json")
        marker = fault.get("fault_checkpoint", {})
        fault_valid = (
            fault.get("exit_code", 0) != 0
            and marker.get("completed_decoder_forwards") == 3
            and marker.get("hip_zero") is True
            and fault.get("native_return_before_kill") is False
        )
        matches = [compare(baseline[i], recovery[i]) for i in (0, 1)]
        golden = read(ROOT / ".b/i3010/physical2/gpu/pages_2_run_1/diagnostics/native-001.json")
        write(
            args.output / "restart-categories.json",
            clean(
                dict(
                    external_process_termination="PASS" if fault_valid else "FAIL",
                    descendant_cleanup="PASS"
                    if fault["accounting_after_cleanup"]["active_processes"] == 0
                    else "FAIL",
                    functional_ocr_recovery="PASS"
                    if all(
                        read(d / "recovery" / f"worker/request-{i}/summary.json")["correctness"][
                            "status"
                        ]
                        == "PASS"
                        for i in (0, 1)
                    )
                    else "FAIL",
                    matched_invocation_determinism=[v["acceptance"] for v in matches],
                    matched_invocations=matches,
                    no_kill_cold_warm=compare(*baseline),
                    post_kill_cold_warm=compare(*recovery),
                    exact_ordered_golden=[compare(golden, v)["acceptance"] for v in recovery],
                    accounting=dict(
                        planned_processes=3,
                        completed_ocr_requests=4,
                        externally_terminated_requests=1,
                        native_partial_return="unavailable",
                        new_production_ledger="See new-strict-ledger.json; strict first-request attempts, warm diagnostic requests receive no credit",
                        historical_campaign="PARTIAL unchanged; FAIL1 / NOT_RUN1",
                    ),
                    conclusion=(
                        "Matched cold and warm outputs reproduce with and without kill; no evidence kill caused bbox difference. No overall P3 qualification."
                        if all(v["acceptance"] == "EXACT_PASS" for v in matches)
                        else "Matched recovery differs; investigate before attributing cause. No P3 qualification."
                    ),
                )
            ),
        )
        ledger_path = local / "D-ledger-attempt1/ledger-summary.json"
        if not ledger_path.exists():
            raise ValueError("Actual new reliability ledger missing")
        ledger = read(ledger_path)
        docs = {v["doc_id"]: v for v in ledger["documents"]}
        if (
            set(docs) != {"no_kill", "post_kill"}
            or ledger["expected_documents"] != 2
            or ledger["expected_pages"] != 4
        ):
            raise ValueError("New reliability ledger scope mismatch")
        if len(docs["no_kill"]["attempts"]) != 1 or len(docs["post_kill"]["attempts"]) != 2:
            raise ValueError("Unlaunched jobs or warm diagnostics incorrectly credited")
        if docs["post_kill"]["attempts"][0].get("reason") != "EXTERNAL_TERMINATION":
            raise ValueError("Fault ledger classification mismatch")
        write(args.output / "new-strict-ledger.json", ledger)
    records = []
    for path in sorted(local.glob("*-attempt*/*/worker/identity.json")):
        ident = read(path)
        folder = path.parent
        for summary_path in sorted(folder.glob("request-*/summary.json")):
            summary = read(summary_path)
            idx = summary_path.parent.name.split("-")[-1]
            observed_path = folder / f"observation-{idx}.json"
            if not observed_path.exists():
                continue
            observed = read(observed_path)
            dispatch = observed["dispatch"]
            mode = observed["mode"]
            actual_ops = {
                name: {op: n for op, n in ops.items() if "_scaled_dot_product" in op}
                for name, ops in dispatch.items()
            }
            dispatch_ok = (
                set(actual_ops) == {"sam_model", "vision_model"}
                and all(actual_ops.values())
                and all(
                    ("math" in op if mode == "MATH" else "math" not in op)
                    for ops in actual_ops.values()
                    for op in ops
                )
            )
            kernel_ok = len(summary["kernels"]) == 12 and all(
                k["dtype"] == "torch.bfloat16"
                and k["device"].startswith("cuda")
                and k["synchronized"]
                for k in summary["kernels"]
            )
            completed = "native" in summary
            gate = (
                process_gate(folder.parent)
                and completed
                and summary["status"] == "PASS"
                and summary["hip_before"]["status"] == summary["hip_after"]["status"] == "PASS"
                and kernel_ok
                and dispatch_ok
                and summary["runtime_state_restored"]
            )
            records.append(
                dict(
                    case=summary_path.relative_to(local).as_posix(),
                    mode=mode,
                    physical_gate="PASS" if gate else "FAIL",
                    process_completion_cleanup_gate="PASS"
                    if process_gate(folder.parent)
                    else "FAIL",
                    completed=completed,
                    actual_dispatch=actual_ops,
                    dash_calls=len(summary["kernels"]),
                    worker_pid=ident["pid"],
                    worker_created=ident["created"],
                    wheel_sha256=ident["wheel_sha256"],
                    producer_sha256=ident["producer_sha256"],
                    installed_manifest_sha256=__import__("worker").digest(
                        ident["installed_members"]
                    ),
                    model_hashes=ident["model"]["source_and_weight_hashes"],
                    input_hashes=[
                        dict(name=Path(v["path"]).name, sha256=v["sha256"])
                        for v in summary.get("wrapper", {}).get("inputs", [])
                    ],
                    prompt_sha256=__import__("hashlib")
                    .sha256(summary.get("native", {}).get("prompt", "").encode())
                    .hexdigest(),
                    prompt_ids_sha256=__import__("worker").digest(
                        summary.get("native", {}).get("prompt_token_ids", [])
                    ),
                    generation_options=summary.get("effective_options"),
                    wrapper_source_hashes=ident["source_hashes"],
                    loaded_module_hashes={
                        Path(v["path"]).name: v["sha256"] for v in ident["loaded_modules"]
                    },
                    raw_sha256=summary.get("raw_sha256"),
                    output_ids_sha256=__import__("worker").digest(summary["native"]["token_ids"])
                    if completed
                    else None,
                    memory_after=summary.get("memory_after"),
                    parameter_versions_unchanged=observed["before"]["parameters"]
                    == observed["after"]["parameters"],
                    buffers_unchanged=observed["before"]["buffers"] == observed["after"]["buffers"],
                )
            )
    # Imported helper's physical_contract is MATH-specific; the independent gate
    # above uses actual backend operators and unchanged native DASH requirements.
    write(args.output / "invocation-index.json", records)
    candidate_records = [r for r in records if r["case"].startswith("B-attempt1/")]
    if b.exists():
        declaration = read(b / "predeclared.json")
        for name, *_ in declaration["schedule"]:
            ident = read(b / name / "worker/identity.json")
            if (
                ident["producer_sha256"] != declaration["worker_sha256"]
                or ident["wheel_sha256"] != __import__("worker").CANDIDATE
            ):
                raise ValueError("Candidate producer/wheel differs from predeclaration")
        if len(candidate_records) != 48 or any(
            r["physical_gate"] != "PASS" for r in candidate_records
        ):
            raise ValueError("Candidate publication requires all 48 qualified physical requests")
        if (
            len(
                {__import__("worker").digest(r["wrapper_source_hashes"]) for r in candidate_records}
            )
            != 1
            or len({__import__("worker").digest(r["model_hashes"]) for r in candidate_records}) != 1
            or len(
                {__import__("worker").digest(r["loaded_module_hashes"]) for r in candidate_records}
            )
            != 1
        ):
            raise ValueError("Candidate source/model/native identities differ between arms")
    publication_status = dict(
        decision="KEEP EXPERIMENTAL",
        candidate_qualified_requests=len(candidate_records),
        historical_PR15="EXACT_PARITY_FAIL_UNCHANGED",
        historical_PR21="STRICT_RESTART_FAIL_PARTIAL_UNCHANGED",
        production_changes=False,
    )
    manifests = {}
    walk_root = "\\\\?\\" + str(local.resolve()) if os.name == "nt" else str(local)
    for directory, dirs, names in os.walk(walk_root):
        dirs[:] = [name for name in dirs if name not in {"tmp", "__pycache__", "build"}]
        for name in sorted(names):
            path = Path(directory) / name
            if path.suffix in (".json", ".py", ".txt", ".log"):
                logical = Path(str(path).removeprefix("\\\\?\\"))
                manifests[logical.relative_to(ROOT).as_posix()] = sha(path)
    write(args.output / "local-artifact-hashes.json", manifests)
    write(
        args.output / "producer-hashes.json",
        {str(p.relative_to(ROOT)): sha(p) for p in (ROOT / "experiments/cold_start").glob("*.py")},
    )
    write(args.output / "publication-status.json", publication_status)
    write(
        args.output / "SEAL.json",
        {p.name: sha(p) for p in sorted(args.output.iterdir()) if p.is_file()},
    )


if __name__ == "__main__":
    main()
