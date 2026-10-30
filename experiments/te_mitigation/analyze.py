"""Strict additive qualification; imports no torch and never changes old receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/cold_start"))
from worker import CANDIDATE, ORIGINAL, digest, load, sha, write  # noqa: E402

bbox = load("te_bbox_comparison", ROOT / "experiments/cold_start/analyze.py")


def read(path):
    return json.loads(Path(path).read_text("utf-8"))


def request_gates(summary, observation, events, profiles, option, mode, pages, shapes=None):
    """Fail closed on physical execution, completion and declared activation scope."""
    shapes = shapes if shapes is not None else [[1, 257, 4096] for _ in range(pages)]
    calls = len(shapes)
    checks = {
        "wrapper": summary["status"] == "PASS",
        "functional": summary["correctness"]["status"] == "PASS",
        "stop": summary["validated_metadata"]["finish_reason"] == "stop",
        "state_restored": summary["runtime_state_restored"] is True,
        "hip": all(summary[k]["status"] == "PASS" for k in ("hip_before", "hip_after")),
        "dash": len(summary["kernels"]) == 12
        and all(
            k["dtype"] == "torch.bfloat16"
            and k["device"].startswith("cuda")
            and k["synchronized"] is True
            for k in summary["kernels"]
        ),
        "guard": bool(summary["guard_branch_counts"]),
        "observation": observation["status"] == "PASS" and observation["mode"] == mode,
        "layers": Counter(e["layer"] for e in events) == Counter(dict.fromkeys(range(24), calls)),
        "layer_calls": Counter((e["layer"], e["call"]) for e in events)
        == Counter((layer, call) for layer in range(24) for call in range(1, calls + 1)),
        "actual_graphs": all(bool(e["graph"]) for e in events) if option != "eager" else True,
        "profile_linkage": len(profiles) == calls * 2
        and len({p["serial"] for p in profiles}) == len(profiles)
        and all(e["profile"] in {p["serial"] for p in profiles} for e in events),
        "activation_contract": all(
            e["dtype"] == "torch.bfloat16"
            and e["device"] == "cuda:0"
            and 1 <= e["call"] <= calls
            and e["shape"] == shapes[e["call"] - 1]
            and e["stride"] == [e["shape"][1] * 4096, 4096, 1]
            and e["graph_optimize"] is True
            and e["te_enabled"] is (option != "te-off")
            for e in events
        ),
    }
    actual = observation["dispatch"]
    checks["attention"] = set(actual) == {"sam_model", "vision_model"} and all(
        "aten::_scaled_dot_product_attention_math" in actual[m]
        if mode == "MATH"
        else (
            "aten::_scaled_dot_product_attention_math" not in actual[m]
            and (
                "aten::_scaled_dot_product_efficient_attention"
                if m == "sam_model"
                else "aten::_scaled_dot_product_flash_attention"
            )
            in actual[m]
        )
        for m in actual
    )
    if mode == "MATH":
        checks["physical_contract"] = summary["physical_contract"] == "PASS"
    # The frozen MATH-only helper labels AUTO physical_contract FAIL. Independent
    # gates above verify the same HIP/DASH/guard/state properties and actual AUTO.
    if option != "default":
        checks["unfused_graph"] = all(
            e["graph"] is None
            if option == "eager"
            else e["graph"] is not None and "TensorExprGroup" not in e["graph"]
            for e in events
        )
        by_profile = Counter(e["profile"] for e in events)
        observed = {p["serial"]: p for p in profiles}
        checks["unfused_dispatch"] = all(
            serial in observed
            and sum(
                op["count"] for op in observed[serial]["operators"] if op["name"] == "aten::sigmoid"
            )
            >= count
            and not any(
                "fused_mul_sigmoid_mul" in op["name"] for op in observed[serial]["operators"]
            )
            for serial, count in by_profile.items()
        )
    return checks


def independent_coverage(processes, schedule):
    expected = Counter((p["option"], p["modes"]) for p in schedule)
    actual = Counter((p["option"], p["modes"]) for p in processes)
    return (
        bool(expected)
        and actual == expected
        and all(n == 2 for n in expected.values())
        and len({(p["pid"], p["created"]) for p in processes}) == len(processes)
        and len({p["pid"] for p in processes}) == len(processes)
    )


def campaign(root, crop_contract=None):
    declaration = read(root / "predeclared.json")
    contract = read(crop_contract) if crop_contract else None
    if contract and (
        contract["declaration_sha256"] != sha(root / "predeclared.json")
        or contract["input_hashes"] != declaration["input_hashes"]
        or contract["producer_sha256"] != sha(ROOT / "experiments/te_mitigation/crop_contract.py")
    ):
        raise ValueError("Crop analysis contract identity mismatch")
    rows, processes, identities = [], [], []
    for item_index, item in enumerate(declaration["schedule"]):
        label = f"{item['option']}-{item['modes'] or 'MATH'}-{item['repeat']}"
        folder = root / label
        data = folder / "data"
        worker = data / "worker"
        finished = read(folder / "finished.json")
        identity = read(worker / "identity.json")
        identities.append(
            {k: identity[k] for k in ("wheel_sha256", "source_hashes", "determinism")}
            | {k: identity["model"][k] for k in ("model_revision", "source_and_weight_hashes")}
        )
        scope = read(data / "scope.json")
        observations = read(data / "observers.json")
        complete = read(worker / "complete.json")
        expected_wheel = CANDIDATE if declaration["backend"] else ORIGINAL
        process_checks = {
            "exit_zero": finished["exit_code"] == 0,
            "no_descendants": finished["accounting"]["active_processes"] == 0,
            "no_failure": not (data / "failure.json").exists(),
            "complete": complete == {"status": "COMPLETED", "requests": len(declaration["cases"])},
            "wheel": identity["wheel_sha256"] == expected_wheel,
            "cases": observations["cases"] == [c["name"] for c in declaration["cases"]],
            "producer": observations["producer_sha256"] == declaration["worker_sha256"],
            "scope": scope["option"] == item["option"] and scope["layers"] == 24,
            "restored": observations["te_before"] is observations["te_after"] is True
            and observations["graph_optimize_after"] is True,
            "single_model": observations["model_initializations"] == 1,
            "owned_pid": identity["pid"] == finished["pid"] or identity["ppid"] == finished["pid"],
        }
        if contract:
            process_checks["crop_source"] = (
                contract["source_sha256"]
                == identity["model"]["source_and_weight_hashes"]["modeling_unlimitedocr.py"]
            )
        if declaration.get("protocol_version", 1) >= 3:
            process_checks["declaration_binding"] = (
                observations["declaration_sha256"] == sha(root / "predeclared.json")
                and observations["declared_item"] == item_index
            )
        processes.append(
            dict(
                label=label,
                checks=process_checks,
                pid=identity["pid"],
                created=identity["created"],
                option=item["option"],
                modes=item["modes"],
            )
        )
        modes = item["modes"].split(",") if item["modes"] else ["MATH"] * len(declaration["cases"])
        for index, case in enumerate(declaration["cases"]):
            summary = read(worker / f"request-{index}/summary.json")
            observation = read(worker / f"observation-{index}.json")
            events = [e for e in observations["q_events"] if e["request"] == index]
            profiles = [p for p in observations["profiles"] if p["request"] == index]
            gates = request_gates(
                summary,
                observation,
                events,
                profiles,
                item["option"],
                modes[index],
                len(case["truth_keys"]),
                contract["cases"][case["name"]] if contract else None,
            )
            gates["case_identity"] = summary["case"] == case
            gates["generation_options"] = summary["effective_options"] == dict(
                crop_mode=None if len(case["truth_keys"]) == 1 else False,
                eval_mode=True,
                max_length=131072,
                max_new_tokens=case["max_new_tokens"],
                temperature=0.0,
                page_count_guard=True,
                max_page_tokens=20000,
                dash_topm=16,
                return_dict=True,
            )
            stages = [s for s in observations["stages"] if s["request"] == index]
            if declaration["tensors"]:
                gates["stage_alignment"] = Counter(
                    (s["call"], s["stage"]) for s in stages
                ) == Counter(
                    (call, stage)
                    for call in range(1, len(case["truth_keys"]) + 1)
                    for stage in (
                        "mlp-input",
                        "activation-input",
                        "activation-output",
                        "mlp-output",
                    )
                )
                gates["tensor_artifacts"] = all(
                    sha(data / s["path"]) == s["file_sha256"] for s in stages
                )
            native = summary["native"]
            geometry = bbox.compare(native, native)
            gates["geometry_valid"] = not geometry["invalid_boxes"]
            gates["raw_integrity"] = (
                sha(worker / f"request-{index}/diagnostics/native-001.raw.txt")
                == hashlib.sha256(native["text"].encode()).hexdigest()
                == summary["raw_sha256"]
                == observation["raw_sha256"]
            )
            guard = [
                json.loads(line)
                for line in (worker / f"request-{index}/diagnostics/guard.jsonl")
                .read_text("utf-8")
                .splitlines()
            ]
            gates["guard_contract"] = bool(guard) and all(
                g["expected_pages"] == len(case["truth_keys"])
                and g["max_page_tokens"] == summary["effective_options"]["max_page_tokens"]
                for g in guard
            )
            gates["mutable_state"] = (
                all(
                    observation["before"][k] == observation["after"][k]
                    for k in ("parameters", "buffers")
                )
                and observation["before"]["training"] is False
            )
            rows.append(
                dict(
                    label=label,
                    option=item["option"],
                    repeat=item["repeat"],
                    ordinal=index,
                    mode=modes[index],
                    case=case["name"],
                    case_equivalence=[case["truth_keys"], case["max_new_tokens"]],
                    gates=gates,
                    raw_sha256=summary["raw_sha256"],
                    output_ids_sha256=digest(native["token_ids"]),
                    prompt_ids_sha256=digest(native["prompt_token_ids"]),
                    prompt_sha256=hashlib.sha256(native["prompt"].encode()).hexdigest(),
                    finish_reason=native["finish_reason"],
                    page_markers=native["text"].count("<PAGE>"),
                    block_markers=native["text"].count("<BLOCK>"),
                    invalid_boxes=geometry["invalid_boxes"],
                    boxes=geometry["boxes"],
                    functional=summary["correctness"],
                    latency_seconds=summary["synchronized_wall_seconds"],
                    memory=summary["memory_after"],
                    dispatch=observation["dispatch"],
                    effective_options=summary["effective_options"],
                    graph_sha256=sorted(
                        {
                            hashlib.sha256(e["graph"].encode()).hexdigest()
                            for e in events
                            if e["graph"]
                        }
                    ),
                    fused_events=sum("TensorExprGroup" in (e["graph"] or "") for e in events),
                    stages=[{k: v for k, v in s.items() if k != "path"} for s in stages],
                    native=native,
                )
            )
    pairs = []
    for option in {r["option"] for r in rows}:
        selected = [r for r in rows if r["option"] == option]
        for right in selected[1:]:
            left = next(
                (r for r in selected if r["case_equivalence"] == right["case_equivalence"]), None
            )
            if left is not None and left is not right:
                pairs.append(
                    dict(
                        option=option,
                        left=[left["label"], left["ordinal"]],
                        right=[right["label"], right["ordinal"]],
                        comparison=bbox.compare(left["native"], right["native"]),
                        stage_hashes_exact=[
                            (s["call"], s["stage"], s["sha256"]) for s in left["stages"]
                        ]
                        == [(s["call"], s["stage"], s["sha256"]) for s in right["stages"]],
                    )
                )
    cross = []
    for case in {r["case"] for r in rows}:
        same = [r for r in rows if r["case"] == case]
        for right in same[1:]:
            if right["option"] != same[0]["option"]:
                cross.append(
                    dict(
                        case=case,
                        left=[same[0]["label"], same[0]["ordinal"]],
                        right=[right["label"], right["ordinal"]],
                        comparison=bbox.compare(same[0]["native"], right["native"]),
                    )
                )
    for row in rows:
        del row["native"]
        # Published summaries retain metrics/hashes, never synthetic payloads.
        row["functional"] = {k: v for k, v in row["functional"].items() if k != "pages"} | {
            "pages": [
                {
                    k: v
                    for k, v in page.items()
                    if k not in {"ground_truth", "normalized_truth", "normalized_prediction"}
                }
                for page in row["functional"]["pages"]
            ]
        }
    coverage = independent_coverage(processes, declaration["schedule"])
    same_identity = bool(identities) and all(i == identities[0] for i in identities)
    physical = (
        coverage
        and same_identity
        and all(all(p["checks"].values()) for p in processes)
        and all(all(r["gates"].values()) for r in rows)
    )
    return dict(
        producer_sha256=sha(__file__),
        declaration_sha256=sha(root / "predeclared.json"),
        crop_contract_sha256=sha(crop_contract) if crop_contract else None,
        processes=processes,
        requests=rows,
        repeat_pairs=pairs,
        cross_options=cross,
        all_physical_and_functional_gates=physical,
        independent_process_coverage=coverage,
        matched_model_source_wheel_determinism=same_identity,
        option_stage_repeatability={
            option: coverage
            and any(p["option"] == option for p in pairs)
            and all(p["stage_hashes_exact"] for p in pairs if p["option"] == option)
            for option in {r["option"] for r in rows}
        }
        if declaration["tensors"]
        else None,
        option_exact_repeatability={
            option: coverage
            and any(p["option"] == option for p in pairs)
            and all(
                p["comparison"]["acceptance"] == "EXACT_PASS"
                for p in pairs
                if p["option"] == option
            )
            for option in {r["option"] for r in rows}
        },
        performance_acceptance="UNQUALIFIED: instrumented descriptive measurements only",
        recommendation="KEEP EXPERIMENTAL",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crop-contract", type=Path)
    args = parser.parse_args()
    write(args.output, campaign(args.root, args.crop_contract))


if __name__ == "__main__":
    main()
