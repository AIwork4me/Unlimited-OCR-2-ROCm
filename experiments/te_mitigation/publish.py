"""Allowlist-only publication of hashes, graphs and metrics, never raw OCR/tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/cold_start"))
from worker import sha, write  # noqa: E402


def read(path):
    return json.loads(path.read_text("utf-8"))


def clean_graph(graph):
    return "\n".join(line.split(" #", 1)[0] for line in graph.splitlines()) + "\n"


def pick(value, keys):
    return {k: value[k] for k in keys.split() if k in value}


def validate_public_text(text):
    if (
        re.search(r"(?i)[a-z]:[/\\]|/(?:users|home)/", text)
        or re.search(
            r'"(?:text|prompt|raw_output|token_ids|prompt_token_ids|native_generation)"\s*:', text
        )
        or any(
            token.lower() in text.lower()
            for token in ("API_KEY", "normalized_prediction", '"ground_truth":')
        )
    ):
        raise ValueError("Private payload or path in publication")


def comparison(value):
    return pick(
        value,
        "raw_exact output_ids_exact prompt_ids_exact raw_hashes token_hashes "
        "first_differing_token alignment_content_labels_reading_order_exact "
        "exact_nonbbox_bytes page_marker_counts signed_delta_distribution acceptance",
    ) | {
        "boxes": [
            pick(
                b,
                "index page label left right signed_deltas iou valid_left valid_right "
                "boundary_touch",
            )
            for b in value["boxes"]
        ],
        "invalid_boxes": [pick(b, "arm index bbox") for b in value["invalid_boxes"]],
    }


def qualification_public(value):
    result = pick(
        value,
        "producer_sha256 declaration_sha256 crop_contract_sha256 all_physical_and_functional_gates "
        "independent_process_coverage matched_model_source_wheel_determinism "
        "option_stage_repeatability option_exact_repeatability performance_acceptance "
        "recommendation",
    )
    result["processes"] = [
        pick(p, "label checks pid created option modes") for p in value["processes"]
    ]
    result["requests"] = []
    for row in value["requests"]:
        selected = pick(
            row,
            "label option repeat ordinal mode case case_equivalence gates raw_sha256 "
            "output_ids_sha256 prompt_ids_sha256 prompt_sha256 finish_reason page_markers "
            "block_markers latency_seconds dispatch graph_sha256 fused_events",
        )
        quality = row["functional"]
        selected["memory"] = pick(
            row["memory"],
            "process_rss_bytes process_peak_rss_bytes "
            "system_total_bytes system_available_bytes system_memory_percent "
            "gpu_allocated_bytes gpu_reserved_bytes gpu_peak_allocated_bytes "
            "gpu_peak_reserved_bytes gpu_free_total_bytes",
        )
        selected["effective_options"] = pick(
            row["effective_options"],
            "crop_mode eval_mode max_length max_new_tokens temperature page_count_guard "
            "max_page_tokens dash_topm return_dict",
        )
        selected["functional"] = pick(
            quality,
            "status threshold_max_cer input_page_count raw_page_markers "
            "content_page_count terminal_empty_marker raw_marker_count_matches_input "
            "page_count_match no_substantive_preamble ordered_page_content_match",
        ) | {
            "pages": [
                pick(p, "page character_error_rate important_numbers_match structure_valid status")
                for p in quality["pages"]
            ]
        }
        selected["geometry"] = comparison(row)
        selected["stages"] = [
            pick(s, "request call stage shape stride dtype device sha256 file_sha256")
            for s in row["stages"]
        ]
        result["requests"].append(selected)
    for category in ("repeat_pairs", "cross_options"):
        result[category] = [
            pick(pair, "option case left right stage_hashes_exact")
            | {"comparison": comparison(pair["comparison"])}
            for pair in value[category]
        ]
    return result


def public_identity(identity):
    model = identity["model"]
    env = model["environment"]
    return dict(
        wheel_sha256=identity["wheel_sha256"],
        source_commit=identity["source_commit"],
        source_hashes=identity["source_hashes"],
        installed_member_count=len(identity["installed_members"]),
        installed_member_manifest_sha256=hashlib.sha256(
            json.dumps(identity["installed_members"], sort_keys=True).encode()
        ).hexdigest(),
        loaded_native_hashes=sorted(
            [
                dict(name=Path(m["path"]).name, sha256=m["sha256"], bytes=m["bytes"])
                for m in identity["loaded_modules"]
            ],
            key=lambda m: (m["name"], m["sha256"]),
        ),
        model={
            k: model[k]
            for k in (
                "model_revision",
                "code_revision",
                "source_and_weight_hashes",
                "source_patch",
                "parameter_device",
                "parameter_dtype",
            )
        },
        runtime={
            k: env[k] for k in ("platform", "windows_version", "python", "packages", "hip", "rocm")
        },
        device={k: env["gpu"][k] for k in ("name", "arch", "memory_note")},
        determinism=identity["determinism"],
        cache_policy=identity["cache_policy"],
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--campaigns", nargs="+", required=True)
    p.add_argument("--mission-supplements", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    graphs, execution, artifacts, identity_hashes = {}, [], {}, {}
    for name in args.campaigns:
        root = ROOT / ".b/te-e2e" / name
        declaration = read(root / "predeclared.json")
        qualification = qualification_public(read(root / "qualification.json"))
        write(args.output / (name + "-qualification.json"), qualification)
        write(
            args.output / (name + "-declaration.json"),
            pick(
                declaration,
                "protocol_version cases source_hashes input_hashes schedule suite backend tensors "
                "worker_sha256 controller_sha256 protocol_sha256 frozen_helper_sha256 "
                "wheel_sha256 model_revision model_manifest_sha256 admission_receipts",
            ),
        )
        if (root / "stage-comparison.json").exists():
            stage_data = read(root / "stage-comparison.json")
            write(
                args.output / (name + "-stages.json"),
                dict(
                    producer_sha256=stage_data["producer_sha256"],
                    rows=[
                        pick(
                            s,
                            "process call stage contracts_exact input_hashes shape stride "
                            "dtype device "
                            "different_elements max_abs mean_abs first_flat_index",
                        )
                        for s in stage_data["rows"]
                    ],
                ),
            )
        for folder in sorted(root.glob("*/data")):
            identity = read(folder / "worker/identity.json")
            sanitized_identity = public_identity(identity)
            identity_digest = hashlib.sha256(
                json.dumps(sanitized_identity, sort_keys=True).encode()
            ).hexdigest()
            if identity_digest not in identity_hashes:
                filename = "identity-" + identity_digest[:12] + ".json"
                write(args.output / filename, sanitized_identity)
                identity_hashes[identity_digest] = filename
            observer = read(folder / "observers.json")
            aggregates = {}
            for event in observer["q_events"]:
                graph = clean_graph(event["graph"]) if event["graph"] else None
                graph_id = hashlib.sha256(graph.encode()).hexdigest() if graph else None
                if graph:
                    graphs[graph_id] = graph
                key = (
                    event["request"],
                    event["call"],
                    event["profile"],
                    graph_id,
                    event["te_enabled"],
                    event["graph_optimize"],
                )
                aggregates.setdefault(key, []).append(event["layer"])
            execution.append(
                dict(
                    campaign=name,
                    process=folder.parent.name,
                    identity_sha256=identity_digest,
                    identity_file=identity_hashes[identity_digest],
                    scope=pick(
                        read(folder / "scope.json"),
                        "option source_sha256 script_schema "
                        "original_python_ast_sha256 layers graph_optimize te_enabled scope",
                    ),
                    invocations=[
                        dict(
                            request=k[0],
                            call=k[1],
                            profile=k[2],
                            graph_sha256=k[3],
                            te_enabled=k[4],
                            graph_optimize=k[5],
                            layers=v,
                        )
                        for k, v in aggregates.items()
                    ],
                    operator_counts=[
                        dict(
                            request=profile["request"],
                            serial=profile["serial"],
                            operators=[
                                op
                                for op in profile["operators"]
                                if any(
                                    token in op["name"]
                                    for token in (
                                        "sigmoid",
                                        "fused_",
                                        "aten::mul",
                                        "addmm",
                                        "scaled_dot_product",
                                    )
                                )
                            ],
                        )
                        for profile in observer["profiles"]
                    ],
                )
            )
        for path in root.rglob("*"):
            if path.is_file() and "tmp" not in path.relative_to(root).parts:
                artifacts[path.relative_to(ROOT).as_posix()] = dict(
                    sha256=sha(path), bytes=path.stat().st_size
                )
    if args.mission_supplements:
        suite = ROOT / ".b/te-e2e/suite-1"
        write(
            args.output / "suite-initial-checker-rejection.json",
            qualification_public(read(suite / "qualification-initial.json")),
        )
        write(
            args.output / "suite-crop-contract.json",
            pick(
                read(suite / "crop-contract-final.json"),
                "producer_sha256 declaration_sha256 source_sha256 input_hashes cases basis scope",
            ),
        )
        write(
            args.output / "scalar-prototypes.json",
            read(ROOT / ".b/te-e2e/scalar-2/public-result-final.json"),
        )
        write(
            args.output / "backend-matrix.json",
            read(ROOT / ".b/te-e2e/backend-1/matrix-qualified.json"),
        )
        write(
            args.output / "backend-numerics.json", read(ROOT / ".b/te-e2e/backend-1/numerics.json")
        )
        cpu = read(ROOT / ".b/te-e2e/cpu-final3/result.json")
        for check in cpu["checks"]:
            check["command"] = [s.replace(str(ROOT), "<REPOSITORY>") for s in check["command"]]
        write(args.output / "cpu-checks.json", cpu)
        for name in (
            "producers",
            "scalar-1",
            "scalar-2",
            "p0-1",
            "cpu-final",
            "cpu-final2",
            "cpu-final3",
            "publication-before-hardening",
        ):
            for path in (ROOT / ".b/te-e2e" / name).rglob("*"):
                if (
                    path.is_file()
                    and "tmp" not in path.relative_to(ROOT / ".b/te-e2e" / name).parts
                ):
                    artifacts[path.relative_to(ROOT).as_posix()] = dict(
                        sha256=sha(path), bytes=path.stat().st_size
                    )
    write(args.output / "execution.json", dict(graphs=graphs, processes=execution))
    write(
        args.output / "artifact-hashes.json",
        dict(
            files=artifacts,
            identity_files=identity_hashes,
            note="Hash-only private inventory; no raw OCR, weights or tensor values published",
        ),
    )
    # Fail closed against accidentally broadening the publication allowlist.
    for path in args.output.glob("*.json"):
        text = path.read_text("utf-8")
        validate_public_text(text)
    write(
        args.output / "SHA256SUMS.json",
        dict(files={path.name: sha(path) for path in sorted(args.output.glob("*.json"))}),
    )


if __name__ == "__main__":
    main()
