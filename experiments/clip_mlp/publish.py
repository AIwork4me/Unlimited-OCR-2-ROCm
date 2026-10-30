"""Sanitized numerical and graph publication; private tensor files never copied."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from analyze import (
    difference,
    graph_without_locations,
    require_reduction,
    require_same_inputs,
    tensor_hash,
)
from replay import ROOT, sha, write


def read(path):
    return json.loads(Path(path).read_text("utf-8"))


def public_review(value, key=""):
    if isinstance(value, dict):
        return {k: public_review(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [public_review(v, key) for v in value]
    if isinstance(value, str) and "graph" in key:
        return graph_without_locations(value)
    return value


def process_ok(folder):
    row = read(folder / "finished.json")
    if (
        row["exit_code"] != 0
        or row["accounting"]["active_processes"] != 0
        or (folder / "worker/failure.json").exists()
    ):
        raise ValueError("Unsuccessful owned process")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch

    local = ROOT / ".b/mlp"
    capture = local / "capture-1"
    supervisor = read(local / "capture-supervisor-1/finished.json")
    if (
        supervisor["exit_code"] != 0
        or supervisor["accounting"]["active_processes"] != 0
        or not (capture / "worker/complete.json").exists()
    ):
        raise ValueError("Capture incomplete")
    expected_raw = [
        "e6408c4a652f7a738abdb14299041a346136528098eca7337443e4a00210e49d",
        "c08e7f728ea1e46dc20c2958bf6d3ac857902542935e257a0b64525b7d7b2a8e",
    ]
    if [
        read(capture / f"worker/observation-{i}.json")["raw_sha256"] for i in (0, 1)
    ] != expected_raw:
        raise ValueError("Capture perturbed prior OCR outputs")
    rows = read(capture / "captures.json")
    stages = {}
    private = []
    for stage in (
        "pre_norm",
        "mlp_input",
        "activation_input",
        "activation_output",
        "fc2_input",
        "mlp_output",
    ):
        a, b = [torch.load(capture / f"call-{i:02d}/{stage}.pt", weights_only=True) for i in (1, 3)]
        metas = [next(r for r in rows if r["stage"] == stage and r["call"] == i) for i in (1, 3)]
        for i, meta in zip((1, 3), metas, strict=True):
            if sha(capture / f"call-{i:02d}/{stage}.pt") != meta["file_sha256"]:
                raise ValueError("Captured artifact altered")
        metrics = difference(a, b)
        stages[stage] = dict(
            metadata=[
                {k: v for k, v in m.items() if k not in ("file_sha256", "cpu_stride")}
                for m in metas
            ],
            comparison=metrics,
        )
        first = metrics["first_differing_flat_index"]
        if first is not None:
            private.append(
                dict(
                    stage=stage,
                    index=first,
                    left_value=float(a.flatten()[first]),
                    right_value=float(b.flatten()[first]),
                )
            )
    for stage in ("pre_norm", "mlp_input", "activation_input"):
        require_same_inputs(*stages[stage]["metadata"])
    write(args.output / "boundary.json", stages)
    private_path = local / "first-differing-values.json"
    if not private_path.exists():
        write(private_path, private)
    write(
        args.output / "capture-graphs.json",
        {
            str(i): graph_without_locations(read(capture / f"call-{i:02d}/graph.json")["graph"])
            for i in (1, 3)
        },
    )
    records = []
    groups = (
        "replay-1",
        "narrow-1",
        "kernel-log-1",
        "prefix1-1",
        "prefix2-1",
        "reviewer-1",
        "reviewer-prefix2",
    )
    expected_counts = {
        "replay-1": 8,
        "narrow-1": 12,
        "kernel-log-1": 2,
        "prefix1-1": 4,
        "prefix2-1": 4,
        "reviewer-1": 6,
        "reviewer-prefix2": 2,
    }
    for group in groups:
        group_count = 0
        for folder in sorted((local / group).iterdir()):
            if not folder.is_dir():
                continue
            process_ok(folder)
            group_count += 1
            identity = read(folder / "worker/identity.json")
            result = read(folder / "worker/result.json")
            setup = read(folder / "worker/setup.json")
            require_same_inputs(
                stages["activation_input"]["metadata"][0], setup["input"], result["input_after"]
            )
            if (
                result["hip_before"]["status"] != result["hip_after"]["status"]
                or result["hip_after"]["status"] != "PASS"
            ):
                raise ValueError("HIP state failed")
            if (
                identity["wheel_sha256"]
                != "415d2dbd4b6ab002cff1c54795fdd66322abd9a46bf982cb64eebb4493ac2e7c"
            ):
                raise ValueError("Wrong wheel")
            if setup["model_loaded"] or setup["weights_loaded"]:
                raise ValueError("Replay loaded a model or weights")
            calls = [
                dict(
                    index=v["index"],
                    role=v["role"],
                    output=v["output"],
                    graph=graph_without_locations(v["graph"]),
                    operators=v["operators"],
                )
                for v in result["calls"]
            ]
            records.append(
                dict(
                    group=group,
                    condition=folder.name,
                    worker_pid=identity["pid"],
                    producer_sha256=identity["producer_sha256"],
                    torch_git_version=identity["torch_git_version"],
                    wheel_sha256=identity["wheel_sha256"],
                    device=identity["device"],
                    loaded_native_hashes={
                        Path(m["path"]).name: m["sha256"] for m in identity["loaded"]
                    },
                    input=setup["input"],
                    source_sha256=setup["source_sha256"],
                    schema=setup["schema"],
                    prefix=setup.get("original_operator_prefix", 3),
                    model_loaded=setup["model_loaded"],
                    weights_loaded=setup["weights_loaded"],
                    control=setup["control"],
                    position=setup["position"],
                    calls=calls,
                    cpu_fp32_reference=result["cpu_fp32_reference"],
                    kernel_visibility=result["kernel_visibility"],
                    hip_zero=True,
                    active_owned_processes=0,
                )
            )
        if group_count != expected_counts[group]:
            raise ValueError("Incomplete predeclared process matrix")
    write(args.output / "replay-matrix.json", records)
    cold = torch.load(local / "replay-1/normal-cold-0/worker/output-0.pt", weights_only=True)
    warm = torch.load(local / "replay-1/normal-warm-0/worker/output-1.pt", weights_only=True)
    x = torch.load(capture / "call-01/activation_input.pt", weights_only=True).float()
    references = {}
    for alpha in (1.702, 1.703125):
        scaled = x * alpha
        for staged in (False, True):
            sigmoid = torch.sigmoid(scaled.bfloat16().float() if staged else scaled)
            predicted = (x * (sigmoid.bfloat16().float() if staged else sigmoid)).bfloat16()
            references[f"alpha={alpha};intermediate_bf16={staged}"] = dict(
                cold=difference(cold, predicted),
                warm=difference(warm, predicted),
                prediction_sha256=tensor_hash(predicted),
            )
    write(args.output / "reference-factorial.json", references)
    reduced = []
    for group in ("multiply-1", "reviewer-multiply"):
        group_count = 0
        for folder in sorted((local / group).iterdir()):
            if not folder.is_dir():
                continue
            process_ok(folder)
            group_count += 1
            result = read(folder / "worker/result.json")
            require_same_inputs(result["input"], stages["activation_input"]["metadata"][0])
            require_reduction(result)
            kernel = result.pop("kernel_code")
            reduced.append(
                dict(
                    group=group,
                    condition=folder.name,
                    **{k: v for k, v in result.items() if k not in ("hip_before", "hip_after")},
                    kernel_body=kernel[kernel.index('extern "C"') :],
                    kernel_source_sha256=__import__("hashlib").sha256(kernel.encode()).hexdigest(),
                )
            )
            reduced[-1]["graph"] = graph_without_locations(reduced[-1]["graph"])
        if group_count != 2:
            raise ValueError("Exactly two independent reductions required per group")
    if len(reduced) != 4:
        raise ValueError("Incomplete reduction matrix")
    write(args.output / "single-multiply.json", reduced)
    kernel_log = (local / "kernel-log-1/normal-warm-0/stderr.log").read_text("utf-8")
    code = "\n".join(
        line.split("] ", 1)[-1]
        for line in kernel_log.splitlines()
        if line.startswith("[DEBUG cuda_codegen")
    )
    code = code[code.index('extern "C"') :]
    write(
        args.output / "generated-kernel.json",
        dict(
            name="fused_mul_sigmoid_mul",
            body=code,
            provenance="Normal warm compile log; CPU profiler only; GPU timeline unavailable",
            log_sha256=sha(local / "kernel-log-1/normal-warm-0/stderr.log"),
        ),
    )
    write(args.output / "source-provenance.json", read(local / "source-provenance.json"))
    write(
        args.output / "producer-hashes.json",
        {p.relative_to(ROOT).as_posix(): sha(p) for p in Path(__file__).parent.glob("*.py")},
    )
    write(
        args.output / "independent-review.json",
        dict(
            audit_10067ee=dict(
                candidate_requests=48,
                processes=24,
                cpu_passed=298,
                cpu_skipped=7,
                historical_hashes=406,
                findings=[],
            ),
            operator_reproduction=read(local / "reviewer-summary.json"),
            minimal_natural_prefix=public_review(read(local / "reviewer-prefix2-summary.json")),
        ),
    )
    hashes = {}
    for directory, dirs, names in os.walk(
        "\\\\?\\" + str(local) if os.name == "nt" else str(local)
    ):
        dirs[:] = [d for d in dirs if d not in {"tmp", "__pycache__", "kernel-cache", "extensions"}]
        for name in names:
            path = Path(directory) / name
            if path.suffix in (".json", ".log", ".py", ".cpp", ".h", ".pt"):
                logical = Path(str(path).removeprefix("\\\\?\\"))
                hashes[logical.relative_to(ROOT).as_posix()] = sha(path)
    write(args.output / "local-artifact-hashes.json", hashes)
    write(
        args.output / "status.json",
        dict(
            decision="KEEP EXPERIMENTAL",
            new_full_ocr_requests=2,
            replay_processes=len(records),
            single_multiply_processes=len(reduced),
            input_identity="EXACT",
            historical_receipts="UNCHANGED",
            production_changes=False,
            measured_P2_requests=0,
        ),
    )
    for path in args.output.iterdir():
        if re.search(
            r"Users|USERPROFILE|C:|left_value|right_value|token_ids|prompt_token",
            path.read_text("utf-8"),
        ):
            raise ValueError("Publication privacy gate rejected private payload or path")
    write(args.output / "SEAL.json", {p.name: sha(p) for p in args.output.iterdir() if p.is_file()})


if __name__ == "__main__":
    main()
