"""Independent scalar-correction prototypes, not a custom wheel or production patch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/clip_mlp"))
from replay import identity, metadata, sha, write  # noqa: E402

SOURCE = """
def scalar_only(x):
    y = x.float()
    return (y * torch.sigmoid(1.702 * y)).to(torch.bfloat16)
def explicit_boundaries(x):
    scaled = (1.702 * x.float()).to(torch.bfloat16)
    activated = torch.sigmoid(scaled.float()).to(torch.bfloat16)
    return (x.float() * activated.float()).to(torch.bfloat16)
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--declaration", type=Path, required=True)
    p.add_argument("--execute-gpu", action="store_true", required=True)
    args = p.parse_args()
    if Path.cwd().resolve() != ROOT or not args.output.resolve().is_relative_to(ROOT / ".b/te-e2e"):
        raise ValueError("Local fresh output required")
    args.output.mkdir(parents=True, exist_ok=False)
    declaration = json.loads(args.declaration.read_text("utf-8"))
    for path, expected in declaration["hashes"].items():
        if sha(ROOT / path) != expected:
            raise ValueError("Scalar experiment predeclared identity changed")
    import torch

    helper, hip = identity(args.output)
    tensor = ROOT / ".b/mlp/capture-1/call-01/activation_input.pt"
    cpu = torch.load(tensor, weights_only=True)
    x = torch.empty_strided(cpu.shape, cpu.stride(), dtype=cpu.dtype, device="cuda")
    x.copy_(cpu)
    captured = json.loads((ROOT / ".b/mlp/capture-1/captures.json").read_text("utf-8"))
    expected = next(c for c in captured if c["call"] == 1 and c["stage"] == "activation_input")
    if metadata(x) != {k: expected[k] for k in metadata(x)}:
        raise ValueError("Original fixed tensor contract changed")
    module = torch.jit.CompilationUnit(SOURCE)
    rows = []
    with torch.inference_mode():
        eager = x * torch.sigmoid(1.702 * x)
        cpu_scaled = (1.702 * cpu.float()).bfloat16()
        cpu_activated = torch.sigmoid(cpu_scaled.float()).bfloat16()
        cpu_reference = (cpu.float() * cpu_activated.float()).bfloat16()
        cpu_scalar_only = (cpu.float() * torch.sigmoid(1.702 * cpu.float())).bfloat16()
        for name in ("scalar_only", "explicit_boundaries"):
            function = getattr(module, name)
            for ordinal in (0, 1):
                before = helper.hip_probe(hip)
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU]
                ) as profiler:
                    result = function(x)
                    torch.cuda.synchronize()
                graph = str(torch.jit.last_executed_optimized_graph())
                difference = (result.float() - eager.float()).abs()
                torch.save(result.cpu(), args.output / f"{name}-{ordinal}.pt")
                rows.append(
                    dict(
                        name=name,
                        ordinal=ordinal,
                        input=metadata(x),
                        output=metadata(result),
                        graph=graph,
                        operators=[
                            dict(name=e.key, count=e.count) for e in profiler.key_averages()
                        ],
                        different_elements=int((result != eager).sum()),
                        max_abs=float(difference.max()),
                        first_differing_flat_index=(
                            int((result != eager).flatten().nonzero()[0])
                            if bool((result != eager).any())
                            else None
                        ),
                        cpu_staged_exact=torch.equal(result.cpu(), cpu_reference),
                        cpu_single_float_expression_exact=torch.equal(
                            result.cpu(), cpu_scalar_only
                        ),
                        hip_before=before,
                        hip_after=helper.hip_probe(hip),
                    )
                )
    write(
        args.output / "result.json",
        dict(
            producer_sha256=sha(__file__),
            source=SOURCE,
            input_artifact_sha256=sha(tensor),
            source_declaration=(
                "External BF16 unchanged; internal float casts explicit; "
                "separate experimental functions"
            ),
            eager=metadata(eager),
            declaration_sha256=sha(args.declaration),
            gpu_eager_cpu_staged_exact=torch.equal(eager.cpu(), cpu_reference),
            cpu_staged=metadata(cpu_reference),
            cpu_single_float=metadata(cpu_scalar_only),
            rows=rows,
        ),
    )
    if any(
        r["hip_before"]["status"] != r["hip_after"]["status"] or r["hip_after"]["status"] != "PASS"
        for r in rows
    ):
        raise ValueError("HIP failed")


if __name__ == "__main__":
    main()
