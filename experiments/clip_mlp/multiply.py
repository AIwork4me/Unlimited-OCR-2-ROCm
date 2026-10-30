"""Reduce the exact original ScriptFunction graph to its first scalar multiply."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from replay import ROOT, identity, metadata, sha, tensor_hash, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-gpu", action="store_true", required=True)
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT or not args.output.resolve().is_relative_to(ROOT / ".b/mlp"):
        raise ValueError("Primary root/local fresh output required")
    args.output.mkdir(parents=True, exist_ok=False)
    import torch

    helper, hip = identity(args.output)
    capture = ROOT / ".b/mlp/capture-1"
    expected = next(
        r
        for r in json.loads((capture / "captures.json").read_text("utf-8"))
        if r["call"] == 1 and r["stage"] == "activation_input"
    )
    cpu = torch.load(capture / "call-01/activation_input.pt", weights_only=True)
    x = torch.empty_strided(cpu.shape, cpu.stride(), dtype=cpu.dtype, device="cuda")
    x.copy_(cpu)
    if metadata(x) != {
        k: expected[k] for k in ("shape", "stride", "dtype", "device", "storage_offset", "sha256")
    }:
        raise ValueError("Fixed input differs")
    lock = json.loads((ROOT / "src/unlimited_ocr_2_rocm/upstream-lock.json").read_text("utf-8-sig"))
    source = (
        ROOT
        / ".b/h/models--baidu--Unlimited-OCR-2/snapshots"
        / lock["model_revision"]
        / "deepencoder.py"
    )
    if sha(source) != lock["files"]["deepencoder.py"]:
        raise ValueError("Source mismatch")
    spec = importlib.util.spec_from_file_location("clip_mlp_scalar_original", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    graph = module.quick_gelu.graph.copy()
    node = next(n for n in graph.nodes() if n.kind() == "aten::mul")
    original_schema = str(node.schema())
    graph.eraseOutput(0)
    graph.registerOutput(node.output())
    torch._C._jit_pass_dce(graph)
    graph.inputs().__next__().setType(torch._C.TensorType.create_from_tensor(x))
    node.output().setType(torch._C.TensorType.create_from_tensor(x))
    if len([n for n in graph.nodes() if n.kind().startswith("aten::")]) != 1:
        raise ValueError("Expected exactly the original first multiply")
    reference_function = torch._C._create_function_from_graph(
        "isolated_original_multiply", graph.copy()
    )
    before = helper.hip_probe(hip)
    with torch.inference_mode():
        with torch.jit.optimized_execution(False):
            reference = reference_function(x)
        # Prevent a silently interpreted fallback during the direct codegen test.
        torch._C._jit_texpr_set_fallback_allowed(False)
        kernel = torch._C._te.TensorExprKernel(graph.copy())
        code = kernel.get_code_text()
        if "__global__" not in code:
            raise ValueError("No GPU kernel generated")
        actual = kernel.run((x,))
        torch.cuda.synchronize()
    if actual.device != x.device or actual.dtype != x.dtype:
        raise ValueError("GPU BF16 identity changed")
    a, b = reference.cpu(), actual.cpu()
    delta = (a.float() - b.float()).abs()
    indices = (a != b).flatten().nonzero().flatten()
    torch.save(a, args.output / "reference.pt")
    torch.save(b, args.output / "tensor-expr.pt")
    expected_original = (cpu.float() * 1.702).bfloat16()
    expected_codegen = (cpu.float() * 1.703125).bfloat16()
    write(
        args.output / "result.json",
        dict(
            producer_sha256=sha(__file__),
            input=metadata(x),
            operator_schema=original_schema,
            graph=str(graph),
            kernel_code=code,
            reference=metadata(reference),
            tensor_expr=metadata(actual),
            different_elements=indices.numel(),
            max_abs=float(delta.max()),
            first_differing_flat_index=int(indices[0]) if len(indices) else None,
            cpu_original_exact=torch.equal(a, expected_original),
            cpu_emitted_coefficient_exact=torch.equal(b, expected_codegen),
            hip_before=before,
            hip_after=helper.hip_probe(hip),
            fallback_allowed=torch._C._jit_texpr_fallback_allowed(),
        ),
    )
    if (
        not torch.equal(a, expected_original)
        or not torch.equal(b, expected_codegen)
        or tensor_hash(x) != expected["sha256"]
    ):
        raise ValueError("Numerical/input gates failed")


if __name__ == "__main__":
    main()
