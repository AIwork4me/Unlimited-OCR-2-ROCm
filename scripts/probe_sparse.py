"""Triton-3: reviewed official DASH prefill kernel versus independent math."""

import argparse
import importlib.util
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unlimited_ocr_2_rocm.config import read_lock, verify_files  # noqa: E402
from unlimited_ocr_2_rocm.doctor import (  # noqa: E402
    collect_environment,
    memory_snapshot,
    require_stack,
    write_json,
)


def probe(source: Path):
    import torch

    lock = read_lock()
    hashes = verify_files(source, lock["files"])
    spec = importlib.util.spec_from_file_location(
        "reviewed_sparse_e2e", source / "sparse_gather_e2e_triton.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    torch.manual_seed(395)
    tests = []
    # Exact checkpoint head layout; length > local window exercises remote bias.
    # The odd length also exercises masked partial blocks.
    for length in (160, 173):
        shape = (1, 10, length, 128)
        q, k, v = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
        selection = torch.tensor([0, 1, 1, -1], device="cuda", dtype=torch.int32)
        selection = selection.view(1, 1, 1, 4).expand(1, 1, length, 4).contiguous()
        bias = torch.tensor([0.25, -0.5, -0.5, 0], device="cuda")
        bias = bias.view(1, 1, 4).expand(1, length, 4).contiguous()
        positions = torch.arange(length, device="cuda").unsqueeze(0)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            out = module.triton_gather_attention_e2e(
                q, k, v, selection, bias, positions, None, 32, 128**-0.5, local_window=128
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if out is None or out.device.type != "cuda":
            raise RuntimeError("Required native sparse kernel declined input")
        # Independent token-wise reference: causal local window wins; selected
        # remote blocks 0 and 1 each contribute once with their own prior bias.
        query_pos = torch.arange(length, device="cuda")[:, None]
        key_pos = torch.arange(length, device="cuda")[None, :]
        causal = key_pos <= query_pos
        local = causal & (key_pos > query_pos - 128)
        remote = causal & ~local & (key_pos < 64)
        mask = torch.full((length, length), float("-inf"), device="cuda")
        remote_bias = torch.where(key_pos < 32, 0.25, -0.5).expand(length, -1)
        mask = torch.where(remote, remote_bias, mask)
        mask = torch.where(local, 0.0, mask)
        expected = (
            torch.softmax(q.float() @ k.float().transpose(-1, -2) * (128**-0.5) + mask, dim=-1)
            @ v.float()
        )
        error = (out.float() - expected).abs()
        # Fixed before first kernel execution; BF16 pre-scaled Q and probability
        # products differ from a FP32 reference. Reject material mismatches.
        torch.testing.assert_close(out.float(), expected, rtol=0.03, atol=0.03)
        tests.append(
            {
                "query_shape": list(shape),
                "status": "PASS",
                "device": str(out.device),
                "max_abs_error": error.max().item(),
                "mean_abs_error": error.mean().item(),
                "rtol": 0.03,
                "atol": 0.03,
                "compile_and_run_seconds": elapsed,
            }
        )
    return {
        "status": "PASS",
        "kernel": "triton_gather_attention_e2e",
        "source_hashes": hashes,
        "cases": tests,
        "memory_after": memory_snapshot(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("eval/results/sparse.json"))
    args = parser.parse_args()
    report = {}
    try:
        report["environment"] = collect_environment()
        require_stack(report["environment"])
        report["probe"] = probe(args.source)
        report["status"] = "PASS"
    except Exception:
        report["status"] = "FAIL"
        report["traceback"] = traceback.format_exc()
    write_json(args.output, report)
    print(f"Triton-3: {report['status']} — {args.output}", flush=True)
    if report["status"] != "PASS":
        print(report["traceback"], file=sys.stderr)
        sys.exit(1)
