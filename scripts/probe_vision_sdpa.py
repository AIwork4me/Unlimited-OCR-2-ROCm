"""Regression for the exact-stack Windows SDPA error; default mode may fail."""

import argparse
import ctypes
import importlib.util
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unlimited_ocr_2_rocm.compat import wrap_vision_forward  # noqa: E402
from unlimited_ocr_2_rocm.config import read_lock, verify_files  # noqa: E402
from unlimited_ocr_2_rocm.doctor import collect_environment, require_stack, write_json  # noqa: E402


def probe(source, mode):
    import torch

    verify_files(source, read_lock()["files"])
    spec = importlib.util.spec_from_file_location(
        "reviewed_vision_encoder", source / "deepencoder.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if sys.platform == "linux":
        candidates = sorted(
            (Path(sys.prefix) / "lib/python3.10/site-packages/torch/lib").glob("libamdhip64.so*")
        )
        if not candidates:
            raise RuntimeError("bundled libamdhip64 not found under torch/lib")
        hip = ctypes.CDLL(str(candidates[0]))
    else:
        hip = ctypes.WinDLL(
            str(Path(sys.prefix) / "Lib/site-packages/_rocm_sdk_core/bin/amdhip64_7.dll")
        )
    hip.hipPeekAtLastError.restype = ctypes.c_int
    block = (
        module.Block(
            dim=768,
            num_heads=12,
            mlp_ratio=4,
            use_rel_pos=True,
            input_size=(64, 64),
            window_size=14,
        )
        .eval()
        .to(device="cuda", dtype=torch.bfloat16)
    )
    events = []
    forward = block.forward
    if mode == "math":
        forward = wrap_vision_forward(forward, "sam_block_regression", events)
    original_flags = (
        torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.mem_efficient_sdp_enabled(),
    )
    with torch.inference_mode():
        x = torch.randn(6, 40, 40, 768, device="cuda", dtype=torch.bfloat16)
        for _ in range(3):
            x = forward(x)
            torch.cuda.synchronize()
            if hip.hipPeekAtLastError() != 0:
                raise RuntimeError("SDPA left a HIP runtime error; refusing to clear or ignore it")
        # Independent attention math at the failing window layout.
        q, k, v = [
            torch.randn(54, 12, 196, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3)
        ]
        bias = torch.randn(54, 12, 196, 196, device="cuda", dtype=torch.bfloat16) * 0.1
        function = torch.nn.functional.scaled_dot_product_attention
        if mode == "math":
            function = wrap_vision_forward(function, "sdpa_reference_regression", events)
        actual = function(q, k, v, attn_mask=bias)
        torch.cuda.synchronize()
        if hip.hipPeekAtLastError() != 0:
            raise RuntimeError("Attention reference call left a HIP error")
        reference = (
            torch.softmax(q.float() @ k.float().transpose(-1, -2) * 0.125 + bias.float(), dim=-1)
            @ v.float()
        )
        torch.testing.assert_close(actual.float(), reference, rtol=0.03, atol=0.03)
        error = float((actual.float() - reference).abs().max())
    if original_flags != (
        torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.mem_efficient_sdp_enabled(),
    ):
        raise RuntimeError("Vision SDPA context leaked into global attention settings")
    return {
        "status": "PASS",
        "mode": mode,
        "block_calls": 3,
        "events": events,
        "max_abs_error": error,
        "rtol": 0.03,
        "atol": 0.03,
        "hip_error_after": 0,
        "attention_flags_restored": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--mode", choices=["default", "math"], default="math")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {}
    try:
        report["environment"] = collect_environment()
        require_stack(report["environment"])
        report["probe"] = probe(args.source, args.mode)
        report["status"] = "PASS"
    except Exception:
        report["status"] = "FAIL"
        report["traceback"] = traceback.format_exc()
    write_json(args.output, report)
    print(f"Vision SDPA {args.mode}: {report['status']}", flush=True)
    sys.exit(0 if report["status"] == "PASS" else 1)
