"""One isolated Windows gfx1151 SDPA experiment; requires torch and stdlib only."""

import argparse
import contextlib
import ctypes
import hashlib
import importlib.metadata
import os
import platform
import subprocess
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

from reporting import (
    classify,
    hip_failed,
    numerical_stats,
    observed_backend,
    reference_attention,
    sha256,
    write_json,
)

# Frozen before the first acceptance matrix, not adjusted to measured outcomes.
TOLERANCES = {"bfloat16": (0.03, 0.03), "float16": (0.005, 0.005), "float32": (1e-4, 1e-4)}


def make_fixture(shape, dtype, mask, seed, noncontiguous):
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    dt = getattr(torch, dtype)
    tensors = [torch.randn(shape, generator=generator).to(dt) for _ in range(3)]
    bias_shape = [*shape[:2], shape[2], shape[2]]
    bias = None
    if mask == "bias":
        bias = (torch.randn(bias_shape, generator=generator) * 0.1).to(dt)
    elif mask == "bool":
        bias = torch.rand(bias_shape, generator=generator) > 0.25
        bias[..., 0] = True  # avoid an entirely masked row in the randomized matrix
    tensors.append(bias)
    digest = hashlib.sha256()
    for tensor in tensors:
        if tensor is not None:
            digest.update(str((list(tensor.shape), str(tensor.dtype))).encode())
            digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
        else:
            digest.update(b"None")
    if noncontiguous:
        # Preserve values and last-dimension stride=1; change outer strides only.
        tensors[:3] = [t.transpose(1, 2).contiguous().transpose(1, 2) for t in tensors[:3]]
    return tensors, digest.hexdigest()


def execute(args, report):
    import torch
    import torch.nn.functional as functional
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.utils._python_dispatch import TorchDispatchMode

    torch.set_num_threads(4)
    props = torch.cuda.get_device_properties(0)
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    report["environment"] = {
        "os": platform.platform(),
        "windows_build": platform.version(),
        "python": sys.version,
        "machine": platform.machine(),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "torch_git": torch.version.git_version,
        "gpu": props.name,
        "architecture": props.gcnArchName,
        "packages": packages,
        "runtime_overrides": {
            k: os.environ[k]
            for k in (
                "AMD_SERIALIZE_KERNEL",
                "HIP_LAUNCH_BLOCKING",
                "HIP_VISIBLE_DEVICES",
                "HSA_OVERRIDE_GFX_VERSION",
                "AOTRITON_DEFAULT_TARGET_GPU",
                "AOTRITON_TORCH_ONLY_USE_FALLBACK",
                "AOTRITON_FORCE_DEFAULT_STREAM",
            )
            if k in os.environ
        },
    }
    expected = {
        "torch": "2.12.0+rocm7.14.1",
        "torchvision": "0.27.0+rocm7.14.1",
        "torchaudio": "2.11.0+rocm7.14.1",
    }
    if not (
        sys.platform == "win32"
        and sys.version_info[:3] == (3, 10, 20)
        and torch.version.hip == "7.14.60850"
        and props.gcnArchName == "gfx1151"
        and all(packages.get(k) == v for k, v in expected.items())
    ):
        raise RuntimeError("Mandated baseline identity mismatch; experiment refused")
    if "HSA_OVERRIDE_GFX_VERSION" in os.environ:
        raise RuntimeError("Architecture override refused")
    hip_path = Path(sys.prefix) / "Lib/site-packages/_rocm_sdk_core/bin/amdhip64_7.dll"
    hip = ctypes.WinDLL(str(hip_path))
    hip.hipPeekAtLastError.restype = ctypes.c_int
    hip.hipGetLastError.restype = ctypes.c_int
    report["hip_library"] = str(hip_path)
    tensors, fixture_hash = make_fixture(
        args.shape, args.dtype, args.mask, args.seed, args.noncontiguous
    )
    report["fixture_sha256"] = fixture_hash
    reference = reference_attention(*tensors, args.causal, args.scale)
    report["tensors"] = [
        None
        if t is None
        else {
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "cpu_stride": list(t.stride()),
        }
        for t in tensors
    ]
    gpu = [None if t is None else t.to("cuda") for t in tensors]
    report["gpu_strides"] = [None if t is None else list(t.stride()) for t in gpu]
    q, k, v, mask = gpu
    events, operators = report["events"], report["operators"]

    def checkpoint(name, synchronize=True):
        event = {"stage": name, "hip_before_sync": hip.hipPeekAtLastError()}
        if synchronize:
            try:
                torch.cuda.synchronize()
                event["synchronize"] = "OK"
            except Exception:
                event["traceback"] = traceback.format_exc()
        event["hip_after_sync"] = hip.hipPeekAtLastError()
        events.append(event)
        print(event, flush=True)

    class Trace(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            operators.append(str(func))
            return func(*args, **(kwargs or {}))

    checkpoint("before_sdpa")
    if hip_failed(events):
        raise RuntimeError("GPU not clean before experiment")
    backend = getattr(SDPBackend, args.backend) if args.backend != "DEFAULT" else None
    params = torch.backends.cuda.SDPAParams(q, k, v, mask, 0.0, args.causal, False)
    report["capabilities"] = {
        "flash": torch.backends.cuda.can_use_flash_attention(params),
        "efficient": torch.backends.cuda.can_use_efficient_attention(params),
        "cudnn": torch.backends.cuda.can_use_cudnn_attention(params),
    }
    trace = Trace() if args.trace == "dispatch" else contextlib.nullcontext()
    profiler = None
    if args.trace == "profiler":
        profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU])
        trace = profiler
    output = None
    try:
        if args.operation != "copy_only":
            with sdpa_kernel(backend) if backend is not None else contextlib.nullcontext(), trace:
                output = functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=args.causal, scale=args.scale
                )
    except RuntimeError as exc:
        message = str(exc)
        if backend is not None and (
            "No available kernel" in message or "No viable backend" in message
        ):
            report["status"] = "UNSUPPORTED"
        raise
    finally:
        if profiler is not None:
            operators.extend(e.key for e in profiler.key_averages())
        report["backend_observed"] = observed_backend(operators)
        checkpoint("after_sdpa")
    pending = hip_failed(events)
    if pending and args.clear_error_nonacceptance:
        cleared = hip.hipGetLastError()
        events.append({"stage": "EXPLICIT_NON_ACCEPTANCE_CLEAR", "cleared_error": cleared})
        print("NON-ACCEPTANCE hipGetLastError:", cleared, flush=True)
    if args.operation in ("copy", "copy_only", "arithmetic", "cast"):
        try:
            if args.operation in ("copy", "copy_only"):
                result = q.transpose(-1, -2).contiguous()
            elif args.operation == "cast":
                result = q.to(torch.float8_e4m3fn)
            else:
                result = q + 1
            report["downstream_shape"] = list(result.shape)
        except Exception:
            report["downstream_traceback"] = traceback.format_exc()
            print(report["downstream_traceback"], file=sys.stderr, flush=True)
        checkpoint("after_" + args.operation)
    pending = hip_failed(events)
    if output is not None and (not pending or args.clear_error_nonacceptance):
        actual = output.cpu()
        report["numerical"] = numerical_stats(actual, reference, *TOLERANCES[args.dtype])
    if args.operation == "copy_only" and not pending:
        if not torch.equal(result.cpu(), tensors[0].transpose(-1, -2)):
            raise RuntimeError("Copy does not match CPU input")
        report["copy_correct"] = True
        report["status"] = "PASS"
    else:
        report["status"] = classify(report.get("numerical"), int(pending))
    if report.get("downstream_traceback") or any(e.get("traceback") for e in events):
        report["status"] = "FAIL"
    report["prohibited_imports_loaded"] = [
        name
        for name in ("triton", "transformers", "huggingface_hub", "sglang")
        if name in sys.modules
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", nargs=4, type=int, default=[54, 12, 196, 64])
    parser.add_argument("--dtype", choices=list(TOLERANCES), default="bfloat16")
    parser.add_argument("--mask", choices=["none", "bias", "bool"], default="bias")
    parser.add_argument(
        "--backend",
        choices=["DEFAULT", "MATH", "FLASH_ATTENTION", "EFFICIENT_ATTENTION", "CUDNN_ATTENTION"],
        default="DEFAULT",
    )
    parser.add_argument(
        "--operation", choices=["sdpa", "copy_only", "copy", "arithmetic", "cast"], default="copy"
    )
    parser.add_argument("--trace", choices=["dispatch", "profiler", "none"], default="dispatch")
    parser.add_argument("--noncontiguous", action="store_true")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--scale", type=float)
    parser.add_argument("--seed", type=int, default=7992)
    parser.add_argument("--clear-error-nonacceptance", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    report = {
        "run_id": str(uuid.uuid4()),
        "utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "script_sha256": sha256(__file__),
        "reporting_sha256": sha256(Path(__file__).with_name("reporting.py")),
        "arguments": vars(args) | {"output": str(args.output)},
        "command": sys.argv,
        "acceptance": not args.clear_error_nonacceptance,
        "backend_requested": args.backend,
        "backend_observed": "UNOBSERVED",
        "events": [],
        "operators": [],
        "status": "INCOMPLETE",
        "tolerances": dict(zip(("atol", "rtol"), TOLERANCES[args.dtype], strict=True)),
    }
    try:
        execute(args, report)
    except Exception:
        report["traceback"] = traceback.format_exc()
        print(report["traceback"], file=sys.stderr, flush=True)
        if report["status"] != "UNSUPPORTED":
            report["status"] = "FAIL"
    code = {"PASS": 0, "UNSUPPORTED": 2}.get(report["status"], 1)
    report["exit_code"] = code
    write_json(args.output, report)
    print(
        report["status"],
        "requested:",
        args.backend,
        "observed:",
        report["backend_observed"],
        "run:",
        report["run_id"],
        "numerics:",
        report.get("numerical"),
        flush=True,
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
