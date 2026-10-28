"""Strict environment checks. Importing this module never imports torch."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STACK = {
    "torch": "2.12.0+rocm7.14.1",
    "torchvision": "0.27.0+rocm7.14.1",
    "torchaudio": "2.11.0+rocm7.14.1",
}


def environment_identity() -> str:
    return hashlib.sha256(str(Path(sys.prefix).resolve()).casefold().encode()).hexdigest()


def require_stack(report: dict[str, Any]) -> None:
    """Reject version drift, alternate hardware and CPU execution."""
    errors = []
    if report.get("platform") != "Windows":
        errors.append("Phase 1 requires native Windows")
    hardware = report.get("windows_hardware", {})
    os_info = hardware.get("os", {})
    build = str(os_info.get("BuildNumber", ""))
    if (
        "Windows 11" not in str(os_info.get("Caption", ""))
        or not build.isdigit()
        or int(build) < 22000
    ):
        errors.append("Windows 11 client build evidence required")
    cpu = str(hardware.get("cpu", {}).get("Name", ""))
    if re.search(r"AMD.*RYZEN\s+AI\s+MAX\+?\s+(?:PRO\s+)?395\b", cpu, re.IGNORECASE) is None:
        errors.append("Ryzen AI MAX+ 395 / confirmed PRO 395 required")
    if not str(report.get("python", "")).startswith("3.10."):
        errors.append("Phase 1 requires the verified CPython 3.10 environment")
    if str(report.get("machine", "")).lower() not in {"amd64", "x86_64"}:
        errors.append("64-bit x86 CPython required")
    for name, version in STACK.items():
        if report.get("packages", {}).get(name) != version:
            errors.append(f"{name} must be {version}")
    if report.get("rocm") != "7.14.1" or report.get("hip") != "7.14.60850":
        errors.append("ROCm 7.14.1 and its HIP build 7.14.60850 required")
    if report.get("cuda_build") is not None:
        errors.append("NVIDIA CUDA build refused")
    if not report.get("gpu_available"):
        errors.append("GPU unavailable; CPU fallback refused")
    gpu = report.get("gpu", {})
    if str(gpu.get("arch", "")).split(":")[0] != "gfx1151":
        errors.append("physical gfx1151 required")
    if "8060S" not in str(gpu.get("name", "")):
        errors.append("Radeon 8060S required")
    if errors:
        raise RuntimeError("; ".join(errors))


def sanitize(text: str) -> str:
    """Redact local home paths, including slash variants in tracebacks."""
    home = str(Path.home())
    for prefix in {home, home.replace("\\", "/"), home.replace("\\", "\\\\")}:
        text = re.sub(re.escape(prefix), "<USERPROFILE>", text, flags=re.IGNORECASE)
    return text


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(sanitize(json.dumps(value, ensure_ascii=False, indent=2)), encoding="utf-8")


def memory_snapshot() -> dict[str, Any]:
    import psutil

    vm = psutil.virtual_memory()
    process_memory = psutil.Process().memory_info()
    result: dict[str, Any] = {
        "process_rss_bytes": process_memory.rss,
        "process_peak_rss_bytes": getattr(process_memory, "peak_wset", process_memory.rss),
        "system_total_bytes": vm.total,
        "system_available_bytes": vm.available,
        "system_memory_percent": vm.percent,
    }
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        result.update(
            gpu_allocated_bytes=torch.cuda.memory_allocated(),
            gpu_reserved_bytes=torch.cuda.memory_reserved(),
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            gpu_free_total_bytes=list(torch.cuda.mem_get_info()),
        )
    return result


def collect_environment() -> dict[str, Any]:
    result: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.system(),
        "windows_version": platform.version(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "executable": sys.executable,
        "environment_id": environment_identity(),
        "isolated_environment": sys.prefix != sys.base_prefix,
        "installed_packages": {
            distribution.metadata["Name"].lower().replace("_", "-"): distribution.version
            for distribution in importlib.metadata.distributions()
            if distribution.metadata["Name"]
        },
        "packages": {},
        "environment": {
            key: value
            for key, value in os.environ.items()
            if key.startswith(("DASH_", "UOCR_", "TRITON_", "HSA_", "ROCR_", "PYTORCH_"))
            or key in {"HIP_PATH", "CC", "CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE"}
        },
    }
    for name in STACK:
        result["packages"][name] = importlib.metadata.version(name)
        module = importlib.import_module(name)
        if module.__version__ != STACK[name]:
            raise RuntimeError(f"{name} imported version differs from required distribution")
    torch = importlib.import_module("torch")
    result.update(
        hip=torch.version.hip,
        rocm=getattr(torch.version, "rocm", None),
        cuda_build=torch.version.cuda,
        gpu_available=torch.cuda.is_available(),
    )
    if result["gpu_available"]:
        props = torch.cuda.get_device_properties(0)
        result["gpu"] = {
            "name": props.name,
            "arch": getattr(props, "gcnArchName", None),
            "pytorch_visible_budget_bytes": props.total_memory,
            "properties": str(props),
            "memory_note": "APU shared system memory; this is not dedicated physical VRAM",
        }
    if platform.system() == "Windows":
        command = (
            "$OutputEncoding=[Console]::OutputEncoding=[Text.UTF8Encoding]::new(); "
            "@{machine_uuid=(Get-CimInstance Win32_ComputerSystemProduct).UUID;"
            "os=Get-CimInstance Win32_OperatingSystem | "
            "Select-Object Caption,Version,BuildNumber,TotalVisibleMemorySize,FreePhysicalMemory;"
            "cpu=Get-CimInstance Win32_Processor | Select-Object Name;"
            "gpu=Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion} "
            "| ConvertTo-Json -Depth 5 -Compress"
        )
        proc = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            encoding="utf-8",
            check=True,
        )
        hardware = json.loads(proc.stdout.lstrip("\ufeff"))
        machine_uuid = hardware.pop("machine_uuid", "")
        hardware["machine_uuid_sha256"] = (
            hashlib.sha256(machine_uuid.strip().lower().encode()).hexdigest()
            if machine_uuid
            else None
        )
        result["windows_hardware"] = hardware
    result["memory"] = memory_snapshot()
    return result


def bf16_probe() -> dict[str, Any]:
    """Execute integer-valued BF16 matmul and compare a separate FP32 reference."""
    import torch

    before = memory_snapshot()
    x_cpu = (torch.arange(1024).reshape(32, 32) % 7 - 3).float()
    y_cpu = (torch.arange(1024).reshape(32, 32) % 5 - 2).float()
    reference = x_cpu @ y_cpu
    x = x_cpu.to(device="cuda:0", dtype=torch.bfloat16)
    y = y_cpu.to(device="cuda:0", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        output = x @ y
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if output.device.type != "cuda" or output.dtype != torch.bfloat16:
        raise RuntimeError("BF16 operation changed device/dtype")
    # BF16 GEMM produced 4.77e-7 cancellation residue for exact zero on this
    # stack. An absolute 1e-5 bound covers FP32 accumulation roundoff; it is
    # far below the 0.5 spacing of BF16 values near the largest result here.
    torch.testing.assert_close(output.float().cpu(), reference, rtol=0, atol=1e-5)
    return {
        "status": "PASS",
        "operation": "32x32 BF16 GPU matmul versus independent CPU FP32 matmul",
        "device": str(output.device),
        "dtype": str(output.dtype),
        "max_abs_error": float((output.float().cpu() - reference).abs().max()),
        "absolute_tolerance": 1e-5,
        "synchronized_seconds": elapsed,
        "memory_before": before,
        "memory_after": memory_snapshot(),
    }
