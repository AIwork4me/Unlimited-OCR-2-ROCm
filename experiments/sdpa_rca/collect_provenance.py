"""Read-only Windows package, loaded-module and AOTriton image provenance."""

import argparse
import ctypes
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from reporting import sha256, write_json


def parse_header(text):
    return dict(re.findall(r"^#define\s+(AOTRITON_(?:VERSION_\w+|GIT_SHA1))\s+(\S+)", text, re.M))


def loaded_modules():
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.K32EnumProcessModules.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_void_p,
    ]
    kernel.K32GetModuleFileNameExW.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_wchar_p,
        ctypes.c_ulong,
    ]
    process = kernel.GetCurrentProcess()
    modules = (ctypes.c_void_p * 4096)()
    needed = ctypes.c_ulong()
    if not kernel.K32EnumProcessModules(
        process, modules, ctypes.sizeof(modules), ctypes.byref(needed)
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    paths = []
    for module in modules[: needed.value // ctypes.sizeof(ctypes.c_void_p)]:
        buffer = ctypes.create_unicode_buffer(32768)
        if kernel.K32GetModuleFileNameExW(process, module, buffer, len(buffer)):
            paths.append(Path(buffer.value))
    return paths


def inventory(path):
    return {"path": str(path), "size": path.stat().st_size, "sha256": sha256(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--install-report", type=Path)
    args = parser.parse_args()
    import torch

    props = torch.cuda.get_device_properties(0)
    base = Path(torch.__file__).parent
    header = base / "include/aotriton/config.h"
    signatures = list((base / "lib/aotriton.images").rglob("__signature__"))
    image_files = list((base / "lib/aotriton.images").rglob("*"))
    important = ("torch_hip", "aotriton", "amdhip", "hsa-runtime", "dl.dll", "liblzma")
    modules = [p for p in loaded_modules() if any(s in p.name.lower() for s in important)]
    system = subprocess.check_output(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "@{os=Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber; "
            "cpu=Get-CimInstance Win32_Processor | Select-Object Name; "
            "gpu=Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,DriverDate} "
            "| ConvertTo-Json -Depth 5",
        ],
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    report = {
        "run_id": str(uuid.uuid4()),
        "utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "script_sha256": sha256(__file__),
        "python": sys.version,
        "os": platform.platform(),
        "system": json.loads(system),
        "gpu": props.name,
        "architecture": props.gcnArchName,
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "torch_commit": torch.version.git_version,
        "torch_config": torch.__config__.show(),
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "runtime_header": {
            "text": header.read_text(),
            "parsed": parse_header(header.read_text()),
            **inventory(header),
        },
        "image_signatures": [
            {"signature": json.loads(p.read_text()), **inventory(p)} for p in signatures
        ],
        "loaded_modules": [inventory(p) for p in modules],
        "image_files": [inventory(p) for p in sorted(image_files) if p.is_file()],
        "image_archive_samples": [],
        "source_claim_limit": "Headers identify packaged build metadata; DLL hash and imports "
        "must corroborate actual runtime. No DLL source SHA export assumed.",
    }
    for archive in sorted((base / "lib/aotriton.images").rglob("attn_fwd.zip")):
        with zipfile.ZipFile(archive) as file:
            report["image_archive_samples"].append(
                {
                    "path": str(archive),
                    "entries": len(file.namelist()),
                    "first_names": file.namelist()[:3],
                }
            )
    if args.install_report:
        install = json.loads(args.install_report.read_text(encoding="utf-8-sig"))
        report["wheel_origins"] = [
            {
                "name": entry["metadata"]["name"],
                "version": entry["metadata"]["version"],
                "download_info": entry["download_info"],
            }
            for entry in install["install"]
        ]
    write_json(args.output, report)
    print("Provenance:", report["run_id"], "modules:", len(modules), flush=True)


if __name__ == "__main__":
    main()
