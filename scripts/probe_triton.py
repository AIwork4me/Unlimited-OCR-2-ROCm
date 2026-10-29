"""Triton-0/1/2: compile and execute on physical gfx1151, fail closed."""

import argparse
import hashlib
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unlimited_ocr_2_rocm.doctor import (  # noqa: E402
    collect_environment,
    memory_snapshot,
    require_stack,
    write_json,
)


def probe():
    import torch
    import triton
    import triton.language as tl

    target = triton.runtime.driver.active.get_current_target()
    expected = "3.5.0" if sys.platform == "linux" else "3.7.1"
    if triton.__version__ != expected or target.backend != "hip" or target.arch != "gfx1151":
        raise RuntimeError(f"Unexpected Triton backend/version: {triton.__version__} / {target}")

    @triton.jit
    def add(x, y, out, n, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        tl.store(
            out + offsets,
            tl.load(x + offsets, mask, other=0) + tl.load(y + offsets, mask, other=0),
            mask,
        )

    n = 4097
    x = torch.arange(n, device="cuda", dtype=torch.float32) / 4096
    y = torch.ones_like(x) * 0.25
    out = torch.empty_like(x)
    before = memory_snapshot()
    torch.cuda.synchronize()
    start = time.perf_counter()
    kernel = add[(triton.cdiv(n, 256),)](x, y, out, n, BLOCK=256)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if out.device.type != "cuda":
        raise RuntimeError("CPU fallback refused")
    torch.testing.assert_close(out, x + y, rtol=0, atol=0)
    torch.testing.assert_close(out.cpu(), torch.arange(n).float() / 4096 + 0.25, rtol=0, atol=0)
    return {
        "status": "PASS",
        "version": triton.__version__,
        "target": str(target),
        "device": str(out.device),
        "elements": n,
        "max_abs_error": 0.0,
        "compile_and_run_seconds": elapsed,
        "assembly_keys": list(kernel.asm),
        "amdgcn_sha256": hashlib.sha256(kernel.asm.get("amdgcn", "").encode()).hexdigest(),
        "memory_before": before,
        "memory_after": memory_snapshot(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("eval/results/triton.json"))
    args = parser.parse_args()
    report = {}
    try:
        report["environment"] = collect_environment()
        require_stack(report["environment"])
        report["probe"] = probe()
        report["status"] = "PASS"
    except Exception:
        report["status"] = "FAIL"
        report["traceback"] = traceback.format_exc()
    write_json(args.output, report)
    print(f"Triton-0/1/2: {report['status']} — {args.output}", flush=True)
    if report["status"] != "PASS":
        print(report["traceback"], file=sys.stderr)
        sys.exit(1)
