"""Fault-only child: publish synchronized checkpoint without changing decoder results."""

import runpy
import sys

from worker import ROOT, write

sys.path.insert(0, str(ROOT / "src"))
from unlimited_ocr_2_rocm import model  # noqa: E402

original_load = model.load_model


def fault_load(*args, **kwargs):
    native = original_load(*args, **kwargs)
    original = native.model.model.forward
    count = 0

    def forward(*a, **kw):
        nonlocal count
        result = original(*a, **kw)
        count += 1
        if count == 3:
            import ctypes

            import torch

            torch.cuda.synchronize()
            output = __import__("pathlib").Path(sys.argv[sys.argv.index("--output") + 1])
            hip = ctypes.WinDLL(
                str(
                    __import__("pathlib").Path(sys.prefix)
                    / "Lib/site-packages/_rocm_sdk_core/bin/amdhip64_7.dll"
                )
            )
            hip.hipPeekAtLastError.restype = ctypes.c_int
            write(
                output / "kill-marker.json",
                dict(
                    completed_decoder_forwards=3,
                    synchronized=True,
                    hip_zero=hip.hipPeekAtLastError() == 0,
                ),
            )
        return result

    native.model.model.forward = forward
    return native


model.load_model = fault_load
runpy.run_path(str(ROOT / "experiments/cold_start/worker.py"), run_name="__main__")
