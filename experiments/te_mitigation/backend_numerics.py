"""CPU-only comparison of retained backend tensors; never substitute CPU inference."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/cold_start"))
from worker import load, sha, write  # noqa: E402

analysis = load("te_numerics_analysis", ROOT / "experiments/te_mitigation/analyze.py")
numerics = load("te_numerics_difference", ROOT / "experiments/clip_mlp/analyze.py")


def main():
    import torch

    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    comparisons = []
    for name, left, right, li, ri in (
        ("matched_cold_backend", "MATH,MATH", "AUTO,AUTO", 0, 0),
        ("matched_warm_after_math", "MATH,MATH", "MATH,AUTO", 1, 1),
        ("math_cold_warm", "MATH,MATH", "MATH,MATH", 0, 1),
        ("auto_cold_warm", "AUTO,AUTO", "AUTO,AUTO", 0, 1),
    ):
        dirs = [args.root / f"eager-{arm}-0/data" for arm in (left, right)]
        stages = [
            [s for s in analysis.read(d / "observers.json")["stages"] if s["request"] == i]
            for d, i in zip(dirs, (li, ri), strict=True)
        ]
        rows = []
        for a, b in zip(*stages, strict=True):
            if any(a[k] != b[k] for k in ("call", "stage", "shape", "stride", "dtype", "device")):
                raise ValueError("Backend tensor contracts differ")
            tensors = []
            for directory, s in zip(dirs, (a, b), strict=True):
                path = directory / s["path"]
                if sha(path) != s["file_sha256"]:
                    raise ValueError("Retained tensor hash changed")
                tensors.append(torch.load(path, map_location="cpu", weights_only=True))
            rows.append(
                dict(
                    call=a["call"],
                    stage=a["stage"],
                    shape=a["shape"],
                    stride=a["stride"],
                    original_dtype=a["dtype"],
                    original_device=a["device"],
                    comparison=numerics.difference(*tensors),
                )
            )
        comparisons.append(dict(name=name, rows=rows))
    write(
        args.output,
        dict(
            producer_sha256=sha(__file__),
            comparisons=comparisons,
            scope="Descriptive numerical deltas only; no operator/bbox tolerance acceptance",
        ),
    )


if __name__ == "__main__":
    main()
