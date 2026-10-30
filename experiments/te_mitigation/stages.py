"""Local tensor analysis: publish contracts/deltas, never element values."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/cold_start"))
from worker import sha, write  # noqa: E402


def main():
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for folder in sorted(args.root.glob("*/data")):
        observer = json.loads((folder / "observers.json").read_text("utf-8"))
        stages = observer["stages"]
        for right in (s for s in stages if s["request"] == 1):
            left = next(
                s
                for s in stages
                if s["request"] == 0 and s["call"] == right["call"] and s["stage"] == right["stage"]
            )
            for s in (left, right):
                if sha(folder / s["path"]) != s["file_sha256"]:
                    raise ValueError("Private tensor seal changed")
            a, b = [
                torch.load(folder / s["path"], weights_only=True, map_location="cpu")
                for s in (left, right)
            ]
            changed = (a.view(torch.int16) != b.view(torch.int16)).flatten()
            indexes = changed.nonzero().flatten()
            delta = b.float() - a.float()
            rows.append(
                dict(
                    process=folder.parent.name,
                    call=right["call"],
                    stage=right["stage"],
                    contracts_exact=all(
                        left[k] == right[k] for k in ("shape", "stride", "dtype", "device")
                    ),
                    input_hashes=[left["sha256"], right["sha256"]],
                    shape=left["shape"],
                    stride=left["stride"],
                    dtype=left["dtype"],
                    device=left["device"],
                    different_elements=int(changed.sum()),
                    max_abs=float(delta.abs().max()),
                    mean_abs=float(delta.abs().mean()),
                    first_flat_index=int(indexes[0]) if indexes.numel() else None,
                )
            )
    write(args.output, dict(producer_sha256=sha(__file__), rows=rows))


if __name__ == "__main__":
    main()
