"""Derive expected invocation shapes from pinned source and PNG headers, no torch."""

import argparse
import ast
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/cold_start"))
from worker import sha, write  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("declaration", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    declaration = json.loads(args.declaration.read_text("utf-8"))
    lock = json.loads((ROOT / "src/unlimited_ocr_2_rocm/upstream-lock.json").read_text("utf-8"))
    source = (
        ROOT
        / ".b/h/models--baidu--Unlimited-OCR-2/snapshots"
        / lock["model_revision"]
        / "modeling_unlimitedocr.py"
    )
    if sha(source) != lock["files"][source.name]:
        raise ValueError("Pinned crop source changed")
    tree = ast.parse(source.read_text("utf-8"))
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "find_closest_aspect_ratio"
    )
    namespace = {}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), "<verified-pure-crop-ratio>", "exec"),
        namespace,
    )
    truth = json.loads((ROOT / "tests/fixtures/ground_truth.json").read_text("utf-8"))["cases"]
    ratios = {
        (i, j)
        for n in range(2, 17)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if 2 <= i * j <= 16
    }
    ratios = sorted(ratios, key=lambda x: x[0] * x[1])
    cases = {}
    for case in declaration["cases"]:
        inputs = case["truth_keys"]
        if len(inputs) == 1:
            relative = "tests/fixtures/" + truth[inputs[0]]["image"]
            path = ROOT / relative
            if sha(path) != declaration["input_hashes"][relative]:
                raise ValueError("Frozen PNG changed")
            data = path.read_bytes()
            if data[:8] != b"\x89PNG\r\n\x1a\n":
                raise ValueError("PNG header required")
            width, height = struct.unpack(">II", data[16:24])
            grid = namespace["find_closest_aspect_ratio"](
                width / height, ratios, width, height, 640
            )
            shapes = [
                [grid[0] * grid[1], (640 // 64) ** 2 + 1, 4096],
                [1, (1024 // 64) ** 2 + 1, 4096],
            ]
        else:
            shapes = [[1, 257, 4096] for _ in inputs]
        cases[case["name"]] = shapes
    write(
        args.output,
        dict(
            producer_sha256=sha(__file__),
            declaration_sha256=sha(args.declaration),
            source_sha256=sha(source),
            input_hashes=declaration["input_hashes"],
            cases=cases,
            basis=(
                "Pinned _uocr_build_inputs/forward single-image auto crop; "
                "SAM stride64 + CLIP CLS; independent source review"
            ),
            scope=(
                "Invocation/layout checker correction; no numerical/semantic/bbox tolerance change"
            ),
        ),
    )


if __name__ == "__main__":
    main()
