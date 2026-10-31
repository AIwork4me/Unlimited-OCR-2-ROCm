"""Compare sealed captured GPU outputs with independent CPU attention references."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from reference import attention_reference, error_summary


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def compare_capture(folder, subsystem, output):
    import torch

    index = json.loads((folder / "capture-index.json").read_text("utf-8"))
    records = {r["stage"]: r for r in index["records"] if r["request"] == 0}
    prefix = subsystem + ".first_sdpa."
    metadata = next(
        m
        for m in index["sdpa_metadata"]
        if m.get("request", 0) == 0 and m.get("subsystem", "sam") == subsystem
    )
    options = metadata["non_tensor_kwargs"]
    allowed = {"attn_mask", "dropout_p", "is_causal", "scale", "enable_gqa"}
    if (
        metadata["positional_count"] != 3
        or options.get("enable_gqa", False)
        or set(options) - allowed
    ):
        raise ValueError("Unsupported or incomplete SDPA invocation contract")
    tensors = {}
    for name in ("q", "k", "v", "mask", "output"):
        if prefix + name not in records:
            if name == "mask":
                continue
            raise ValueError("Missing required capture")
        record = records[prefix + name]
        path = folder / record["path"]
        if sha(path) != record["file_sha256"]:
            raise ValueError("Changed tensor file")
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        raw = tensor.contiguous().view(torch.uint8).numpy().tobytes()
        if hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError("Changed tensor bytes")
        tensors[name] = tensor.float().numpy()
    if metadata["mask_present"] != ("mask" in tensors):
        raise ValueError("Mask capture mismatch")
    contract = dict(
        scale=options.get("scale"),
        causal=options.get("is_causal", False),
        dropout=options.get("dropout_p", 0.0),
    )
    ref32 = attention_reference(
        tensors["q"],
        tensors["k"],
        tensors["v"],
        mask=tensors.get("mask"),
        dtype=np.float32,
        **contract,
    )
    ref64 = attention_reference(
        tensors["q"],
        tensors["k"],
        tensors["v"],
        mask=tensors.get("mask"),
        dtype=np.float64,
        **contract,
    )
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "reference-fp32.npy", ref32)
    np.save(output / "reference-fp64.npy", ref64)
    rounded = torch.from_numpy(ref64).to(torch.bfloat16).float().numpy()
    observed = tensors["output"]
    return dict(
        source_index_sha256=sha(folder / "capture-index.json"),
        subsystem=subsystem,
        contracts={
            n: {k: records[prefix + n][k] for k in ("shape", "stride", "dtype", "device", "sha256")}
            | {"original_storage_offset": records[prefix + n].get("storage_offset")}
            for n in tensors
        },
        fp32=error_summary(observed, ref32),
        fp64=error_summary(observed, ref64),
        fp32_vs_fp64=error_summary(ref32, ref64),
        differs_from_bf16_rounded_fp64_reference=int((observed != rounded).sum()),
        operator_contract=contract
        | {"enable_gqa": False, "mask_present": metadata["mask_present"]},
        rounded_fp64=error_summary(observed, rounded),
        reference_hashes={p.name: sha(p) for p in output.iterdir()},
        source=sha(__file__),
        reference_source=sha(Path(__file__).with_name("reference.py")),
        acceptance="DESCRIPTIVE ONLY: no numerical or OCR acceptance threshold",
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = Path.cwd()
    old = root / ".b/te-e2e/mixed-rca-20261030-v1"
    iso = root / ".b/te-e2e/sam-clip-isolation-20261031-v1"
    pairs = [
        ("SAM-MATH", old / "fine-0/data", "sam"),
        ("SAM-efficient", old / "fine-1/data", "sam"),
        ("CLIP-MATH-own-input", iso / "process-0/data", "clip"),
        ("CLIP-flash-own-input", iso / "process-2/data", "clip"),
    ]
    results = {
        label: compare_capture(folder, sub, args.output / label) for label, folder, sub in pairs
    }
    summary = dict(
        numpy=np.__version__,
        results=results,
        limitation=(
            "Old CLIP inputs differ; own-input errors are not a direct backend comparison. "
            "Old GPU storage offsets unmeasured."
        ),
    )
    (args.output / "retained-reference.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v["fp64"] for k, v in results.items()}))


if __name__ == "__main__":
    main()
