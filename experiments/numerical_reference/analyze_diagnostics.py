"""CPU-only audit of matched GPU attention, logits and supported cache captures."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from reference import error_summary
from retained import compare_capture, sha


def read(path):
    return json.loads(path.read_text("utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def main():
    import torch
    from safetensors import safe_open

    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, evidence, output = Path.cwd(), args.evidence, args.output
    output.mkdir(parents=True, exist_ok=False)
    labels = ["MM", "AA", "AM", "MA"]
    dirs = {name: evidence / f"process-{i}/data" for i, name in enumerate(labels)}
    indices = {name: read(d / "capture-index.json") for name, d in dirs.items()}
    records = {name: {r["stage"]: r for r in idx["records"]} for name, idx in indices.items()}
    arrays = {}

    def tensor(label, stage):
        key = label, stage
        if key not in arrays:
            row = records[label][stage]
            path = dirs[label] / row["path"]
            if sha(path) != row["file_sha256"]:
                raise ValueError("Changed capture file")
            value = torch.load(path, map_location="cpu", weights_only=True)
            raw = value.contiguous().view(torch.uint8).numpy().tobytes()
            if hashlib.sha256(raw).hexdigest() != row["sha256"]:
                raise ValueError("Changed tensor bytes")
            arrays[key] = value
        return arrays[key]

    spec = importlib.util.spec_from_file_location(
        "nr_gate", root / "experiments/te_mitigation/analyze.py"
    )
    gates = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gates)
    checks = []
    for i, label in enumerate(labels):
        data = dirs[label]
        summary = read(data / "worker/request-0/summary.json")
        ob = read(data / "worker/observation-0.json")
        obs = read(data / "observers.json")
        condition = read(evidence / "predeclared.json")["schedule"][i]["modes"]
        common = gates.request_gates(
            summary,
            ob,
            obs["q_events"],
            obs["profiles"],
            "eager",
            "AUTO",
            1,
            [[6, 101, 4096], [1, 257, 4096]],
        )
        expected = {
            "sam_model": "math" if condition.split("/")[0] == "MATH" else "efficient_attention",
            "vision_model": "math" if condition.split("/")[1] == "MATH" else "flash_attention",
        }
        common["observation"] = ob["status"] == "PASS" and ob["mode"] == condition
        common["attention"] = set(ob["dispatch"]) == set(expected) and all(
            any("_scaled_dot_product" in op for op in ops)
            and all(expected[module] in op for op in ops if "_scaled_dot_product" in op)
            for module, ops in ob["dispatch"].items()
        )
        common["mutable_state"] = all(
            ob["before"][k] == ob["after"][k] for k in ["parameters", "buffers"]
        )
        common["admission"] = all(read(data.parent / "admission.json").values())
        finish = read(data.parent / "finished.json")
        common["cleanup"] = (
            finish["exit_code"] == 0 and finish["accounting"]["active_processes"] == 0
        )
        if not all(common.values()):
            raise ValueError((label, common))
        checks.append(
            dict(
                label=label,
                gates=common,
                dispatch=ob["dispatch"],
                runtime=indices[label]["runtime"],
            )
        )
        for stage in records[label]:
            tensor(label, stage)

    refs = {}
    for label in labels:
        refs[label] = {
            sub: compare_capture(dirs[label], sub, output / f"{label}-{sub}")
            for sub in ["sam", "clip"]
        }
    paired = []
    for left, right, sub in [("MM", "AA", "sam"), ("MM", "MA", "clip"), ("AM", "AA", "clip")]:
        prefix = sub + ".first_sdpa."
        names = ["q", "k", "v"] + (["mask"] if sub == "sam" else [])
        contracts = ["shape", "stride", "storage_offset", "dtype", "device", "sha256"]
        if any(
            records[left][prefix + n][k] != records[right][prefix + n][k]
            for n in names
            for k in contracts
        ):
            raise ValueError("Unmatched operator input/contract")
        lm = next(m for m in indices[left]["sdpa_metadata"] if m["subsystem"] == sub)
        rm = next(m for m in indices[right]["sdpa_metadata"] if m["subsystem"] == sub)
        if lm != rm:
            raise ValueError("Unmatched invocation options")
        stages = [prefix + "output"]
        if sub == "clip":
            stages += [
                "clip." + n
                for n in [
                    "norm1",
                    "attention_projected",
                    "residual",
                    "norm2",
                    "mlp",
                    "block0.output",
                ]
            ]
        paired.append(
            dict(
                left=left,
                right=right,
                subsystem=sub,
                inputs_exact=True,
                metadata_exact=True,
                stages=[
                    dict(
                        stage=n,
                        left_contract=records[left][n],
                        right_contract=records[right][n],
                        logical_bytes_exact=records[left][n]["sha256"]
                        == records[right][n]["sha256"],
                        metrics=error_summary(
                            tensor(left, n).float().numpy(), tensor(right, n).float().numpy()
                        ),
                    )
                    for n in stages
                ],
            )
        )

    lock = read(root / "src/unlimited_ocr_2_rocm/upstream-lock.json")
    snapshot = root / ".b/h/models--baidu--Unlimited-OCR-2/snapshots" / lock["model_revision"]
    mapping = read(snapshot / "model.safetensors.index.json")["weight_map"]
    weight_path = snapshot / mapping["lm_head.weight"]
    if sha(weight_path) != lock["weights"][weight_path.name]:
        raise ValueError("Changed projection weights")
    entries = [(label, step) for label in labels for step in [7, 10]]
    hidden = np.stack(
        [
            tensor(label, f"decoder.hidden.{step}").float().numpy().reshape(-1)
            for label, step in entries
        ]
    )
    with safe_open(weight_path, framework="pt", device="cpu") as weights:
        w = weights.get_tensor("lm_head.weight")
        projection32 = np.empty((len(entries), w.shape[0]), dtype=np.float32)
        projection64 = np.empty((len(entries), w.shape[0]), dtype=np.float64)
        for start in range(0, w.shape[0], 4096):
            block = w[start : start + 4096].float().numpy()
            projection32[:, start : start + 4096] = (
                hidden.astype(np.float32) @ block.astype(np.float32).T
            )
            projection64[:, start : start + 4096] = (
                hidden.astype(np.float64) @ block.astype(np.float64).T
            )
    np.save(output / "projection-fp32.npy", projection32)
    np.save(output / "projection-fp64.npy", projection64)
    logits = []
    for row, (label, step) in enumerate(entries):
        raw = tensor(label, f"logits.raw.{step}").float().numpy().reshape(-1)
        selection = tensor(label, f"logits.selection.{step}").numpy().reshape(-1)
        decisions = indices[label]["teacher_forced_decisions"]
        order = np.lexsort((np.arange(len(raw)), -raw))[:2]
        reforder = np.lexsort((np.arange(len(raw)), -projection64[row]))[:2]
        actual = decisions[step]["actual_selected"]
        if not np.array_equal(raw, selection) or int(np.argmax(raw)) != actual:
            raise ValueError("Unverified post-guard selection")
        logits.append(
            dict(
                label=label,
                step=step,
                raw_logit_sha256=records[label][f"logits.raw.{step}"]["sha256"],
                selection_logit_sha256=records[label][f"logits.selection.{step}"]["sha256"],
                dtype_transition=(
                    "GPU BF16 lm_head -> GPU FP32 cast -> unchanged page guard -> argmax"
                ),
                top2_token_ids=[int(x) for x in order],
                top2_values=[float(raw[x]) for x in order],
                signed_margin=float(raw[order[0]] - raw[order[1]]),
                exact_maxima=int((raw == raw.max()).sum()),
                actual_argmax=actual,
                fed_token=decisions[step]["fed_token"],
                actual_argmax_sha256=digest(actual),
                fp64_top2_token_ids=[int(x) for x in reforder],
                fp64_top2_values=[float(projection64[row, x]) for x in reforder],
                fp64_margin=float(projection64[row, reforder[0]] - projection64[row, reforder[1]]),
                fp64_argmax_sha256=digest(int(reforder[0])),
                projection_error=error_summary(raw, projection64[row]),
                projection32_vs64=error_summary(projection32[row], projection64[row]),
                scope=(
                    "FP64 projection from fixed captured BF16 hidden state and immutable "
                    "BF16 lm_head weights; not higher-precision decoder execution."
                ),
            )
        )
    cache = []
    for label in labels:
        append_checks = []
        for previous, current in [(0, 7), (7, 10)]:
            for layer in range(12):
                for kind in ["K", "V"]:
                    old_name = f"cache.{kind}.{previous}.{layer}"
                    new_name = f"cache.{kind}.{current}.{layer}"
                    old = tensor(label, old_name)
                    prefix = tensor(label, new_name)[..., : old.shape[-2], :]
                    prefix_hash = hashlib.sha256(
                        prefix.contiguous().view(torch.uint8).numpy().tobytes()
                    ).hexdigest()
                    append_checks.append(
                        dict(
                            previous=previous,
                            current=current,
                            layer=layer,
                            kind=kind,
                            prefix_bytes_exact=prefix_hash == records[label][old_name]["sha256"],
                        )
                    )
        cache.append(
            dict(
                label=label,
                cache=indices[label]["cache"],
                append_checks=append_checks,
                contracts=[r for n, r in records[label].items() if n.startswith("cache.")],
            )
        )
    write(output / "attention-reference.json", dict(checks=checks, references=refs, paired=paired))
    write(output / "token-logits-private.json", dict(requests=logits))
    write(output / "cache-private.json", dict(requests=cache))
    print(
        json.dumps(
            dict(
                all_gates=True,
                matched_pairs=[(x["left"], x["right"], x["subsystem"]) for x in paired],
                logits=[
                    {
                        k: x[k]
                        for k in ["label", "step", "signed_margin", "exact_maxima", "fp64_margin"]
                    }
                    for x in logits
                ],
            )
        )
    )


if __name__ == "__main__":
    main()
