"""Offline diagnostic only; never imported by production. Raw evidence stays local."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import random
import subprocess
import sys
import traceback
import zipfile
from pathlib import Path

ROOT = Path.cwd().resolve()
FROZEN = ROOT
from isolation_control import module_mode, dispatch_valid
ORIGINAL = "415d2dbd4b6ab002cff1c54795fdd66322abd9a46bf982cb64eebb4493ac2e7c"
CANDIDATE = "6a67d15861538ee217afdccd1ccc09ff318d7f44bff29a10f74c0d51730e6e1e"
BASE = "82951c73b28bf735f5655e6f5c3306cd170a21c2"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run(args):
    import numpy as np
    import psutil
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    if Path.cwd().resolve() != ROOT or psutil.virtual_memory().available < 32 * 1024**3:
        raise RuntimeError("Primary-root cwd and 32 GiB available admission required")
    source_diff = subprocess.check_output(
        ["git", "diff", BASE, "--", "src", "tests/fixtures", "requirements"], cwd=ROOT
    )
    if source_diff:
        raise RuntimeError("Frozen integration production source changed")
    sys.path.insert(0, str(FROZEN / "src"))
    helper = load("cold_start_observer", FROZEN / "experiments/native_regression/runner.py")
    expected = ORIGINAL if args.wheel == "original" else CANDIDATE
    wheel = ROOT / ".b/wheels/torch-2.12.0+rocm7.14.1-cp310-cp310-win_amd64.whl"
    if args.wheel == "candidate":
        wheels = list((ROOT / ".b/out/wheels").glob("torch*.whl"))
        wheel = next(p for p in wheels if sha(p) == CANDIDATE)
    if sha(wheel) != expected:
        raise RuntimeError("Wrong wheel artifact")
    members = {}
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.startswith("torch/") and name.lower().endswith((".dll", ".pyd", ".py")):
                h = hashlib.sha256(archive.read(name)).hexdigest()
                if sha(Path(sys.prefix) / "Lib/site-packages" / name) != h:
                    raise RuntimeError("Installed wheel member mismatch: " + name)
                members[name] = h
    if args.init:
        torch.cuda.init()
        torch.cuda.synchronize()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    from unlimited_ocr_2_rocm import inference
    from unlimited_ocr_2_rocm.model import load_model

    native = load_model(ROOT / ".b/h", offline=True)
    loaded = helper.loaded_modules()
    for m in loaded:
        name = Path(m["path"]).name.lower()
        matches = [h for n, h in members.items() if Path(n).name.lower() == name]
        if matches and (
            m["sha256"] not in matches
            or not Path(m["path"]).resolve().is_relative_to(Path(sys.prefix).resolve())
        ):
            raise RuntimeError("Loaded wheel identity mismatch")
    hip = helper.open_loaded_hip(loaded)
    module = __import__(native.model.__class__.__module__, fromlist=["_"])
    source = {str(p.relative_to(FROZEN)): sha(p) for p in (FROZEN / "src").rglob("*.py")}
    write(
        args.output / "identity.json",
        dict(
            pid=os.getpid(),
            ppid=os.getppid(),
            created=psutil.Process().create_time(),
            executable=sys.executable,
            prefix=sys.prefix,
            cwd=str(Path.cwd()),
            producer_sha256=sha(__file__),
            source_commit=BASE,
            source_hashes=source,
            helper_sha256=sha(FROZEN / "experiments/native_regression/runner.py"),
            wheel_sha256=expected,
            installed_members=members,
            loaded_modules=loaded,
            model=native.evidence,
            condition=vars(args) | {"output": str(args.output)},
            environment={
                k: v
                for k, v in os.environ.items()
                if any(
                    s in k
                    for s in (
                        "HIP",
                        "HSA",
                        "TRITON",
                        "DASH",
                        "CUBLAS",
                        "DETERMIN",
                        "PYTHONHASH",
                        "OMP",
                        "MKL",
                    )
                )
            },
            determinism=dict(
                enabled=torch.are_deterministic_algorithms_enabled(),
                warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
            ),
            cache_policy="Shared disk Triton cache; fresh process/module/model/KV; no deletion",
        ),
    )
    truths = json.loads((FROZEN / "tests/fixtures/ground_truth.json").read_text("utf-8"))["cases"]
    mapping = {"pages_2": "pages_2_run_1"}
    case = next(c for c in helper.normal_cases() if c["name"] == mapping.get(args.case, args.case))
    events, dispatch, handles = [], {}, []
    active = [None]

    def fingerprint(value):
        if isinstance(value, torch.Tensor):
            cpu = value.detach().contiguous().cpu()
            return dict(
                shape=list(cpu.shape),
                dtype=str(cpu.dtype),
                sha256=hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest(),
            )
        if isinstance(value, (list, tuple)):
            return [fingerprint(v) for v in value]
        if isinstance(value, dict):
            return {
                str(k): fingerprint(v)
                for k, v in value.items()
                if isinstance(v, (torch.Tensor, list, tuple))
            }
        return type(value).__name__

    def observe(name, mod):
        def pre(_mod, inputs, kwargs):
            events.append(dict(stage=name + ".input", value=fingerprint((inputs, kwargs))))

        def post(_mod, inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            if name == "lm_head":
                cpu = value.detach()[0, -1].float().cpu()
                values, ids = cpu.topk(2)
                events.append(
                    dict(
                        stage=name,
                        value=fingerprint(cpu),
                        top_ids=ids.tolist(),
                        top_values=values.tolist(),
                        margin=float(values[0] - values[1]),
                    )
                )
            else:
                events.append(dict(stage=name + ".output", value=fingerprint(value)))

        handles.extend(
            (mod.register_forward_pre_hook(pre, with_kwargs=True), mod.register_forward_hook(post))
        )

    if args.trace:
        build_inputs = module._uocr_build_inputs

        def observed_inputs(*a, **kw):
            result = build_inputs(*a, **kw)
            events.append(dict(stage="preprocessing.output", value=fingerprint(result)))
            return result

        module._uocr_build_inputs = observed_inputs
        for name in ("sam_model", "vision_model", "projector"):
            observe(name, getattr(native.model.model, name))
        for i, layer in enumerate(native.model.model.layers):
            observe(f"layer.{i}", layer)
        observe("lm_head", native.model.lm_head)
        if args.deep_trace:
            for name, child in native.model.model.vision_model.named_modules():
                if name:
                    observe("clip." + name, child)

    from torch.overrides import TorchFunctionMode

    deep_trace = args.deep_trace

    class Attention(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            if func is not torch.nn.functional.scaled_dot_product_attention:
                return func(*args, **(kwargs or {}))
            if active[0] == "vision_model" and deep_trace:
                events.append(
                    dict(stage="clip.sdpa.input", value=fingerprint((args, kwargs or {})))
                )
                result = func(*args, **(kwargs or {}))
                events.append(dict(stage="clip.sdpa.output", value=fingerprint(result)))
                return result
            return func(*args, **(kwargs or {}))

    originals = {
        n: inspect.unwrap(getattr(native.model.model, n).forward)
        for n in ("sam_model", "vision_model")
    }

    class Dispatch(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            name = str(func)
            if "scaled_dot_product" in name:
                dispatch.setdefault(active[0], {}).setdefault(name, 0)
                dispatch[active[0]][name] += 1
            return func(*args, **(kwargs or {}))

    for index, mode in enumerate(args.modes.split(",")):
        from torch.nn.attention import SDPBackend, sdpa_kernel

        events.clear()
        dispatch.clear()
        for name, original in originals.items():

            def scoped(*a, _name=name, _original=original, _mode=module_mode(mode, name), **kw):
                active[0] = _name
                backends = (
                    SDPBackend.MATH
                    if _mode == "MATH"
                    else [
                        SDPBackend.FLASH_ATTENTION,
                        SDPBackend.EFFICIENT_ATTENTION,
                        SDPBackend.MATH,
                    ]
                )
                with sdpa_kernel(backends):
                    with torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU]
                    ) as profile:
                        with Attention():
                            result = _original(*a, **kw)
                        torch.cuda.synchronize()
                    for entry in profile.key_averages():
                        if "_scaled_dot_product" in entry.key:
                            dispatch.setdefault(_name, {}).setdefault(entry.key, 0)
                            dispatch[_name][entry.key] += entry.count
                active[0] = None
                native.vision.append(
                    dict(
                        module=_name,
                        backend="GPU MATH SDPA" if _mode == "MATH" else "AUTO",
                        status="PASS",
                    )
                )
                return result

            getattr(native.model.model, name).forward = scoped
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        before = dict(
            cpu_rng=fingerprint(torch.get_rng_state()),
            gpu_rng=fingerprint(torch.cuda.get_rng_state()),
            training=native.model.training,
            parameters={n: p._version for n, p in native.model.named_parameters()},
            buffers={n: fingerprint(b) for n, b in native.model.named_buffers()},
            dash_globals={k: repr(getattr(module, k)) for k in ("_DASH_INFER",)},
        )
        with Dispatch():
            summary = helper.run_case(
                native,
                module,
                inference,
                case,
                truths,
                FROZEN / "tests/fixtures",
                args.output / f"request-{index}",
                hip=hip,
            )
        write(
            args.output / f"observation-{index}.json",
            dict(
                mode=mode,
                before=before,
                after=dict(
                    cpu_rng=fingerprint(torch.get_rng_state()),
                    gpu_rng=fingerprint(torch.cuda.get_rng_state()),
                    parameters={n: p._version for n, p in native.model.named_parameters()},
                    buffers={n: fingerprint(b) for n, b in native.model.named_buffers()},
                ),
                events=events.copy(),
                dispatch=dispatch.copy(),
                raw_sha256=summary.get("raw_sha256"),
                status=summary["status"],
            ),
        )
        required = {"sam_model", "vision_model"}
        if set(dispatch) != required or any(
            not any("_scaled_dot_product" in op for op in v) for v in dispatch.values()
        ):
            raise RuntimeError("Actual per-encoder SDPA unobserved")
        if not dispatch_valid(dispatch, mode):
            raise RuntimeError("Independent backend mismatch or unexpected fallback")
        if summary["status"] != "PASS" or summary["hip_after"]["status"] != "PASS":
            raise RuntimeError("Request failed; retain evidence")
    for handle in handles:
        handle.remove()
    write(
        args.output / "complete.json", dict(status="COMPLETED", requests=len(args.modes.split(",")))
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wheel", choices=["original", "candidate"], required=True)
    parser.add_argument("--case", default="pages_2")
    parser.add_argument("--modes", default="MATH,MATH")
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--deep-trace", action="store_true")
    parser.add_argument("--jit-unoptimized", action="store_true")
    parser.add_argument("--execute-gpu", action="store_true", required=True)
    options = parser.parse_args()
    if not options.output.resolve().is_relative_to(ROOT / ".b/cold-start-rca"):
        parser.error("Fresh local experiment output required")
    options.output.mkdir(parents=True, exist_ok=False)
    try:
        if options.jit_unoptimized:
            import torch

            with torch.jit.optimized_execution(False):
                run(options)
        else:
            run(options)
    except BaseException:
        write(options.output / "failure.json", dict(traceback=traceback.format_exc()))
        raise
