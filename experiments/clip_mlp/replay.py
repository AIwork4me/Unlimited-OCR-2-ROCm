"""Model-free fixed-tensor replay of the pinned official quick_gelu ScriptFunction."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import traceback
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/cold_start"))
from worker import ORIGINAL, load, sha, write  # noqa: E402


def tensor_hash(tensor):
    import torch

    return hashlib.sha256(
        tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def metadata(tensor):
    return dict(
        shape=list(tensor.shape),
        stride=list(tensor.stride()),
        dtype=str(tensor.dtype),
        device=str(tensor.device),
        storage_offset=tensor.storage_offset(),
        sha256=tensor_hash(tensor),
    )


def identity(output):
    import torch

    wheel = ROOT / ".b/wheels/torch-2.12.0+rocm7.14.1-cp310-cp310-win_amd64.whl"
    if sha(wheel) != ORIGINAL:
        raise ValueError("Original published wheel required")
    members = {}
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.startswith("torch/") and name.endswith((".dll", ".pyd", ".py")):
                expected = hashlib.sha256(archive.read(name)).hexdigest()
                if sha(Path(sys.prefix) / "Lib/site-packages" / name) != expected:
                    raise ValueError("Installed original member differs: " + name)
                members[name] = expected
    helper = load("clip_mlp_identity_helper", ROOT / "experiments/native_regression/runner.py")
    torch.cuda.init()
    props = torch.cuda.get_device_properties(0)
    if props.gcnArchName != "gfx1151" or torch.__version__ != "2.12.0+rocm7.14.1":
        raise ValueError("Original Windows gfx1151 identity required")
    modules = helper.loaded_modules()
    for item in modules:
        matches = [
            h for n, h in members.items() if Path(n).name.lower() == Path(item["path"]).name.lower()
        ]
        if matches and (
            item["sha256"] not in matches
            or not Path(item["path"]).resolve().is_relative_to(Path(sys.prefix).resolve())
        ):
            raise ValueError("Loaded original member differs")
    write(
        output / "identity.json",
        dict(
            wheel_sha256=ORIGINAL,
            torch_git_version=torch.version.git_version,
            executable=sys.executable,
            pid=os.getpid(),
            producer_sha256=sha(__file__),
            installed_manifest=members,
            loaded=modules,
            device=dict(name=props.name, arch=props.gcnArchName),
            jit_api_sha256=sha(Path(torch.__file__).parent / "jit/_fuser.py"),
            determinism=torch.are_deterministic_algorithms_enabled(),
            supported_activities=[str(a) for a in torch.profiler.supported_activities()],
        ),
    )
    return helper, helper.open_loaded_hip(modules)


def run(args):
    import torch

    helper, hip = identity(args.output)
    captures = json.loads((args.capture / "captures.json").read_text("utf-8"))
    expected = next(v for v in captures if v["call"] == 1 and v["stage"] == "activation_input")
    path = args.capture / "call-01/activation_input.pt"
    if sha(path) != expected["file_sha256"]:
        raise ValueError("Frozen input artifact changed")
    cpu = torch.load(path, map_location="cpu", weights_only=True)
    if tensor_hash(cpu) != expected["sha256"] or list(cpu.stride()) != expected["stride"]:
        raise ValueError("Captured input bytes/layout changed")
    x = torch.empty_strided(cpu.shape, cpu.stride(), dtype=cpu.dtype, device="cuda")
    x.copy_(cpu)
    if metadata(x) != {
        k: expected[k] for k in ("shape", "stride", "dtype", "device", "storage_offset", "sha256")
    }:
        raise ValueError("Replay layout/input identity differs")
    lock = json.loads((ROOT / "src/unlimited_ocr_2_rocm/upstream-lock.json").read_text("utf-8-sig"))
    source = (
        ROOT
        / ".b/h/models--baidu--Unlimited-OCR-2/snapshots"
        / lock["model_revision"]
        / "deepencoder.py"
    )
    if sha(source) != lock["files"]["deepencoder.py"]:
        raise ValueError("Immutable official source mismatch")
    if args.control == "te-off":
        torch._C._jit_set_texpr_fuser_enabled(False)
    elif args.control == "fusion-off":
        torch._C._jit_override_can_fuse_on_gpu(False)
    elif args.control == "profiling-off":
        torch._C._jit_set_profiling_mode(False)
    spec = importlib.util.spec_from_file_location("clip_mlp_original_deepencoder", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = module.quick_gelu
    if args.prefix < 3:
        graph = function.graph.copy()
        operators = [n for n in graph.nodes() if n.kind().startswith("aten::")]
        graph.eraseOutput(0)
        graph.registerOutput(operators[args.prefix - 1].output())
        torch._C._jit_pass_dce(graph)
        function = torch._C._create_function_from_graph(f"original_prefix_{args.prefix}", graph)
    write(
        args.output / "setup.json",
        dict(
            input=metadata(x),
            input_artifact_sha256=sha(path),
            source_sha256=sha(source),
            schema=str(function.schema),
            static_graph=str(function.graph),
            control=args.control,
            position=args.position,
            graph_executor_optimize_before=torch._C._get_graph_executor_optimize(),
            can_fuse_gpu=torch._C._jit_can_fuse_on_gpu(),
            tensor_expr_fuser=torch._C._jit_texpr_fuser_enabled(),
            model_loaded=False,
            weights_loaded=False,
            original_operator_prefix=args.prefix,
        ),
    )
    activity = [torch.profiler.ProfilerActivity.CPU]
    if torch.profiler.ProfilerActivity.CUDA in torch.profiler.supported_activities():
        activity.append(torch.profiler.ProfilerActivity.CUDA)
    values = []
    before = helper.hip_probe(hip)
    with torch.inference_mode(), torch.jit.optimized_execution(args.control != "no-opt"):
        for i in range(1 if args.position == "cold" else 2):
            with torch.profiler.profile(activities=activity, record_shapes=True) as profile:
                result = function(x)
                torch.cuda.synchronize()
            graph = str(torch.jit.last_executed_optimized_graph())
            torch.save(result.cpu(), args.output / f"output-{i}.pt")
            profile.export_chrome_trace(str(args.output / f"trace-{i}.json"))
            values.append(
                dict(
                    index=i,
                    role="MEASURED"
                    if i == (0 if args.position == "cold" else 1)
                    else "FIXED_INPUT_INITIALIZATION_CALL",
                    graph=graph,
                    output=metadata(result),
                    operators=[dict(name=e.key, count=e.count) for e in profile.key_averages()],
                    events=[
                        dict(name=e.name, device_type=str(e.device_type)) for e in profile.events()
                    ],
                )
            )
    after = helper.hip_probe(hip)
    out = result.float().cpu()
    reference_x = cpu.float()
    reference = 1.702 * reference_x
    if args.prefix > 1:
        reference = torch.sigmoid(reference)
    if args.prefix > 2:
        reference = reference_x * reference
    error = (out - reference).abs()
    write(
        args.output / "result.json",
        dict(
            calls=values,
            input_after=metadata(x),
            hip_before=before,
            hip_after=after,
            cpu_fp32_reference=dict(
                max_abs=float(error.max()),
                mean_abs=float(error.mean()),
                elements=error.numel(),
                nonfinite=int((~torch.isfinite(out)).sum()),
            ),
            graph_executor_optimize_after=torch._C._get_graph_executor_optimize(),
            memory=dict(
                allocated=torch.cuda.max_memory_allocated(),
                reserved=torch.cuda.max_memory_reserved(),
            ),
            kernel_visibility="GPU activity"
            if len(activity) > 1
            else "CPU profiler only; GPU kernel symbols unavailable",
        ),
    )
    if (
        before["status"] != after["status"]
        or after["status"] != "PASS"
        or tensor_hash(x) != expected["sha256"]
    ):
        raise ValueError("HIP/input integrity gate failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--control",
        choices=["normal", "no-opt", "te-off", "fusion-off", "profiling-off"],
        required=True,
    )
    parser.add_argument("--position", choices=["cold", "warm"], required=True)
    parser.add_argument("--prefix", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument("--execute-gpu", action="store_true", required=True)
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT or not args.output.resolve().is_relative_to(ROOT / ".b/mlp"):
        parser.error("Use a fresh repository-local output")
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        run(args)
    except BaseException:
        write(args.output / "failure.json", dict(traceback=traceback.format_exc()))
        raise
