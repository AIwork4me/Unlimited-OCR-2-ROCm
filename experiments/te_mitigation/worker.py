"""Experimental end-to-end scopes only; production sources remain unchanged."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import sys
import traceback
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/cold_start"))
import worker as frozen  # noqa: E402
from worker import sha, write  # noqa: E402


def run(args):
    import torch

    sys.path.insert(0, str(ROOT / "src"))
    from unlimited_ocr_2_rocm import model

    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    declaration = json.loads(args.declaration.read_text("utf-8"))
    item = declaration["schedule"][args.item]
    if (
        args.option != item["option"]
        or args.modes != item["modes"]
        or args.wheel != ("candidate" if declaration["backend"] else "original")
        or args.tensors != declaration["tensors"]
        or args.suite != declaration["suite"]
    ):
        raise ValueError("Actual execution differs from declared arm")
    for inventory in ("source_hashes", "input_hashes"):
        for relative, expected in declaration[inventory].items():
            if sha(ROOT / relative) != expected:
                raise ValueError("Predeclared identity changed: " + relative)
    flag_before = torch._C._jit_texpr_fuser_enabled()
    if not flag_before or not torch._C._get_graph_executor_optimize():
        raise ValueError("Fresh default JIT state required")
    if args.option == "te-off":
        torch._C._jit_set_texpr_fuser_enabled(False)
    original_load = model.load_model
    original_module_load = frozen.load
    original_profile = torch.profiler.profile
    q_events = []
    profiles = []
    stages = []
    request = [-1]
    q_counts = {}
    active_profile = [None]
    native_ref = []
    case_names = []

    def tensor(stage, tensor, layer=0, call_offset=0):
        if not args.tensors or layer != 0:
            return
        call = q_counts.get(0, 0) + call_offset
        slot = output / f"stages/request-{request[0]}/call-{call}"
        slot.mkdir(parents=True, exist_ok=True)
        cpu = torch.empty_strided(tensor.shape, tensor.stride(), dtype=tensor.dtype, device="cpu")
        cpu.copy_(tensor.detach())
        path = slot / (stage + ".pt")
        torch.save(cpu, path)
        stages.append(
            dict(
                request=request[0],
                call=call,
                stage=stage,
                path=path.relative_to(output).as_posix(),
                shape=list(tensor.shape),
                stride=list(tensor.stride()),
                dtype=str(tensor.dtype),
                device=str(tensor.device),
                sha256=hashlib.sha256(
                    cpu.contiguous().view(torch.uint8).numpy().tobytes()
                ).hexdigest(),
                file_sha256=sha(path),
            )
        )

    def loaded(*a, **kw):
        native = original_load(*a, **kw)
        native_ref.append(native)
        deep = importlib.import_module(
            native.model.__class__.__module__.rsplit(".", 1)[0] + ".deepencoder"
        )
        if sha(deep.__file__) != native.evidence["source_and_weight_hashes"]["deepencoder.py"]:
            raise ValueError("Loaded activation source differs from verified model")
        script = deep.quick_gelu
        tree = ast.parse(Path(deep.__file__).read_text("utf-8"))
        node = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "quick_gelu"
        )
        node.decorator_list = []
        namespace = {"torch": torch}
        exec(
            compile(
                ast.Module(body=[node], type_ignores=[]), "<verified-original-quick-gelu>", "exec"
            ),
            namespace,
        )
        eager = namespace["quick_gelu"]
        write(
            output / "scope.json",
            dict(
                option=args.option,
                source_sha256=sha(deep.__file__),
                script_schema=str(script.schema),
                original_python_ast_sha256=hashlib.sha256(ast.dump(node).encode()).hexdigest(),
                layers=len(native.model.model.vision_model.transformer.layers),
                graph_optimize=torch._C._get_graph_executor_optimize(),
                te_enabled=torch._C._jit_texpr_fuser_enabled(),
                scope="isolated process TE flag"
                if args.option == "te-off"
                else "CLIP MLP instance globals only",
                script_function_id=id(script),
            ),
        )
        for i, block in enumerate(native.model.model.vision_model.transformer.layers):

            def make_q(layer):
                def observed(x):
                    q_counts[layer] = q_counts.get(layer, 0) + 1
                    tensor("activation-input", x, layer)
                    result = (eager if args.option == "eager" else script)(x)
                    graph = (
                        None
                        if args.option == "eager"
                        else str(torch.jit.last_executed_optimized_graph())
                    )
                    tensor("activation-output", result, layer)
                    q_events.append(
                        dict(
                            request=request[0],
                            layer=layer,
                            call=q_counts[layer],
                            profile=active_profile[0],
                            shape=list(x.shape),
                            stride=list(x.stride()),
                            dtype=str(x.dtype),
                            device=str(x.device),
                            graph=graph,
                            te_enabled=torch._C._jit_texpr_fuser_enabled(),
                            graph_optimize=torch._C._get_graph_executor_optimize(),
                        )
                    )
                    return result

                return observed

            bound = block.mlp.forward
            original = bound.__func__
            if original.__code__.co_names != ("fc2", "quick_gelu", "fc1"):
                raise ValueError("Unexpected original MLP forward")
            clone = types.FunctionType(
                original.__code__,
                dict(original.__globals__, quick_gelu=make_q(i)),
                original.__name__,
                original.__defaults__,
                original.__closure__,
            )
            block.mlp.forward = types.MethodType(clone, block.mlp)
            if i == 0:
                block.mlp.fc1.register_forward_pre_hook(
                    lambda m, a: tensor("mlp-input", a[0], call_offset=1)
                )
                block.mlp.fc2.register_forward_hook(lambda m, a, r: tensor("mlp-output", r))
        return native

    @contextmanager
    def profile(*a, **kw):
        serial = len(profiles)
        active_profile[0] = serial
        current = None
        try:
            with original_profile(*a, **kw) as current:
                yield current
        finally:
            try:
                operators = (
                    [dict(name=e.key, count=e.count) for e in current.key_averages()]
                    if current is not None
                    else []
                )
                profiles.append(dict(request=request[0], serial=serial, operators=operators))
            finally:
                active_profile[0] = None

    def module_load(name, path):
        helper = original_module_load(name, path)
        if name == "cold_start_observer":
            cases = helper.normal_cases()
            schedule = (
                cases
                if args.suite
                else [next(c for c in cases if c["name"] == "pages_2_run_1")] * 2
            )
            if schedule != declaration["cases"]:
                raise ValueError("Actual cases differ from frozen schedule")
            case_names.extend(c["name"] for c in schedule)
            original_case = helper.run_case

            def execute(native, module, caller, case, truth, fixtures, slot, **kw):
                request[0] += 1
                q_counts.clear()
                return original_case(
                    native, module, caller, schedule[request[0]], truth, fixtures, slot, **kw
                )

            helper.run_case = execute
        return helper

    model.load_model = loaded
    frozen.load = module_load
    torch.profiler.profile = profile
    count = 13 if args.suite else 2
    modes = args.modes.split(",") if args.modes else ["MATH"] * count
    if len(modes) != count:
        raise ValueError("Wrong invocation schedule")
    worker = output / "worker"
    worker.mkdir()
    try:
        frozen.run(
            SimpleNamespace(
                output=worker,
                wheel=args.wheel,
                modes=",".join(modes),
                case="pages_2",
                init=False,
                trace=False,
                deep_trace=False,
                jit_unoptimized=False,
            )
        )
    finally:
        model.load_model = original_load
        frozen.load = original_module_load
        torch.profiler.profile = original_profile
        torch._C._jit_set_texpr_fuser_enabled(flag_before)
        write(
            output / "observers.json",
            dict(
                producer_sha256=sha(__file__),
                declaration_sha256=sha(args.declaration),
                declared_item=args.item,
                cases=case_names,
                q_events=q_events,
                profiles=profiles,
                stages=stages,
                te_before=flag_before,
                te_after=torch._C._jit_texpr_fuser_enabled(),
                graph_optimize_after=torch._C._get_graph_executor_optimize(),
                model_initializations=len(native_ref),
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--item", type=int, required=True)
    parser.add_argument("--option", choices=["default", "te-off", "eager"], required=True)
    parser.add_argument("--wheel", choices=["original", "candidate"], default="original")
    parser.add_argument("--modes")
    parser.add_argument("--suite", action="store_true")
    parser.add_argument("--tensors", action="store_true")
    parser.add_argument("--execute-gpu", action="store_true", required=True)
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT or not args.output.resolve().is_relative_to(ROOT / ".b/te-e2e"):
        parser.error("Primary root and fresh local evidence required")
    try:
        run(args)
    except BaseException:
        if args.output.exists():
            write(args.output / "failure.json", dict(traceback=traceback.format_exc()))
        raise
