"""Explicitly gated GPU child; importing it never imports torch or model code."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import json
import shutil
import sys
import time
from functools import wraps
from pathlib import Path
from typing import Any

from protocol import BASE, CANDIDATE, digest, file_hash, publish, read_plan


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load frozen diagnostic helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def execution_gate(plan: dict[str, Any], receipt: Path) -> None:
    gate = json.loads(receipt.read_text(encoding="utf-8"))
    if gate.get("matrix", {}).get("P0") != "PASS":
        raise ValueError("Physical execution requires an explicit P0 PASS receipt")
    if gate.get("source_commit") != BASE or plan.get("base_source_commit") != BASE:
        raise ValueError("P0 receipt source does not match the integration base")
    if not __debug__:
        raise ValueError("Optimized Python disables historical assertion gates")


def run(plan_path: Path, output: Path, phase: str, gate: Path) -> None:
    plan = read_plan(plan_path, Path.cwd().resolve(), Path(__file__).resolve().parents[2])
    execution_gate(plan, gate)
    root, work = Path(plan["root"]), Path(plan["worktree"])
    if Path.cwd().resolve() != root.resolve():
        raise ValueError("Run only from the primary repository root")
    if output.exists() or not output.resolve().is_relative_to(root.resolve()):
        raise ValueError("Use a fresh repository-local worker output")
    output.mkdir(parents=True)
    for path, expected in plan["source_hashes"].items():
        if file_hash(Path(path)) != expected:
            raise ValueError("Source/helper changed after preparation")
    sys.path.insert(0, str(work / "src"))
    refs = Path(plan["references"])
    diagnostic = load_module("p2_frozen_diagnostic", refs / "validate.py")
    observer = load_module("p2_frozen_observer", refs / "observer.py")
    compat = load_module("p2_frozen_compat", refs / "compat.py")
    diagnostic.verify_candidate(Path(plan["identity_profile"]))
    import torch

    from unlimited_ocr_2_rocm.completion import require_complete_pages
    from unlimited_ocr_2_rocm.inference import run_native
    from unlimited_ocr_2_rocm.model import load_model

    before_load = time.perf_counter()
    native = load_model(Path(plan["cache_dir"]), offline=True)
    loaded = observer.loaded_modules()
    profile = json.loads(Path(plan["identity_profile"]).read_text(encoding="utf-8"))
    expected_modules = {Path(k).name.lower(): v for k, v in profile["binaries"].items()}
    observed_modules = {Path(r["path"]).name.lower(): r for r in loaded}
    for name, expected in expected_modules.items():
        if name not in observed_modules:
            raise ValueError("Required candidate native module is not loaded: " + name)
        member = observed_modules[name]
        if member["sha256"] != expected or not Path(member["path"]).resolve().is_relative_to(
            Path(sys.prefix).resolve()
        ):
            raise ValueError("Loaded candidate native identity mismatch: " + name)
    publish(
        output / "load.json",
        dict(
            seconds=time.perf_counter() - before_load, model=native.evidence, loaded_modules=loaded
        ),
    )
    originals = {
        name: inspect.unwrap(getattr(native.model.model, name).forward)
        for name in ("sam_model", "vision_model")
    }
    hip = observer.open_loaded_hip(loaded)
    truths = json.loads(Path(plan["ground_truth"]).read_text(encoding="utf-8"))["cases"]
    identity = dict(
        wheel_sha256=CANDIDATE,
        model_sha256=digest(native.evidence["source_and_weight_hashes"]),
        config_sha256=digest(plan["generation"]),
        producer_sha256=digest(plan["source_hashes"]),
        stack_sha256=digest(
            dict(
                packages=native.evidence["environment"]["packages"],
                dependencies=native.evidence["native_dependencies"],
                modules={k: v["sha256"] for k, v in observed_modules.items()},
            )
        ),
    )
    for item in plan["schedule"]:
        if (item["phase"] == "diagnostic") != (phase == "diagnostic"):
            continue
        trial_dir = output / item["id"]
        trial_dir.mkdir()
        publish(trial_dir / "attempt.json", dict(item, state="RUNNING"))
        case = plan["cases"][item["case"]]
        images = [Path(path) for path in case["images"]]
        for image, expected in zip(images, case["hashes"], strict=True):
            if file_hash(image) != expected:
                raise ValueError("Synthetic input changed after preparation")
        if file_hash(Path(plan["ground_truth"])) != plan["ground_truth_sha256"]:
            raise ValueError("Synthetic ground truth changed after preparation")
        if item["case"] == "repeat_unicode_recovery":
            image = trial_dir / "中文 input" / "page with spaces.png"
            image.parent.mkdir()
            shutil.copyfile(images[0], image)
            images = [image]
        for name, original in originals.items():
            getattr(native.model.model, name).forward = compat.wrap_vision_forward(
                original, name, native.vision, backend=item["mode"]
            )
        calls = diagnostic.instrument(native, phase == "diagnostic", trial_dir)
        measured_dispatch: dict[str, set[str]] = {name: set() for name in originals}
        if phase != "diagnostic":
            from torch.utils._python_dispatch import TorchDispatchMode

            def observe(forward: Any, name: str, events: Any = measured_dispatch) -> Any:
                class Dispatch(TorchDispatchMode):
                    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                        backend = diagnostic.backend_name([str(func)])
                        if backend != "UNOBSERVED":
                            events[name].add(backend)
                        return func(*args, **(kwargs or {}))

                @wraps(forward)
                def observed(*args: Any, **kwargs: Any) -> Any:
                    with Dispatch():
                        return forward(*args, **kwargs)

                return observed

            for name in originals:
                encoder = getattr(native.model.model, name)
                encoder.forward = observe(encoder.forward, name)
        actual = []
        original_infer = native.model.infer

        @wraps(original_infer)
        def capture(
            *args: Any, _original: Any = original_infer, _records: Any = actual, **kwargs: Any
        ) -> Any:
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = _original(*args, **kwargs)
            torch.cuda.synchronize()
            _records.append((result, time.perf_counter() - started))
            return result

        row = dict(item, **identity, status="FAIL")
        native.model.infer = capture
        try:
            torch.manual_seed(plan["seed"])
            hip_before = observer.hip_probe(hip)
            report = run_native(
                native,
                images,
                trial_dir / "private-output",
                multi=len(images) > 1,
                truth=[truths[k] for k in case["truth_keys"]],
                max_new_tokens=case["max_new_tokens"],
            )
            hip_after = observer.hip_probe(hip)
            if len(actual) != 1:
                raise ValueError("Expected exactly one unchanged native infer call")
            result, elapsed = actual[0]
            publish(trial_dir / "private-native.json", result)
            require_complete_pages(result["text"], len(images), max_chars=1_000_000)
            memory = report["memory_after"]
            row.update(
                status=report["status"],
                input_sha256=digest(case["hashes"]),
                raw_sha256=report["raw_sha256"],
                prompt_ids_sha256=digest(result["prompt_token_ids"]),
                output_ids_sha256=digest(result["token_ids"]),
                output_tokens=len(result["token_ids"]),
                finish_reason=report["finish_reason"],
                structure_pass=True,
                correctness_pass=report.get("correctness", {}).get("status") == "PASS",
                dash_calls=report["native_kernel_count"],
                hip_zero=hip_before["status"] == hip_after["status"] == "PASS",
                inference_seconds=elapsed,
                e2e_seconds=report["synchronized_wall_seconds"],
                vision_seconds=sum(e["synchronized_seconds"] for e in report["vision_inputs"]),
                peak_allocated_bytes=memory["gpu_peak_allocated_bytes"],
                peak_reserved_bytes=memory["gpu_peak_reserved_bytes"],
                process_rss_bytes=memory["process_rss_bytes"],
            )
            if phase == "diagnostic":
                row["dispatch"] = {
                    name: sorted(
                        {
                            e.get("observed_backend", "UNOBSERVED")
                            for e in calls
                            if e["module"] == name
                        }
                    )
                    for name in originals
                }
                row["numerical_pass"] = bool(calls) and all(e["status"] == "PASS" for e in calls)
            else:
                row["dispatch"] = {
                    name: sorted(values) for name, values in measured_dispatch.items()
                }
        finally:
            native.model.infer = original_infer
            publish(trial_dir / "trial.json", row)
        print(item["id"], row["status"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("diagnostic", "measurement"), required=True)
    parser.add_argument("--p0-receipt", type=Path, required=True)
    parser.add_argument("--execute-gpu", action="store_true", required=True)
    args = parser.parse_args()
    run(args.plan, args.output, args.phase, args.p0_receipt)
