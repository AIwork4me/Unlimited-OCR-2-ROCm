"""Opt-in physical before/after diagnostics; importing this module never imports torch.

Run from the repository root with the physically validated original environment.
Every arm loads once in a fresh process. Outputs are local unsanitized evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import shutil
import sys
import time
import traceback
import uuid
from contextlib import contextmanager, nullcontext
from functools import wraps
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_bytes(path: Path, payload: bytes) -> None:
    """Publish complete same-volume evidence atomically, refusing any overwrite."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # link creates the final name atomically and fails if that name exists.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    write_bytes(path, payload.encode("utf-8"))


def fresh_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError("Use a fresh output directory; existing evidence is immutable")
    path.mkdir(parents=True, exist_ok=True)


def callable_source(function: Any) -> dict[str, Any]:
    function = inspect.unwrap(function)
    try:
        filename = Path(inspect.getsourcefile(function) or "")
        _, line = inspect.getsourcelines(function)
        return {
            "module": function.__module__,
            "function": function.__qualname__,
            "path": str(filename),
            "line": line,
            "sha256": sha256(filename),
        }
    except (OSError, TypeError, AttributeError):
        return {"source_unavailable": True}


def guard_branch(state: dict[str, int], expected_pages: int, max_page_tokens: int) -> str:
    """Classify the exact pinned guard's branch from its post-call scalar state."""
    if expected_pages <= 0:
        return "disabled_expected_pages_zero"
    markers = state["marker_cum"]
    if markers > expected_pages:
        return "force_eos_excess_pages"
    if 0 < max_page_tokens <= state["page_len"]:
        return "force_page_at_page_cap" if markers < expected_pages else "force_eos_at_page_cap"
    return "suppress_eos_missing_pages" if markers < expected_pages else "unmodified"


class NativeObserver:
    """Observe infer's unchanged return and guard's unchanged logits/state operation."""

    def __init__(self, model: Any, module: Any, output: Path):
        self.model, self.module, self.output = model, module, output
        self.records: list[dict[str, Any]] = []
        self.guard_events: list[dict[str, Any]] = []
        self.original_infer = model.infer
        self.original_guard = getattr(module, "_uocr_page_count_guard", None)

    def __enter__(self) -> NativeObserver:
        fresh_output(self.output)
        original = self.original_infer

        @wraps(original)
        def infer(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            # Official dictionaries are JSON primitives. A deep JSON copy freezes
            # evidence without changing any object returned to the native caller.
            frozen = json.loads(json.dumps(result, ensure_ascii=False, allow_nan=False))
            index = len(self.records) + 1
            write_json(self.output / f"native-{index:03d}.json", frozen)
            if isinstance(result, dict) and isinstance(result.get("text"), str):
                write_bytes(self.output / f"native-{index:03d}.raw.txt", result["text"].encode())
            record = {
                "native": frozen,
                "source": callable_source(original),
                "options": {
                    key: value
                    for key, value in kwargs.items()
                    if key not in {"image_file", "image_files"}
                },
                "image_file": kwargs.get("image_file"),
                "identity_preserved": True,
            }
            bound = inspect.signature(original).bind(*args, **kwargs)
            bound.apply_defaults()
            record["effective_options"] = {
                key: value
                for key, value in bound.arguments.items()
                if key
                in {
                    "max_new_tokens",
                    "max_length",
                    "max_page_tokens",
                    "page_count_guard",
                    "crop_mode",
                    "return_dict",
                    "eval_mode",
                    "temperature",
                    "dash_topm",
                }
            }
            self.records.append(record)
            write_json(self.output / f"call-{index:03d}.json", record)
            return result

        self.model.infer = infer
        if self.original_guard is not None:
            original_guard = self.original_guard

            @wraps(original_guard)
            def guard(
                logits: Any,
                gen_ids: Any,
                expected_pages: int,
                page_id: int,
                eos_id: int,
                max_page_tokens: int,
                st: dict[str, int],
            ) -> Any:
                before = dict(st)
                result = original_guard(
                    logits, gen_ids, expected_pages, page_id, eos_id, max_page_tokens, st
                )
                event = {
                    "step": len(self.guard_events),
                    "generated_ids_count": len(gen_ids),
                    "expected_pages": expected_pages,
                    "page_id": page_id,
                    "eos_id": eos_id,
                    "max_page_tokens": max_page_tokens,
                    "state_before": before,
                    "state_after": dict(st),
                    "branch": guard_branch(st, expected_pages, max_page_tokens),
                }
                self.guard_events.append(event)
                # Flush each event so an interrupted/failing run retains counters.
                with (self.output / "guard.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(event, ensure_ascii=False) + "\n")
                return result

            self.module._uocr_page_count_guard = guard
        return self

    def __exit__(self, *exc: Any) -> None:
        self.model.infer = self.original_infer
        if self.original_guard is not None:
            self.module._uocr_page_count_guard = self.original_guard
        write_json(
            self.output / "observer.json",
            {
                "infer_source": callable_source(self.original_infer),
                "guard_source": callable_source(self.original_guard),
                "calls_returned": len(self.records),
                "guard_events": len(self.guard_events),
                "observation": (
                    "Original return object and guard result retained unchanged; "
                    "no logits read or edited"
                ),
            },
        )


@contextmanager
def interrupt_after_forward(decoder: Any, synchronize: Any, count: int = 3):
    """Disclosed fault injection, once, after an actual completed decoder forward."""
    original = decoder.forward
    state = {
        "completed_forward_calls": 0,
        "injected": False,
        "target": count,
        "kind": "controlled_physical_KeyboardInterrupt_after_completed_forward",
    }

    @wraps(original)
    def forward(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        state["completed_forward_calls"] += 1
        if state["completed_forward_calls"] == count and not state["injected"]:
            synchronize()
            state["injected"] = True
            raise KeyboardInterrupt("G1 controlled fault injection after completed decoder forward")
        return result

    decoder.forward = forward
    try:
        yield state
    finally:
        decoder.forward = original


def normal_cases() -> list[dict[str, Any]]:
    cases = [
        {"name": key, "truth_keys": [key], "max_new_tokens": 1024}
        for key in ("english", "chinese", "mixed", "table", "math")
    ]
    cases.append(
        {"name": "repeat_unicode_recovery", "truth_keys": ["english"], "max_new_tokens": 1024}
    )
    for count in (2, 3, 5):
        for repeat in (1, 2):
            cases.append(
                {
                    "name": f"pages_{count}_run_{repeat}",
                    "truth_keys": [f"page_{page}" for page in range(1, count + 1)],
                    "max_new_tokens": 1024 * count,
                }
            )
    cases.append(
        {
            "name": "table_2_pages",
            "truth_keys": ["table_page_1", "table_page_2"],
            "max_new_tokens": 2048,
        }
    )
    return cases


def load_before(path: Path) -> Any:
    name = "unlimited_ocr_2_rocm._g1_before"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load historical caller")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def helper_proof(before_helpers: Path, current: Path) -> dict[str, Any]:
    proof = {}
    for name in ("model.py", "output.py", "compat.py", "config.py", "doctor.py"):
        baseline, candidate = before_helpers / name, current / name
        left, right = sha256(baseline), sha256(candidate)
        if left != right:
            raise RuntimeError(f"Historical helper differs: {name}")
        proof[name] = {"before_sha256": left, "after_sha256": right, "byte_identical": True}
    return proof


def baseline_proof(source: Path, helpers: Path, current: Path, receipt: Path) -> dict[str, Any]:
    expected = json.loads(receipt.read_text(encoding="utf-8"))
    if sha256(source) != expected["inference"]:
        raise RuntimeError("Historical caller hash differs from extracted baseline receipt")
    proof = helper_proof(helpers, current)
    for name, record in proof.items():
        if record["before_sha256"] != expected[Path(name).stem]:
            raise RuntimeError(f"Historical helper hash differs from receipt: {name}")
    return {
        "caller_verified": True,
        "caller_sha256": expected["inference"],
        "receipt_sha256": sha256(receipt),
        "helpers": proof,
    }


def loaded_modules() -> list[dict[str, Any]]:
    """Read loaded Windows native images without loading project DLLs."""
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    process = kernel.GetCurrentProcess()
    modules = (wintypes.HMODULE * 4096)()
    needed = wintypes.DWORD()
    psapi.EnumProcessModulesEx.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.HMODULE),
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.DWORD,
    ]
    psapi.GetModuleFileNameExW.argtypes = [
        wintypes.HANDLE,
        wintypes.HMODULE,
        wintypes.LPWSTR,
        wintypes.DWORD,
    ]
    if not psapi.EnumProcessModulesEx(
        process, modules, ctypes.sizeof(modules), ctypes.byref(needed), 3
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    if needed.value > ctypes.sizeof(modules):
        raise RuntimeError("Loaded module inventory buffer insufficient")
    result = []
    for module in modules[: needed.value // ctypes.sizeof(wintypes.HMODULE)]:
        name = ctypes.create_unicode_buffer(32768)
        if not psapi.GetModuleFileNameExW(process, module, name, len(name)):
            raise ctypes.WinError(ctypes.get_last_error())
        path = Path(name.value)
        lower = str(path).lower()
        if (
            "site-packages" in lower and any(part in lower for part in ("torch", "rocm", "triton"))
        ) or path.name.lower().startswith(("amdhip", "amd_comgr", "hsa-runtime")):
            result.append({"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size})
    return result


def open_loaded_hip(modules: list[dict[str, Any]]) -> Any:
    import ctypes
    from ctypes import wintypes

    paths = [
        item["path"] for item in modules if Path(item["path"]).name.lower().startswith("amdhip64")
    ]
    if len(paths) != 1:
        raise RuntimeError("Expected exactly one already-loaded amdhip64 runtime")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
    kernel.GetModuleHandleW.restype = wintypes.HMODULE
    handle = kernel.GetModuleHandleW(paths[0])
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    # Existing handle avoids loading a second runtime or changing DLL search.
    api = ctypes.CDLL(paths[0], handle=handle)
    for name in ("hipRuntimeGetVersion", "hipDriverGetVersion"):
        function = getattr(api, name)
        function.argtypes, function.restype = [ctypes.POINTER(ctypes.c_int)], ctypes.c_int
    api.hipPeekAtLastError.argtypes, api.hipPeekAtLastError.restype = [], ctypes.c_int
    return api


def hip_probe(api: Any) -> dict[str, Any]:
    """Read versions and peek at sticky errors; never call hipGetLastError."""
    import ctypes

    result: dict[str, Any] = {"status": "FAIL", "clears_error": False}
    try:
        # Peek first: version queries must not conceal a pre-existing error.
        result["hipPeekAtLastError"] = int(api.hipPeekAtLastError())
        for name in ("hipRuntimeGetVersion", "hipDriverGetVersion"):
            version = ctypes.c_int()
            code = int(getattr(api, name)(ctypes.byref(version)))
            result[name] = {"return_code": code, "version": version.value}
        result["status"] = (
            "PASS"
            if (
                result["hipPeekAtLastError"] == 0
                and all(
                    result[name]["return_code"] == 0
                    for name in ("hipRuntimeGetVersion", "hipDriverGetVersion")
                )
            )
            else "FAIL"
        )
    except Exception:
        result["error"] = traceback.format_exc()
    return result


def runtime_state(native: Any, module: Any) -> dict[str, Any]:
    config = native.model.config
    decoder_module = importlib.import_module(native.model.model.__class__.__module__)
    base = getattr(decoder_module, "_base", None)
    return {
        "ring_window": getattr(config, "_ring_window", None),
        "sliding_window": getattr(config, "sliding_window", None),
        "dash_infer_on": dict(getattr(module, "_DASH_INFER", {})).get("on"),
        "page_scope_prefill": getattr(base, "_PAGE_SCOPE_PREFILL", None),
        "routers": [
            {
                "type": router.__class__.__name__,
                "use_triton": bool(router.use_triton),
                "stage0": router.stage0,
            }
            for router in native.model.modules()
            if router.__class__.__name__ == "DashAttention"
        ],
    }


def hash_manifest(output: Path) -> None:
    write_json(
        output / "hashes.json",
        {
            path.relative_to(output).as_posix(): sha256(path)
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name != "hashes.json"
        },
    )


def compare_case(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    differences = []
    required = {
        "text",
        "token_ids",
        "prompt_token_ids",
        "finish_reason",
        "num_pages_out",
        "expected_pages",
    }
    if any(
        not required <= item.get("native", {}).keys() or not item.get("raw_sha256")
        for item in (before, after)
    ):
        differences.append("missing_native_evidence")
    for key in (
        "text",
        "token_ids",
        "prompt_token_ids",
        "finish_reason",
        "num_pages_out",
        "expected_pages",
        "prompt",
    ):
        if before.get("native", {}).get(key) != after.get("native", {}).get(key):
            differences.append("native." + key)
    for key in ("raw_sha256", "kernels"):
        if before.get(key) != after.get(key):
            differences.append(key)
    ground_truth_pass = all(
        item.get("correctness", {}).get("status") == "PASS" for item in (before, after)
    )
    if not ground_truth_pass:
        differences.append("ground_truth_gate")
    if any(item.get("status") != "PASS" for item in (before, after)):
        differences.append("wrapper_status")
    if any(
        not item.get("kernels")
        or item.get("runtime_state_restored") is not True
        or item.get("physical_contract") != "PASS"
        for item in (before, after)
    ):
        differences.append("physical_contract")
    return {
        "status": "PASS" if not differences else "FAIL",
        "differences": differences,
        "ground_truth_pass": ground_truth_pass,
        "timing_comparison": "recorded, no tolerance gate",
        "wrapper_reported_finish": {
            "before": before.get("wrapper", {}).get("finish_reason"),
            "after": after.get("wrapper", {}).get("finish_reason"),
        },
    }


def read_case(output: Path) -> dict[str, Any]:
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    native_path = output / "diagnostics/native-001.json"
    raw_path = output / "diagnostics/native-001.raw.txt"
    if not raw_path.is_file() or sha256(raw_path) != summary.get("raw_sha256"):
        raise ValueError(f"Missing or changed raw artifact: {output.name}")
    native = json.loads(native_path.read_text(encoding="utf-8"))
    if native != summary.get("native") or raw_path.read_bytes() != native["text"].encode():
        raise ValueError(f"Native dictionary and raw artifact disagree: {output.name}")
    return summary


def control_contracts(summaries: dict[str, Any]) -> dict[str, str]:
    maximum = summaries.get("control_max_length_prompt_plus_8", {})
    cap = summaries.get("control_page_cap_8", {})
    length_pass = (
        maximum.get("status") == "DIAGNOSTIC_ONLY"
        and maximum.get("physical_contract") == "PASS"
        and maximum.get("native", {}).get("finish_reason") == "length"
        and len(maximum.get("native", {}).get("token_ids", [])) == 8
        and bool(maximum.get("raw_sha256"))
    )
    branches = cap.get("guard_branch_counts", {})
    cap_pass = (
        cap.get("status") == "DIAGNOSTIC_ONLY"
        and cap.get("physical_contract") == "PASS"
        and cap.get("native", {}).get("finish_reason") == "stop"
        and (branches.get("force_page_at_page_cap", 0) + branches.get("force_eos_at_page_cap", 0))
        > 0
        and cap.get("correctness", {}).get("status") != "PASS"
        and bool(cap.get("raw_sha256"))
    )
    return {
        "max_length_control": "PASS" if length_pass else "FAIL",
        "page_cap_control": "PASS" if cap_pass else "FAIL",
    }


def comparison_arm_gate(before: Path, after: Path) -> dict[str, Any]:
    differences = []
    receipts = []
    for directory, expected_arm in ((before, "before"), (after, "after")):
        receipt = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        suite = json.loads((directory / "suite.json").read_text(encoding="utf-8"))
        receipts.append(receipt)
        if receipt.get("arm") != expected_arm:
            differences.append(expected_arm + ".arm_identity")
        if suite.get("status") != "PASS":
            differences.append(expected_arm + ".suite_status")
        manifest = directory / "hashes.json"
        if not manifest.is_file():
            differences.append(expected_arm + ".missing_hash_manifest")
        else:
            for relative, expected in json.loads(manifest.read_text(encoding="utf-8")).items():
                path = directory / relative
                if not path.is_file() or sha256(path) != expected:
                    differences.append(expected_arm + ".artifact_hash:" + relative)
    if receipts[0].get("harness_sha256") != receipts[1].get("harness_sha256"):
        differences.append("harness_source_changed_between_arms")
    return {"status": "PASS" if not differences else "FAIL", "differences": differences}


def suite_contract(
    arm: str, summaries: dict[str, Any], retry_comparison: Any, controls_requested: bool = False
) -> dict[str, Any]:
    normal_pass = all(
        summaries.get(case["name"], {}).get("correctness", {}).get("status") == "PASS"
        and summaries[case["name"]].get("status") == "PASS"
        and summaries[case["name"]].get("runtime_state_restored") is True
        and summaries[case["name"]].get("physical_contract") == "PASS"
        for case in normal_cases()
    )
    length = summaries.get("length_2_pages_8_tokens", {})
    length_native = length.get("native", {})
    length_pass = (
        length.get("status") == "FAIL"
        and length_native.get("finish_reason") == "length"
        and len(length_native.get("token_ids", [])) == 8
        and bool(length.get("raw_sha256"))
        and length.get("runtime_state_restored") is True
        and length.get("physical_contract") == "PASS"
        and "telemetry_error" not in length
    )
    checks = {
        "normal_acceptance": "PASS" if normal_pass else "FAIL",
        "length_failure_contract": "PASS" if length_pass else "FAIL",
    }
    if arm == "after":
        interrupted = summaries.get("controlled_interrupt_2_pages", {})
        injection = interrupted.get("fault_injection") or {}
        interrupted_pass = (
            interrupted.get("status") == "FAIL"
            and interrupted.get("error") == "INTERRUPTED"
            and injection.get("injected") is True
            and injection.get("completed_forward_calls") == 3
            and interrupted.get("runtime_state_restored") is True
            and interrupted.get("physical_contract") == "PASS"
            and "telemetry_error" not in interrupted
            and "native" not in interrupted
        )
        checks["interruption_failure_contract"] = "PASS" if interrupted_pass else "FAIL"
        retry_pass = retry_comparison is not None and retry_comparison.get("status") == "PASS"
        checks["retry_identity_contract"] = "PASS" if retry_pass else "FAIL"
    if controls_requested:
        checks.update(control_contracts(summaries))
    return {
        "status": "PASS" if all(value == "PASS" for value in checks.values()) else "FAIL",
        **checks,
        "cases": {
            name: {"status": value["status"], "correctness": value.get("correctness")}
            for name, value in summaries.items()
        },
    }


def run_case(
    native: Any,
    module: Any,
    caller: Any,
    case: dict[str, Any],
    truth_map: dict[str, Any],
    fixtures: Path,
    output: Path,
    *,
    interrupt: bool = False,
    direct_options: dict[str, Any] | None = None,
    hip: Any = None,
) -> dict[str, Any]:
    import torch

    from unlimited_ocr_2_rocm.completion import completion_metadata, require_complete_pages
    from unlimited_ocr_2_rocm.doctor import memory_snapshot
    from unlimited_ocr_2_rocm.output import assess

    fresh_output(output)
    truths = [truth_map[key] for key in case["truth_keys"]]
    images = [fixtures / truth["image"] for truth in truths]
    for image, truth in zip(images, truths, strict=True):
        if sha256(image) != truth["sha256"]:
            raise RuntimeError("Fixture SHA256 mismatch")
    if case["name"] == "repeat_unicode_recovery":
        unicode_dir = output / "中文 input"
        unicode_dir.mkdir()
        image = unicode_dir / "page with spaces.png"
        shutil.copyfile(images[0], image)
        try:
            caller.validate_inputs([unicode_dir / "missing.png"], multi=False)
        except FileNotFoundError:
            pass
        else:
            raise RuntimeError("Missing-input guard failed")
        images = [image]
    kernel_start, vision_start = len(native.kernels), len(native.vision)
    summary: dict[str, Any] = {
        "case": case,
        "status": "FAIL",
        "state_before": runtime_state(native, module),
        "caller_source": callable_source(caller.run_native),
        "native_source_map": {
            name: callable_source(getattr(native.model, name))
            for name in ("infer", "infer_multi", "_uocr_generate")
        },
        "direct_diagnostic_options": direct_options,
        "hip_before": hip_probe(hip),
    }
    context = (
        interrupt_after_forward(native.model.model, torch.cuda.synchronize)
        if interrupt
        else nullcontext(None)
    )
    start = time.perf_counter()
    observer = NativeObserver(native.model, module, output / "diagnostics")
    try:
        # HIP-before was sampled before any initial CUDA telemetry call, and
        # failed initial synchronization must still produce a complete FAIL file.
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        summary["memory_before"] = memory_snapshot()
        with observer, context as injection:
            summary["fault_injection"] = injection
            if direct_options is None:
                summary["wrapper"] = caller.run_native(
                    native,
                    images,
                    output / "wrapper",
                    multi=len(images) > 1,
                    truth=truths,
                    max_new_tokens=case["max_new_tokens"],
                )
                summary["status"] = summary["wrapper"]["status"]
            else:
                with torch.inference_mode():
                    native.model.infer(
                        native.tokenizer,
                        image_file=[str(p) for p in images],
                        crop_mode=False,
                        save_results=False,
                        eval_mode=True,
                        return_dict=True,
                        max_new_tokens=case["max_new_tokens"],
                        **direct_options,
                    )
                summary["status"] = "DIAGNOSTIC_ONLY"
    except KeyboardInterrupt:
        summary.update(status="FAIL", error="INTERRUPTED", traceback=traceback.format_exc())
        # The official generation loop did not return: do not invent partial text/IDs.
        summary["partial_native_return"] = "unavailable; official loop raised before return"
    except Exception:
        summary.update(status="FAIL", traceback=traceback.format_exc())
    finally:
        try:
            torch.cuda.synchronize()
            summary["memory_after"] = memory_snapshot()
        except Exception:
            summary["status"] = "FAIL"
            summary["telemetry_error"] = traceback.format_exc()
        summary.update(
            state_after=runtime_state(native, module),
            synchronized_wall_seconds=time.perf_counter() - start,
            kernels=native.kernels[kernel_start:],
            vision=native.vision[vision_start:],
        )
        summary["runtime_state_restored"] = summary["state_before"] == summary["state_after"]
        summary["hip_after"] = hip_probe(hip)
        summary["guard_branch_counts"] = {
            branch: sum(event["branch"] == branch for event in observer.guard_events)
            for branch in sorted({event["branch"] for event in observer.guard_events})
        }
        if observer.records:
            summary["effective_options"] = observer.records[-1]["effective_options"]
        vision = summary["vision"]
        kernel_pass = len(summary["kernels"]) >= 12 and all(
            str(item.get("device", "")).startswith("cuda")
            and item.get("dtype") == "torch.bfloat16"
            and item.get("synchronized") is True
            for item in summary["kernels"]
        )
        vision_pass = {item.get("module") for item in vision} == {
            "sam_model",
            "vision_model",
        } and all(
            item.get("status") == "PASS" and item.get("backend") == "GPU MATH SDPA"
            for item in vision
        )
        hip_pass = all(summary[key]["status"] == "PASS" for key in ("hip_before", "hip_after"))
        guard_pass = bool(observer.guard_events) and all(
            event["expected_pages"] == len(images)
            and event["max_page_tokens"] == (direct_options or {}).get("max_page_tokens", 20000)
            for event in observer.guard_events
        )
        summary["physical_contract"] = (
            "PASS"
            if kernel_pass
            and vision_pass
            and hip_pass
            and guard_pass
            and summary["runtime_state_restored"]
            and "telemetry_error" not in summary
            else "FAIL"
        )
        if not hip_pass:
            summary["status"] = "FAIL"
        if observer.records:
            summary["native"] = result = observer.records[-1]["native"]
            raw_path = output / "diagnostics/native-001.raw.txt"
            if raw_path.is_file():
                summary["raw_sha256"] = sha256(raw_path)
            # Independent post-observation gate uses the same fixed acceptance
            # rules for BOTH callers, while retaining their original own status.
            try:
                metadata = completion_metadata(
                    result, expected_pages=len(images), max_new_tokens=case["max_new_tokens"]
                )
                summary["validated_metadata"] = metadata
                if metadata["finish_reason"] == "length":
                    raise ValueError("Native length truncation")
                require_complete_pages(result["text"], len(images), max_chars=1_000_000)
                summary["correctness"] = assess(result["text"], truths)
            except Exception:
                summary["acceptance_error"] = traceback.format_exc()
        write_json(output / "summary.json", summary)
        hash_manifest(output)
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--arm", choices=("before", "after"))
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--cache-dir", type=Path, default=Path(".b/h"))
    result.add_argument("--identity", type=Path)
    result.add_argument("--before-source", type=Path, default=Path(".b/g1/before-inference.py"))
    result.add_argument("--before-helpers", type=Path, default=Path(".b/g1/before-helpers"))
    result.add_argument(
        "--baseline-hashes", type=Path, default=Path(".b/g1/baseline-source-hashes.json")
    )
    result.add_argument("--fixtures", type=Path, default=Path("tests/fixtures"))
    result.add_argument("--controls", action="store_true")
    result.add_argument("--compare", type=Path, nargs=2, metavar=("BEFORE", "AFTER"))
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    if Path.cwd().resolve() != root:
        raise RuntimeError("Run only from the primary repository root")
    if not args.output.resolve().is_relative_to(root):
        raise ValueError("Evidence must stay inside repository root")
    fresh_output(args.output)
    if args.compare:
        arms = comparison_arm_gate(*args.compare)
        comparison = {}
        for case in normal_cases():
            before, after = [read_case(arm / case["name"]) for arm in args.compare]
            comparison[case["name"]] = compare_case(before, after)
        write_json(args.output / "comparison.json", comparison)
        write_json(args.output / "arm-gate.json", arms)
        hash_manifest(args.output)
        return (
            0
            if (
                arms["status"] == "PASS"
                and all(row["status"] == "PASS" for row in comparison.values())
            )
            else 1
        )
    if args.arm is None:
        raise ValueError("--arm or --compare is required")
    proof = baseline_proof(
        args.before_source,
        args.before_helpers,
        root / "src/unlimited_ocr_2_rocm",
        args.baseline_hashes,
    )
    from unlimited_ocr_2_rocm import inference
    from unlimited_ocr_2_rocm.model import load_model

    caller = load_before(args.before_source) if args.arm == "before" else inference
    receipt = {
        "arm": args.arm,
        "helper_proof": proof,
        "caller": callable_source(caller.run_native),
        "harness_sha256": sha256(Path(__file__)),
        "identity_receipt": None,
        "memory_meanings": {
            "gpu": (
                "APU allocator allocated/reserved and peaks; not dedicated VRAM or total system use"
            ),
            "process": "RSS working set and peak working set; not GPU-only usage",
        },
    }
    if args.identity:
        receipt["identity_receipt"] = {
            "path": str(args.identity),
            "sha256": sha256(args.identity),
            "content": json.loads(args.identity.read_text(encoding="utf-8")),
        }
    write_json(args.output / "run.json", receipt)
    truths = json.loads((args.fixtures / "ground_truth.json").read_text(encoding="utf-8"))
    write_json(args.output / "ground_truth.json", truths)
    native = load_model(args.cache_dir, offline=True)
    module = importlib.import_module(native.model.__class__.__module__)
    modules = loaded_modules()
    write_json(args.output / "loaded-modules-after-load.json", modules)
    hip = open_loaded_hip(modules)
    prefix = native.model.__class__.__module__.rsplit(".", 1)[0]
    write_json(
        args.output / "loaded-native-sources.json",
        {
            name: {
                "path": str(Path(importlib.import_module(prefix + "." + name).__file__)),
                "sha256": sha256(Path(importlib.import_module(prefix + "." + name).__file__)),
            }
            for name in (
                "modeling_unlimitedocr",
                "modeling_deepseekv2",
                "deepencoder",
                "dash_attention",
                "sparse_gather_e2e_triton",
            )
        },
    )
    write_json(args.output / "load.json", native.evidence)
    write_json(args.output / "topology.json", runtime_state(native, module))
    summaries = {}
    for case in normal_cases():
        print(f"G1 {args.arm}: {case['name']}", flush=True)
        summaries[case["name"]] = run_case(
            native,
            module,
            caller,
            case,
            truths["cases"],
            args.fixtures,
            args.output / case["name"],
            hip=hip,
        )
    short = {
        "name": "length_2_pages_8_tokens",
        "truth_keys": ["page_1", "page_2"],
        "max_new_tokens": 8,
    }
    summaries[short["name"]] = run_case(
        native,
        module,
        caller,
        short,
        truths["cases"],
        args.fixtures,
        args.output / short["name"],
        hip=hip,
    )
    retry_comparison = None
    if args.arm == "after":
        normal = next(case for case in normal_cases() if case["name"] == "pages_2_run_1")
        interrupted = {**normal, "name": "controlled_interrupt_2_pages"}
        summaries[interrupted["name"]] = run_case(
            native,
            module,
            caller,
            interrupted,
            truths["cases"],
            args.fixtures,
            args.output / interrupted["name"],
            interrupt=True,
            hip=hip,
        )
        retry = {**normal, "name": "retry_after_controlled_interrupt_2_pages"}
        summaries[retry["name"]] = run_case(
            native,
            module,
            caller,
            retry,
            truths["cases"],
            args.fixtures,
            args.output / retry["name"],
            hip=hip,
        )
        retry_comparison = compare_case(summaries[normal["name"]], summaries[retry["name"]])
        write_json(args.output / "retry-comparison.json", retry_comparison)
    if args.controls:
        prompt_length = len(
            summaries["pages_2_run_1"].get("native", {}).get("prompt_token_ids", [])
        )
        if not prompt_length:
            raise RuntimeError("Cannot lower max_length without observed native prompt length")
        for name, options in (
            ("control_max_length_prompt_plus_8", {"max_length": prompt_length + 8}),
            ("control_page_cap_8", {"max_page_tokens": 8}),
        ):
            case = {"name": name, "truth_keys": ["page_1", "page_2"], "max_new_tokens": 64}
            summaries[name] = run_case(
                native,
                module,
                caller,
                case,
                truths["cases"],
                args.fixtures,
                args.output / name,
                direct_options=options,
                hip=hip,
            )
    suite = suite_contract(args.arm, summaries, retry_comparison, args.controls)
    write_json(args.output / "suite.json", suite)
    write_json(args.output / "loaded-modules-after-suite.json", loaded_modules())
    hash_manifest(args.output)
    return 0 if suite["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    raise SystemExit(main())
