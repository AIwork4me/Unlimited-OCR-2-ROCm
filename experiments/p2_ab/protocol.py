"""Synthetic-only, fail-closed candidate-wheel A/B design and metric export."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import statistics
from pathlib import Path
from typing import Any

CANDIDATE = "6a67d15861538ee217afdccd1ccc09ff318d7f44bff29a10f74c0d51730e6e1e"
REFERENCE = "cfd684224fd4b306cdc22837b4abcf9c432a1c35"
MODES = ("math", "auto")
BASE = "82951c73b28bf735f5655e6f5c3306cd170a21c2"
TRIAL_KEYS = ("id", "phase", "case", "repeat", "mode")


def definitions() -> dict[str, list[str]]:
    result = {k: [k] for k in ("english", "chinese", "mixed", "table", "math")}
    result["repeat_unicode_recovery"] = ["english"]
    for count in (2, 3, 5):
        result[f"pages_{count}"] = [f"page_{i}" for i in range(1, count + 1)]
    result["table_2_pages"] = ["table_page_1", "table_page_2"]
    return result


def read_plan(path: Path, root: Path, work: Path) -> dict[str, Any]:
    import subprocess

    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
        seal = json.loads(path.with_name("public-preparation.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("Missing or invalid sealed preparation") from error
    if seal.get("protocol_sha256") != digest(plan):
        raise ValueError("Prepared plan hash changed")
    if (
        plan.get("root") != str(root.resolve())
        or plan.get("worktree") != str(work.resolve())
        or plan.get("base_source_commit") != BASE
        or plan.get("workload") != "SYNTHETIC_ONLY"
        or plan.get("real_documents") != "BLOCKED_DATA"
        or plan.get("references") != str((path.parent / "references").resolve())
        or plan.get("schedule") != schedule(list(definitions()), plan["repeats"], plan["seed"])
    ):
        raise ValueError("Plan is outside the fixed synthetic protocol")
    gt_path = work / "tests/fixtures/ground_truth.json"
    locked_gt = json.loads(
        subprocess.check_output(
            ["git", "show", BASE + ":tests/fixtures/ground_truth.json"], cwd=root
        )
    )
    if json.loads(gt_path.read_text(encoding="utf-8")) != locked_gt:
        raise ValueError("Synthetic ground truth differs from immutable source")
    if (
        plan.get("ground_truth") != str(gt_path)
        or file_hash(gt_path) != plan["ground_truth_sha256"]
    ):
        raise ValueError("Synthetic ground truth binding changed")
    cases = {}
    for name, keys in definitions().items():
        truths = [locked_gt["cases"][key] for key in keys]
        images = [gt_path.parent / truth["image"] for truth in truths]
        hashes = [file_hash(p) for p in images]
        if hashes != [truth["sha256"] for truth in truths]:
            raise ValueError("Original synthetic input changed")
        cases[name] = dict(
            images=list(map(str, images)),
            hashes=hashes,
            truth_keys=keys,
            max_new_tokens=1024 * len(keys),
        )
    if plan.get("cases") != cases:
        raise ValueError("Only immutable original synthetic inputs are authorized")
    refs = path.parent / "references"
    for commit, source, name in (
        (REFERENCE, "experiments/native_ocr/validate.py", "validate.py"),
        (REFERENCE, "src/unlimited_ocr_2_rocm/compat.py", "compat.py"),
        (BASE, "experiments/native_regression/runner.py", "observer.py"),
    ):
        original = subprocess.check_output(["git", "show", commit + ":" + source], cwd=root)
        if (refs / name).read_bytes() != original:
            raise ValueError("Diagnostic helper differs from immutable Git blob")
    sources = list((work / "src").rglob("*.py")) + list((work / "experiments/p2_ab").glob("*.py"))
    sources += list(refs.glob("*.py")) + [Path(plan["identity_profile"]).resolve(), gt_path]
    sources += [work / "src/unlimited_ocr_2_rocm/upstream-lock.json"]
    if plan.get("source_hashes") != {str(p.resolve()): file_hash(p) for p in sources}:
        raise ValueError("Source/profile/helper set changed after preparation")
    return plan


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()


def file_hash(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def validate_timeout(value: float) -> None:
    if not math.isfinite(value) or not 0 < value <= 7200:
        raise ValueError("Timeout must be finite and between 0 and 7200 seconds")


def publish(path: Path, value: Any) -> None:
    """Atomic create-only publication; never overwrite a prior receipt."""
    import os
    import uuid

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write((json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def schedule(cases: list[str], repeats: int = 6, seed: int = 0) -> list[dict[str, Any]]:
    if repeats < 6 or repeats % 2 or not cases or len(set(cases)) != len(cases):
        raise ValueError("Require unique cases and an even repeat count >=6")
    if any(re.fullmatch(r"[a-z0-9_]+", case) is None for case in cases):
        raise ValueError("Unsafe case name")
    order = list(cases)
    random.Random(seed).shuffle(order)
    rows = []
    for phase in ("diagnostic", "warmup", "measurement"):
        for repeat in range(repeats if phase == "measurement" else 1):
            for index, case in enumerate(order):
                modes = MODES[::-1] if phase == "measurement" and (repeat + index) % 2 else MODES
                for mode in modes:
                    rows.append(
                        dict(
                            id=f"{phase}_{case}_{repeat}_{mode}",
                            phase=phase,
                            case=case,
                            repeat=repeat,
                            mode=mode,
                        )
                    )
    return rows


def distribution(values: list[float]) -> dict[str, float]:
    return dict(
        median=statistics.median(values),
        minimum=min(values),
        maximum=max(values),
        sample_stdev=statistics.stdev(values) if len(values) > 1 else 0.0,
    )


def summarize(plan: list[dict[str, Any]], trials: list[dict[str, Any]]) -> dict[str, Any]:
    errors = []
    if any(set(row) != set(TRIAL_KEYS) for row in plan):
        errors.append("unknown_schedule_field")
    if not plan or [r.get("id") for r in trials] != [r["id"] for r in plan]:
        errors.append("empty_plan_or_execution_order")
    for key in ("model_sha256", "config_sha256", "producer_sha256", "stack_sha256"):
        identities = {r.get(key) for r in trials}
        if len(identities) != 1 or any(
            not isinstance(h, str) or re.fullmatch("[0-9a-f]{64}", h) is None for h in identities
        ):
            errors.append("model_config_source_or_stack_mismatch")
    expected = {row["id"]: row for row in plan}
    received = {row.get("id"): row for row in trials}
    if len(received) != len(trials) or received.keys() != expected.keys():
        errors.append("missing_duplicate_or_extra_trial")
    for name, trial in received.items():
        if name not in expected:
            continue
        if any(trial.get(key) != value for key, value in expected[name].items()):
            errors.append("schedule_mismatch")
        if (
            trial.get("status") != "PASS"
            or trial.get("wheel_sha256") != CANDIDATE
            or trial.get("finish_reason") != "stop"
            or trial.get("dash_calls") != 12
            or any(
                trial.get(k) is not True for k in ("structure_pass", "correctness_pass", "hip_zero")
            )
        ):
            errors.append("quality_identity_or_execution_gate")
        for key in ("inference_seconds", "vision_seconds", "e2e_seconds"):
            value = trial.get(key)
            if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
                errors.append("invalid_timing")
        for key in ("peak_allocated_bytes", "peak_reserved_bytes", "process_rss_bytes"):
            value = trial.get(key)
            if type(value) is not int or value < 0:
                errors.append("invalid_memory")
        wanted = (
            {"sam_model": ["MATH"], "vision_model": ["MATH"]}
            if trial["mode"] == "math"
            else {"sam_model": ["EFFICIENT_ATTENTION"], "vision_model": ["FLASH_ATTENTION"]}
        )
        if trial.get("dispatch") != wanted:
            errors.append("actual_trial_dispatch_gate")
        if trial.get("phase") == "diagnostic" and trial.get("numerical_pass") is not True:
            errors.append("numerical_gate")
    for case in {row["case"] for row in plan}:
        rows = [row for row in trials if row.get("case") == case]
        for key in ("input_sha256", "raw_sha256", "prompt_ids_sha256", "output_ids_sha256"):
            hashes = {row.get(key) for row in rows}
            if len(hashes) != 1 or any(
                not isinstance(h, str) or re.fullmatch("[0-9a-f]{64}", h) is None for h in hashes
            ):
                errors.append("synthetic_pair_parity")
        counts = {row.get("output_tokens") for row in rows}
        if len(counts) != 1 or any(type(n) is not int or n <= 0 for n in counts):
            errors.append("unexpected_token_count")
        measured = [r for r in rows if r.get("phase") == "measurement"]
        if measured:
            peaks = {
                mode: max(
                    (r.get("peak_allocated_bytes", -1) for r in measured if r.get("mode") == mode),
                    default=-1,
                )
                for mode in MODES
            }
            if peaks["auto"] > peaks["math"] * 1.05:
                errors.append("auto_peak_allocation_degradation_over_5_percent")
    result: dict[str, Any] = dict(
        status="FAIL" if errors else "PASS",
        workload="SYNTHETIC_ONLY",
        real_documents="BLOCKED_DATA",
        wheel_sha256=CANDIDATE,
        expected_trials=len(plan),
        received_trials=len(trials),
        failures=sorted(set(errors)),
        cases={},
        trials=[
            dict(
                {k: item[k] for k in TRIAL_KEYS},
                status="PASS" if received.get(item["id"], {}).get("status") == "PASS" else "FAIL",
                metrics={
                    key: value if type(value) in (int, float) and math.isfinite(value) else None
                    for key in (
                        "inference_seconds",
                        "vision_seconds",
                        "e2e_seconds",
                        "peak_allocated_bytes",
                        "peak_reserved_bytes",
                        "process_rss_bytes",
                    )
                    for value in [received.get(item["id"], {}).get(key)]
                },
                hashes={
                    key: value
                    if isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value)
                    else None
                    for key in (
                        "input_sha256",
                        "raw_sha256",
                        "prompt_ids_sha256",
                        "output_ids_sha256",
                        "model_sha256",
                        "stack_sha256",
                    )
                    for value in [received.get(item["id"], {}).get(key)]
                },
            )
            for item in plan
        ],
    )
    if errors:
        return result
    if all(r["phase"] == "diagnostic" for r in plan):
        return result
    for case in sorted({row["case"] for row in plan}):
        measured = [r for r in trials if r["case"] == case and r["phase"] == "measurement"]
        entry = {}
        for mode in MODES:
            samples = [r for r in measured if r["mode"] == mode]
            entry[mode] = dict(
                samples=len(samples),
                inference_seconds=distribution([r["inference_seconds"] for r in samples]),
                vision_seconds=distribution([r["vision_seconds"] for r in samples]),
                e2e_seconds=distribution([r["e2e_seconds"] for r in samples]),
                peak_allocated_bytes=max(r["peak_allocated_bytes"] for r in samples),
                peak_reserved_bytes=max(r["peak_reserved_bytes"] for r in samples),
                process_rss_bytes=max(r["process_rss_bytes"] for r in samples),
            )
        ratios = []
        for repeat in sorted({r["repeat"] for r in measured}):
            pair = {r["mode"]: r for r in measured if r["repeat"] == repeat}
            ratios.append(pair["math"]["inference_seconds"] / pair["auto"]["inference_seconds"])
        entry["paired_inference_ratio"] = distribution(ratios)
        result["cases"][case] = entry
    return result
