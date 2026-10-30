import importlib.util
import sys
from pathlib import Path

import pytest

PATH = Path(__file__).parents[2] / "experiments/p2_ab/protocol.py"


def protocol():
    assert PATH.exists(), "synthetic A/B protocol is not implemented"
    spec = importlib.util.spec_from_file_location("p2_protocol", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_schedule_balances_each_case_and_excludes_warmup_from_measurements():
    module = protocol()
    schedule = module.schedule(["english", "pages_5"], repeats=6, seed=17)
    assert len(schedule) == 32
    for case in ("english", "pages_5"):
        rows = [r for r in schedule if r["case"] == case]
        assert [r["mode"] for r in rows if r["phase"] == "diagnostic"] == ["math", "auto"]
        samples = [r for r in rows if r["phase"] == "measurement"]
        assert len(samples) == 12
        first = [r["mode"] for r in samples[::2]]
        assert first.count("math") == first.count("auto") == 3
    assert module.schedule(["english", "pages_5"], 6, 17) == schedule


@pytest.mark.parametrize("repeats", [0, 3, 4, 5, -2])
def test_unbalanced_or_insufficient_repeat_design_is_rejected(repeats):
    with pytest.raises(ValueError):
        protocol().schedule(["english"], repeats, 0)


def trial(item):
    row = dict(item)
    row.update(
        status="PASS",
        wheel_sha256="6a67d15861538ee217afdccd1ccc09ff318d7f44bff29a10f74c0d51730e6e1e",
        input_sha256="a" * 64,
        raw_sha256="b" * 64,
        prompt_ids_sha256="c" * 64,
        output_ids_sha256="d" * 64,
        output_tokens=80,
        finish_reason="stop",
        structure_pass=True,
        correctness_pass=True,
        dash_calls=12,
        hip_zero=True,
        inference_seconds=4.0 if item["mode"] == "math" else 2.0,
        e2e_seconds=4.1 if item["mode"] == "math" else 2.1,
        vision_seconds=2.0 if item["mode"] == "math" else 1.0,
        peak_allocated_bytes=100,
        peak_reserved_bytes=200,
        process_rss_bytes=300,
        model_sha256="1" * 64,
        config_sha256="2" * 64,
        producer_sha256="3" * 64,
        stack_sha256="4" * 64,
    )
    row["dispatch"] = (
        {"sam_model": ["MATH"], "vision_model": ["MATH"]}
        if item["mode"] == "math"
        else {"sam_model": ["EFFICIENT_ATTENTION"], "vision_model": ["FLASH_ATTENTION"]}
    )
    if item["phase"] == "diagnostic":
        row["numerical_pass"] = True
    return row


def test_summary_uses_measurements_only_and_exports_no_payload_or_paths():
    module = protocol()
    plan = module.schedule(["english"], 6, 0)
    rows = [trial(item) for item in plan]
    for row in rows:
        row["raw_text"] = "private payload"
        row["path"] = "private path"
        if row["phase"] != "measurement":
            row["inference_seconds"] = 999.0
    result = module.summarize(plan, rows)
    assert result["status"] == "PASS"
    assert result["real_documents"] == "BLOCKED_DATA"
    assert result["cases"]["english"]["math"]["inference_seconds"]["median"] == 4.0
    assert result["cases"]["english"]["auto"]["samples"] == 6
    assert result["cases"]["english"]["paired_inference_ratio"]["median"] == 2.0
    assert "private" not in str(result)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("wheel_sha256", "415d2dbd4b6ab002cff1c54795fdd66322abd9a46bf982cb64eebb4493ac2e7c"),
        ("raw_sha256", "e" * 64),
        ("output_ids_sha256", "e" * 64),
        ("input_sha256", "e" * 64),
        ("model_sha256", "e" * 64),
        ("config_sha256", "e" * 64),
        ("producer_sha256", "e" * 64),
        ("stack_sha256", "e" * 64),
        ("output_tokens", 81),
        ("finish_reason", "length"),
        ("structure_pass", False),
        ("correctness_pass", False),
        ("hip_zero", False),
        ("dash_calls", 0),
        ("inference_seconds", float("nan")),
        ("peak_reserved_bytes", -1),
    ],
)
def test_invalid_or_nonmatching_trials_block_speedup(field, value):
    module = protocol()
    plan = module.schedule(["english"], 6, 0)
    rows = [trial(item) for item in plan]
    rows[-1][field] = value
    result = module.summarize(plan, rows)
    assert result["status"] == "FAIL"
    assert result["cases"] == {}


def test_missing_duplicate_and_silent_fallback_are_not_accepted():
    module = protocol()
    plan = module.schedule(["english"], 6, 0)
    rows = [trial(item) for item in plan]
    assert module.summarize(plan, rows[:-1])["status"] == "FAIL"
    assert module.summarize(plan, rows + rows[:1])["status"] == "FAIL"
    rows[1]["dispatch"]["sam_model"] = ["MATH"]
    assert module.summarize(plan, rows)["status"] == "FAIL"


def test_importing_protocol_does_not_load_torch():
    before = "torch" in sys.modules
    protocol()
    assert ("torch" in sys.modules) == before


def test_reordered_measurements_and_empty_plan_do_not_claim_pass():
    module = protocol()
    plan = module.schedule(["english"], 6, 0)
    rows = [trial(item) for item in plan]
    rows[-2:] = rows[-2:][::-1]
    assert module.summarize(plan, rows)["status"] == "FAIL"
    assert module.summarize([], [])["status"] == "FAIL"


def test_publication_refuses_overwrite_and_failed_publish_leaves_no_final(tmp_path, monkeypatch):
    import os

    module = protocol()
    target = tmp_path / "trial.json"
    module.publish(target, {"state": "FAIL"})
    with pytest.raises(FileExistsError):
        module.publish(target, {"state": "PASS"})
    assert target.read_text(encoding="utf-8").find("FAIL") >= 0

    def interrupted(*args):
        raise OSError("injected publication failure")

    monkeypatch.setattr(os, "link", interrupted)
    with pytest.raises(OSError):
        module.publish(tmp_path / "second.json", {"state": "PASS"})
    assert not (tmp_path / "second.json").exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_cli_help_and_existing_output_refusal_do_not_import_torch(tmp_path):
    import subprocess

    entry = PATH.with_name("run.py")
    assert entry.exists(), "A/B entry point not implemented"
    result = subprocess.run(
        [sys.executable, str(entry), "--help"], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0
    result = subprocess.run(
        [
            sys.executable,
            str(entry),
            "prepare",
            "--output",
            str(tmp_path),
            "--identity-profile",
            str(tmp_path / "absent.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "fresh" in result.stderr.lower()


def test_gpu_worker_module_import_is_inert():
    worker_path = PATH.with_name("worker.py")
    assert worker_path.exists(), "GPU worker not implemented"
    sys.path.insert(0, str(PATH.parent))
    try:
        spec = importlib.util.spec_from_file_location("p2_worker", worker_path)
        module = importlib.util.module_from_spec(spec)
        before = "torch" in sys.modules
        spec.loader.exec_module(module)
        assert ("torch" in sys.modules) == before
    finally:
        sys.path.pop(0)


def test_diagnostics_gate_does_not_produce_a_timing_comparison():
    module = protocol()
    plan = [r for r in module.schedule(["english"], 6, 0) if r["phase"] == "diagnostic"]
    result = module.summarize(plan, [trial(r) for r in plan])
    assert result["status"] == "PASS"
    assert result["cases"] == {}


def test_auto_peak_allocation_degradation_blocks_speedup():
    module = protocol()
    plan = module.schedule(["english"], 6, 0)
    rows = [trial(r) for r in plan]
    for row in rows:
        if row["mode"] == "auto" and row["phase"] == "measurement":
            row["peak_allocated_bytes"] = 120
    assert module.summarize(plan, rows)["status"] == "FAIL"


def test_p0_hold_blocks_worker_before_torch_import(tmp_path):
    import json

    sys.path.insert(0, str(PATH.parent))
    try:
        from worker import execution_gate

        receipt = tmp_path / "p0.json"
        receipt.write_text(json.dumps({"matrix": {"P0": "HOLD"}}), encoding="utf-8")
        before = "torch" in sys.modules
        with pytest.raises(ValueError, match="P0 PASS"):
            execution_gate({"base_source_commit": "a" * 40}, receipt)
        assert ("torch" in sys.modules) == before
    finally:
        sys.path.pop(0)


def test_measured_math_fallback_is_rejected_and_plan_fields_cannot_leak():
    module = protocol()
    plan = module.schedule(["english"], 6, 0)
    rows = [trial(r) for r in plan]
    next(r for r in rows if r["phase"] == "measurement" and r["mode"] == "auto")["dispatch"][
        "sam_model"
    ] = ["MATH"]
    assert module.summarize(plan, rows)["status"] == "FAIL"
    plan[0]["private_text"] = "secret payload"
    result = module.summarize(plan, [trial(r) for r in plan])
    assert result["status"] == "FAIL"
    assert "secret payload" not in str(result)


def test_unsealed_or_modified_plan_is_rejected_before_execution(tmp_path):
    import json

    module = protocol()
    assert hasattr(module, "read_plan"), "sealed-plan validation not implemented"
    target = tmp_path / "plan.json"
    target.write_text(json.dumps({"root": "private"}), encoding="utf-8")
    with pytest.raises(ValueError):
        module.read_plan(target, tmp_path, tmp_path)
    module.publish(tmp_path / "public-preparation.json", {"protocol_sha256": "0" * 64})
    with pytest.raises(ValueError, match="hash"):
        module.read_plan(target, tmp_path, tmp_path)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0, -1, 7201])
def test_invalid_timeout_cannot_disable_the_deadline(value):
    module = protocol()
    assert hasattr(module, "validate_timeout"), "bounded-time validation not implemented"
    with pytest.raises(ValueError):
        module.validate_timeout(value)


@pytest.mark.skipif(sys.platform != "win32", reason="Native Windows job lifecycle")
def test_job_contains_and_terminates_an_immediate_grandchild():
    import subprocess

    path = PATH.with_name("windows_job.py")
    assert path.exists(), "Race-free Windows process ownership not implemented"
    spec = importlib.util.spec_from_file_location("p2_windows_job", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    job = module.Job()
    child = None
    try:
        child = job.spawn(
            [
                sys.executable,
                "-c",
                "import subprocess,sys; p=subprocess.Popen("
                "[sys.executable,'-c','import time; time.sleep(30)']); print(p.pid,flush=True)",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert int(child.stdout.readline()) > 0
        assert child.wait(timeout=10) == 0
        assert job.accounting()["total_processes"] >= 2
        assert job.accounting()["active_processes"] >= 1
        job.terminate()
        assert job.accounting()["active_processes"] == 0
    finally:
        job.close()
        if child is not None:
            child.wait(timeout=10)
