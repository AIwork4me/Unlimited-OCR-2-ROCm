"""P2-A opt-in budget policy gates (CPU only; no model, no GPU)."""

import json
from pathlib import Path

from unlimited_ocr_2_rocm.inference import validate_budget

POLICY = {
    "schema": 1,
    "experiment": "P2A_OUTPUT_TOKEN_BUDGET_RCA",
    "budgets_new_tokens": [12288, 16384],
}


def policy_file(tmp_path: Path, payload: dict | None = None) -> Path:
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(payload or POLICY), encoding="utf-8")
    return path


def test_production_budgets_need_no_policy(tmp_path):
    assert validate_budget(1) is None
    assert validate_budget(8192) is None
    assert validate_budget(8192, policy_file(tmp_path)) is None


def test_budget_above_production_cap_requires_policy():
    try:
        validate_budget(12288)
    except ValueError as error:
        assert "production cap" in str(error)
    else:
        raise AssertionError("cap must be enforced without the frozen policy")


def test_policy_authorizes_only_listed_budgets(tmp_path):
    path = policy_file(tmp_path)
    assert validate_budget(12288, path) is None
    assert validate_budget(16384, path) is None
    try:
        validate_budget(32768, path)
    except ValueError as error:
        assert "not authorized" in str(error)
    else:
        raise AssertionError("unlisted budgets must be refused")


def test_malformed_policy_is_refused(tmp_path):
    for payload in ({"schema": 2}, {"schema": 1}, {"schema": 1, "budgets_new_tokens": "12288"}):
        path = policy_file(tmp_path, payload)
        try:
            validate_budget(12288, path)
        except ValueError:
            continue
        raise AssertionError(f"malformed policy must be refused: {payload}")
    missing = tmp_path / "absent.json"
    try:
        validate_budget(12288, missing)
    except ValueError as error:
        assert "Unreadable" in str(error)
    else:
        raise AssertionError("missing policy must be refused")


def test_p2a_driver_admission_arithmetic():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_p2a_budget", Path(__file__).parents[2] / "scripts/run_p2a_budget.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    policy = {
        "admission": {
            "context_cap_tokens": 131072,
            "prompt_upper_bound_tokens_per_page": 300,
            "required_margin_tokens": 4096,
        }
    }
    admitted = module.admission_report(policy, pages=5, budget=16384)
    assert admitted["admitted"] is True
    assert admitted["total_upper_bound_tokens"] == 5 * 300 + 16384 + 4096
    tight = dict(policy["admission"])
    tight["context_cap_tokens"] = 20000
    refused = module.admission_report({"admission": tight}, pages=5, budget=16384)
    assert refused["admitted"] is False
