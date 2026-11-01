"""P2-A opt-in budget policy gates (CPU only; no model, no GPU).

The gate pins the frozen policy by exact content digest: a schema-valid but
unpinned policy file must be refused (forgery regression), and the production
default (8192) needs no policy at all.
"""

import json
from pathlib import Path

from unlimited_ocr_2_rocm.inference import validate_budget

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_POLICY = REPO_ROOT / "eval" / "experiments" / "p2a-token-budget" / "policy.json"


def policy_file(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_production_budgets_need_no_policy():
    assert validate_budget(1) is None
    assert validate_budget(8192) is None


def test_budget_above_production_cap_requires_policy():
    try:
        validate_budget(12288)
    except ValueError as error:
        assert "production cap" in str(error)
    else:
        raise AssertionError("cap must be enforced without the frozen policy")


def test_frozen_policy_authorizes_only_its_listed_budgets():
    assert validate_budget(12288, FROZEN_POLICY) is None
    assert validate_budget(16384, FROZEN_POLICY) is None
    for unlisted in (32768, 8193):
        try:
            validate_budget(unlisted, FROZEN_POLICY)
        except ValueError as error:
            assert "not authorized" in str(error)
        else:
            raise AssertionError(f"unlisted budget must be refused: {unlisted}")


def test_forgery_and_tamper_fail_closed(tmp_path):
    """The core regression: a hand-made policy with the right schema and the
    right budget list must NOT pass just because it parses."""
    for payload in (
        {
            "schema": 1,
            "experiment": "P2A_OUTPUT_TOKEN_BUDGET_RCA",
            "budgets_new_tokens": [12288, 16384],
        },
        {
            "schema": 1,
            "experiment": "P2A_OUTPUT_TOKEN_BUDGET_RCA",
            "budgets_new_tokens": [12288, 16384, 32768],
        },
        {"schema": 2},
        {"schema": 1, "budgets_new_tokens": "12288"},
        {},
    ):
        path = policy_file(tmp_path, payload)
        try:
            validate_budget(12288, path)
        except ValueError as error:
            assert "not pinned" in str(error), f"{payload}: {error}"
        else:
            raise AssertionError(f"unpinned policy must be refused: {payload}")
    # a byte-edited copy of the real policy (e.g. trailing newline) is tampered
    tampered = tmp_path / "tampered.json"
    tampered.write_text(FROZEN_POLICY.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    try:
        validate_budget(12288, tampered)
    except ValueError as error:
        assert "not pinned" in str(error)
    else:
        raise AssertionError("byte-edited frozen policy must be refused")
    # invalid JSON also fails closed (digest refusal precedes parsing)
    garbage = tmp_path / "garbage.json"
    garbage.write_text("{not json", encoding="utf-8")
    try:
        validate_budget(12288, garbage)
    except ValueError as error:
        assert "not pinned" in str(error)
    else:
        raise AssertionError("unparseable policy must be refused")
    # missing file
    try:
        validate_budget(12288, tmp_path / "absent.json")
    except ValueError as error:
        assert "Unreadable" in str(error)
    else:
        raise AssertionError("missing policy must be refused")


def test_experiment_identity_is_bound():
    try:
        validate_budget(12288, FROZEN_POLICY, experiment="SOME_OTHER_EXPERIMENT")
    except ValueError as error:
        assert "does not match the requested experiment" in str(error)
    else:
        raise AssertionError("wrong experiment identity must be refused")
    assert validate_budget(12288, FROZEN_POLICY, experiment="P2A_OUTPUT_TOKEN_BUDGET_RCA") is None


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
