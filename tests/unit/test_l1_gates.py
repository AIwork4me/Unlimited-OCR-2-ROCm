"""L1 gate evaluation: the replay hard gate and single-attempt rule (CPU)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def _driver():
    spec = importlib.util.spec_from_file_location(
        "run_l1_fifteen_page", REPO_ROOT / "scripts/run_l1_fifteen_page.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _policy() -> dict[str, Any]:
    return json.loads(
        (REPO_ROOT / "eval/experiments/l1-fifteen-page/policy.json").read_text(encoding="utf-8")
    )


def _summary(state: str, attempts: int = 1, finish: str = "stop", pages_out: int = 15) -> dict:
    return {
        "documents": [
            {
                "doc_id": "book_eng_77552588",
                "state": state,
                "attempts": [{"attempt": f"{i:06d}"} for i in range(1, attempts + 1)],
                "metadata": {
                    "finish_reason": finish,
                    "experiment": {"page_guard": {"num_pages_out": pages_out}},
                },
            }
        ],
        "counts": {state: 1, "NOT_RUN": 419},
    }


def _report(gpu_gib: float = 11.0, rss_gib: float = 2.5, **experiment: Any) -> dict:
    return {
        "memory_after": {"gpu_peak_allocated_bytes": int(gpu_gib * 1024**3)},
        "experiment": {"host_peak_rss_gib": rss_gib, **experiment},
    }


def _replay(mismatches: int = 0) -> dict:
    return {"guard": {"expected_pages": 15, "forced_mismatches": [{"step": 1}] * mismatches}}


def test_promotion_requires_replay_evidence() -> None:
    driver = _driver()
    verdict = driver.evaluate_l1_gates(_summary("PASS"), _policy(), [_report()], replay=None)
    assert verdict["promotion_admitted"] is False
    assert any("replay evidence missing" in failure for failure in verdict["failures"])


def test_promotion_blocked_by_replay_mismatch() -> None:
    driver = _driver()
    verdict = driver.evaluate_l1_gates(
        _summary("PASS"), _policy(), [_report()], replay=_replay(mismatches=1)
    )
    assert verdict["promotion_admitted"] is False
    assert verdict["forced_marker_mismatches"] == 1
    assert any("forced-marker mismatches" in failure for failure in verdict["failures"])


def test_promotion_passes_on_clean_complete_attempt() -> None:
    driver = _driver()
    verdict = driver.evaluate_l1_gates(_summary("PASS"), _policy(), [_report()], replay=_replay(0))
    assert verdict["promotion_admitted"] is True
    assert verdict["failures"] == []


def test_truncated_attempt_is_not_promoted() -> None:
    driver = _driver()
    verdict = driver.evaluate_l1_gates(
        _summary("TRUNCATED", finish="length"),
        _policy(),
        [_report()],
        replay=_replay(0),
    )
    assert verdict["promotion_admitted"] is False
    assert any("natural completion" in failure for failure in verdict["failures"])


def test_second_attempt_blocks_promotion() -> None:
    driver = _driver()
    verdict = driver.evaluate_l1_gates(
        _summary("PASS", attempts=2), _policy(), [_report()], replay=_replay(0)
    )
    assert verdict["promotion_admitted"] is False
    assert any("exactly one supervised attempt" in failure for failure in verdict["failures"])


def test_resource_and_observer_breaches_block_promotion() -> None:  # noqa
    driver = _driver()
    verdict = driver.evaluate_l1_gates(
        _summary("PASS"),
        _policy(),
        [
            _report(gpu_gib=16.5),
            _report(observer={"aborted": True}, watchdog={"breach": {"kind": "BREACH"}}),
        ],
        replay=_replay(0),
    )
    assert verdict["promotion_admitted"] is False
    assert verdict["observer_aborted"] is True
    assert verdict["watchdog_breach"] is True
    assert any("exceeds 16.0" in failure for failure in verdict["failures"])


def test_replay_page_count_disagreement_fails() -> None:
    driver = _driver()
    replay = _replay(0)
    replay["guard"]["expected_pages"] = 5
    verdict = driver.evaluate_l1_gates(_summary("PASS"), _policy(), [_report()], replay=replay)
    assert verdict["promotion_admitted"] is False
    assert any("expected_pages disagrees" in failure for failure in verdict["failures"])


def test_missing_document_blocks_promotion() -> None:
    driver = _driver()
    summary = _summary("NOT_RUN", finish="stop")
    summary["documents"][0]["state"] = "NOT_RUN"
    verdict = driver.evaluate_l1_gates(summary, _policy(), [], replay=_replay(0))
    assert verdict["promotion_admitted"] is False
    assert any("did not complete" in failure for failure in verdict["failures"])


if __name__ == "__main__":
    sys.exit(pytest_main := 0)
