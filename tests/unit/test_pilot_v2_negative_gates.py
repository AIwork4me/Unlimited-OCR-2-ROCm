"""Pilot-v2 authorization-readiness negative gates.

Ten independent fail-closed checks over the REAL frozen policy file, the
REAL plan/budget/n-gram gates, the REAL ledger integrity checks and the
REAL watchdog — every invalid configuration must be refused. CPU-only.

Gate map (mission P1 list):
  1 wrong policy digest        -> validate_budget / validate_ngram pin
  2 modified selection         -> execution_plan expected_class_counts binding
  3 modified token budget      -> validate_budget budget-set equality
  4 changed n-gram parameters  -> validate_ngram sizes/window equality
  5 missing/stale telemetry    -> ledger PASS seal requires heartbeat.json
  6 watchdog threshold breach  -> ResourceWatchdog abort + breach receipt
  7 timeout / termination      -> supervise_worker state machine (P1.5 suite)
  8 unauthorized execution     -> execution_plan authorization gate
  9 wrong dataset identity     -> manifest validation refuses hash drift
 10 resume with drift          -> Run identity digest + attempt seal checks
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[2]
CAMPAIGN_POLICY = REPO_ROOT / "eval/experiments/mpdoc-full-campaign/policy.json"

from unlimited_ocr_2_rocm.benchmark.campaign import (  # noqa: E402
    CAMPAIGN_POLICY_SHA256,
    execution_plan,
)
from unlimited_ocr_2_rocm.inference import validate_budget, validate_ngram  # noqa: E402

AUTHORIZED_CONFIG = {
    "schema": 1,
    "authorization": {
        "schema": 1,
        "campaign_policy_sha256": CAMPAIGN_POLICY_SHA256,
        "authorized_by": "owner",
    },
    "budget_policy_sha256": CAMPAIGN_POLICY_SHA256,
    "expected_class_counts": {"PRODUCTION_ADMITTED": 1, "CAMPAIGN_POLICY_REQUIRED": 1},
}


def _manifest(prod_pages: int, camp_pages: int):
    return [
        {"id": "prod_a", "pages": [{"path": "p.png"} for _ in range(prod_pages)]},
        {"id": "camp_b", "pages": [{"path": "p.png"} for _ in range(camp_pages)]},
    ]


# 1 — wrong policy digest (forged file, both gates)
def test_gate_1_forged_policy_digest_is_refused(tmp_path: Path) -> None:
    forged = tmp_path / "forged.json"
    payload = json.loads(CAMPAIGN_POLICY.read_text("utf-8"))
    payload["budgets_new_tokens"] = [39321]
    payload["note"] = "one byte differs"
    forged.write_text(json.dumps(payload, indent=1), "utf-8")
    with pytest.raises(ValueError, match="not pinned by this gate"):
        validate_budget(39321, policy_path=forged)
    with pytest.raises(ValueError, match="not one of the pinned frozen identities"):
        validate_ngram(35, 1024, policy_path=forged)
    # the REAL file passes both gates
    validate_budget(39321, policy_path=CAMPAIGN_POLICY)
    validate_ngram(35, 1024, policy_path=CAMPAIGN_POLICY)


# 2 — modified selection (live manifest no longer matches frozen counts)
def test_gate_2_modified_selection_blocks_the_campaign() -> None:
    plan = execution_plan(_manifest(3, 8), config=AUTHORIZED_CONFIG)
    assert plan["executable"] == ["camp_b", "prod_a"]
    drifted = _manifest(3, 8) + [{"id": "camp_x", "pages": [{"path": "p.png"} for _ in range(9)]}]
    plan = execution_plan(drifted, config=AUTHORIZED_CONFIG)
    assert plan["executable"] == ["prod_a"]
    assert set(plan["blocked"].values()) == {"CAMPAIGN_POLICY_MANIFEST_MISMATCH"}


# 3 — modified token budget
def test_gate_3_other_budgets_are_refused() -> None:
    for budget in (39320, 39322, 49152, 65536):
        with pytest.raises(ValueError, match="not authorized by the frozen policy"):
            validate_budget(budget, policy_path=CAMPAIGN_POLICY)


# 4 — changed n-gram parameters
def test_gate_4_ngram_variants_are_refused() -> None:
    for size, window in ((35, 100), (35, 2048), (36, 1024), (0, 1024)):
        with pytest.raises(ValueError):
            validate_ngram(size, window, policy_path=CAMPAIGN_POLICY)


# 5 — missing telemetry: a PASS attempt without a sealed heartbeat refuses
def test_gate_5_pass_attempt_without_heartbeat_receipt_refuses(tmp_path: Path) -> None:
    from unlimited_ocr_2_rocm.benchmark import ledger as ledger_module

    run = _tiny_run(tmp_path)
    doc_dir = run.path / "documents" / "doc_a"
    attempt = doc_dir / "000001"
    attempt.mkdir(parents=True)
    for name, content in (
        ("raw.txt", "raw"),
        ("metadata.json", json.dumps({"state": "PASS", "doc_id": "doc_a", "attempt": "000001"})),
        ("started.json", "{}"),
        ("prediction.md", "pred"),
        # heartbeat.json deliberately ABSENT
    ):
        (attempt / name).write_text(content, "utf-8")
    seal = {p.name: ledger_module.file_hash(p) for p in sorted(attempt.iterdir()) if p.is_file()}
    (attempt / "COMMIT.json").write_text(json.dumps(seal), "utf-8")
    with pytest.raises(ValueError, match="heartbeat receipt"):
        run.attempts("doc_a")


def _tiny_run(tmp_path: Path):
    from unlimited_ocr_2_rocm.benchmark.ledger import Run

    manifest = {
        "schema": 1,
        "dataset": "T",
        "revision": "a" * 40,
        "ground_truth_sha256": "b" * 64,
        "expected_documents": 1,
        "expected_pages": 1,
        "documents": [
            {
                "id": "doc_a",
                "scorer_name": "doc_a.md",
                "pages": [{"id": 1, "path": "images/doc_a/page_1.png", "sha256": "c" * 64}],
            }
        ],
    }
    run = Run.create(tmp_path / "runs", manifest, {"schema": 1}, {"python": "3"})
    return run


# 6 — watchdog threshold breach aborts exactly once, with the breach receipt
def test_gate_6_watchdog_breach_aborts_with_receipt(tmp_path: Path) -> None:
    from unlimited_ocr_2_rocm.benchmark.watchdog import WATCHDOG_EXIT_CODE, ResourceWatchdog

    limit = 16 * 1024**3
    allocations = iter([limit - 1, limit - 1, limit + 1, limit + 1])
    aborts: list[dict] = []
    wd = ResourceWatchdog(
        sampler=lambda: next(allocations),
        limit_bytes=limit,
        heartbeat_path=tmp_path / "heartbeat.json",
        abort=aborts.append,
    )
    assert wd.check_once()["over_limit"] is False
    assert wd.check_once()["over_limit"] is False
    assert wd.check_once()["over_limit"] is True
    assert len(aborts) == 1
    assert aborts[0]["kind"] == "GPU_ALLOCATED_LIMIT_BREACH"
    assert aborts[0]["allocated_bytes"] == limit + 1
    assert WATCHDOG_EXIT_CODE == 86
    assert wd.receipt()["watchdog"]["peak_allocated_bytes"] == limit + 1


# 7 — timeout/termination classification is covered by the P1.5 injection
# suite; assert it is present and green here (run inline).
def test_gate_7_timeout_and_termination_suite_is_green() -> None:
    rc = pytest.main(["-q", str(REPO_ROOT / "tests/unit/test_p15_runner_hardening.py")])
    assert rc == 0


# 8 — unauthorized execution
def test_gate_8_unauthorized_execution_is_refused() -> None:
    docs = _manifest(3, 8)
    plan = execution_plan(docs, config={"schema": 1})
    assert plan["executable"] == ["prod_a"]
    assert plan["blocked"]["camp_b"] == "CAMPAIGN_POLICY_REQUIRED_NO_AUTHORIZATION"
    # an authorization naming a foreign digest is equally refused
    foreign = {
        **AUTHORIZED_CONFIG,
        "authorization": {"campaign_policy_sha256": "f" * 64, "authorized_by": "imposter"},
    }
    plan = execution_plan(docs, config=foreign)
    assert plan["executable"] == ["prod_a"]
    assert plan["blocked"]["camp_b"] == "CAMPAIGN_POLICY_DIGEST_MISMATCH"


# 9 — wrong dataset identity: manifest validation refuses drifted page bytes
def test_gate_9_dataset_identity_drift_refuses(tmp_path: Path) -> None:
    from unlimited_ocr_2_rocm.benchmark.manifest import validate_manifest
    from unlimited_ocr_2_rocm.config import file_sha256

    images = tmp_path / "images" / "doc_a"
    images.mkdir(parents=True)
    page = images / "page_1.png"
    page.write_bytes(b"original-bytes")
    manifest = {
        "schema": 1,
        "dataset": "T",
        "revision": "a" * 40,
        "ground_truth_sha256": "b" * 64,
        "expected_documents": 1,
        "expected_pages": 1,
        "documents": [
            {
                "id": "doc_a",
                "scorer_name": "doc_a.md",
                "pages": [
                    {"id": 1, "path": "images/doc_a/page_1.png", "sha256": file_sha256(page)}
                ],
            }
        ],
    }
    assert validate_manifest(manifest, tmp_path)["status"] == "PASS"
    page.write_bytes(b"tampered-bytes")  # dataset identity drift
    report = validate_manifest(manifest, tmp_path)
    assert report["status"] != "PASS"


# 10 — resume with mismatched frozen identities: run.json tampering refuses
def test_gate_10_run_identity_tamper_refuses(tmp_path: Path) -> None:
    from unlimited_ocr_2_rocm.benchmark.ledger import Run

    run = _tiny_run(tmp_path)
    data = json.loads((run.path / "run.json").read_text("utf-8"))
    data["config"]["timeout_seconds_per_document"] = 1.0  # silent protocol drift
    (run.path / "run.json").write_text(json.dumps(data), "utf-8")
    with pytest.raises(ValueError, match="identity hash mismatch"):
        Run(run.path)


def test_observer_required_for_campaign_class_only() -> None:
    """The fifth policy's collision-observer contract: campaign-class attempts
    get an observer; production-class attempts must not silently grow one."""
    from unlimited_ocr_2_rocm.benchmark.campaign import new_observer_if_required

    assert new_observer_if_required("PRODUCTION_ADMITTED") is None
    observer = new_observer_if_required("CAMPAIGN_POLICY_REQUIRED")
    assert observer is not None
    seen: list[dict] = []
    observer2 = new_observer_if_required("CAMPAIGN_POLICY_REQUIRED", seen.append)
    assert observer2 is not None and seen == []
