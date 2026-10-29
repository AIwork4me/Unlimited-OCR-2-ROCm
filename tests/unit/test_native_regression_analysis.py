"""CPU-only analysis/replay tests; authored values are never GPU evidence."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def analysis():
    path = Path(__file__).resolve().parents[2] / "experiments/native_regression/analyze.py"
    assert path.is_file(), "CPU analysis implementation is missing"
    spec = importlib.util.spec_from_file_location("g1_analysis_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_comparison_ignores_process_cache_paths_but_rejects_sha_change(analysis):
    before = {
        "infer": {
            "path": "old-cache/source.py",
            "sha256": "aaa",
            "line": 2,
            "function": "Model.infer",
        }
    }
    after = {
        "infer": {
            "path": "new-cache/source.py",
            "sha256": "aaa",
            "line": 2,
            "function": "Model.infer",
        }
    }
    assert analysis.source_identity(before) == analysis.source_identity(after)
    after["infer"]["sha256"] = "bbb"
    assert analysis.source_identity(before) != analysis.source_identity(after)


def test_memory_uses_native_snapshot_and_explicitly_excludes_launcher_rss(analysis):
    value = analysis.memory_report(
        {
            "memory_after": {
                "process_rss_bytes": 12_000_000_000,
                "process_peak_rss_bytes": 13_000_000_000,
                "gpu_peak_allocated_bytes": 9_000_000_000,
                "gpu_peak_reserved_bytes": 11_000_000_000,
            }
        }
    )
    assert value["native_process"]["process_rss_bytes"] == 12_000_000_000
    assert value["launcher_rss_used"] is False


def test_missing_physical_failure_captures_are_not_reported_as_pass(analysis, tmp_path):
    result = analysis.physical_failure_report(tmp_path / "before", tmp_path / "after")
    assert result["status"] == "NOTRUN"
    assert result["interruption"]["status"] == "NOTRUN"


def test_real_ledger_replay_retains_all_attempts_and_prior_commit_on_rename_failure(
    analysis, tmp_path
):
    from experiments.native_regression.runner import sha256, write_bytes, write_json

    source = tmp_path / "CPU_AUTHORED_TEST_INPUT_NOT_GPU"
    raw = "<PAGE><BLOCK>text[0,0,1,1]<FORK>Page one<PAGE><BLOCK>text[0,0,1,1]<FORK>Page two"
    for name, text, ids, finish, budget in (
        ("pages_2_run_1", raw, [128827, 1, 128827, 2], "stop", 2048),
        ("length_2_pages_8_tokens", "<PAGE><BLOCK>", list(range(8)), "length", 8),
    ):
        case = source / name
        native = {
            "text": text,
            "token_ids": ids,
            "prompt_token_ids": [128827, 128827],
            "finish_reason": finish,
            "expected_pages": 2,
            "num_pages_out": text.count("<PAGE>"),
            "time": 1.0,
            "prefill_time": 0.1,
        }
        write_bytes(case / "diagnostics/native-001.raw.txt", text.encode())
        write_json(case / "diagnostics/native-001.json", native)
        write_json(
            case / "summary.json",
            {
                "case": {"max_new_tokens": budget},
                "native": native,
                "raw_sha256": sha256(case / "diagnostics/native-001.raw.txt"),
                "memory_after": {"process_rss_bytes": 123456},
            },
        )
    fixtures = Path(__file__).resolve().parents[1] / "fixtures"
    result = analysis.replay_failures(source, fixtures, tmp_path / "replay")
    assert result["status"] == "PASS"
    assert result["accounting"]["expected_documents"] == 7
    assert result["accounting"]["expected_pages"] == 14
    assert result["total_attempts"] == 8
    assert result["accounting"]["counts"] == {
        "PASS": 2,
        "FAIL": 3,
        "TIMEOUT": 0,
        "OOM": 0,
        "TRUNCATED": 1,
        "NOT_RUN": 1,
    }
    assert result["checks"]["rename_prior_commit_unchanged"] is True
    assert result["checks"]["rename_new_pending_raw_preserved"] is True
    assert result["labels"]["missing_page"] == "SYNTHETIC_REPLAY_NOT_GPU_CORRUPTION: DERIVATIVE"
    assert result["labels"]["malformed"] == "SYNTHETIC_REPLAY_NOT_GPU_CORRUPTION: AUTHORED"
