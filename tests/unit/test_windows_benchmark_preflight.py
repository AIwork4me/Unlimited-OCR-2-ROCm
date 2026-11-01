"""Reject changed annotations before inventorying or launching any worker."""

import argparse
import importlib.util
import json
from pathlib import Path

import pytest


def test_prepare_rejects_ground_truth_that_does_not_match_frozen_lock(tmp_path):
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "windows_shared_prepare", root / "scripts/run_mpdoc_sample.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"doc": {"reason": "test"}}))
    gt = tmp_path / "gt.json"
    gt.write_text("[]")
    lock = tmp_path / "lock.json"
    lock.write_text(
        json.dumps(
            {"datasets": {"MPDocBench": {"revision": "a" * 40, "ground_truth_sha256": "0" * 64}}}
        )
    )
    args = argparse.Namespace(
        max_new_tokens=8192,
        selection=selection,
        lock=lock,
        ground_truth=gt,
        dataset_root=tmp_path / "absent",
    )
    with pytest.raises(ValueError, match="frozen ground-truth"):
        module.prepare_command(args)
