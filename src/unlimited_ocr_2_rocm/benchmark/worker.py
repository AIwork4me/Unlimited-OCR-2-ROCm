"""Optional isolated CPU process importing the unchanged, verified official scorer.

All shell commands still run from the repository root. Original scorer files
write under ./result/<save_name>; a relative save_name directs them to the
requested repository-local output. No upstream upload entry point is executed.
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .ledger import json_bytes, write_bytes
from .manifest import file_hash
from .scorer import (
    METRICS,
    aggregate,
    check_dependencies,
    check_sources,
    metric_values,
    sample_coverage,
)


def pipeline_fixture(output: Path, save_name: str) -> dict[str, Any]:
    from dataset.end2end_dataset import End2EndDataset
    from task.end2end_run_eval import End2EndEval

    from .._upstream.process_unlimitedOCR2 import convert

    fixtures_root = Path("tests/fixtures/benchmark").resolve()
    target = output / "pipeline"
    predictions = target / "predictions"
    predictions.mkdir(parents=True)
    write_bytes(
        predictions / "synthetic.md",
        str(convert((fixtures_root / "synthetic.raw.txt").read_text(encoding="utf-8"))).encode(
            "utf-8"
        ),
    )
    metrics = {
        task: {"metric": [metric]}
        for task, metric, _, _ in METRICS.values()
        if task != "display_formula"
    }
    gt = fixtures_root / "synthetic.json"
    cfg = {
        "metrics": metrics,
        "dataset": {
            "ground_truth": {"data_path": str(gt)},
            "prediction": {"data_path": str(predictions)},
            "match_method": "quick_match",
        },
    }
    End2EndEval(End2EndDataset(cfg), metrics, str(gt), save_name + "/pipeline")
    values = metric_values(json.loads((target / "metric_result.json").read_text(encoding="utf-8")))
    passed = all(value == 100 for name, value in values.items() if name != "FormulaCDM")
    return {
        "status": "PASS" if passed else "FAIL",
        "metrics": values,
        "ground_truth_sha256": file_hash(gt),
        "raw_sha256": file_hash(fixtures_root / "synthetic.raw.txt"),
    }


def environment() -> dict[str, Any]:
    return {
        "python": sys.version,
        "python_sha256": file_hash(Path(sys.executable)),
        "platform": platform.platform(),
        "packages": dict(
            sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions())
        ),
        "controls": {
            name: value for name, value in sorted(os.environ.items()) if name.startswith("MPDOC_")
        },
        "producer_sha256": {
            p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))
        },
        "postprocessor_sha256": file_hash(
            Path(__file__).parents[1] / "_upstream/process_unlimitedOCR2.py"
        ),
    }


def cdm_tools() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for tool, option in (
        ("node", "--version"),
        ("magick", "-version"),
        ("xelatex", "--version"),
        ("kpsewhich", "--version"),
    ):
        path = shutil.which(tool)
        if path is None:
            result[tool] = {"status": "BLOCKED", "reason": "NOT_INSTALLED"}
        else:
            proc = subprocess.run([path, option], capture_output=True, timeout=30, check=False)
            result[tool] = {
                "status": "PASS" if proc.returncode == 0 else "BLOCKED",
                "sha256": file_hash(Path(path)),
                "version": (proc.stdout + proc.stderr)
                .decode("utf-8", errors="replace")
                .splitlines()[:3],
            }
    return result


def fixtures() -> dict[str, list[dict[str, Any]]]:
    """Authored synthetic samples at the official matched-sample API boundary."""

    def sample(gt: Any, pred: Any, doc: str) -> dict[str, Any]:
        return {
            "gt": gt,
            "pred": pred,
            "norm_gt": gt,
            "norm_pred": pred,
            "gt_idx": 1,
            "img_id": doc + ".pdf",
            "gt_attribute": [{}],
        }

    text = [sample("abcd", "abxd", "a"), sample("ab", "", "b")]
    table = "<html><body><table><tr><td>cell</td></tr></table></body></html>"
    tables = [sample(table, table, "a"), sample(table, "", "b")]
    return {
        "text_block": text,
        "merged_text_block": copy.deepcopy(text),
        "reading_order": [sample([1, 2, 3], [1, 3, 2], "a"), sample([1], [1], "b")],
        "table": tables,
        "merged_table": copy.deepcopy(tables),
        "head": [sample(["# A", "## B"], ["# A", "## B"], "a"), sample(["# C"], [], "b")],
        "figure": [
            {"img_id": "a.pdf", "gt_count": 2, "pred_count": 2, "matched_count": 1},
            {"img_id": "b.pdf", "gt_count": 1, "pred_count": 0, "matched_count": 0},
        ],
        "display_formula": [sample("x+y=z", "x+y=z", "a")],
    }


def selftest(output: Path, save_name: str) -> dict[str, Any]:
    from registry.registry import METRIC_REGISTRY

    expected = {
        "TextEdit": 37.5,
        "T-TextEdit": 37.5,
        "TableTEDS": 50,
        "T-TableTEDS": 50,
        "ReadOrderEdit": 100 * (1 - 1 / 3),
        "FigureF1": 25,
        "HeadingTEDS": 75,
        "FormulaCDM": 100,
    }
    tools = cdm_tools()
    samples = fixtures()
    result: dict[str, Any] = {}
    checks: dict[str, Any] = {}
    for name, (task, metric, _, _) in METRICS.items():
        if name == "FormulaCDM" and any(v["status"] != "PASS" for v in tools.values()):
            checks[name] = {"status": "BLOCKED", "reason": "CDM_RENDERER_UNAVAILABLE"}
            continue
        try:
            _, scores = METRIC_REGISTRY.get(metric)(copy.deepcopy(samples[task])).evaluate(
                [], f"{save_name}/{task}"
            )
            result[task] = {"all": scores}
            actual = metric_values(result)[name]
            passed = actual is not None and math.isclose(actual, expected[name], abs_tol=1e-7)
            checks[name] = {
                "status": "PASS" if passed else "FAIL",
                "expected": expected[name],
                "actual": actual,
            }
        except Exception as error:
            checks[name] = {"status": "BLOCKED", "reason": f"{type(error).__name__}: {error}"}
    values = metric_values(result)
    pipeline = pipeline_fixture(output, save_name)
    report = {
        "status": "PASS"
        if all(v["status"] == "PASS" for v in checks.values()) and pipeline["status"] == "PASS"
        else "BLOCKED",
        "checks": checks,
        "pipeline": pipeline,
        "tools": tools,
        "metrics": values,
        "overall": aggregate(values),
        "official_result": result,
        "fixtures": samples,
        "environment": environment(),
    }
    write_bytes(output / "selftest.json", json_bytes(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--profile", choices=["qwen", "marathon"], default="qwen")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(Path.cwd().resolve()):
        raise ValueError("Output must stay inside the development root")
    output.mkdir(parents=True, exist_ok=False)
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    profile = copy.deepcopy(lock["qwen"])
    if args.profile == "marathon":
        profile["files"].update(lock["datasets"]["MarathonDocBench"]["patch"])
    check_sources(args.source, profile)
    requirements = Path("requirements/benchmark-scorer.lock.txt")
    if file_hash(requirements) != lock["scorer_requirements_sha256"]:
        raise ValueError("Frozen scorer dependency lock hash mismatch")
    check_dependencies(
        environment()["packages"], requirements.read_text(encoding="utf-8").splitlines()
    )
    sys.path.insert(0, str(args.source.resolve()))
    # Import registers original classes. Never invoke pdf_validation.py's CLI.
    import metrics  # noqa: F401

    Path("result").mkdir(exist_ok=True)
    save_name = os.path.relpath(output, Path("result").resolve()).replace("\\", "/")
    report = selftest(output, save_name)
    if args.ground_truth or args.predictions:
        if not args.ground_truth or not args.predictions or report["status"] != "PASS":
            return 2
        if args.profile == "marathon" and os.name == "nt":
            raise ValueError("Official Marathon TEDS timeout requires SIGALRM; score on Linux")
        from dataset.end2end_dataset import End2EndDataset
        from task.end2end_run_eval import End2EndEval

        gt = json.loads(args.ground_truth.read_text(encoding="utf-8"))
        names = {Path(row["page_info"]["image_path"]).stem + ".md" for row in gt}
        if names != {p.name for p in args.predictions.glob("*.md")}:
            raise ValueError(
                "Scorer prediction inventory must equal the full ground-truth inventory"
            )
        cfg = {
            "metrics": {task: {"metric": [metric]} for task, metric, _, _ in METRICS.values()},
            "dataset": {
                "ground_truth": {"data_path": str(args.ground_truth.resolve())},
                "prediction": {"data_path": str(args.predictions.resolve())},
                "match_method": "quick_match",
            },
        }
        write_bytes(output / "config.json", json_bytes(cfg))
        official = output / "official"
        official.mkdir()
        name = save_name + "/official"
        matched = End2EndDataset(cfg)
        coverage = sample_coverage([row["page_info"]["image_path"] for row in gt], matched.samples)
        write_bytes(output / "coverage.json", json_bytes(coverage))
        End2EndEval(matched, cfg["metrics"], str(args.ground_truth.resolve()), name)
        values = metric_values(
            json.loads((official / "metric_result.json").read_text(encoding="utf-8"))
        )
        write_bytes(
            output / "scores.json",
            json_bytes(
                {
                    "metrics": values,
                    "overall": aggregate(values),
                    "qualification": "SCORER_ONLY; review coverage and failure catalog",
                }
            ),
        )
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
