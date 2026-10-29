"""Run real CDM controls against a verified official scorer on Linux.

Run with the isolated frozen CPU scorer Python, from the repository root.
The caller supplies process-local XeLaTeX, fonts, Node and ImageMagick.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path


def prepare_pipeline(output: Path) -> tuple[Path, Path]:
    """Extend the authored relation fixture with one displayed formula."""
    from unlimited_ocr_2_rocm._upstream.process_unlimitedOCR2 import convert

    fixtures = Path("tests/fixtures/benchmark")
    gt = json.loads((fixtures / "synthetic.json").read_text(encoding="utf-8"))
    gt[0]["layout_dets"].append(
        {
            "anno_id": 7,
            "category_type": "equation_isolated",
            "latex": "x+y=z",
            "order": 7,
            "page_id": 2,
            "poly": [300, 300, 500, 300, 500, 400, 300, 400],
            "ignore": False,
            "attribute": {},
        }
    )
    raw = (fixtures / "synthetic.raw.txt").read_text(encoding="utf-8").rstrip()
    raw += "\n<BLOCK>equation[300,300,500,400]<FORK>$$x+y=z$$\n"
    gt_path = output / "ground-truth.json"
    gt_path.write_text(json.dumps(gt, indent=2), encoding="utf-8")
    (output / "prediction.raw.txt").write_text(raw, encoding="utf-8")
    predictions = output / "predictions"
    predictions.mkdir()
    (predictions / "synthetic.md").write_text(str(convert(raw)), encoding="utf-8")
    return gt_path, predictions


def classify_cdm(
    name: str,
    score: float | None,
    rendered: bool,
    timed_out: bool,
    timeout_renderer_started: bool = False,
    error_observed: bool = False,
) -> str:
    if name == "timeout" and timed_out and timeout_renderer_started:
        return "EXPECTED_TIMEOUT"
    if timed_out:
        return "FAIL"
    if name == "error" and score == 0 and not rendered and error_observed:
        return "EXPECTED_RENDER_ERROR"
    if name == "exact" and rendered and score == 1:
        return "PASS"
    if (
        name == "mismatch"
        and rendered
        and score is not None
        and math.isclose(score, 0.4, abs_tol=1e-7)
    ):
        return "PASS"
    return "FAIL"


def pipeline_passed(scores: dict, coverage: dict) -> bool:
    from unlimited_ocr_2_rocm.benchmark.scorer import METRICS

    return (
        all(scores.get("metrics", {}).get(name) == 100 for name in METRICS)
        and scores.get("overall") == 100
        and coverage.get("tasks", {}).get("display_formula", {}).get("sample_count") == 1
    )


def render_artifacts(output: Path, name: str, subset: str) -> dict:
    from PIL import Image

    result = {"rendered": False, "valid_boxes": 0, "png_valid": False}
    try:
        rows = [
            json.loads(line)
            for line in (output / subset / "bbox" / f"{name}.jsonl").read_text().splitlines()
        ]
        result["valid_boxes"] = sum(
            1
            for row in rows
            if len(row.get("bbox", [])) == 4
            and row["bbox"][2] > row["bbox"][0]
            and row["bbox"][3] > row["bbox"][1]
            and row.get("token")
        )
        image_path = output / subset / "vis" / f"{name}_base.png"
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            result["png_valid"] = (
                image.width > 0
                and image.height > 0
                and any(low < 255 for low, _ in image.convert("RGB").getextrema())
            )
            result["image_size"] = list(image.size)
        result["rendered"] = result["valid_boxes"] > 0 and result["png_valid"]
    except (OSError, ValueError, TypeError, KeyError):
        pass
    return result


def find_renderer(process_group: int) -> dict | None:
    """Observe the actual prediction renderer, including unflushed TeX logs."""
    for directory in Path("/proc").iterdir():
        if not directory.name.isdecimal():
            continue
        try:
            pid = int(directory.name)
            if os.getpgid(pid) != process_group:
                continue
            executable = os.readlink(directory / "exe")
            command = (directory / "cmdline").read_bytes().decode().strip("\0").split("\0")
            if Path(executable).name == "xetex" and any(
                "temp_dir_pred_timeout" in arg for arg in command
            ):
                return {
                    "pid": pid,
                    "process_group": process_group,
                    "executable": executable,
                    "command": command,
                }
        except (OSError, ValueError):
            continue
    return None


def active_group_processes(process_group: int) -> list[int]:
    result = []
    for directory in Path("/proc").iterdir():
        if not directory.name.isdecimal():
            continue
        try:
            pid = int(directory.name)
            state = (directory / "stat").read_text().rpartition(")")[2].split()[0]
            if state not in {"Z", "X"} and os.getpgid(pid) == process_group:
                result.append(pid)
        except (OSError, ValueError, IndexError):
            continue
    return result


def cleanup_process_group(proc: subprocess.Popen) -> list[int]:
    """Clean up descendants even if their leader exited or observation failed."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()
    deadline = time.monotonic() + 2
    while survivors := active_group_processes(proc.pid):
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Renderer processes survived cleanup: {survivors}")
        time.sleep(0.01)
    return []


def child(source: Path, output: Path, name: str, profile_name: str) -> None:
    import importlib.metadata

    from unlimited_ocr_2_rocm.benchmark.manifest import file_hash
    from unlimited_ocr_2_rocm.benchmark.scorer import check_dependencies, check_sources

    lock = json.loads(Path("eval/benchmarks/benchmark-lock.json").read_text(encoding="utf-8"))
    profile = lock["qwen"]
    if profile_name == "marathon":
        profile["files"].update(lock["datasets"]["MarathonDocBench"]["patch"])
    check_sources(source, profile)
    requirements = Path("requirements/benchmark-scorer.lock.txt")
    if file_hash(requirements) != lock["scorer_requirements_sha256"]:
        raise ValueError("Frozen scorer dependency lock hash mismatch")
    check_dependencies(
        {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        requirements.read_text(encoding="utf-8").splitlines(),
    )
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source.resolve()))
    from metrics.cdm_metric import CDM

    pairs = {
        "exact": ("x+y=z", "x+y=z"),
        "mismatch": ("x+y=z", "a+b=c"),
        "error": ("x+y=z", r"\ThisCommandDoesNotExist"),
        "timeout": ("x+y=z", r"\loop\iftrue x\repeat"),
    }
    output.mkdir(parents=True, exist_ok=False)
    gt, pred = pairs[name]
    (output / "input.json").write_text(json.dumps({"gt": gt, "pred": pred}), encoding="utf-8")
    result = CDM(output_root=str(output)).evaluate(gt, pred, name)
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if name == "error":
        # CDM deletes its temporary TeX logs and swallows renderer exceptions.
        # Reproduce at the unchanged renderer boundary to retain the full error.
        from metrics.cdm.modules.latex2bbox_color import latex2bbox_color

        diagnostic = output / "diagnostic"
        for folder in ("bbox", "vis", "temp"):
            (diagnostic / folder).mkdir(parents=True)
        try:
            latex2bbox_color(
                (pred, name, str(diagnostic), str(diagnostic / "temp"), CDM.gen_color_list(5800))
            )
        except Exception:
            traceback.print_exc()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=["qwen", "marathon"], default="qwen")
    parser.add_argument("--pipeline", action="store_true")
    parser.add_argument("--case", choices=["exact", "mismatch", "error", "timeout"])
    args = parser.parse_args()
    args.output = args.output.resolve()
    if not args.output.is_relative_to(Path.cwd().resolve()):
        raise ValueError("Output must stay inside the development root")
    if args.case:
        child(args.source, args.output, args.case, args.profile)
        return 0
    if os.name != "posix":
        raise RuntimeError("Run in the isolated Linux CPU environment")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.pipeline:
        gt_path, predictions = prepare_pipeline(args.output)
        code = subprocess.call(
            [
                sys.executable,
                "-m",
                "unlimited_ocr_2_rocm.benchmark.worker",
                "--source",
                str(args.source),
                "--output",
                str(args.output / "worker"),
                "--lock",
                "eval/benchmarks/benchmark-lock.json",
                "--profile",
                args.profile,
                "--ground-truth",
                str(gt_path),
                "--predictions",
                str(predictions),
            ]
        )
        if code:
            return code
        scores = json.loads((args.output / "worker/scores.json").read_text())
        coverage = json.loads((args.output / "worker/coverage.json").read_text())
        if not pipeline_passed(scores, coverage):
            raise RuntimeError("Full eight-metric formula pipeline failed; inspect scores/coverage")
        return 0
    results = {}
    for name in ("exact", "mismatch", "error", "timeout"):
        output = args.output / name
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--source",
            str(args.source),
            "--output",
            str(output),
            "--case",
            name,
            "--profile",
            args.profile,
        ]
        started = time.monotonic()
        timed_out = False
        renderer = None
        with (args.output / f"{name}.log").open("wb") as log:
            proc = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            try:
                if name != "timeout":
                    proc.wait(timeout=120)
                else:
                    deadline = started + 120
                    while proc.poll() is None:
                        if renderer is None:
                            renderer = find_renderer(proc.pid)
                            if renderer:
                                renderer["observed_after_seconds"] = time.monotonic() - started
                                (output / "timeout-process.json").write_text(json.dumps(renderer))
                                deadline = time.monotonic() + 0.05
                        if time.monotonic() >= deadline:
                            raise subprocess.TimeoutExpired(command, 0.05 if renderer else 120)
                        try:
                            proc.wait(timeout=0.01)
                        except subprocess.TimeoutExpired:
                            pass
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                survivors = cleanup_process_group(proc)
        result_path = output / "result.json"
        original = json.loads(result_path.read_text()) if result_path.exists() else None
        artifacts = {subset: render_artifacts(output, name, subset) for subset in ("gt", "pred")}
        rendered = all(row["rendered"] for row in artifacts.values())
        score = original.get("F1_score") if original else None
        timeout_renderer_started = artifacts["gt"]["rendered"] and renderer is not None
        error_observed = (
            artifacts["gt"]["rendered"]
            and not artifacts["pred"]["rendered"]
            and any(
                any(
                    marker in path.read_text(encoding="utf-8", errors="replace")
                    for marker in (
                        "Emergency stop",
                        "Runaway argument",
                        "Fatal error occurred",
                        "Undefined control sequence",
                    )
                )
                for path in (output / "diagnostic/temp").glob("*.log")
            )
        )
        results[name] = {
            "status": classify_cdm(
                name, score, rendered, timed_out, timeout_renderer_started, error_observed
            ),
            "original_result": original,
            "rendered_both": rendered,
            "artifacts": artifacts,
            "error_observed": error_observed,
            "timed_out": timed_out,
            "timeout_renderer_started": timeout_renderer_started,
            "returncode": proc.returncode,
            "active_processes_after_cleanup": survivors,
            "elapsed_seconds": time.monotonic() - started,
            "timeout_seconds": 0.05 if name == "timeout" else 120,
            "startup_timeout_seconds": 120,
            "command": command,
        }
    (args.output / "cases.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return int(any(row["status"] == "FAIL" for row in results.values()))


if __name__ == "__main__":
    raise SystemExit(main())
