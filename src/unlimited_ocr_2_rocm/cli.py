"""Command line entry point; model imports and downloads are explicit actions."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

from . import __version__
from .config import download_snapshot, read_lock, verify_files
from .doctor import bf16_probe, collect_environment, require_stack, sanitize, write_json
from .inference import run_native, validate_inputs
from .model import load_model


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="unlimited-ocr2-rocm")
    result.add_argument("--version", action="store_true")
    commands = result.add_subparsers(dest="command")
    doctor = commands.add_parser("doctor", help="strict identity and real BF16 GPU check")
    doctor.add_argument("--output", type=Path, default=Path("eval/results/doctor.json"))
    download = commands.add_parser("download", help="cache and verify immutable official files")
    download.add_argument("--cache-dir", type=Path)
    download.add_argument("--offline", action="store_true")
    for name in ("single", "multi", "smoke-single", "smoke-multi"):
        command = commands.add_parser(name)
        if name in {"single", "multi"}:
            command.add_argument("images", type=Path, nargs="+")
        else:
            command.add_argument("--fixtures", type=Path, default=Path("tests/fixtures"))
            if name == "smoke-single":
                command.add_argument(
                    "--cases", nargs="+", default=["english", "chinese", "mixed", "table"]
                )
            else:
                command.add_argument(
                    "--pages", nargs="+", type=int, choices=[2, 3, 5], default=[2, 3, 5]
                )
        command.add_argument("--cache-dir", type=Path)
        command.add_argument("--offline", action="store_true")
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--max-new-tokens", type=int, default=1024)
    return result


def smoke(args: argparse.Namespace) -> int:
    ground_truth = json.loads((args.fixtures / "ground_truth.json").read_text(encoding="utf-8"))
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a fresh suite output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    native = load_model(args.cache_dir, args.offline)
    write_json(args.output / "load.json", native.evidence)
    results = []
    if args.command == "smoke-single":
        for name in args.cases:
            truth = ground_truth["cases"][name]
            result = run_native(
                native,
                [args.fixtures / truth["image"]],
                args.output / name,
                truth=[truth],
                max_new_tokens=args.max_new_tokens,
            )
            results.append({"case": name, "status": result["status"]})
            print(f"{name}: {result['status']}", flush=True)
            if result["status"] != "PASS":
                break
        if results and all(item["status"] == "PASS" for item in results):
            truth = ground_truth["cases"][args.cases[0]]
            unicode_inputs = args.output / "中文 input"
            unicode_inputs.mkdir()
            image = unicode_inputs / "page with spaces.png"
            shutil.copyfile(args.fixtures / truth["image"], image)
            # A recoverable invalid-input error must not poison the loaded model.
            try:
                validate_inputs([unicode_inputs / "missing.png"], multi=False)
            except FileNotFoundError:
                pass
            else:
                raise RuntimeError("Missing-input guard failed")
            repeat_result = run_native(
                native,
                [image],
                args.output / "repeat_unicode_recovery",
                truth=[truth],
                max_new_tokens=args.max_new_tokens,
            )
            results.append({"case": "repeat_unicode_recovery", "status": repeat_result["status"]})
    else:
        for count in args.pages:
            truth = [ground_truth["cases"][f"page_{i}"] for i in range(1, count + 1)]
            images = [args.fixtures / page["image"] for page in truth]
            for repeat in range(2):
                name = f"pages_{count}_run_{repeat + 1}"
                result = run_native(
                    native,
                    images,
                    args.output / name,
                    multi=True,
                    truth=truth,
                    max_new_tokens=args.max_new_tokens * count,
                )
                results.append({"case": name, "status": result["status"]})
                print(f"{name}: {result['status']}", flush=True)
                if result["status"] != "PASS":
                    break
            if results[-1]["status"] != "PASS":
                break
    passed = bool(results) and all(item["status"] == "PASS" for item in results)
    write_json(
        args.output / "suite.json", {"status": "PASS" if passed else "FAIL", "cases": results}
    )
    return 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    if args.version:
        print(__version__)
        return 0
    if args.command is None:
        argument_parser.print_help()
        return 0
    try:
        if args.command == "doctor":
            report = collect_environment()
            require_stack(report)
            report["bf16"] = bf16_probe()
            report["status"] = "PASS"
            write_json(args.output, report)
        elif args.command == "download":
            snapshot = download_snapshot(args.cache_dir, args.offline)
            lock = read_lock()
            verify_files(snapshot, {**lock["files"], **lock["weights"]})
            print(sanitize(str(snapshot)))
        elif args.command.startswith("smoke-"):
            return smoke(args)
        else:
            images = validate_inputs(args.images, multi=args.command == "multi")
            native = load_model(args.cache_dir, args.offline)
            report = run_native(
                native,
                images,
                args.output,
                multi=args.command == "multi",
                max_new_tokens=args.max_new_tokens,
            )
            print(f"Inference: {report['status']} — {args.output / 'manifest.json'}")
            return 0 if report["status"] == "INFERENCE_ONLY" else 1
    except Exception as error:
        print(f"Error: {sanitize(str(error))}", file=sys.stderr)
        if hasattr(args, "output"):
            failure: dict[str, Any] = {"status": "FAIL", "traceback": traceback.format_exc()}
            path = args.output if args.command == "doctor" else args.output / "error.json"
            write_json(path, failure)
        return 1
    return 0
