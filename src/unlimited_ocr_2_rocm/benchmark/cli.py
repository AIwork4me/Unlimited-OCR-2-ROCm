"""Standalone benchmark tooling. No inference, model download or torch import."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .ledger import Run, json_bytes, write_bytes
from .manifest import DENOMINATORS, file_hash, inventory, local_path, validate_manifest
from .scorer import check_sources


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def fetch_scorer(source: Path, lock: dict[str, Any], profile: str) -> None:
    source.mkdir(parents=True, exist_ok=False)
    files = copy.deepcopy(lock["qwen"]["files"])
    patch = lock["datasets"]["MarathonDocBench"]["patch"] if profile == "marathon" else {}
    # Resolve the complete destination set before workers create directories.
    # Windows resolution of absent parents must not race those mutations.
    destinations = {relative: local_path(source, relative) for relative in {**files, **patch}}

    def download(item: tuple[str, str]) -> None:
        relative, sha = item
        if relative in patch:
            rev = lock["datasets"]["MarathonDocBench"]["revision"]
            url = f"https://huggingface.co/datasets/baidu/MarathonDocBench/resolve/{rev}/utils/mpdocbench_patch/{relative}"
        else:
            rev = lock["qwen"]["revision"]
            url = f"https://raw.githubusercontent.com/Tongyi-Zhiwen/Qwen-Doc/{rev}/MPDocBench/{relative}"
        path = destinations[relative]
        path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(1, 4):
            try:
                with urllib.request.urlopen(url, timeout=90) as response:
                    data = response.read()
                break
            except OSError as error:
                print(f"Source transport failure {relative}, attempt {attempt}: {error}")
                if attempt == 3:
                    raise
        if hashlib.sha256(data).hexdigest() != sha:
            raise ValueError(f"Downloaded source hash mismatch: {relative}")
        write_bytes(path, data)

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(download, {**files, **patch}.items()))
    check_sources(source, {"files": {**files, **patch}})


def score(args: argparse.Namespace) -> int:
    args.output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {"status": "BLOCKED", "overall": None}
    try:
        lock = read(args.lock)
        profile = copy.deepcopy(lock["qwen"])
        if args.profile == "marathon":
            profile["files"].update(lock["datasets"]["MarathonDocBench"]["patch"])
        check_sources(args.source, profile)
        if args.ground_truth:
            dataset = "MarathonDocBench" if args.profile == "marathon" else "MPDocBench"
            if file_hash(args.ground_truth) != lock["datasets"][dataset]["ground_truth_sha256"]:
                raise ValueError("Ground-truth hash does not match the frozen dataset")
        command = [
            str(args.python),
            "-m",
            "unlimited_ocr_2_rocm.benchmark.worker",
            "--source",
            str(args.source.resolve()),
            "--lock",
            str(args.lock.resolve()),
            "--output",
            str((args.output / "worker").resolve()),
            "--profile",
            args.profile,
        ]
        if args.ground_truth or args.predictions:
            if not args.ground_truth or not args.predictions:
                raise ValueError("Both ground truth and full prediction directory are required")
            command += [
                "--ground-truth",
                str(args.ground_truth.resolve()),
                "--predictions",
                str(args.predictions.resolve()),
            ]
        env = {**os.environ, "PYTHONUTF8": "1", "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
        with (
            (args.output / "stdout.log").open("xb") as stdout,
            (args.output / "stderr.log").open("xb") as stderr,
        ):
            proc = subprocess.run(
                command, stdout=stdout, stderr=stderr, timeout=args.timeout, env=env, check=False
            )
        report.update(
            exit_code=proc.returncode, scorer_lock_sha256=file_hash(args.lock), profile=args.profile
        )
        fixture_path = args.output / "worker/selftest.json"
        if fixture_path.exists():
            report["selftest"] = read(fixture_path)
            if proc.returncode == 0 and report["selftest"]["status"] == "PASS":
                report["status"] = "PASS"
                scored = args.output / "worker/scores.json"
                if scored.exists():
                    report.update(read(scored))
                    if report["overall"] is None:
                        report["status"] = "BLOCKED"
        else:
            report["reason"] = "SCORER_DID_NOT_COMPLETE_SELFTEST; inspect stderr.log"
    except subprocess.TimeoutExpired:
        report.update(status="TIMEOUT", reason="SCORER_PROCESS_TIMEOUT")
    except Exception as error:
        report["reason"] = f"{type(error).__name__}: {error}"
    write_bytes(args.output / "report.json", json_bytes(report))
    write_bytes(
        args.output / "SHA256SUMS.json",
        json_bytes(
            {
                p.relative_to(args.output).as_posix(): file_hash(p)
                for p in sorted(args.output.rglob("*"))
                if p.is_file()
            }
        ),
    )
    print(json.dumps({"status": report["status"], "overall": report["overall"]}))
    return 0 if report["status"] == "PASS" else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inv = commands.add_parser("inventory")
    inv.add_argument("--dataset", choices=DENOMINATORS, required=True)
    inv.add_argument("--ground-truth", type=Path, required=True)
    inv.add_argument("--dataset-root", type=Path, required=True)
    inv.add_argument("--lock", type=Path, required=True)
    inv.add_argument("--output", type=Path, required=True)
    val = commands.add_parser("validate")
    val.add_argument("--manifest", type=Path, required=True)
    val.add_argument("--dataset-root", type=Path, required=True)
    init = commands.add_parser("init")
    for flag in ("manifest", "dataset-root", "config", "environment", "output"):
        init.add_argument("--" + flag, type=Path, required=True)
    for name in ("todo", "begin", "finish", "report", "export"):
        command = commands.add_parser(name)
        command.add_argument("--run", type=Path, required=True)
        if name in {"begin", "finish"}:
            command.add_argument("--document", required=True)
        if name == "finish":
            command.add_argument("--attempt", required=True)
            command.add_argument("--state", required=True)
            command.add_argument("--raw", type=Path, required=True)
            command.add_argument("--metadata", type=Path, required=True)
        if name in {"report", "export"}:
            command.add_argument("--output", type=Path, required=True)
    for name in ("fetch-scorer", "score"):
        command = commands.add_parser(name)
        command.add_argument("--source", type=Path, required=True)
        command.add_argument("--lock", type=Path, required=True)
        command.add_argument("--profile", choices=["qwen", "marathon"], default="qwen")
        if name == "score":
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--python", type=Path, default=Path(sys.executable))
            command.add_argument("--ground-truth", type=Path)
            command.add_argument("--predictions", type=Path)
            command.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args(argv)
    if args.command == "score":
        return score(args)
    if args.command == "fetch-scorer":
        fetch_scorer(args.source, read(args.lock), args.profile)
    elif args.command == "inventory":
        pin = read(args.lock)["datasets"][args.dataset]
        if file_hash(args.ground_truth) != pin["ground_truth_sha256"]:
            raise ValueError("Frozen ground-truth hash mismatch")
        docs, pages = DENOMINATORS[args.dataset]
        manifest = inventory(
            args.ground_truth, args.dataset_root, args.dataset, pin["revision"], docs, pages
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_bytes(args.output, json_bytes(manifest))
        report = validate_manifest(manifest, args.dataset_root)
        print(json.dumps(report))
        return 0 if report["status"] == "PASS" else 2
    elif args.command == "validate":
        report = validate_manifest(read(args.manifest), args.dataset_root)
        print(json.dumps(report))
        return 0 if report["status"] == "PASS" else 2
    elif args.command == "init":
        manifest = read(args.manifest)
        if validate_manifest(manifest, args.dataset_root)["status"] != "PASS":
            raise ValueError("Run initialization requires verified local page bytes")
        print(Run.create(args.output, manifest, read(args.config), read(args.environment)).path)
    else:
        run = Run(args.run)
        if args.command == "todo":
            print(json.dumps(run.todo()))
        elif args.command == "begin":
            print(run.begin(args.document))
        elif args.command == "finish":
            run.finish(
                args.document,
                args.attempt,
                args.raw.read_bytes().decode("utf-8"),
                args.state,
                read(args.metadata),
            )
        elif args.command == "export":
            run.export(args.output)
        else:
            summary = run.summary()
            args.output.mkdir(parents=True, exist_ok=False)
            write_bytes(args.output / "summary.json", json_bytes(summary))
            stream = io.StringIO(newline="")
            writer = csv.writer(stream)
            writer.writerow(["document", "pages", "state", "attempts", "reason"])
            for row in summary["documents"]:
                writer.writerow(
                    [row["doc_id"], row["pages"], row["state"], len(row["attempts"]), row["reason"]]
                )
            write_bytes(args.output / "documents.csv", stream.getvalue().encode("utf-8"))
            write_bytes(
                args.output / "SHA256SUMS.json",
                json_bytes({p.name: file_hash(p) for p in args.output.iterdir() if p.is_file()}),
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
