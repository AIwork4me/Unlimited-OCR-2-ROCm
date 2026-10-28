"""Evidence aggregation; Gate E requires an actual comparable baseline pair."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any

from .config import file_sha256, read_lock
from .doctor import write_json


def compare_reproduction(
    baseline: dict[str, Any] | None, current: dict[str, Any]
) -> dict[str, Any]:
    failures = []
    if baseline is None:
        return {"status": "NOT PROVEN", "reasons": ["No linked successful baseline"]}
    for label, run in (("baseline", baseline), ("current", current)):
        if any(run.get("gates", {}).get(gate) != "PASS" for gate in "ABCD"):
            failures.append(f"{label} did not pass all A-D gates")
        if run.get("required_cases_passed") is not True:
            failures.append(f"{label} lacks required cases")
        if not run.get("environment_id"):
            failures.append(f"{label} lacks environment identity")
    if baseline.get("environment_id") == current.get("environment_id"):
        failures.append("Environment identities are not distinct")
    if current.get("fresh_environment") is not True:
        failures.append("Current run was not created as a fresh environment")
    for name in ("machine_identity", "artifact_hashes", "package_versions"):
        if not baseline.get(name) or baseline.get(name) != current.get(name):
            failures.append(f"Missing or different {name}")
    return {"status": "NOT PROVEN" if failures else "PASS", "reasons": failures}


def summarize(
    root: Path, fixtures: Path, fresh: bool, baseline_path: Path | None
) -> dict[str, Any]:
    def read(relative: str) -> dict[str, Any]:
        path = root / relative
        return json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else {}

    record = read("reproduction.json")
    doctor, triton, sparse = read("doctor.json"), read("triton.json"), read("sparse.json")
    load = read("single/load.json")
    single, multi = read("single/suite.json"), read("multi/suite.json")
    expected_single = {"english", "chinese", "mixed", "table", "math", "repeat_unicode_recovery"}
    expected_multi = {f"pages_{n}_run_{r}" for n in (2, 3, 5) for r in (1, 2)}
    expected_multi.add("table_2_pages")
    singles = {case["case"]: case["status"] for case in single.get("cases", [])}
    multis = {case["case"]: case["status"] for case in multi.get("cases", [])}
    single_ok = all(singles.get(case) == "PASS" for case in expected_single)
    multi_ok = all(multis.get(case) == "PASS" for case in expected_multi)
    record["required_cases_passed"] = single_ok and multi_ok
    english = read("single/english/manifest.json")
    record["gates"] = {
        "A": "PASS" if doctor.get("status") == "PASS" else "FAIL",
        "B": "PASS"
        if (
            load.get("status") == "PASS"
            and sparse.get("status") == "PASS"
            and triton.get("status") == "PASS"
            and english.get("native_kernel_count", 0) >= 12
        )
        else "NOT PROVEN",
        "C": "PASS" if single_ok else "NOT PROVEN",
        "D": "PASS" if multi_ok else "NOT PROVEN",
    }
    hardware = doctor.get("windows_hardware", {})
    uuid = hardware.get("machine_uuid_sha256")
    record["machine_identity"] = (
        {
            "uuid_sha256": uuid,
            "cpu": hardware.get("cpu", {}).get("Name", "").strip(),
            "build": hardware.get("os", {}).get("BuildNumber"),
            "gpu": doctor.get("gpu"),
            "driver": hardware.get("gpu"),
            "ram": doctor.get("memory", {}).get("system_total_bytes"),
        }
        if uuid
        else {}
    )
    record["environment_id"] = hashlib.sha256(
        str(Path(sys.prefix).resolve()).casefold().encode()
    ).hexdigest()
    record["fresh_environment"] = fresh
    package_root = Path(__file__).parent
    hashes = dict(load.get("wrapper_source_hashes", {}))
    hashes["upstream-lock.json"] = file_sha256(package_root / "upstream-lock.json")
    for path in sorted(fixtures.glob("*")):
        if path.is_file():
            hashes["fixture/" + path.name] = file_sha256(path)
    hashes.update(
        {
            "model/" + name: digest
            for name, digest in load.get("source_and_weight_hashes", {}).items()
        }
    )
    record["artifact_hashes"] = hashes
    record["model_revision"] = read_lock()["model_revision"]
    versions = {}
    lock = package_root.parents[1] / "requirements/windows-cp310.lock.txt"
    if lock.is_file():
        for line in lock.read_text(encoding="utf-8-sig").splitlines():
            if "==" in line:
                name = line.split("==", 1)[0]
                try:
                    versions[name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    versions[name] = "MISSING"
    record["package_versions"] = versions
    baseline = None
    if baseline_path is not None:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8-sig"))
        record["baseline_manifest_sha256"] = file_sha256(baseline_path)
    record["gate_E"] = compare_reproduction(baseline, record)
    record["same_physical_machine"] = (
        baseline is not None
        and bool(record["machine_identity"])
        and baseline.get("machine_identity") == record["machine_identity"]
    )
    record["phase1_status"] = "PASS" if record["gate_E"]["status"] == "PASS" else "BLOCKED"
    write_json(root / "reproduction.json", record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    report = summarize(args.root, args.fixtures, args.fresh, args.baseline)
    print(f"Workflow gates: {report['gates']}; Gate E: {report['gate_E']['status']}")
