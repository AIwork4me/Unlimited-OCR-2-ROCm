"""Conservative CPU-only interpretation of immutable RCA process reports."""

import argparse
import json
import re
from pathlib import Path
from typing import Any

TOLERANCES = {"bfloat16": (0.03, 0.03), "float16": (0.005, 0.005), "float32": (1e-4, 1e-4)}
FUSED = {"FLASH_ATTENTION", "EFFICIENT_ATTENTION"}


def read_report(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"Nonfinite JSON constant: {value}")

    result = json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=reject)
    if not isinstance(result, dict):
        raise ValueError("Report must be a JSON object")
    return result


def classify_case(report: dict[str, Any], process_exit: int) -> str:
    if report.get("status") == "UNSUPPORTED" and process_exit == report.get("exit_code") == 2:
        return "UNSUPPORTED"
    events = report.get("events", [])
    if process_exit or report.get("exit_code") != 0 or report.get("status") == "FAIL":
        return "FAIL"
    if report.get("traceback") or report.get("downstream_traceback"):
        return "FAIL"
    if any(
        e.get("hip_before_sync") or e.get("hip_after_sync") or e.get("traceback") for e in events
    ):
        return "FAIL"
    if report.get("acceptance") is not True or report.get("status") != "PASS":
        return "INCONCLUSIVE"
    args = report.get("arguments", {})
    stages = {e.get("stage") for e in events}
    needed = {"before_sdpa", "after_sdpa"}
    if args.get("operation") in ("copy", "arithmetic", "cast", "copy_only"):
        needed.add("after_" + args["operation"])
    if not needed <= stages or any(
        e.get("hip_before_sync") != 0
        or e.get("hip_after_sync") != 0
        or e.get("synchronize") != "OK"
        for e in events
    ):
        return "INCONCLUSIVE"
    if args.get("operation") == "copy_only":
        return "CONTROL_PASS" if report.get("copy_correct") is True else "INCONCLUSIVE"
    stats = report.get("numerical", {})
    tol = TOLERANCES.get(args.get("dtype"))
    if not stats or not tol or (stats.get("atol"), stats.get("rtol")) != tol:
        return "INCONCLUSIVE"
    if any(stats.get(k) for k in ("failing_elements", "nonfinite_actual", "nonfinite_reference")):
        return "FAIL"
    if stats.get("total_elements", 0) <= 0 or any(
        stats.get(k) != 0 for k in ("failing_elements", "nonfinite_actual", "nonfinite_reference")
    ):
        return "INCONCLUSIVE"
    actual, requested = report.get("backend_observed"), report.get("backend_requested")
    if requested != "DEFAULT" and requested != actual:
        return "INCONCLUSIVE"
    if actual in FUSED:
        return "FUSED_PASS"
    return "MATH_PASS" if actual == "MATH" else "INCONCLUSIVE"


def valid_provenance(p: dict[str, Any]) -> bool:
    digest, source = p.get("built_wheel_sha256", ""), p.get("source_sha", "")
    return bool(
        re.fullmatch(r"[0-9a-f]{64}", digest)
        and digest == p.get("installed_wheel_sha256")
        and re.fullmatch(r"[0-9a-f]{40}", source)
        and source == p.get("installed_git_version")
        and p.get("architecture") == "gfx1151"
        and p.get("python_abi") == "cp310"
        and p.get("platform") == "win_amd64"
        and all(
            p.get(k) is True
            for k in (
                "import_within_venv",
                "loaded_binaries_within_venv",
                "runtime_images_verified",
            )
        )
        and p.get("pip_check_exit") == 0
        and p.get("source_overrides") == {}
    )


def decide(
    wheel_built: bool,
    installed: bool,
    provenance_ok: bool,
    statuses: list[str],
    *,
    matrix_complete: bool = False,
) -> str:
    """matrix_complete requires fixture/route/repetition checks by the caller."""
    if not wheel_built:
        return "BUILD_BLOCKED"
    if not installed:
        return "INSTALL_BLOCKED"
    if not provenance_ok:
        return "INCONCLUSIVE"
    if "FAIL" in statuses:
        return "REGRESSION_PERSISTS"
    if (
        matrix_complete
        and "FUSED_PASS" in statuses
        and "MATH_PASS" in statuses
        and all(s in {"FUSED_PASS", "MATH_PASS", "CONTROL_PASS"} for s in statuses)
    ):
        return "FIX_VERIFIED"
    return "INCONCLUSIVE"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    records = []
    for process in read_report(args.directory / "processes.json")["processes"]:
        path = args.directory / process["report"]
        r = read_report(path) if path.exists() else {}
        records.append(
            {
                "case": process["case"],
                "report": process["report"],
                "process_exit": process["exit_code"],
                "status": classify_case(r, process["exit_code"]),
                "requested": r.get("backend_requested"),
                "observed": r.get("backend_observed"),
                "fixture_sha256": r.get("fixture_sha256"),
                "numerical": r.get("numerical"),
                "hip_events": r.get("events"),
            }
        )
    (args.directory / "summary.json").write_text(json.dumps({"cases": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()
