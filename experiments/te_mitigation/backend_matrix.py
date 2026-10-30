"""Separate backend, cold/warm, process repeat and prior-request comparisons."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/cold_start"))
from worker import load, sha, write  # noqa: E402

analysis = load("te_matched_backend_analysis", ROOT / "experiments/te_mitigation/analyze.py")
publication = load("te_matched_backend_publication", ROOT / "experiments/te_mitigation/publish.py")


def matrix(root):
    qualification = analysis.campaign(root)
    declaration = analysis.read(root / "predeclared.json")
    records = {}
    for item in declaration["schedule"]:
        modes = item["modes"]
        folder = root / f"eager-{modes}-{item['repeat']}" / "data"
        observer = analysis.read(folder / "observers.json")
        for ordinal in (0, 1):
            summary = analysis.read(folder / f"worker/request-{ordinal}/summary.json")
            records[modes, item["repeat"], ordinal] = dict(
                native=summary["native"],
                stages=[s for s in observer["stages"] if s["request"] == ordinal],
            )
    categories = {
        k: []
        for k in (
            "same_order_fresh_process_repeat",
            "within_backend_cold_warm",
            "matched_cold_backend",
            "matched_warm_backend",
            "prior_backend_interaction",
        )
    }

    def pair(category, left, right):
        a, b = records[left], records[right]
        stages = []
        for x, y in zip(a["stages"], b["stages"], strict=True):
            stages.append(
                dict(
                    call=x["call"],
                    stage=x["stage"],
                    contracts_exact=x["call"] == y["call"]
                    and x["stage"] == y["stage"]
                    and all(x[k] == y[k] for k in ("shape", "stride", "dtype", "device")),
                    bytes_exact=x["sha256"] == y["sha256"],
                    hashes=[x["sha256"], y["sha256"]],
                )
            )
        categories[category].append(
            dict(
                left=left,
                right=right,
                stages=stages,
                comparison=publication.comparison(analysis.bbox.compare(a["native"], b["native"])),
            )
        )

    for modes in ("MATH,MATH", "AUTO,AUTO", "MATH,AUTO", "AUTO,MATH"):
        for ordinal in (0, 1):
            pair("same_order_fresh_process_repeat", (modes, 0, ordinal), (modes, 1, ordinal))
    for repeat in (0, 1):
        for modes in ("MATH,MATH", "AUTO,AUTO"):
            pair("within_backend_cold_warm", (modes, repeat, 0), (modes, repeat, 1))
        for left, right in (("MATH,MATH", "AUTO,AUTO"), ("MATH,AUTO", "AUTO,MATH")):
            pair("matched_cold_backend", (left, repeat, 0), (right, repeat, 0))
        for left, right in (("MATH,MATH", "MATH,AUTO"), ("AUTO,MATH", "AUTO,AUTO")):
            pair("matched_warm_backend", (left, repeat, 1), (right, repeat, 1))
        for left, right in (("MATH,MATH", "AUTO,MATH"), ("AUTO,AUTO", "MATH,AUTO")):
            pair("prior_backend_interaction", (left, repeat, 1), (right, repeat, 1))
    exact = {
        name: all(p["comparison"]["acceptance"] == "EXACT_PASS" for p in pairs)
        for name, pairs in categories.items()
    }
    stage_exact = {
        name: all(s["bytes_exact"] and s["contracts_exact"] for p in pairs for s in p["stages"])
        for name, pairs in categories.items()
    }
    stable = all(
        exact[k] and stage_exact[k]
        for k in (
            "same_order_fresh_process_repeat",
            "within_backend_cold_warm",
            "prior_backend_interaction",
        )
    )
    return dict(
        producer_sha256=sha(__file__),
        declaration_sha256=sha(root / "predeclared.json"),
        physical_functional=qualification["all_physical_and_functional_gates"],
        categories=categories,
        exact_by_category=exact,
        stage_exact_by_category=stage_exact,
        causal_controls_stable=stable,
        verdict="EXACT_PASS"
        if stable and all(exact.values()) and qualification["all_physical_and_functional_gates"]
        else "EXACT_FAIL_OR_UNCONTROLLED",
        scope=(
            "New candidate-wheel two-page eager-QuickGELU context only; "
            "historical PR15 FAIL unchanged"
        ),
        recommendation="KEEP EXPERIMENTAL",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write(args.output, matrix(args.root))


if __name__ == "__main__":
    main()
