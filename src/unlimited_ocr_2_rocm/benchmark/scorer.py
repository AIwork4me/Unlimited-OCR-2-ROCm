"""Hash gates and presentation of original MPDocBench metric outputs (no metric substitutes)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .manifest import file_hash, local_path

METRICS = {
    "TextEdit": ("text_block", "Edit_dist", "ALL_page_avg", True),
    "T-TextEdit": ("merged_text_block", "Edit_dist", "ALL_page_avg", True),
    "TableTEDS": ("table", "TEDS", "all", False),
    "T-TableTEDS": ("merged_table", "TEDS", "all", False),
    "FormulaCDM": ("display_formula", "CDM", "all", False),
    "ReadOrderEdit": ("reading_order", "Edit_dist", "ALL_page_avg", True),
    "FigureF1": ("figure", "FigureF1", "all", False),
    "HeadingTEDS": ("head", "HeadTEDS", "all", False),
}


def check_dependencies(packages: dict[str, str], requirements: list[str]) -> None:
    from packaging.requirements import Requirement

    observed = {name.lower().replace("_", "-"): version for name, version in packages.items()}
    if "torch" in observed:
        raise ValueError("Official scorer requires a separate torch-free CPU environment")
    for line in requirements:
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        req = Requirement(line)
        if req.marker and not req.marker.evaluate():
            continue
        version = observed.get(req.name.lower().replace("_", "-"))
        if version is None or not req.specifier.contains(version):
            raise ValueError(f"Frozen scorer dependency mismatch: {req.name}: {version}")


def sample_coverage(documents: list[str], samples: dict[str, Any]) -> dict[str, Any]:
    """Expose official sample populations; absence is never called not-applicable."""
    tasks = {}
    for task, rows in samples.items():
        rows = rows if isinstance(rows, list) else rows.samples
        counts = {name: 0 for name in documents}
        for row in rows:
            name = row["img_id"]
            if name not in counts:
                raise ValueError(f"Scorer produced unknown document: {name}")
            counts[name] += 1
        tasks[task] = {
            "sample_count": sum(counts.values()),
            "documents_with_samples": sum(value > 0 for value in counts.values()),
            "documents_without_samples": [name for name, count in counts.items() if not count],
            "per_document_samples": counts,
        }
    return {"expected_documents": len(documents), "tasks": tasks}


def check_sources(source: Path, lock: dict[str, Any]) -> None:
    for relative, sha in lock["files"].items():
        path = local_path(source, relative)
        if not path.is_file() or file_hash(path) != sha:
            raise ValueError(f"Official scorer hash mismatch: {relative}")
    for path in source.rglob("*"):
        if path.name == "__pycache__" or path.suffix.lower() in {
            ".pyc",
            ".pyo",
            ".pyd",
            ".so",
            ".dll",
        }:
            raise ValueError(f"Refusing unlocked import artifact: {path.name}")
        if (
            path.suffix in {".py", ".js"}
            and path.relative_to(source).as_posix() not in lock["files"]
        ):
            raise ValueError(f"Refusing unlocked executable scorer file: {path.name}")


def metric_values(result: dict[str, Any]) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    for name, (task, metric, field, invert) in METRICS.items():
        value = result.get(task, {}).get("all", {}).get(metric, {}).get(field)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and 0 <= value <= 1
        ):
            values[name] = 100 * (1 - value if invert else value)
        else:
            values[name] = None
    return values


def aggregate(values: dict[str, float | None]) -> float | None:
    """Marathon's documented mean of eight higher-is-better percentages."""
    numbers = [values.get(name) for name in METRICS]
    if any(v is None or not math.isfinite(v) or not 0 <= v <= 100 for v in numbers):
        return None
    return sum(v for v in numbers if v is not None) / 8
