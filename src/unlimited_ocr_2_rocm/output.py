"""Non-executing inspection of official raw page and layout markup."""

from __future__ import annotations

import html
import re
import unicodedata
from pathlib import Path
from typing import Any

from ._upstream.process_unlimitedOCR2 import _parse_blocks


def readable_text(text: str) -> str:
    if re.search(r"<\s*BLOCK\s*>", text, flags=re.IGNORECASE):
        # Preserve per-page attribution; official convert() may merge cross-page
        # fragments for Markdown. Use its exact reviewed block parser for CER.
        text = "\n".join(block["content"] for block in _parse_blocks(text))
    text = re.sub(r"<\|ref\|>.*?<\|/ref\|>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|det\|>.*?<\|/det\|>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|.*?\|>", "", text, flags=re.DOTALL)
    text = re.sub(r"</?[^>]+>", " ", text)
    return unicodedata.normalize("NFKC", html.unescape(text))


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", readable_text(text))


def preserve_raw(path: Path, raw: str) -> None:
    path.write_bytes(raw.encode("utf-8"))


def split_pages(raw: str) -> list[str]:
    return raw.split("<PAGE>")[1:] if "<PAGE>" in raw else [raw]


def character_error_rate(expected: str, predicted: str) -> float:
    if not expected:
        return float(bool(predicted))
    previous = list(range(len(predicted) + 1))
    for i, left in enumerate(expected, 1):
        current = [i]
        for j, right in enumerate(predicted, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1] / len(expected)


def assess(raw: str, truth: list[dict[str, Any]], max_cer: float = 0.02) -> dict[str, Any]:
    if not truth or any(not normalize(page["text"]) for page in truth):
        raise ValueError("nonempty ground truth required for every page")
    pages = split_pages(raw)
    raw_markers = raw.count("<PAGE>")
    terminal = raw_markers == len(truth) + 1 and not pages[-1].strip()
    if terminal:
        pages = pages[:-1]
    count_ok = len(pages) == len(truth) and raw_markers == len(truth) + int(terminal)
    prefix_ok = not normalize(raw.split("<PAGE>")[0]) if "<PAGE>" in raw else False
    comparisons = []
    for index, expected in enumerate(truth):
        predicted = normalize(pages[index]) if index < len(pages) else ""
        normalized_truth = normalize(expected["text"])
        cer = character_error_rate(normalized_truth, predicted)
        readable = readable_text(pages[index]) if index < len(pages) else ""
        numbers = re.findall(r"\d+(?:\.\d+)?", readable)
        important_numbers_ok = all(str(number) in numbers for number in expected.get("numbers", []))
        structure_ok = not expected.get("requires_table") or (
            index < len(pages)
            and "<table" in pages[index].lower()
            and "</table>" in pages[index].lower()
        )
        comparisons.append(
            {
                "page": index + 1,
                "ground_truth": expected["text"],
                "normalized_truth": normalized_truth,
                "normalized_prediction": predicted,
                "character_error_rate": cer,
                "important_numbers_match": important_numbers_ok,
                "structure_valid": bool(structure_ok),
                "status": "PASS"
                if cer <= max_cer and important_numbers_ok and structure_ok
                else "FAIL",
            }
        )
    return {
        "status": "PASS"
        if count_ok and prefix_ok and all(page["status"] == "PASS" for page in comparisons)
        else "FAIL",
        "threshold_max_cer": max_cer,
        "input_page_count": len(truth),
        "raw_page_markers": raw.count("<PAGE>"),
        "content_page_count": len(pages),
        "terminal_empty_marker": terminal,
        "raw_marker_count_matches_input": raw_markers == len(truth),
        "page_count_match": count_ok,
        "no_substantive_preamble": prefix_ok,
        "ordered_page_content_match": all(page["status"] == "PASS" for page in comparisons),
        "pages": comparisons,
    }
