"""Retain the official native document result without changing generation."""

from __future__ import annotations

import math
import re
from typing import Any

from ._upstream.process_unlimitedOCR2 import BLOCK_RE, _parse_blocks

_BBOX = r"\[\s*-?\d+(?:\s*[,，]\s*-?\d+){3,}\s*\]"
_LEVEL = r"<LEVEL\d+>"
_CANONICAL_HEADER = re.compile(
    r"<BLOCK>\s*[a-zA-Z_]+"
    rf"(?:\s*{_LEVEL}(?:\s*{_BBOX})?|\s*{_BBOX}(?:\s*{_LEVEL})?)?"
    r"\s*<FORK>"
)


def diagnostic_evidence(value: Any) -> tuple[Any, bool]:
    """Copy JSON values; tag invalid values without calling arbitrary repr/code.

    The boolean identifies malformed evidence, including unsupported extra
    fields. Valid official native results retain their original JSON values.
    """
    invalid = False
    ancestors: set[int] = set()

    def visit(item: Any, depth: int = 0) -> Any:
        nonlocal invalid
        if item is None or type(item) is bool:
            return item
        if type(item) is str:
            try:
                item.encode("utf-8")
            except UnicodeEncodeError:
                invalid = True
                return {"invalid_string": ascii(item)}
            return item
        if type(item) is int:
            try:
                str(item)
            except ValueError:
                invalid = True
                return {"invalid_integer": "serialization_digit_limit"}
            return item
        if type(item) is float:
            if math.isfinite(item):
                return item
            invalid = True
            return {"nonfinite_float": str(item)}
        if type(item) not in (list, dict):
            invalid = True
            kind = type(item)
            return {"invalid_type": kind.__module__ + "." + kind.__qualname__}
        if depth >= 32 or id(item) in ancestors:
            invalid = True
            return {"invalid_container": "depth_limit" if depth >= 32 else "cycle"}
        ancestors.add(id(item))
        try:
            if type(item) is list:
                return [visit(child, depth + 1) for child in item]
            entries = [(visit(key, depth + 1), child) for key, child in item.items()]
            if any(type(key) is not str for key, _ in entries):
                invalid = True
                return {
                    "invalid_mapping": [
                        {"key": key, "value": visit(child, depth + 1)} for key, child in entries
                    ]
                }
            return {key: visit(child, depth + 1) for key, child in entries}
        finally:
            ancestors.remove(id(item))

    result = visit(value)
    return result, invalid


def infer_document(
    model: Any,
    tokenizer: Any,
    image_files: list[str],
    *,
    multi: bool,
    max_new_tokens: int,
    no_repeat_ngram_size: int = 0,
    ngram_window: int = 0,
) -> Any:
    if not image_files or (not multi and len(image_files) != 1):
        raise ValueError("Expected one image or an ordered full document")
    options: dict[str, Any] = {}
    if multi:
        options["crop_mode"] = False
    if no_repeat_ngram_size > 0:
        # Opt-in repetition suppression; the caller's policy gate has already
        # validated the exact (size, window) pair (inference.validate_ngram).
        options["no_repeat_ngram_size"] = no_repeat_ngram_size
        options["ngram_window"] = ngram_window
    return model.infer(
        tokenizer,
        image_file=list(image_files) if multi else image_files[0],
        save_results=False,
        eval_mode=True,
        return_dict=True,
        max_new_tokens=max_new_tokens,
        **options,
    )


def completion_metadata(result: Any, *, expected_pages: int, max_new_tokens: int) -> dict[str, Any]:
    """Validate native metadata; EOS/length is never inferred from token count.

    This proves native termination metadata consistency, not semantic page
    completeness. In particular the official page guard can force EOS.
    """
    _, invalid = diagnostic_evidence(result)
    if invalid:
        raise ValueError("Native result contains invalid JSON metadata")
    required = {
        "text",
        "token_ids",
        "prompt_token_ids",
        "finish_reason",
        "num_pages_out",
        "expected_pages",
        "time",
        "prefill_time",
    }
    if not isinstance(result, dict) or not required <= result.keys():
        raise ValueError("Missing native completion dictionary fields")
    if not isinstance(result["text"], str) or result["finish_reason"] not in ("stop", "length"):
        raise ValueError("Invalid native text or finish reason")
    for name in ("token_ids", "prompt_token_ids"):
        ids = result[name]
        if not isinstance(ids, list) or any(type(token) is not int or token < 0 for token in ids):
            raise ValueError("Invalid native token IDs")
    tokens = len(result["token_ids"])
    if tokens > max_new_tokens:
        raise ValueError("Native output exceeds requested token budget")
    # The pinned loop selects EOS before appending it. A stop needs one more
    # iteration than the returned non-EOS IDs; a stop at the cap is inconsistent.
    if result["finish_reason"] == "stop" and tokens >= max_new_tokens:
        raise ValueError("Native EOS stop has no remaining decode iteration")
    for name in ("expected_pages", "num_pages_out"):
        if type(result[name]) is not int or result[name] < 0:
            raise ValueError("Invalid native page count")
    page_id = 128827  # Frozen checkpoint/config token; no tokenizer substitution.
    if (
        result["expected_pages"] != expected_pages
        or result["prompt_token_ids"].count(page_id) != expected_pages
        or result["num_pages_out"] != result["token_ids"].count(page_id)
        or result["num_pages_out"] != result["text"].count("<PAGE>")
    ):
        raise ValueError("Native page counts disagree with inputs, IDs or raw text")
    for name in ("time", "prefill_time"):
        value = result[name]
        if name == "prefill_time" and value is None:
            continue
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("Invalid native generation timing")
    return {
        "finish_reason": result["finish_reason"],
        "output_tokens": tokens,
        "max_tokens": max_new_tokens,
        "native_generation": result,
    }


def require_complete_pages(raw: str, expected_pages: int, *, max_chars: int | None = None) -> None:
    """Require canonical page structure; the caller owns any resource bound."""
    if max_chars is not None and len(raw) > max_chars:
        raise ValueError("Native page validation exceeds bounded input size")
    pages = raw.split("<PAGE>")
    if pages[-1:] == [""]:
        pages = pages[:-1]
    if not raw.strip() or not pages or pages[0].strip() or len(pages) - 1 != expected_pages:
        raise ValueError("Empty, malformed or missing native output pages")
    for page in pages[1:]:
        headers = list(BLOCK_RE.finditer(page))
        blocks = _parse_blocks(page)
        # The official converter deliberately repairs/falls back on malformed
        # text. Such a useful readable representation cannot prove validity.
        if (
            not headers
            or len(headers) != page.count("<BLOCK>")
            or len(blocks) != len(headers)
            or any(block["label"] == "__raw__" for block in blocks)
            or page[: headers[0].start()].replace("<MERGE>", "").strip()
            or any(not header.group().rstrip().endswith("<FORK>") for header in headers)
            or any(_CANONICAL_HEADER.fullmatch(header.group()) is None for header in headers)
        ):
            raise ValueError("Empty, malformed or missing native output blocks")
