"""Post-hoc replay of the n-gram suppression and page guard over emitted ids.

Reads the token_ids.json persisted by the P2-B worker and deterministically
replays, per decode step, (a) the frozen no-repeat n-gram ban
(docs/rca/NGRAM35_SOURCE_AUDIT.md §2: trailing window over generated ids only,
prefix-match successors) and (b) the native page-guard state machine
(modeling_unlimitedocr.py _uocr_page_count_guard). No logits are persisted, so
the replay yields ban sets/counts and guard decisions, never the
counterfactual un-suppressed token.

The forced-step consistency check detects the audited native corner
(NGRAM35_SOURCE_AUDIT.md §7): when the guard cap branch would force a token,
the emitted token must equal it — an all--inf row after the n-gram ban makes
argmax tie-break arbitrarily instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_PAGE_TOKEN_ID = 128827
DEFAULT_EOS_ID = 1
DEFAULT_MAX_PAGE_TOKENS = 20000


def banned_tokens(history: list[int], ngram_size: int, window: int) -> set[int]:
    """The frozen-model ban set for one step (output-only trailing window)."""
    if ngram_size <= 0 or len(history) < ngram_size:
        return set()
    prefix = tuple(history[-(ngram_size - 1) :]) if ngram_size > 1 else tuple()
    start = max(0, len(history) - window)
    end = len(history) - ngram_size + 1
    banned: set[int] = set()
    for i in range(start, end):
        ngram = history[i : i + ngram_size]
        if ngram_size == 1 or tuple(ngram[:-1]) == prefix:
            banned.add(ngram[-1])
    return banned


def guard_decision(
    state: dict,
    history: list[int],
    expected_pages: int,
    page_token_id: int,
    eos_id: int,
    max_page_tokens: int,
) -> tuple[str, int | None]:
    """One step of the native page guard; returns (kind, forced-or-banned id)."""
    n = len(history)
    if n > state["scanned"]:
        tail = history[state["scanned"] : n]
        markers = tail.count(page_token_id)
        if markers:
            state["marker_cum"] += markers
            state["page_len"] = tail[::-1].index(page_token_id)
        else:
            state["page_len"] += len(tail)
        state["scanned"] = n
    mc = state["marker_cum"]
    if mc > expected_pages:
        return "force", eos_id
    if 0 < max_page_tokens <= state["page_len"]:
        return "force", (page_token_id if mc < expected_pages else eos_id)
    if mc < expected_pages:
        return "ban", eos_id
    return "none", None


def repeated_tail(token_ids: list[int], min_count: int = 3, max_cycle: int = 64) -> dict:
    """Smallest repeated tail cycle occurring at least min_count times, if any."""
    n = len(token_ids)
    for cycle in range(1, max_cycle + 1):
        if cycle * min_count > n:
            break
        matched = True
        for k in range(1, cycle * (min_count - 1) + 1):
            if token_ids[n - k] != token_ids[n - k - cycle]:
                matched = False
                break
        if matched:
            # onset = index where the pure cycle run begins (conservative:
            # first position from which the cycle repeats uninterrupted).
            onset = n - cycle * min_count
            while onset - cycle >= 0 and token_ids[onset - cycle] == token_ids[onset]:
                onset -= cycle
            return {"cycle_len": cycle, "count": (n - onset) // cycle, "onset_index": onset}
    return {}


def replay(
    token_ids: list[int],
    *,
    ngram_size: int,
    window: int,
    expected_pages: int,
    page_token_id: int = DEFAULT_PAGE_TOKEN_ID,
    eos_id: int = DEFAULT_EOS_ID,
    max_page_tokens: int = DEFAULT_MAX_PAGE_TOKENS,
) -> dict:
    ban_sizes: list[int] = []
    first_ban_step = None
    forced_steps: list[dict] = []
    forced_mismatches: list[dict] = []
    state = {"scanned": 0, "marker_cum": 0, "page_len": 0}
    for step, token in enumerate(token_ids):
        banned = banned_tokens(token_ids[:step], ngram_size, window)
        if banned:
            ban_sizes.append(len(banned))
            if first_ban_step is None:
                first_ban_step = step
        kind, target = guard_decision(
            state, token_ids[:step], expected_pages, page_token_id, eos_id, max_page_tokens
        )
        if kind == "force":
            entry = {
                "step": step,
                "forced": target,
                "emitted": token,
                "token_was_banned": token in banned if target == page_token_id else None,
            }
            forced_steps.append(entry)
            if token != target:
                forced_mismatches.append(entry)
    markers = sum(1 for t in token_ids if t == page_token_id)
    return {
        "steps": len(token_ids),
        "ngram": {
            "size": ngram_size,
            "window": window,
            "steps_with_ban": len(ban_sizes),
            "first_ban_step": first_ban_step,
            "max_ban_size": max(ban_sizes) if ban_sizes else 0,
            "mean_ban_size": round(sum(ban_sizes) / len(ban_sizes), 3) if ban_sizes else 0.0,
        },
        "guard": {
            "expected_pages": expected_pages,
            "emitted_markers": markers,
            "marker_overshoot": markers - expected_pages if markers > expected_pages else 0,
            "forced_steps": len(forced_steps),
            "forced_mismatches": forced_mismatches,
        },
        "repeated_tail": repeated_tail(token_ids),
        "verdict": {
            "forced_marker_corner_detected": bool(forced_mismatches),
            "notes": (
                "Forced-step mismatch: the guard would force a token the model did not "
                "emit — the audited native all--inf corner or a guard divergence; the "
                "attempt must not be read as guard-consistent."
            )
            if forced_mismatches
            else "",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--token-ids", required=True, type=Path, help="token_ids.json persisted by the P2-B worker"
    )
    parser.add_argument("--ngram-size", type=int, required=True)
    parser.add_argument("--ngram-window", type=int, required=True)
    parser.add_argument("--expected-pages", type=int, required=True)
    parser.add_argument("--page-token-id", type=int, default=DEFAULT_PAGE_TOKEN_ID)
    parser.add_argument("--eos-id", type=int, default=DEFAULT_EOS_ID)
    parser.add_argument("--max-page-tokens", type=int, default=DEFAULT_MAX_PAGE_TOKENS)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    token_ids = json.loads(args.token_ids.read_text("utf-8"))
    result = replay(
        token_ids,
        ngram_size=args.ngram_size,
        window=args.ngram_window,
        expected_pages=args.expected_pages,
        page_token_id=args.page_token_id,
        eos_id=args.eos_id,
        max_page_tokens=args.max_page_tokens,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", "utf-8")
    print(
        json.dumps(
            {
                "steps": result["steps"],
                "first_ban_step": result["ngram"]["first_ban_step"],
                "forced_mismatches": len(result["guard"]["forced_mismatches"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
