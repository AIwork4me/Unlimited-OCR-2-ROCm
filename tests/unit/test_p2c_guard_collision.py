"""P2-C Phase 1: synthetic guard-collision fixtures against the frozen model.

The audited corner (docs/rca/NGRAM35_SOURCE_AUDIT.md §7/§10): the native loop
applies the n-gram ban BEFORE the page guard
(modeling_unlimitedocr.py:1168-1172), and the guard's forced-token path keeps
the POST-ban logit value — so when the guard force-emits ``<PAGE>`` while
``<PAGE>`` is in the ban set, the row becomes all ``-inf`` and argmax falls to
an implementation-defined tie-break instead of the forced marker.

These tests execute the FROZEN functions themselves (extracted from the pinned
model snapshot by AST, hash-checked, never modified). They run only where both
torch and the frozen snapshot exist (the gfx1151 host); CI documents the
semantics via the replay analyzer equivalents that are always available.
"""

from __future__ import annotations

import ast
import hashlib
import random
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_REVISION = "9cf077c4b206e0e2a31879df937b0b0bcba3661c"
FROZEN_MODELING = (
    REPO_ROOT.parent
    / "hf-cache"
    / "models--baidu--Unlimited-OCR-2"
    / "snapshots"
    / FROZEN_REVISION
    / "modeling_unlimitedocr.py"
)
PINNED_GUARD_SHA256 = "51831ce27446ce6dce8683b4dc1f011be95b1690e91a0760cbe3c3e9f541082d"
PINNED_NGRAM_SHA256 = "c5d546822dad26424aa37e32c361a77d9ebd7786ca903c44da8b5e22b8c14a22"

PAGE_ID = 128827  # frozen config.json page_token_id
EOS_ID = 1  # frozen config.json eos_token_id

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from analyze_ngram_evidence import banned_tokens, guard_decision  # noqa: E402

torch = pytest.importorskip("torch")


def _frozen_functions():
    if not FROZEN_MODELING.exists():
        pytest.skip(f"frozen model snapshot not present: {FROZEN_MODELING}")
    source = FROZEN_MODELING.read_text(encoding="utf-8")
    tree = ast.parse(source)
    extracted: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in (
            "_uocr_page_count_guard",
            "_uocr_no_repeat_ngram",
        ):
            extracted[node.name] = ast.get_source_segment(source, node)
    assert len(extracted) == 2, "both frozen functions must exist in the snapshot"
    assert hashlib.sha256(extracted["_uocr_page_count_guard"].encode()).hexdigest() == (
        PINNED_GUARD_SHA256
    ), "frozen page guard drifted from the pinned source"
    assert hashlib.sha256(extracted["_uocr_no_repeat_ngram"].encode()).hexdigest() == (
        PINNED_NGRAM_SHA256
    ), "frozen n-gram processor drifted from the pinned source"
    namespace = {"torch": torch}
    exec(extracted["_uocr_page_count_guard"], namespace)  # noqa: S102 - pinned source
    exec(extracted["_uocr_no_repeat_ngram"], namespace)  # noqa: S102 - pinned source
    return namespace


def _fresh_state() -> dict:
    return {"marker_cum": 0, "scanned": 0, "page_len": 0}


def _collision_history() -> list[int]:
    """34 id-3 tokens, a page marker, then 34 more id-3 tokens.

    The trailing 34 tokens equal the 35-gram prefix that ended page 1 (the
    post-marker run must be long enough to push PAGE out of the trailing
    window), so the frozen n-gram processor (35/1024) bans PAGE_ID at the next
    step while the guard (page cap reached, markers 1 < expected 5)
    simultaneously forces PAGE_ID.
    """
    return [3] * 34 + [PAGE_ID] + [3] * 34


def _no_collision_history() -> list[int]:
    """Same guard state, but the trailing prefix never preceded a marker."""
    return [3] * 34 + [PAGE_ID] + [7] * 34


def test_collision_forced_page_banned_is_all_inf_row() -> None:
    frozen = _frozen_functions()
    history = _collision_history()
    logits = torch.arange(200000, dtype=torch.float32)
    assert banned_tokens(history, 35, 1024) == {PAGE_ID}
    after_ban = frozen["_uocr_no_repeat_ngram"](logits.clone(), history, 35, 1024, set())
    assert after_ban[PAGE_ID] == -float("inf")
    guarded = frozen["_uocr_page_count_guard"](
        after_ban, history, 5, PAGE_ID, EOS_ID, 8, _fresh_state()
    )
    # The forced marker kept its POST-ban value: the entire row is -inf and
    # argmax no longer selects the token the guard forced.
    assert guarded[PAGE_ID] == -float("inf")
    assert torch.isinf(guarded).all()
    assert int(torch.argmax(guarded)) != PAGE_ID


def test_forced_page_without_ban_selects_marker() -> None:
    frozen = _frozen_functions()
    history = _no_collision_history()
    logits = torch.arange(200000, dtype=torch.float32)
    assert banned_tokens(history, 35, 1024) == set()
    step = frozen["_uocr_no_repeat_ngram"](logits.clone(), history, 35, 1024, set())
    guarded = frozen["_uocr_page_count_guard"](step, history, 5, PAGE_ID, EOS_ID, 8, _fresh_state())
    assert int(torch.argmax(guarded)) == PAGE_ID


def test_forced_eos_never_collides_with_ban() -> None:
    frozen = _frozen_functions()
    rng = random.Random(20261030)
    # EOS terminates the loop before entering gen, so it can never be a
    # learned n-gram continuation; verify over randomized histories anyway.
    for _ in range(200):
        history = [rng.randrange(2, 500) for _ in range(rng.randrange(40, 400))]
        banned = frozen["_uocr_no_repeat_ngram"](
            torch.arange(2000, dtype=torch.float32), history, 35, 1024, set()
        )
        assert banned[EOS_ID] != -float("inf")
    # Overshoot path (marker_cum > expected) forces EOS; the row keeps a
    # finite EOS logit even after any ban.
    history = [PAGE_ID] * 6  # marker_cum 6 > expected 5 at scan time
    wide = torch.arange(200000, dtype=torch.float32)
    step = frozen["_uocr_no_repeat_ngram"](wide, history, 2, 1024, set())
    guarded = frozen["_uocr_page_count_guard"](
        step, history, 5, PAGE_ID, EOS_ID, 20000, _fresh_state()
    )
    assert int(torch.argmax(guarded)) == EOS_ID


def test_order_net_equivalence_at_the_corner() -> None:
    frozen = _frozen_functions()
    history = _collision_history()
    native = torch.arange(200000, dtype=torch.float32)
    native = frozen["_uocr_no_repeat_ngram"](native, history, 35, 1024, set())
    native = frozen["_uocr_page_count_guard"](
        native, history, 5, PAGE_ID, EOS_ID, 8, _fresh_state()
    )
    swapped = torch.arange(200000, dtype=torch.float32)
    swapped = frozen["_uocr_page_count_guard"](
        swapped, history, 5, PAGE_ID, EOS_ID, 8, _fresh_state()
    )
    swapped = frozen["_uocr_no_repeat_ngram"](swapped, history, 35, 1024, set())
    # Both orders end in an all-inf row at the collision: the SGLang processor
    # order and the native order are net-equivalent at temperature 0 here.
    assert torch.isinf(native).all() and torch.isinf(swapped).all()
    assert torch.equal(native, swapped)


def test_ban_scope_output_only_and_windowed() -> None:
    frozen = _frozen_functions()
    logits = torch.arange(200000, dtype=torch.float32)
    # The recurring 35-gram sits beyond the trailing 1024-token window: no ban.
    distant = [9] * 34 + [PAGE_ID] + [5] * 1100 + [9] * 34
    assert banned_tokens(distant, 35, 1024) == set()
    after = frozen["_uocr_no_repeat_ngram"](logits.clone(), distant, 35, 1024, set())
    assert after[PAGE_ID] != -float("inf")
    # Same pattern inside the window: banned.
    near = [9] * 34 + [PAGE_ID] + [5] * 100 + [9] * 34
    assert banned_tokens(near, 35, 1024) == {PAGE_ID}
    after = frozen["_uocr_no_repeat_ngram"](logits.clone(), near, 35, 1024, set())
    assert after[PAGE_ID] == -float("inf")
    # Prompt tokens never enter the scan: only gen ids are passed (structure
    # of the frozen call site), and the replay agrees on both cases.
    assert banned_tokens([4] * 34 + [PAGE_ID] + [4] * 34, 35, 1024) == {PAGE_ID}


def test_replay_matches_frozen_functions_on_collision_sequences() -> None:
    frozen = _frozen_functions()
    rng = random.Random(351024)
    for _ in range(500):
        ngram = rng.choice([2, 3, 5, 35])
        window = rng.choice([16, 128, 1024])
        history = [rng.randrange(2, 300) for _ in range(rng.randrange(0, 120))]
        logits = torch.arange(200000, dtype=torch.float32)
        frozen["_uocr_no_repeat_ngram"](logits, history, ngram, window, set())
        expect = banned_tokens(history, ngram, window)
        banned_ids = {int(i) for i in torch.nonzero(torch.isinf(logits)).flatten()}
        assert banned_ids == expect
        expected_pages = rng.randrange(1, 6)
        cap = rng.choice([8, 64, 20000])
        state = _fresh_state()
        kind, target = guard_decision(state, history, expected_pages, PAGE_ID, EOS_ID, cap)
        guarded = frozen["_uocr_page_count_guard"](
            logits.clone(), history, expected_pages, PAGE_ID, EOS_ID, cap, _fresh_state()
        )
        finite = {int(i) for i in torch.nonzero(torch.isfinite(guarded)).flatten()}
        if kind == "force":
            assert finite == {target}
        elif kind == "ban":
            assert EOS_ID not in finite
            assert guarded[EOS_ID] == -float("inf")


def test_page_cap_reachability_by_budget() -> None:
    # The forced-PAGE path needs page_len >= max_page_tokens (20000): it is
    # structurally unreachable while the output budget stays below the cap.
    from analyze_ngram_evidence import DEFAULT_MAX_PAGE_TOKENS

    assert DEFAULT_MAX_PAGE_TOKENS == 20000
    for budget in (8192, 12288, 16384):
        assert budget < DEFAULT_MAX_PAGE_TOKENS
    assert 39321 >= DEFAULT_MAX_PAGE_TOKENS  # official L1 class: corner is live
