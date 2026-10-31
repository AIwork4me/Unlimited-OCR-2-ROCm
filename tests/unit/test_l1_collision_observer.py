"""L1 qualification: the opt-in guard-collision observer against the frozen loop.

The audited corner (docs/validation/P2C_GUARD_COLLISION_SAFETY.md §3, spec §6):
when the page guard forces ``<PAGE>`` while the n-gram ban has already set it
to ``-inf``, the whole logits row is ``-inf`` and argmax tie-breaks arbitrarily.
The observer wraps the FROZEN module functions by call-through and aborts the
generation the moment such a row exists — BEFORE the unsafe token is selected.
It must never change frozen behavior on safe steps and never edit the frozen
snapshot file.
"""

from __future__ import annotations

import ast
import hashlib
import random
import sys
import types
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

PAGE_ID = 128827
EOS_ID = 1

sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from analyze_ngram_evidence import guard_decision, replay  # noqa: E402

torch = pytest.importorskip("torch")

from unlimited_ocr_2_rocm.collision_observer import (  # noqa: E402
    GuardCollisionObserver,
    UnsafeSelectionCollision,
    locate_decode_module,
)


def _frozen_functions() -> dict:
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
    namespace: dict = {"torch": torch}
    exec(extracted["_uocr_page_count_guard"], namespace)  # noqa: S102 - pinned source
    exec(extracted["_uocr_no_repeat_ngram"], namespace)  # noqa: S102 - pinned source
    return namespace


def _fake_module(frozen: dict) -> types.ModuleType:
    module = types.ModuleType("fake_frozen_decode")
    module._uocr_page_count_guard = frozen["_uocr_page_count_guard"]  # type: ignore[attr-defined]
    module._uocr_no_repeat_ngram = frozen["_uocr_no_repeat_ngram"]  # type: ignore[attr-defined]
    return module


def _fresh_state() -> dict:
    return {"marker_cum": 0, "scanned": 0, "page_len": 0}


def _collision_history() -> list[int]:
    return [3] * 34 + [PAGE_ID] + [3] * 34


def test_observer_is_non_invasive_install_uninstall(tmp_path: Path) -> None:
    frozen = _frozen_functions()
    module = _fake_module(frozen)
    original_ngram = module._uocr_no_repeat_ngram
    original_guard = module._uocr_page_count_guard
    snapshot_sha = hashlib.sha256(FROZEN_MODELING.read_bytes()).hexdigest()
    observer = GuardCollisionObserver()
    observer.install(module)
    assert module._uocr_no_repeat_ngram is not original_ngram
    assert module._uocr_page_count_guard is not original_guard
    assert observer.uninstall() is True
    assert module._uocr_no_repeat_ngram is original_ngram
    assert module._uocr_page_count_guard is original_guard
    # The frozen snapshot file is untouched by installation/uninstallation.
    assert hashlib.sha256(FROZEN_MODELING.read_bytes()).hexdigest() == snapshot_sha
    observer2 = GuardCollisionObserver()
    with observer2.session(module):
        pass
    assert module._uocr_no_repeat_ngram is original_ngram


def test_safe_steps_are_bit_identical_passthrough() -> None:
    frozen = _frozen_functions()
    module = _fake_module(frozen)
    observer = GuardCollisionObserver()
    rng = random.Random(7)
    with observer.session(module):
        for _ in range(50):
            history = [rng.randrange(2, 900) for _ in range(rng.randrange(0, 200))]
            base = torch.arange(200000, dtype=torch.float32)
            frozen_out = frozen["_uocr_no_repeat_ngram"](base.clone(), history, 35, 1024, set())
            frozen_out = frozen["_uocr_page_count_guard"](
                frozen_out, history, 5, PAGE_ID, EOS_ID, 20000, _fresh_state()
            )
            observed = module._uocr_no_repeat_ngram(base.clone(), history, 35, 1024, set())
            observed = module._uocr_page_count_guard(
                observed, history, 5, PAGE_ID, EOS_ID, 20000, _fresh_state()
            )
            assert torch.equal(frozen_out, observed)
            assert int(torch.isfinite(observed).sum()) > 1


def test_collision_aborts_before_selection() -> None:
    frozen = _frozen_functions()
    module = _fake_module(frozen)
    history = _collision_history()
    state = _fresh_state()
    observer = GuardCollisionObserver()
    with observer.session(module), pytest.raises(UnsafeSelectionCollision) as info:
        logits = module._uocr_no_repeat_ngram(
            torch.arange(200000, dtype=torch.float32), history, 35, 1024, set()
        )
        # The observer raises INSIDE this call: argmax never runs, so the
        # unintended tie-break token cannot be selected or committed.
        module._uocr_page_count_guard(logits, history, 5, PAGE_ID, EOS_ID, 8, state)
    error = info.value
    assert error.expected_pages == 5
    assert error.page_len == 34  # 34 tokens since the last marker
    assert error.marker_cum == 1
    assert error.generated_tokens == 69
    assert error.partial_token_ids == history
    events = observer.export_events()
    assert events["aborted"] is True
    assert events["collision"]["step"] == 69


def test_forced_page_without_ban_is_observed_not_aborted() -> None:
    frozen = _frozen_functions()
    module = _fake_module(frozen)
    history = [3] * 34 + [PAGE_ID] + [7] * 34
    observer = GuardCollisionObserver()
    with observer.session(module):
        logits = module._uocr_no_repeat_ngram(
            torch.arange(200000, dtype=torch.float32), history, 35, 1024, set()
        )
        logits = module._uocr_page_count_guard(
            logits, history, 5, PAGE_ID, EOS_ID, 8, _fresh_state()
        )
        assert int(torch.argmax(logits)) == PAGE_ID
    events = observer.export_events()
    assert events["aborted"] is False
    assert events["forced_page_steps"] == [
        {"step": 69, "forced": PAGE_ID, "page_len": 34, "marker_cum": 1}
    ]


def test_forced_eos_overshoot_and_undercount_ban_observed() -> None:
    frozen = _frozen_functions()
    module = _fake_module(frozen)
    observer = GuardCollisionObserver()
    # Overshoot: marker_cum 6 > expected 5 → the guard forces EOS.
    history = [PAGE_ID] * 6
    with observer.session(module):
        logits = module._uocr_no_repeat_ngram(
            torch.arange(200000, dtype=torch.float32), history, 35, 1024, set()
        )
        logits = module._uocr_page_count_guard(
            logits, history, 5, PAGE_ID, EOS_ID, 20000, _fresh_state()
        )
        assert int(torch.argmax(logits)) == EOS_ID
    # Under-count: markers 0 < expected 1 → EOS is banned, row not all--inf.
    history_under = [4] * 40
    with observer.session(module):
        logits = module._uocr_no_repeat_ngram(
            torch.arange(200000, dtype=torch.float32), history_under, 35, 1024, set()
        )
        logits = module._uocr_page_count_guard(
            logits, history_under, 1, PAGE_ID, EOS_ID, 20000, _fresh_state()
        )
        assert logits[EOS_ID] == -float("inf")
        assert int(torch.isfinite(logits).sum()) > 1
    events = observer.export_events()
    assert events["forced_eos_steps"] == [
        {"step": 6, "forced": EOS_ID, "page_len": 0, "marker_cum": 6}
    ]
    assert events["aborted"] is False


def test_multiple_page_histories_track_guard_transitions() -> None:
    frozen = _frozen_functions()
    module = _fake_module(frozen)
    # Two complete pages then a third running page (markers 2 < expected 3);
    # forced markers append and the guard state machine keeps transitioning.
    history: list[int] = [5] * 10 + [PAGE_ID] + [6] * 10 + [PAGE_ID] + [7] * 12
    observer_state = _fresh_state()
    reference_state = _fresh_state()
    observer = GuardCollisionObserver()
    observed: set[tuple[int, int]] = set()
    with observer.session(module):
        for step in range(1, len(history) + 6):
            prefix = history[:step]
            logits = module._uocr_no_repeat_ngram(
                torch.arange(200000, dtype=torch.float32), prefix, 2, 1024, set()
            )
            logits = module._uocr_page_count_guard(
                logits, prefix, 3, PAGE_ID, EOS_ID, 10, observer_state
            )
            kind, target = guard_decision(reference_state, prefix, 3, PAGE_ID, EOS_ID, 10)
            nxt = int(torch.argmax(logits))
            if kind == "force":
                observed.add((step, target))
            if nxt == EOS_ID:
                break
            history.append(nxt)
    events = observer.export_events()
    reported = {
        (event["step"], event["forced"])
        for event in events["forced_page_steps"] + events["forced_eos_steps"]
    }
    assert reported == observed, "observer events must match the frozen guard transitions"
    assert any(target == PAGE_ID for _, target in observed)


def test_window_boundary_stale_gram_does_not_collide() -> None:
    frozen = _frozen_functions()
    module = _fake_module(frozen)
    # The recurring 35-gram is beyond the trailing 1024-token window: the
    # forced <PAGE> survives (no collision) even though the same prefix bans
    # inside the window.
    distant = [9] * 34 + [PAGE_ID] + [5] * 1100 + [9] * 34
    near = [9] * 34 + [PAGE_ID] + [5] * 100 + [9] * 34
    observer = GuardCollisionObserver()
    with observer.session(module):
        for history, expect_collision in ((distant, False), (near, True)):
            state = _fresh_state()
            if expect_collision:
                with pytest.raises(UnsafeSelectionCollision):
                    logits = module._uocr_no_repeat_ngram(
                        torch.arange(200000, dtype=torch.float32), history, 35, 1024, set()
                    )
                    module._uocr_page_count_guard(logits, history, 5, PAGE_ID, EOS_ID, 8, state)
            else:
                logits = module._uocr_no_repeat_ngram(
                    torch.arange(200000, dtype=torch.float32), history, 35, 1024, set()
                )
                logits = module._uocr_page_count_guard(
                    logits, history, 5, PAGE_ID, EOS_ID, 8, state
                )
                assert int(torch.argmax(logits)) == PAGE_ID
    events = observer.export_events()
    assert events["aborted"] is True  # the near-window case aborted
    assert len(events["forced_page_steps"]) == 1  # only the distant case forced


def test_observer_agrees_with_replay_detector() -> None:
    """Differential: a scripted decode through the wrapped functions versus
    scripts/analyze_ngram_evidence.replay over the emitted ids."""
    frozen = _frozen_functions()
    module = _fake_module(frozen)
    rng = random.Random(20261031)
    for case in range(20):
        ngram_size = rng.choice([2, 3, 5, 35])
        window = rng.choice([16, 128, 1024])
        expected_pages = rng.randrange(1, 6)
        cap = rng.choice([8, 16, 64])
        base_logits = torch.randn(200000)
        gen: list[int] = []
        state = _fresh_state()
        observer = GuardCollisionObserver()
        aborted = False
        with observer.session(module):
            for _step in range(400):
                logits = module._uocr_no_repeat_ngram(
                    base_logits.clone(), gen, ngram_size, window, set()
                )
                try:
                    logits = module._uocr_page_count_guard(
                        logits, gen, expected_pages, PAGE_ID, EOS_ID, cap, state
                    )
                except UnsafeSelectionCollision:
                    aborted = True
                    break
                nxt = int(torch.argmax(logits))
                if nxt == EOS_ID:
                    break
                gen.append(nxt)
        events = observer.export_events()
        result = replay(
            gen,
            ngram_size=ngram_size,
            window=window,
            expected_pages=expected_pages,
            max_page_tokens=cap,
        )
        replay_forced = {
            (entry["step"], entry["forced"]) for entry in result["guard"]["forced_step_list"]
        }
        observer_pages = {(event["step"], event["forced"]) for event in events["forced_page_steps"]}
        # Known asymmetry: a FORCED EOS always terminates the loop and EOS
        # never enters token_ids, so the post-hoc replay cannot see it; the
        # observer can (it watches the pre-selection state). Forced PAGE
        # markers are emitted and must agree exactly.
        assert observer_pages == replay_forced, f"case {case}: observer/replay disagree"
        for event in events["forced_eos_steps"]:
            assert event["step"] == len(gen), "forced EOS must be the terminal step"
        assert events["aborted"] == aborted


def test_locate_decode_module_finds_frozen_functions() -> None:
    frozen = _frozen_functions()
    module = _fake_module(frozen)
    saved = sys.modules.get("fake_locate_target")
    try:

        class _FakeAutoModel:
            pass

        _FakeAutoModel.__module__ = "fake_locate_target"
        sys.modules["fake_locate_target"] = module
        assert locate_decode_module(_FakeAutoModel()) is module
    finally:
        if saved is None:
            sys.modules.pop("fake_locate_target", None)
        else:
            sys.modules["fake_locate_target"] = saved
    with pytest.raises(RuntimeError):
        locate_decode_module(object())


def test_observer_events_bounded() -> None:
    frozen = _frozen_functions()
    module = _fake_module(frozen)
    observer = GuardCollisionObserver(max_events=5)
    with observer.session(module):
        for step in range(1, 40):
            history = [PAGE_ID] * step
            logits = module._uocr_no_repeat_ngram(
                torch.arange(200000, dtype=torch.float32), history, 2, 1024, set()
            )
            logits = module._uocr_page_count_guard(
                logits, history, 3, PAGE_ID, EOS_ID, 20000, _fresh_state()
            )
    events = observer.export_events()
    assert len(events["forced_eos_steps"]) <= 5
    assert events["events_dropped"] > 0
