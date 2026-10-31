"""Opt-in experimental observer for the frozen native decode loop.

The audited guard-collision corner (docs/validation/P2C_GUARD_COLLISION_SAFETY.md
§3, docs/validation/L1_FIFTEEN_PAGE_SPEC.md §6) becomes reachable at the
39 321-token budget class: when the page guard forces ``<PAGE>`` while the
n-gram ban has already set that id to ``-inf``, the whole logits row is
``-inf`` and the loop's argmax tie-breaks to an arbitrary token instead of
the forced marker. The existing replay detector identifies this AFTER
execution; this observer detects the all-``-inf`` row BEFORE selection and
aborts the generation.

Non-invasiveness contract:

- the frozen snapshot file is never edited; the observer wraps the two frozen
  module functions by call-through at runtime, exactly like the reviewed
  kernel instrumentation in ``model.load_model``;
- on every safe step the wrapped functions return the frozen result
  bit-for-bit and the only added work is one ``isfinite`` reduction in the
  guard path;
- installation is opt-in (the L1 worker installs it when the pinned policy
  says so), scoped to one generation, and fully reverted afterwards;
- aborting raises before ``argmax`` runs, so an unintended token is never
  selected, never enters the history, and never reaches the output.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

NGRAM_FN = "_uocr_no_repeat_ngram"
GUARD_FN = "_uocr_page_count_guard"
FROZEN_PAIR = (NGRAM_FN, GUARD_FN)

DEFAULT_PAGE_TOKEN_ID = 128827  # frozen config.json sparse_attention_config
DEFAULT_EOS_ID = 1  # frozen config.json eos_token_id


class UnsafeSelectionCollision(RuntimeError):
    """An all-``-inf`` logits row existed at selection time.

    Raised BEFORE the frozen loop's argmax, carrying the guard state and the
    generated ids so far for evidence preservation.
    """

    def __init__(
        self,
        *,
        step: int,
        expected_pages: int,
        page_len: int,
        marker_cum: int,
        generated_tokens: int,
        partial_token_ids: list[int],
    ) -> None:
        self.step = step
        self.expected_pages = expected_pages
        self.page_len = page_len
        self.marker_cum = marker_cum
        self.generated_tokens = generated_tokens
        self.partial_token_ids = partial_token_ids
        super().__init__(
            f"guard-collision at step {step}: forced marker banned by n-gram "
            f"(page_len={page_len}, marker_cum={marker_cum}, "
            f"expected_pages={expected_pages}); {generated_tokens} tokens generated"
        )


class GuardCollisionObserver:
    """Wrap the frozen n-gram/page-guard functions; abort on all-``-inf`` rows.

    ``max_events`` bounds the per-kind forced-event list; dropped events are
    counted, never silent. Aggregates stay exact regardless of the cap.
    """

    def __init__(
        self,
        *,
        page_token_id: int = DEFAULT_PAGE_TOKEN_ID,
        eos_id: int = DEFAULT_EOS_ID,
        max_events: int = 10_000,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.page_token_id = page_token_id
        self.eos_id = eos_id
        self.max_events = max_events
        self._event_sink = event_sink
        self._module: Any = None
        self._originals: dict[str, Any] = {}
        self._wrappers: dict[str, Any] = {}
        self._ngram_steps = 0
        self._guard_steps = 0
        self._normal_steps = 0
        self._forced_page: list[dict[str, int]] = []
        self._forced_eos: list[dict[str, int]] = []
        self._anomalies: list[dict[str, int]] = []
        self._dropped = 0
        self._sink_errors = 0
        self._collision: dict[str, Any] | None = None

    def _emit(self, event: dict[str, Any]) -> None:
        """Deliver an event to the sink; a broken sink never changes frozen
        behavior and never masks a collision abort."""
        if self._event_sink is None:
            return
        try:
            self._event_sink(event)
        except Exception:  # noqa: BLE001 - observation must stay non-invasive
            self._sink_errors += 1

    # -- installation ------------------------------------------------------

    def install(self, module: Any) -> None:
        if self._module is not None:
            raise RuntimeError("Observer already installed")
        for name in FROZEN_PAIR:
            if not callable(getattr(module, name, None)):
                raise RuntimeError(f"Frozen function missing from decode module: {name}")
        self._module = module
        self._originals[NGRAM_FN] = getattr(module, NGRAM_FN)
        self._originals[GUARD_FN] = getattr(module, GUARD_FN)
        self._wrappers[NGRAM_FN] = self._wrap_ngram(self._originals[NGRAM_FN])
        self._wrappers[GUARD_FN] = self._wrap_guard(self._originals[GUARD_FN])
        setattr(module, NGRAM_FN, self._wrappers[NGRAM_FN])
        setattr(module, GUARD_FN, self._wrappers[GUARD_FN])

    def uninstall(self) -> bool:
        """Restore the frozen functions; False when nothing was installed."""
        if self._module is None:
            return False
        for name in FROZEN_PAIR:
            if getattr(self._module, name, None) is not self._wrappers[name]:
                # A third party replaced our wrapper; refuse to claim a clean
                # restore (and do not clobber their change silently).
                raise RuntimeError(f"Observer wrapper was replaced externally: {name}")
            setattr(self._module, name, self._originals[name])
        self._module = None
        self._originals = {}
        self._wrappers = {}
        return True

    def session(self, module: Any) -> _ObserverSession:
        return _ObserverSession(self, module)

    # -- wrappers (call-through, observation only) -------------------------

    def _wrap_ngram(self, original: Any) -> Any:
        def wrapper(logits: Any, gen_ids: list[int], *args: Any) -> Any:
            result = original(logits, gen_ids, *args)
            self._ngram_steps += 1
            return result

        return wrapper

    def _wrap_guard(self, original: Any) -> Any:
        def wrapper(
            logits: Any,
            gen_ids: list[int],
            expected_pages: int,
            page_id: int,
            eos_id: int,
            max_page_tokens: int,
            st: dict,
        ) -> Any:
            result = original(logits, gen_ids, expected_pages, page_id, eos_id, max_page_tokens, st)
            self._after_guard(result, gen_ids, expected_pages, page_id, eos_id, st)
            return result

        return wrapper

    # -- observation --------------------------------------------------------

    def _after_guard(
        self,
        logits: Any,
        gen_ids: list[int],
        expected_pages: int,
        page_id: int,
        eos_id: int,
        st: dict,
    ) -> None:
        import torch

        self._guard_steps += 1
        step = len(gen_ids)
        finite_count = int(torch.isfinite(logits).sum())
        if finite_count == 0:
            self._collision = {
                "step": step,
                "expected_pages": expected_pages,
                "page_len": st.get("page_len"),
                "marker_cum": st.get("marker_cum"),
                "generated_tokens": len(gen_ids),
            }
            self._emit({"kind": "collision", **self._collision, "partial_token_ids": list(gen_ids)})
            raise UnsafeSelectionCollision(
                step=step,
                expected_pages=expected_pages,
                page_len=int(st.get("page_len", 0)),
                marker_cum=int(st.get("marker_cum", 0)),
                generated_tokens=len(gen_ids),
                partial_token_ids=list(gen_ids),
            )
        if finite_count == 1:
            target = int(torch.argmax(logits))
            event = {
                "step": step,
                "forced": target,
                "page_len": int(st.get("page_len", 0)),
                "marker_cum": int(st.get("marker_cum", 0)),
            }
            kind = (
                "forced_page"
                if target == page_id
                else ("forced_eos" if target == eos_id else "anomaly")
            )
            if target == page_id:
                bucket = self._forced_page
            elif target == eos_id:
                bucket = self._forced_eos
            else:
                bucket = self._anomalies
                event["note_step_only_finite_non_marker"] = 1
            if len(bucket) < self.max_events:
                bucket.append(event)
            else:
                self._dropped += 1
            self._emit({"kind": kind, **event})
            return
        self._normal_steps += 1

    # -- reporting ----------------------------------------------------------

    @property
    def aborted(self) -> bool:
        return self._collision is not None

    def export_events(self) -> dict[str, Any]:
        return {
            "schema": 1,
            "label": "L1_GUARD_COLLISION_OBSERVER_EVENTS",
            "installed": self._module is not None,
            "aborted": self.aborted,
            "collision": self._collision,
            "counters": {
                "ngram_steps": self._ngram_steps,
                "guard_steps": self._guard_steps,
                "normal_steps": self._normal_steps,
                "forced_page_steps": len(self._forced_page),
                "forced_eos_steps": len(self._forced_eos),
                "anomalies": len(self._anomalies),
            },
            "forced_page_steps": list(self._forced_page),
            "forced_eos_steps": list(self._forced_eos),
            "anomalies": list(self._anomalies),
            "events_dropped": self._dropped,
            "sink_errors": self._sink_errors,
            "events_cap": self.max_events,
            "page_token_id": self.page_token_id,
            "eos_id": self.eos_id,
        }

    def write_events(self, path: Path) -> None:
        import json

        payload = self.export_events()
        payload["events_truncated"] = self._dropped > 0
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")


class _ObserverSession:
    def __init__(self, observer: GuardCollisionObserver, module: Any) -> None:
        self._observer = observer
        self._module = module

    def __enter__(self) -> GuardCollisionObserver:
        self._observer.install(self._module)
        return self._observer

    def __exit__(self, *_exc: Any) -> None:
        self._observer.uninstall()


def _has_frozen_pair(module: Any) -> bool:
    """True when both frozen functions are real function attributes.

    ``types.FunctionType`` matters: lazy namespaces (torch.ops/torch.classes)
    answer ``hasattr`` for any name, which must not count as a decode module.
    """
    import types as _types

    return all(isinstance(getattr(module, name, None), _types.FunctionType) for name in FROZEN_PAIR)


def locate_decode_module(model: Any) -> Any:
    """Find the loaded trust_remote_code module holding the frozen functions."""
    seen: set[int] = set()
    classes: list[type] = [type(model)]
    modules_method = getattr(model, "modules", None)
    if callable(modules_method):
        classes.extend(type(item) for item in modules_method())
    deduped: list[type] = []
    for cls in classes:
        if id(cls) in seen:
            continue
        seen.add(id(cls))
        deduped.append(cls)
    for cls in deduped:
        module = sys.modules.get(cls.__module__)
        if module is not None and _has_frozen_pair(module):
            return module
    for module in list(sys.modules.values()):
        if module is not None and _has_frozen_pair(module):
            return module
    raise RuntimeError("Frozen decode module (trust_remote_code) not located for observation")
