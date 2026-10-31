import tempfile
import unittest
from pathlib import Path

from diagnostic import clone_for_prefix


def generation_fixture():
    gen = []
    t_first = None
    seen_prefixes = []
    for step in range(3):
        seen_prefixes.append(list(gen))
        nxt = 5 + step
        if t_first is None:
            t_first = 1
        gen.append(nxt)
    return gen, seen_prefixes


def _uocr_page_count_guard(value):
    return value


def guarded_fixture():
    gen = []
    t_first = None
    for step in range(2):
        nxt = _uocr_page_count_guard(step + 5)
        if t_first is None:
            t_first = 1
        gen.append(nxt)
    return gen


class TeacherForcingTests(unittest.TestCase):
    def test_later_guard_observers_remain_visible_to_cloned_globals(self):
        calls = []
        target = [lambda value: value]
        with tempfile.TemporaryDirectory(dir=Path.cwd() / ".b") as temp:
            clone = clone_for_prefix(
                guarded_fixture, [], [], Path(temp) / "clone.py", guard_resolver=lambda: target[0]
            )

            def observed(value):
                calls.append(value)
                return value

            target[0] = observed
            self.assertEqual(clone(), [5, 6])
        self.assertEqual(calls, [5, 6])

    def test_common_prefix_updates_generation_state_without_changing_selection(self):
        decisions = []
        with tempfile.TemporaryDirectory(dir=Path.cwd() / ".b") as temp:
            clone = clone_for_prefix(generation_fixture, [8, 9], decisions, Path(temp) / "clone.py")
            gen, prefixes = clone()
        self.assertEqual(gen, [8, 9, 7])
        self.assertEqual(prefixes, [[], [8], [8, 9]])
        self.assertEqual([d["actual_selected"] for d in decisions], [5, 6, 7])
        self.assertEqual(generation_fixture()[0], [5, 6, 7])


if __name__ == "__main__":
    unittest.main()
