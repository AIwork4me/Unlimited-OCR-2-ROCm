"""Regression for decorated model functions whose wrapper has other globals."""

import tempfile
import unittest
from pathlib import Path

import torch
from diagnostic import clone_for_prefix


@torch.no_grad()
def decorated_fixture():
    gen = []
    t_first = None
    for step in range(2):
        assert not torch.is_grad_enabled()
        nxt = step + 5
        if t_first is None:
            t_first = 1
        gen.append(nxt)
    return gen


class DecoratorTests(unittest.TestCase):
    def test_clone_uses_target_globals_and_retains_no_grad(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd() / ".b") as temp:
            clone = clone_for_prefix(decorated_fixture, [8], [], Path(temp) / "clone.py")
            with torch.enable_grad():
                self.assertEqual(clone(), [8, 6])
                self.assertTrue(torch.is_grad_enabled())


if __name__ == "__main__":
    unittest.main()
