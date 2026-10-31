"""CPU check of the exact supported cache API; no model or GPU used."""

import unittest

import torch
from transformers.cache_utils import DynamicCache


class CacheTests(unittest.TestCase):
    def test_supported_iteration_positions_and_allocation_lifetime(self):
        cache = DynamicCache()
        keys = torch.arange(24).reshape(1, 2, 3, 4).bfloat16()
        cache.update(keys, keys.clone(), 0)
        self.assertEqual(len(cache), 1)
        self.assertEqual(cache.get_seq_length(0), 3)
        first = cache[0][0]
        snapshot = first.clone()
        cache.update(torch.ones((1, 2, 1, 4)).bfloat16(), torch.ones((1, 2, 1, 4)).bfloat16(), 0)
        self.assertEqual(cache.get_seq_length(0), 4)
        self.assertEqual(len(list(cache)), 1)
        self.assertNotEqual(first.data_ptr(), cache[0][0].data_ptr())
        self.assertTrue(torch.equal(first, snapshot))
        self.assertTrue(torch.equal(cache[0][0][..., :3, :], snapshot))


if __name__ == "__main__":
    unittest.main()
