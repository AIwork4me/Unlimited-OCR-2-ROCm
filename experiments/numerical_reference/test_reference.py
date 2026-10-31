"""Model-free CPU reference tests; no torch, weights or network required."""

import math
import unittest

import numpy as np
from reference import attention_reference


class ReferenceTests(unittest.TestCase):
    def test_zero_queries_mean(self):
        q = np.zeros((1, 1, 2, 2))
        v = np.array([[[[2.0, 4.0], [6.0, 8.0]]]])
        got = attention_reference(q, q, v, dtype=np.float64)
        np.testing.assert_array_equal(got, [[[[4.0, 6.0], [4.0, 6.0]]]])

    def test_boolean_mask_and_fully_masked_row(self):
        q = np.zeros((1, 1, 2, 2))
        v = np.array([[[[2.0, 4.0], [6.0, 8.0]]]])
        got = attention_reference(q, q, v, mask=np.array([[True, False], [False, False]]))
        np.testing.assert_array_equal(got, [[[[2.0, 4.0], [0.0, 0.0]]]])

    def test_additive_mask_and_scale_against_scalar(self):
        q = np.array([[[[1.0, -2.0], [3.0, 1.0]]]])
        k = q[..., ::-1].copy()
        v = np.array([[[[2.0, 4.0], [6.0, 8.0]]]])
        mask = np.array([[0.25, -0.5], [0.1, 0.2]])
        got = attention_reference(q, k, v, mask=mask, scale=0.3)
        for i in range(2):
            scores = [
                math.fsum(float(q[0, 0, i, d] * k[0, 0, j, d]) for d in range(2)) * 0.3 + mask[i, j]
                for j in range(2)
            ]
            exps = [math.exp(x - max(scores)) for x in scores]
            expected = [
                math.fsum(exps[j] * v[0, 0, j, d] for j in range(2)) / math.fsum(exps)
                for d in range(2)
            ]
            np.testing.assert_allclose(got[0, 0, i], expected, rtol=1e-14, atol=1e-14)

    def test_strided_inputs_and_causal(self):
        q = np.zeros((1, 1, 4, 2))[..., ::2, :]
        v = np.array([[[[2.0, 4.0], [6.0, 8.0]]]])
        got = attention_reference(q, q, v, causal=True)
        np.testing.assert_array_equal(got, [[[[2.0, 4.0], [4.0, 6.0]]]])

    def test_reject_dropout_and_ambiguous_mask(self):
        q = np.zeros((1, 1, 2, 2))
        with self.assertRaises(ValueError):
            attention_reference(q, q, q, dropout=0.1)
        with self.assertRaises(ValueError):
            attention_reference(q, q, q, causal=True, mask=np.ones((2, 2)))


if __name__ == "__main__":
    unittest.main()
