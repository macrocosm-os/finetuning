import math
import random
import unittest
from finetune.eval.normalization import normalize_score, NormalizationId


class TestNormalizeScore(unittest.TestCase):

    def test_normalize_none(self):
        for _ in range(10):
            score = random.random()
            normalization_id = NormalizationId.NONE
            normalized_score = normalize_score(score, normalization_id)
            self.assertEqual(normalized_score, score)

    def test_normalize_inverse_exponential(self):
        score = 5.0
        normalization_id = NormalizationId.INVERSE_EXPONENTIAL
        norm_kwargs = {"ceiling": 10.0}
        normalized_score = normalize_score(score, normalization_id, norm_kwargs)
        self.assertAlmostEqual(normalized_score, 0.6224593312018546, places=6)

    def test_normalize_inverse_exponential_above_ceiling(self):
        score = 15.0
        normalization_id = NormalizationId.INVERSE_EXPONENTIAL
        norm_kwargs = {"ceiling": 10.0}
        normalized_score = normalize_score(score, normalization_id, norm_kwargs)
        self.assertEqual(normalized_score, 1.0)

    def test_normalize_inverse_exponential_always_between_0_and_1(self):
        for i in range(101):
            score = i / 10
            normalization_id = NormalizationId.INVERSE_EXPONENTIAL
            norm_kwargs = {"ceiling": 10.0}
            normalized_score = normalize_score(score, normalization_id, norm_kwargs)
            self.assertTrue(0 <= normalized_score <= 1)

    def test_unhandled_normalization_method(self):
        score = 5.0
        normalization_id = 999  # Invalid normalization method
        norm_kwargs = {}
        with self.assertRaises(ValueError):
            normalize_score(score, normalization_id, norm_kwargs)


if __name__ == "__main__":
    unittest.main()
