import unittest
from finetune.eval.if_eval.bullet_count import BulletFrequencyRule


class TestBulletCount(unittest.TestCase):
    def test_frequency(self):
        rule = BulletFrequencyRule(count=2)
        self.assertTrue(
            rule.matches("We have the following points: \n* point one \n \n*point two.")
        )
        self.assertTrue(
            rule.matches("We have the following * points \n* point * one \n*point two.")
        )
        self.assertTrue(
            rule.matches("We have two points. \n*point one. \n * point two.")
        )
        self.assertFalse(
            rule.matches("We have the following points \n- point one \n-point two.")
        )
        self.assertFalse(rule.matches("We have the following points \n* point one"))
        self.assertFalse(
            rule.matches(
                "We have the following points \n* point one \n*point two \n* point three"
            )
        )

    def test_invalid_threshold(self):
        with self.assertRaises(ValueError):
            BulletFrequencyRule(count=0)

    def test_get_prompt_frequency_one(self):
        rule = BulletFrequencyRule(count=1)
        self.assertEqual(
            rule.get_prompt(),
            "The response must contain exactly 1 bullet point in markdown format.",
        )

    def test_get_prompt_frequency_two(self):
        rule = BulletFrequencyRule(count=2)
        self.assertEqual(
            rule.get_prompt(),
            "The response must contain exactly 2 bullet points in markdown format.",
        )


if __name__ == "__main__":
    unittest.main()
