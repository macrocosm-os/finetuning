import unittest
from finetune.eval.if_eval.word_count import WordCountAtLeastRule, WordCountAtMostRule


class TestWordCountRule(unittest.TestCase):
    def test_at_least(self):
        rule = WordCountAtLeastRule(count=5)
        self.assertTrue(rule.matches("This  is a\n test, six words."))
        self.assertFalse(rule.matches("Only four words here."))

    def test_at_most(self):
        rule = WordCountAtMostRule(count=5)
        self.assertTrue(rule.matches("This is four words."))
        self.assertFalse(rule.matches("This is a test with six words."))

    def test_invalid_threshold(self):
        with self.assertRaises(ValueError):
            WordCountAtLeastRule(count=0)
        with self.assertRaises(ValueError):
            WordCountAtMostRule(count=0)

    def test_get_prompt_at_least(self):
        rule = WordCountAtLeastRule(count=5)
        self.assertEqual(rule.get_prompt(), "The response must be at least 5 words.")

    def test_get_prompt_at_most(self):
        rule = WordCountAtMostRule(count=5)
        self.assertEqual(
            rule.get_prompt(), "The response must be no more than 5 words."
        )

    def test_get_prompt_at_least_one(self):
        rule = WordCountAtLeastRule(count=1)
        self.assertEqual(rule.get_prompt(), "The response must be at least 1 word.")

    def test_get_prompt_at_most_one(self):
        rule = WordCountAtMostRule(count=1)
        self.assertEqual(rule.get_prompt(), "The response must be no more than 1 word.")


if __name__ == "__main__":
    unittest.main()
