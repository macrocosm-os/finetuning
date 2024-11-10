import unittest
from finetune.eval.if_eval.word_count import WordCountRule
from finetune.eval.if_eval.rule import RuleConstraint


class TestWordCountRule(unittest.TestCase):
    def test_greater_than_eq_constraint(self):
        rule = WordCountRule(count=5, constraint=RuleConstraint.GREATER_THAN_EQ)
        self.assertTrue(rule.matches("This  is a\n test, six words."))
        self.assertFalse(rule.matches("Only four words here."))

    def test_less_than_eq_constraint(self):
        rule = WordCountRule(count=5, constraint=RuleConstraint.LESS_THAN_EQ)
        self.assertTrue(rule.matches("This is four words."))
        self.assertFalse(rule.matches("This is a test with six words."))

    def test_invalid_constraint(self):
        with self.assertRaises(ValueError):
            WordCountRule(count=5, constraint=5)

    def test_get_prompt_greater_than_eq(self):
        rule = WordCountRule(count=5, constraint=RuleConstraint.GREATER_THAN_EQ)
        self.assertEqual(rule.get_prompt(), "The response must be at least 5 words.")

    def test_get_prompt_less_than_eq(self):
        rule = WordCountRule(count=5, constraint=RuleConstraint.LESS_THAN_EQ)
        self.assertEqual(
            rule.get_prompt(), "The response must be no more than 5 words."
        )

    def test_get_prompt_greater_than_eq_one(self):
        rule = WordCountRule(count=1, constraint=RuleConstraint.GREATER_THAN_EQ)
        self.assertEqual(rule.get_prompt(), "The response must be at least 1 word.")

    def test_get_prompt_less_than_eq_one(self):
        rule = WordCountRule(count=1, constraint=RuleConstraint.LESS_THAN_EQ)
        self.assertEqual(rule.get_prompt(), "The response must be no more than 1 word.")


if __name__ == "__main__":
    unittest.main()
