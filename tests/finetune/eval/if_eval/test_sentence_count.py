import unittest
from finetune.eval.if_eval.sentence_count import SentenceCountAtLeastRule, SentenceCountAtMostRule


class TestSentenceCount(unittest.TestCase):
    def test_at_least(self):
        rule = SentenceCountAtLeastRule(count=2)
        self.assertTrue(rule.matches("This  is a\n test.  Woo! We have three sentences, right?", 0))
        self.assertFalse(rule.matches("This is only one sentence with multiple punctuations...", 0))
        # Note that ". ." or "!!" or "!?" all count as two sentences currently.
        self.assertTrue(rule.matches("Hello!?", 0))
        self.assertTrue(rule.matches("Hello!!", 0))
        self.assertTrue(rule.matches("Hello. .", 0))
        self.assertTrue(rule.matches("\"Hello!\" she said.", 0))


    def test_at_most(self):
        rule = SentenceCountAtMostRule(count=2)
        self.assertTrue(rule.matches("This is only one sentence with multiple punctuations...", 0))
        self.assertFalse(rule.matches("This  is a\n test.  Woo! We have three sentences, right?", 0))

    def test_invalid_threshold(self):
        with self.assertRaises(ValueError):
            SentenceCountAtLeastRule(count=0)
        with self.assertRaises(ValueError):
            SentenceCountAtMostRule(count=0)

    def test_get_prompt_at_least(self):
        rule = SentenceCountAtLeastRule(count=2)
        self.assertEqual(rule.get_prompt(0), "The response must be at least 2 sentences.")

    def test_get_prompt_at_most(self):
        rule = SentenceCountAtMostRule(count=2)
        self.assertEqual(
            rule.get_prompt(0), "The response must be no more than 2 sentences."
        )

    def test_get_prompt_at_least_one(self):
        rule = SentenceCountAtLeastRule(count=1)
        self.assertEqual(rule.get_prompt(0), "The response must be at least 1 sentence.")

    def test_get_prompt_at_most_one(self):
        rule = SentenceCountAtMostRule(count=1)
        self.assertEqual(rule.get_prompt(0), "The response must be no more than 1 sentence.")


if __name__ == "__main__":
    unittest.main()
