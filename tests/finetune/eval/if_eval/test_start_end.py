import unittest

from finetune.eval.if_eval.start_end import ENDS_WITH_PHRASES, EndsWithRule


class TestStardEnd(unittest.TestCase):
    def test_ends_with(self):
        for _ in range(100):
            rule = EndsWithRule()
            self.assertTrue(rule.matches(f"lower case response {rule.phrase.lower()}"))
            self.assertTrue(rule.matches(f"UPPER CASE RESPONSE {rule.phrase.upper()}"))
            self.assertFalse(rule.matches(f"Phrase in {rule.phrase} the middle fails."))
            self.assertFalse(rule.matches(f"Extra punctuation fails {rule.phrase}."))

    def test_get_prompt_ends_with(self):
        for _ in range(100):
            rule = EndsWithRule()
            self.assertTrue(
                rule.get_prompt().startswith("End your response with the exact phrase ")
            )
            self.assertTrue(
                rule.get_prompt()
                .removeprefix('End your response with the exact phrase "')
                .removesuffix('"')
                in ENDS_WITH_PHRASES
            )


if __name__ == "__main__":
    unittest.main()
