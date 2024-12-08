import unittest

from finetune.eval.if_eval.start_end import (
    ENDS_WITH_PHRASES,
    EndsWithRule,
    QuotationRule,
)


class TestStartEnd(unittest.TestCase):
    def test_ends_with(self):
        for _ in range(100):
            rule = EndsWithRule()
            self.assertTrue(rule.matches(f"lower case response {rule.phrase.lower()}"))
            self.assertTrue(rule.matches(f"UPPER CASE RESPONSE {rule.phrase.upper()}"))
            self.assertFalse(rule.matches(f"Phrase in {rule.phrase} the middle fails."))
            self.assertFalse(rule.matches(f"Extra punctuation fails {rule.phrase}."))
            self.assertFalse(rule.matches(f"Newline fails {rule.phrase}\n"))

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
            # Also test that it has no commas as this rule is compatible with comma related rules.
            self.assertTrue("," not in rule.get_prompt())

    def test_quotation(self):
        rule = QuotationRule()
        self.assertTrue(rule.matches('"Response in double quotes."'))
        self.assertTrue(rule.matches('"Response in double quotes with newline. \n"'))
        self.assertFalse(rule.matches('"Response starting with a double quote.'))
        self.assertFalse(rule.matches('Response ending with a doubel quote"'))
        self.assertFalse(rule.matches('"Response with space after quotes" '))
        self.assertFalse(rule.matches('"Response with newline after quotes"\n'))

    def test_get_prompt_quotation(self):
        rule = QuotationRule()
        self.assertEqual(
            rule.get_prompt(), "Wrap your entire response in double quotation marks."
        )


if __name__ == "__main__":
    unittest.main()
