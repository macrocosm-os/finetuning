import unittest
from finetune.eval.if_eval.comma import NoCommaRule


class TestComma(unittest.TestCase):
    def test_no_comma(self):
        rule = NoCommaRule()
        self.assertTrue(rule.matches("This  is a\n test; with no comma."))
        self.assertFalse(rule.matches("This is a test, with a comma."))

    def test_get_prompt_no_comma(self):
        rule = NoCommaRule()
        self.assertEqual(
            rule.get_prompt(), "Do not use any commas in your response."
        )

if __name__ == "__main__":
    unittest.main()
