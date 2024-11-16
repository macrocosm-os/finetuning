import unittest
from finetune.eval.if_eval.casing import UppercaseRule, LowercaseRule


class TestCasing(unittest.TestCase):
    def test_uppercase(self):
        rule = UppercaseRule()
        self.assertTrue(rule.matches("THIS  IS A\n TEST 1, ALL UPPERCASE!", 0))
        self.assertFalse(rule.matches("THIS 1 HAS A LOWERCASE LETTEr.", 0))

    def test_lowercase(self):
        rule = LowercaseRule()
        self.assertTrue(rule.matches("this  is a\n test 1, all lowercase.", 0))
        self.assertFalse(rule.matches("This 1 has an uppercase letter.", 0))

    def test_get_prompt_uppercase(self):
        rule = UppercaseRule()
        self.assertEqual(rule.get_prompt(0), "All letters in the response must be uppercase.")

    def test_get_prompt_lowercase(self):
        rule = LowercaseRule()
        self.assertEqual(
            rule.get_prompt(0), "All letters in the response must be lowercase."
        )

if __name__ == "__main__":
    unittest.main()
