import unittest

from finetune.eval.if_eval.keywords import (
    KeywordForbiddenRule,
    KeywordFrequencyRule,
    KeywordInclusionRule,
    interesting_keyword,
)


class TestInterestingKeyword(unittest.TestCase):
    def test_interesting_keyword_with_desirable_tags(self):
        text = "The quick brown fox jumps over the lazy dog."
        for _ in range(100):
            keyword = interesting_keyword(text, forbidden_words=[])
            self.assertIn(keyword, ["quick", "brown", "fox", "jumps", "lazy", "dog"])

    def test_interesting_keyword_with_forbidden_words(self):
        text = "The quick brown fox jumps over the lazy dog."
        for _ in range(100):
            keyword = interesting_keyword(
                text, forbidden_words=["quick", "brown", "fox"]
            )
            self.assertIn(keyword, ["jumps", "lazy", "dog"])

    def test_interesting_keyword_without_desirable_tags(self):
        text = "and or but if"
        for _ in range(100):
            keyword = interesting_keyword(text, forbidden_words=["and"])
            self.assertIn(keyword, ["or", "but", "if"])

    def test_interesting_keyword_with_punctuation(self):
        text = "Hello, world! This is a test."
        for _ in range(100):
            keyword = interesting_keyword(text, forbidden_words=[])
            self.assertIn(keyword, ["hello", "world", "this", "test"])

    def test_interesting_keyword_no_options(self):
        text = "Test"
        for _ in range(100):
            keyword = interesting_keyword(text, forbidden_words=["test"])
            self.assertIn(keyword, ["answer", "question", "response"])


class TestIncludeKeyword(unittest.TestCase):
    def test_get_prompt(self):
        rule = KeywordInclusionRule(["test", "example"])
        self.assertEqual(
            rule.get_prompt(0), 'The word "test" must be included in the response.'
        )
        self.assertEqual(
            rule.get_prompt(1), 'The word "example" must be included in the response.'
        )

    def test_matches(self):
        rule = KeywordInclusionRule(["test", "example"])
        self.assertTrue(rule.matches("This is a test sentence.", 0))
        self.assertFalse(rule.matches("This is a test sentence.", 1))

    def test_matches_not_on_substring(self):
        rule = KeywordInclusionRule(["is"])
        self.assertFalse(rule.matches("This isn't a match.", 0))

    def test_case_insensitive(self):
        rule = KeywordInclusionRule(["test"])
        self.assertTrue(rule.matches("This is a TEST sentence.", 0))

    def test_punctuation(self):
        rule = KeywordInclusionRule(["test"])
        self.assertTrue(rule.matches("This is a test!", 0))


class TestForbiddenKeyword(unittest.TestCase):
    def test_get_prompt(self):
        rule = KeywordForbiddenRule(["test", "example"])
        self.assertEqual(
            rule.get_prompt(0), 'The word "test" must not be the response.'
        )
        self.assertEqual(
            rule.get_prompt(1), 'The word "example" must not be the response.'
        )

    def test_matches(self):
        rule = KeywordForbiddenRule(["test", "example"])
        self.assertFalse(rule.matches("This is a test sentence.", 0))
        self.assertTrue(rule.matches("This is a sentence.", 0))
        self.assertTrue(rule.matches("This is a test sentence.", 1))
        self.assertFalse(rule.matches("This is an example sentence.", 1))

    def test_matches_not_on_substring(self):
        rule = KeywordForbiddenRule(["is"])
        self.assertTrue(rule.matches("This isn't a match.", 0))

    def test_case_insensitive(self):
        rule = KeywordForbiddenRule(["test"])
        self.assertFalse(rule.matches("This is a TEST sentence.", 0))

    def test_punctuation(self):
        rule = KeywordForbiddenRule(["test"])
        self.assertFalse(rule.matches("This is a test!", 0))


class TestKeywordFrequencyRule(unittest.TestCase):
    def test_get_prompt(self):
        rule = KeywordFrequencyRule([("test", 2), ("example", 3)])
        self.assertEqual(rule.get_prompt(0), 'Include the word "test" 2 times.')
        self.assertEqual(rule.get_prompt(1), 'Include the word "example" 3 times.')

    def test_matches(self):
        rule = KeywordFrequencyRule([("test", 2), ("example", 3)])
        self.assertTrue(rule.matches("This is a test test sentence.", 0))
        self.assertFalse(rule.matches("This is a test sentence.", 0))
        self.assertTrue(rule.matches("This is an example example example sentence.", 1))
        self.assertFalse(rule.matches("This is an example example sentence.", 1))

    def test_matches_not_on_substring(self):
        rule = KeywordFrequencyRule([("is", 2)])
        self.assertFalse(rule.matches("This isn't isn't a match.", 0))
        self.assertTrue(
            rule.matches(
                "This is a test. This is only a test. It isn't going to include a submatch.",
                0,
            )
        )

    def test_case_insensitive(self):
        rule = KeywordFrequencyRule([("test", 2)])
        self.assertTrue(rule.matches("This is a TEST test sentence.", 0))

    def test_punctuation(self):
        rule = KeywordFrequencyRule([("test", 2)])
        self.assertTrue(rule.matches("This is a test! Test it again.", 0))
        self.assertFalse(rule.matches("This is a test!", 0))


if __name__ == "__main__":
    unittest.main()
