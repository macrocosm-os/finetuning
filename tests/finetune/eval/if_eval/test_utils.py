import unittest
from finetune.eval.if_eval.utils import get_words


class TestUtils(unittest.TestCase):
    def test_get_words(self):
        self.assertEqual(get_words("Hello, world!"), ["Hello", "world"])
        self.assertEqual(get_words("Python is great."), ["Python", "is", "great"])
        self.assertEqual(
            get_words("No punctuation here"), ["No", "punctuation", "here"]
        )
        self.assertEqual(get_words(""), [])
        self.assertEqual(get_words("Multiple     spaces"), ["Multiple", "spaces"])


if __name__ == "__main__":
    unittest.main()
