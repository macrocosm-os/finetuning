import unittest
from collections import defaultdict

from finetune.datasets.generated.word_sorting_loader import (
    WordSortingLoader,
    WORD_SORTING_CHALLENGE_PROMPT,
)


class TestWordSortingOrder(unittest.TestCase):
    def test_bad_word_length(self):
        """Tests that instantiating a loader with the min word count greater than the max word count fails."""

        with self.assertRaises(ValueError):
            _ = WordSortingLoader(min_word_count=21, max_word_count=20)

    def test_generate_even_distribution_word_count(self):
        """Tests that the loader creates a roughly even distribution of word count lengths"""

        loader = WordSortingLoader(
            samples=100000,
            min_word_count=1,
            max_word_count=5,
        )

        lengths = defaultdict(int)

        for challenge, _ in loader:
            # Remove the prepended prompt.
            reference_words = challenge[len(WORD_SORTING_CHALLENGE_PROMPT) :].split()
            lengths[len(reference_words)] += 1

        # Check that we have exactly 5 keys and they are the expected word counts.
        self.assertEqual(lengths.keys(), {1, 2, 3, 4, 5})

        # Check that we have a roughly equal distribtion of key lengths.
        for value in lengths.values():
            self.assertAlmostEqual(value, 20000, delta=1000)

    def test_generate_reference_answers_are_correctly_sorted(self):
        """Tests that the loader creates correctly sorted reference answers"""

        loader = WordSortingLoader(
            samples=100000,
        )

        for _, reference in loader:
            reference_words = reference.split()
            self.assertEqual(reference_words, sorted(reference_words))

    def test_generate_reference_answers_are_always_lowercase(self):
        """Tests that the loader creates correctly lowercased reference answers"""

        loader = WordSortingLoader(
            samples=100000,
        )

        for _, reference in loader:
            self.assertTrue(reference.islower())

    def test_length(self):
        """Tests that the loader correctly reports the length of samples generated."""
        loader = WordSortingLoader(samples=100)

        self.assertEqual(len(loader), 100)

    def test_iterate(self):
        """Tests that the loader correctly iterates across all the samples."""
        loader = WordSortingLoader(samples=100)

        iterated = 0
        for _, _ in loader:
            iterated += 1

        self.assertEqual(iterated, 100)

    def test_get_sample(self):
        """Tests that the loader can successfully get a single sample."""
        loader = WordSortingLoader(samples=10)

        for _ in range(100):
            _ = loader.get_sample()

    # Tokenization test deliberately elided for performance.
    # def test_tokenize(self):
    #     pass
