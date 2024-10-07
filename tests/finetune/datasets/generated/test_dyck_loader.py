import unittest
from collections import defaultdict

from finetune.datasets.generated.dyck_loader import (
    DYCK_CHARACTER_PAIRS,
    DyckLoader,
    complete_dyck,
    generate_dyck,
)


class TestDyckLoader(unittest.TestCase):
    def test_generate_dyck(self):
        """Tests that generate dyck only generates dycks and they are of the expected length."""

        for i in range(100):
            dyck = generate_dyck(DYCK_CHARACTER_PAIRS, i)
            self.assertEqual(len(dyck), 2 * i)
            self.assertEqual(len(complete_dyck(DYCK_CHARACTER_PAIRS, dyck)), 0)

    def test_complete_dyck_bad_chars(self):
        """Tests that complete dyck raises an error if a bad character is found."""

        with self.assertRaises(ValueError):
            _ = complete_dyck([("(", ")")], "(<>)")

    def test_complete_dyck_bad_dyck(self):
        """Tests that complete dyck raises an error if the incomplete dyck is already invalid."""

        with self.assertRaises(ValueError):
            _ = complete_dyck(DYCK_CHARACTER_PAIRS, "(<)>")

    def test_complete_dyck_empty(self):
        """Tests that complete dyck returns and empty string for already complete dycks."""

        completing_chars = complete_dyck(DYCK_CHARACTER_PAIRS, "(<>)")

        self.assertEqual(completing_chars, "")

    def test_complete_dyck(self):
        """Tests that complete dyck correctly completes dycks."""

        completing_chars = complete_dyck(DYCK_CHARACTER_PAIRS, "(<{[]")

        self.assertEqual(completing_chars, "}>)")

    def test_bad_word_length(self):
        """Tests that instantiating a loader with the min word length greater than the max word length fails."""

        with self.assertRaises(ValueError):
            _ = DyckLoader(
                min_length_pairs=20,
                max_length_pairs=2,
                min_length_answer=1,
                max_length_answer=3,
            )

    def test_bad_answer_length(self):
        """Tests that instantiating a loader with the min answer length greater than the max answer length fails."""

        with self.assertRaises(ValueError):
            _ = DyckLoader(
                min_length_pairs=2,
                max_length_pairs=20,
                min_length_answer=3,
                max_length_answer=1,
            )

    def test_bad_word_to_answer_length(self):
        """Tests that instantiating a loader with the max answer length greater than the max word length fails."""

        with self.assertRaises(ValueError):
            _ = DyckLoader(
                min_length_pairs=2,
                max_length_pairs=20,
                min_length_answer=1,
                max_length_answer=21,
            )

    def test_generate_even_distribution_pair_count(self):
        """Tests that the loader creates a roughly even distribution of dyck prompt lengths"""

        loader = DyckLoader(
            samples=100000,
            min_length_pairs=1,
            max_length_pairs=5,
            min_length_answer=1,
            max_length_answer=1,
        )

        lengths = defaultdict(int)

        # Challenge length includes the prompt and spaces but not the reference.
        # So we just check for exactly 5 different lengths.
        for challenge, _ in loader:
            lengths[len(challenge)] += 1

        self.assertEqual(len(lengths), 5)

        for value in lengths.values():
            self.assertAlmostEqual(value, 20000, delta=200)

    def test_generate_even_distribution_answer_length(self):
        """Tests that the loader creates a roughly even distribution of dyck answer lengths"""

        loader = DyckLoader(
            samples=100000,
            min_length_pairs=1,
            max_length_pairs=5,
            min_length_answer=1,
            max_length_answer=5,
        )

        lengths = defaultdict(int)

        # Reference length includes spaces so we just check for exactly 5 different lengths.
        for _, reference in loader:
            lengths[len(reference)] += 1

        self.assertEqual(len(lengths), 5)

        for value in lengths.values():
            self.assertAlmostEqual(value, 20000, delta=200)

    def test_length(self):
        """Tests that the loader correctly reports the length of samples generated."""
        loader = DyckLoader(samples=100)

        self.assertEqual(len(loader), 100)

    def test_iterate(self):
        """Tests that the loader correctly iterates across all the samples."""
        loader = DyckLoader(samples=100)

        iterated = 0
        for _, _ in loader:
            iterated += 1

        self.assertEqual(iterated, 100)

    def test_get_sample(self):
        """Tests that the loader can successfully get a single sample."""
        loader = DyckLoader(samples=10)

        for _ in range(100):
            _ = loader.get_sample()

    # Tokenization test deliberately elided for performance.
    # def test_tokenize(self):
    #     pass
