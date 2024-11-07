import unittest
from collections import defaultdict

from transformers import AutoTokenizer

from finetune.datasets.hugging_face.hugging_face_loader import (
    FALCON_NAME,
    FINEWEB_EDU_SCORE_2_NAME,
    HuggingFaceLoader,
)


class TestHuggingFaceLoader(unittest.TestCase):
    def test_unique_pages_fineweb(self):
        """Tests that the hugging face loader only loads unique pages from fine web."""
        # Ensure we get all the possible pages of the aritificially shortened data.
        NUM_PAGES = 5
        CONFIG_DATA = {"CC-MAIN-2013-20": {"num_rows": 499, "split": "train"}}

        # Avoid loading pages until we override the config data for testing.
        dataloader = HuggingFaceLoader(name=FINEWEB_EDU_SCORE_2_NAME, num_pages=None)
        dataloader.configs_data = CONFIG_DATA

        # Only fetch these once for performance, although for better correctness should consider running in a loop.
        # We check for max pages or max pages - 1 to handle the random offset.
        dataloader._fetch_data_to_buffer(NUM_PAGES)
        self.assertIn(len(dataloader.pages), [NUM_PAGES, NUM_PAGES - 1])
        self.assertIn(len(set(dataloader.pages)), [NUM_PAGES, NUM_PAGES - 1])

        self.assertIn(
            len(dataloader.buffer),
            [
                NUM_PAGES * dataloader.num_rows_per_page,
                (NUM_PAGES - 1) * dataloader.num_rows_per_page,
            ],
        )
        self.assertEqual(len(dataloader.buffer), len(set(dataloader.buffer)))

    def test_duplicate_threshold_fineweb(self):
        """Tests that the hugging face loader stops loading after hitting too many duplicate pages from fine web."""
        # Try to get 6 pages from a set that only contains 5 pages worth.
        NUM_PAGES = 6
        NUM_PAGES_ACTUAL = 5
        CONFIG_DATA = {"CC-MAIN-2013-20": {"num_rows": 499, "split": "train"}}

        # Avoid loading pages until we override the config data for testing.
        dataloader = HuggingFaceLoader(name=FINEWEB_EDU_SCORE_2_NAME, num_pages=None)
        dataloader.configs_data = CONFIG_DATA

        # Only fetch these once for performance, although for better correctness should consider running in a loop.
        # We check for actual pages or actual pages - 1 to handle the random offset.
        dataloader._fetch_data_to_buffer(NUM_PAGES)
        self.assertIn(len(dataloader.pages), [NUM_PAGES_ACTUAL, NUM_PAGES_ACTUAL - 1])
        self.assertIn(
            len(set(dataloader.pages)), [NUM_PAGES_ACTUAL, NUM_PAGES_ACTUAL - 1]
        )

        self.assertIn(
            len(dataloader.buffer),
            [
                NUM_PAGES_ACTUAL * dataloader.num_rows_per_page,
                (NUM_PAGES_ACTUAL - 1) * dataloader.num_rows_per_page,
            ],
        )
        self.assertEqual(len(dataloader.buffer), len(set(dataloader.buffer)))

    def test_page_offset_fineweb(self):
        """Tests that the hugging face loader will only generate page starts that are num rows per pages apart from fineweb."""

        # Avoid loading pages until we override the config data for testing.
        dataloader = HuggingFaceLoader(
            name=FINEWEB_EDU_SCORE_2_NAME, num_pages=None, num_rows_per_page=100
        )

        # Create a fake configs data with only 599 rows.
        dataloader.configs_data = {
            "CC-MAIN-2013-20": {"num_rows": 599, "split": "train"}
        }

        # Ensure get random pages returns only 0, 100, 200, 300, 400 and 500.
        expected_page_starts_1 = {0, 100, 200, 300, 400, 500}
        page_starts_1 = defaultdict(int)
        for _ in range(1000):
            random_pages = dataloader.get_random_pages(1, initial_offset=0)
            _, page_start, _ = random_pages[0]
            page_starts_1[page_start] += 1

        self.assertEqual(set(page_starts_1.keys()), expected_page_starts_1)

        # Create a fake configs data with only 598 rows.
        dataloader.configs_data = {
            "CC-MAIN-2013-20": {"num_rows": 598, "split": "train"}
        }

        # Ensure get random pages returns only 0, 100, 200, 300, and 400 (since 500-598 is not 100 rows).
        expected_page_starts_2 = {0, 100, 200, 300, 400}
        page_starts_2 = defaultdict(int)
        for _ in range(1000):
            random_pages = dataloader.get_random_pages(1, initial_offset=0)
            _, page_start, _ = random_pages[0]
            page_starts_2[page_start] += 1

        self.assertEqual(set(page_starts_2.keys()), expected_page_starts_2)

    def test_page_initial_offset_fineweb(self):
        """Tests that the hugging face loader correctly applies an initial offset to the page starts from fineweb."""
        # Avoid loading pages until we override the config data for testing.
        dataloader = HuggingFaceLoader(name=FINEWEB_EDU_SCORE_2_NAME, num_pages=None)

        # Ensure we know the num_rows_per_page.
        test_num_rows_per_page = 100
        dataloader.num_rows_per_page = test_num_rows_per_page

        # Create a fake configs data with only 599 rows.
        dataloader.configs_data = {
            "CC-MAIN-2013-20": {"num_rows": 599, "split": "train"}
        }

        # Define initial offset of 50.
        initial_offset = 50
        # Ensure get random pages returns only 50, 150, 250, 350, and 450.
        expected_page_starts_1 = {50, 150, 250, 350, 450}
        page_starts_1 = defaultdict(int)
        for _ in range(1000):
            random_pages = dataloader.get_random_pages(1, initial_offset=initial_offset)
            _, page_start, _ = random_pages[0]
            page_starts_1[page_start] += 1

        self.assertEqual(set(page_starts_1.keys()), expected_page_starts_1)

        # Create a fake configs data with only 548 rows.
        dataloader.configs_data = {
            "CC-MAIN-2013-20": {"num_rows": 548, "split": "train"}
        }

        # Ensure get random pages returns only 50, 150, 250, and 350 (since 450-548 is not 100 rows)
        expected_page_starts_2 = {50, 150, 250, 350}
        page_starts_2 = defaultdict(int)
        for _ in range(1000):
            random_pages = dataloader.get_random_pages(1, initial_offset=initial_offset)
            _, page_start, _ = random_pages[0]
            page_starts_2[page_start] += 1

        self.assertEqual(set(page_starts_2.keys()), expected_page_starts_2)

    def test_seed_fineweb(self):
        """Tests that the hugging face data loader fetches the same data with the same seed (barring retries) from fineweb."""

        # Use the same seed for each loader.
        RANDOM_SEED = 1
        # Fetch just two pages to keep the test reasonably fast.
        NUM_PAGES = 2

        # First dataloader.
        dataloader_1 = HuggingFaceLoader(
            name=FINEWEB_EDU_SCORE_2_NAME,
            num_pages=NUM_PAGES,
            random_seed=RANDOM_SEED,
        )

        # Assert that the number of pages requested were loaded.
        self.assertEqual(len(dataloader_1.pages), NUM_PAGES)

        # Now create a second loader with the same seed.
        dataloader_2 = HuggingFaceLoader(
            name=FINEWEB_EDU_SCORE_2_NAME,
            num_pages=NUM_PAGES,
            random_seed=RANDOM_SEED,
        )

        # Assert both dataloaders have the same pages
        self.assertEqual(set(dataloader_1.pages), set(dataloader_2.pages))

        # Assert that both have the same buffers
        self.assertEqual(dataloader_1.buffer, dataloader_2.buffer)

    def test_seed_falcon(self):
        """Tests that the hugging face data loader fetches the same data with the same seed (barring retries) from falcon."""

        # Use the same seed for each loader.
        RANDOM_SEED = 1
        # Fetch just two pages to keep the test reasonably fast.
        NUM_PAGES = 2

        # First dataloader.
        dataloader_1 = HuggingFaceLoader(
            name=FALCON_NAME,
            num_pages=NUM_PAGES,
            random_seed=RANDOM_SEED,
        )

        # Assert that the number of pages requested were loaded.
        self.assertEqual(len(dataloader_1.pages), NUM_PAGES)

        # Now create a second loader with the same seed.
        dataloader_2 = HuggingFaceLoader(
            name=FALCON_NAME,
            num_pages=NUM_PAGES,
            random_seed=RANDOM_SEED,
        )

        # Assert both dataloaders have the same pages
        self.assertEqual(set(dataloader_1.pages), set(dataloader_2.pages))

        # Assert that both have the same buffers
        self.assertEqual(dataloader_1.buffer, dataloader_2.buffer)

    def test_length_fineweb(self):
        """Tests that the hugging face loader correctly reports the length of samples generated from fineweb."""
        # Fetch just two pages to keep the test reasonably fast.
        NUM_PAGES = 2
        NUM_ROWS_PER_PAGE = 100
        loader = HuggingFaceLoader(
            name=FINEWEB_EDU_SCORE_2_NAME,
            num_pages=NUM_PAGES,
            num_rows_per_page=NUM_ROWS_PER_PAGE,
        )

        self.assertEqual(len(loader), NUM_PAGES * NUM_ROWS_PER_PAGE)

    def test_iterate_fineweb(self):
        """Tests that the hugging face loader correctly iterates across all the samples from fineweb."""
        NUM_PAGES = 2
        NUM_ROWS_PER_PAGE = 100
        loader = HuggingFaceLoader(
            name=FINEWEB_EDU_SCORE_2_NAME,
            num_pages=NUM_PAGES,
            num_rows_per_page=NUM_ROWS_PER_PAGE,
        )

        iterated = 0
        for _ in loader:
            iterated += 1

        self.assertEqual(iterated, NUM_PAGES * NUM_ROWS_PER_PAGE)

    def test_get_sample_fineweb(self):
        """Tests that the hugging face loader can successfully get a single sample from fineweb."""
        NUM_PAGES = 2
        NUM_ROWS_PER_PAGE = 100
        loader = HuggingFaceLoader(
            name=FINEWEB_EDU_SCORE_2_NAME,
            num_pages=NUM_PAGES,
            num_rows_per_page=NUM_ROWS_PER_PAGE,
        )

        for _ in range(100):
            _ = loader.get_sample()

    def test_tokenize_fineweb(self):
        """Tests that the hugging face loader can successfully tokenize the data in the buffer from fineweb."""
        NUM_PAGES = 2
        NUM_ROWS_PER_PAGE = 100
        loader = HuggingFaceLoader(
            name=FINEWEB_EDU_SCORE_2_NAME,
            num_pages=NUM_PAGES,
            num_rows_per_page=NUM_ROWS_PER_PAGE,
        )

        tokenizer = AutoTokenizer.from_pretrained("Xenova/gpt-4")

        batches = loader.tokenize(tokenizer, sequence_length=4096)

        self.assertEqual(len(batches), NUM_PAGES * NUM_ROWS_PER_PAGE)

    def test_tokenize_truncation_fineweb(self):
        """Tests that the hugging face loader can successfully tokenize and truncate the data in the buffer from fineweb."""
        NUM_PAGES = 2
        NUM_ROWS_PER_PAGE = 100
        loader = HuggingFaceLoader(
            name=FINEWEB_EDU_SCORE_2_NAME,
            num_pages=NUM_PAGES,
            num_rows_per_page=NUM_ROWS_PER_PAGE,
        )

        tokenizer = AutoTokenizer.from_pretrained("Xenova/gpt-4")

        batches_regular = loader.tokenize(tokenizer, sequence_length=4096)
        batches_truncation = loader.tokenize(tokenizer, sequence_length=5)

        # Check that at least one of the regular batches is > 5 tokens long.
        self.assertTrue(any(len(batch > 5) for batch in batches_regular))

        # Check that all of the truncated batches are <= 5 tokens long.
        self.assertTrue(all(len(batch <= 5) for batch in batches_truncation))
