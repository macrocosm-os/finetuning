import unittest

from transformers import AutoTokenizer

from finetune.datasets.hugging_face.hugging_face_subset_loader import (
    HuggingFaceSubsetLoader,
)


class TestHuggingFaceSubsetLoader(unittest.TestCase):

    def test_load_falcon_data(self):
        """Tests we can load data from the hugging face falcon dataset."""
        tokenizer = AutoTokenizer.from_pretrained("Xenova/gpt-4")
        batches = list(
            HuggingFaceSubsetLoader(
                dataset_name="tiiuae/falcon-refinedweb",
                batch_size=1,
                sequence_length=1024,
                page_row_offsets=[0, 100],
                page_row_count=100,
                tokenizer=tokenizer,
            )
        )

        self.assertEqual(len(batches), 155)
