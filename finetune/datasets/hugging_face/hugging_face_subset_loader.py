# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 const

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import torch
import typing
import requests
import bittensor as bt
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer
import time


class HuggingFaceSubsetLoader(IterableDataset):
    # TODO: Add a map of dataset name to max rows. Also generator that separates pages by rows per page.
    def __init__(
        self,
        dataset_name: str,
        batch_size,
        sequence_length,
        page_row_offsets: typing.List[int],
        page_row_count: int,
        tokenizer: AutoTokenizer,
    ):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.tokenizer = tokenizer
        self.base_url = "https://datasets-server.huggingface.co/rows"
        self.params = {
            "dataset": dataset_name,
            "config": "default",
            "split": "train",
        }
        self.page_row_offsets = page_row_offsets
        self.page_row_count = page_row_count
        self.buffer = []
        self.retry_limit = 10  # Number of retries
        self.retry_delay = 5  # Seconds to wait between retries

        for page_row_offset in self.page_row_offsets:
            self.fetch_data_for_page(page_row_offset)

    def fetch_data_for_page(self, page_row_offset):
        self.params["offset"] = page_row_offset
        self.params["limit"] = self.page_row_count
        attempt = 0
        while attempt < self.retry_limit:
            try:
                response = requests.get(self.base_url, params=self.params)
                response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code
                for row in response.json()["rows"]:
                    content = row["row"]["content"]
                    # Truncate/Pad the content to the desired sequence length. This avoid context breaks.
                    self.buffer += self.tokenizer(
                        content,
                        truncation=True,
                        padding="max_length",
                        max_length=self.sequence_length,
                    )["input_ids"]
                break  # If the request was successful, break out of the retry loop
            except requests.exceptions.RequestException as e:
                attempt += 1
                bt.logging.warning(
                    f"Failed to fetch data, retrying. Attempt {attempt}/{self.retry_limit}"
                )
                if attempt < self.retry_limit:
                    time.sleep(self.retry_delay)  # Wait before the next retry
                else:
                    bt.logging.error(
                        "Maximum retry limit reached. Unable to fetch data."
                    )
                    raise

    def __iter__(self):
        while len(self.buffer) >= self.sequence_length * self.batch_size:
            batch = []
            for _ in range(self.batch_size):
                batch.append(torch.tensor(self.buffer[: self.sequence_length]))
                self.buffer = self.buffer[self.sequence_length :]
            yield torch.stack(batch)

    def __next__(self):
        batch = []
        for _ in range(self.batch_size):
            batch.append(torch.tensor(self.buffer[: self.sequence_length]))
            self.buffer = self.buffer[self.sequence_length :]
        return torch.stack(batch)
