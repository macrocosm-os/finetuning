# The MIT License (MIT)

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
import typing
import random
import time
import requests
import torch

import bittensor as bt
from transformers import PreTrainedTokenizerBase


FINEWEB_EDU_SCORE_2_NAME = "HuggingFaceFW/fineweb-edu-score-2"
FALCON_NAME = "tiiuae/falcon-refinedweb"


class HuggingFaceLoader:
    rows_base_url: str = "https://datasets-server.huggingface.co/rows"
    size_base_url: str = "https://datasets-server.huggingface.co/size"

    retry_limit: int = 10  # Number of retries
    retry_delay: int = 5  # Seconds to wait between retries

    def __init__(
        self,
        name: str,
        num_pages: int = 4,
        num_rows_per_page: int = 100,
        random_seed: typing.Optional[int] = None,
    ):
        """Loads text data from hugging face datasets.

        Args:
            name (str): Name of the dataset from hugging face. Must match url path.
            num_pages (int, optional): Number of pages of data to fetch. Defaults to 4.
            num_rows_per_page (int, optional): Number of rows to read from each fetched page. Defaults to 100.
            random_seed (typing.Optional[int], optional): Seed to use for all random operations. Defaults to None.
        """
        self.name = name
        self.num_pages = num_pages

        # Initialize with seed if provided.
        if random_seed is not None:
            random.seed(random_seed)

        # Note that the number of 'samples' is equal to pages * rows per page since each sample is a single row.
        self.num_rows_per_page = num_rows_per_page
        self.duplicate_page_threshold = 100

        # Buffer to hold rows of pages loaded from the api
        self.buffer: typing.List[str] = []

        # Get the dataset configs and their row sizes
        self.configs_data = self.fetch_dataset_configs()

        # We first need to fetch the data and fill the loader buffer.
        # Since some sample files are broken, we first try to find `num_pages`
        # responsive samples, then we add them to the found pages `self.pages`
        if self.num_pages:
            self._fetch_data_to_buffer(self.num_pages)

    def _fetch_data_to_buffer(self, num_pages):
        """
        Randomly sample unique pages and add their data to the buffer.
        If a page is inaccessible, another one is sampled.
        this method sets the `pages` property
        """

        self.pages = []
        attempts = 0
        duplicates = 0

        # Choose a consistent initial offset for the random pages so we do not overlap on each page get.
        initial_offset = random.randint(0, self.num_rows_per_page - 1)

        while len(self.pages) < num_pages:

            # randomly sample one page
            page = self.get_random_pages(num_pages=1, initial_offset=initial_offset)[0]

            # skip the page if we already have it
            if page in self.pages:
                duplicates += 1
                if duplicates >= self.duplicate_page_threshold:
                    bt.logging.debug(
                        f"Hit duplicate page threshold of {self.duplicate_page_threshold}. Stopping early at: {len(self.pages)} pages."
                    )
                    break
                else:
                    continue

            config_name, page_row_start, split = page

            # Create the request parameters
            params = dict(
                dataset=self.name,
                config=config_name,
                split=split,
                offset=page_row_start,
                limit=self.num_rows_per_page,
            )

            try:
                response = requests.get(self.rows_base_url, params=params)

                response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code

                # Add the page since the request was successful
                self.pages.append(page)

                for row in response.json()["rows"]:
                    # TODO: Consider taking this as a parameter instead.
                    # For now we either get the "text" column from fineweb or "content" from falcon.
                    if self.name == FINEWEB_EDU_SCORE_2_NAME:
                        content = row["row"]["text"]
                    elif self.name == FALCON_NAME:
                        content = row["row"]["content"]
                    else:
                        raise NotImplementedError(
                            f"Unable to parse rows from hugging face dataset: {self.name}"
                        )

                    # Append the content to the buffer without tokenization.
                    self.buffer.append(content)

                response.close()

            except requests.exceptions.RequestException:
                response.close()
                attempts += 1
                bt.logging.warning(
                    f"Failed to fetch data, retrying with a newly sampled page. Attempt {attempts}/{self.retry_limit * num_pages}"
                )
                if attempts < num_pages * self.retry_limit:
                    pass

                else:
                    bt.logging.error(
                        "Maximum retry limit reached. Unable to fetch data."
                    )
                    raise

    def get_random_pages(self, num_pages, initial_offset):
        """
        Randomly sample num_pages with an intiial offset.
        A page is a row number of a given split of a given dataset dump offset by num_rows_per_page.
        """
        pages = []

        for _ in range(num_pages):

            # Choose a random config.
            config_name = random.choice(list(self.configs_data.keys()))

            # Choose a random page start.
            # We do so by chunking the rows of the config data into N pages of length num_rows_per_page.
            # We remove the initial offset from the total rows in doing this calculation to ensure we don't go over.
            data_row_count = self.configs_data[config_name]["num_rows"] - initial_offset
            # Add 1 to the row count as we are 0 indexed.
            data_page_count = (data_row_count + 1) // self.num_rows_per_page
            # Select a random page start by taking the randomly selected page and multiplying by num_rows_per_page.
            selected_page_start = initial_offset + (
                random.randint(0, data_page_count - 1) * self.num_rows_per_page
            )

            split = self.configs_data[config_name]["split"]

            pages.append((config_name, selected_page_start, split))

        return pages

    def get_page_names(self):
        """
        This is a utility function that returns the page names that were used.
        Each page as a single string instead of a tuple
        """

        page_names = []

        if hasattr(self, "pages"):
            page_names = [
                f"{cfg_name}_{num_rows}_{split}"
                for cfg_name, num_rows, split in self.pages
            ]

        return page_names

    def fetch_dataset_configs(self) -> typing.Dict[str, typing.Dict]:
        """
        Fetch the different dump names, aka configs, aka samples, of the
        dataset.
        The returned value is a dictionary with dump names as keys and
        a dict of the number of rows and the split as values.
        """
        # Request parameters
        params = dict(dataset=self.name)

        attempt = 0
        while attempt < self.retry_limit:
            try:
                response = requests.get(self.size_base_url, params=params)
                response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code

                # Extract the configs dict
                configs_dict = response.json()["size"]["splits"]

                # If the dict only has one entry, use the default (as in Falcon). Else use the rest (as in Fineweb).
                # Create a new dict with config names as keys and the number of rows as values.
                if len(configs_dict) == 1:
                    configs_data = {
                        entry["config"]: {
                            "num_rows": entry["num_rows"],
                            "split": entry["split"],
                        }
                        for entry in configs_dict
                    }
                else:
                    configs_data = {
                        entry["config"]: {
                            "num_rows": entry["num_rows"],
                            "split": entry["split"],
                        }
                        for entry in configs_dict
                        if entry["config"] != "default"
                    }

                return configs_data

            except requests.exceptions.RequestException:
                attempt += 1
                bt.logging.warning(
                    f"Failed to fetch dataset configs, retrying. Attempt {attempt}/{self.retry_limit}"
                )
                if attempt < self.retry_limit:
                    time.sleep(self.retry_delay)  # Wait before the next retry
                else:
                    bt.logging.error(
                        "Maximum retry limit reached. Unable to fetch data."
                    )
                    raise

    def tokenize(
        self, tokenizer: PreTrainedTokenizerBase, sequence_length: int
    ) -> typing.List[torch.Tensor]:
        # Each batch is a tokenized row of content up to sequence length.
        batches = []

        for content in self:
            input_ids = tokenizer(content, max_length=sequence_length, truncation=True)[
                "input_ids"
            ]

            batches.append(torch.tensor(input_ids + [tokenizer.eos_token_id]))

        return batches

    def get_sample(self) -> typing.Tuple[str]:
        return random.choice(self.buffer)

    def __iter__(self):
        return self.buffer.__iter__()

    def __len__(self):
        return len(self.buffer)
