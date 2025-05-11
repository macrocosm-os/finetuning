import inspect
import json
import random
import re
import time
import typing
from typing import List

import numpy as np
import requests
import taoverse.utilities.logging as logging
import torch
from transformers import PreTrainedTokenizerBase

from finetune.datasets.loader import DatasetLoader

FINEWEB_EDU_SCORE_2_NAME = "HuggingFaceFW/fineweb-edu-score-2"
FALCON_NAME = "tiiuae/falcon-refinedweb"
SYNTHETIC_1_SFT_NAME = "PrimeIntellect/SYNTHETIC-1-SFT-Data"
CODEFORCES_COTS_NAME = "open-r1/codeforces-cots"


class HuggingFaceLoader(DatasetLoader):
    rows_base_url: str = "https://datasets-server.huggingface.co/rows"
    size_base_url: str = "https://datasets-server.huggingface.co/size"

    retry_limit: int = 10  # Number of retries
    initial_retry_delay: int = 5
    max_retry_delay: int = 60

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
        # For SYNTHETIC-1-SFT, we'll store dicts with content and metadata
        self.buffer = []

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
        retry_delay = self.initial_retry_delay

        # Choose a consistent initial offset for the random pages
        initial_offset = random.randint(0, self.num_rows_per_page - 1)

        while len(self.pages) < num_pages:

            # randomly sample one page
            page = self.get_random_pages(num_pages=1, initial_offset=initial_offset)[0]

            # skip the page if we already have it
            if page in self.pages:
                duplicates += 1
                if duplicates >= self.duplicate_page_threshold:
                    logging.debug(
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

                # Check specifically for rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", retry_delay))
                    logging.warning(
                        f"Rate limited. Waiting {retry_after} seconds before retry. Attempt {attempts + 1}/{self.retry_limit}"
                    )
                    time.sleep(retry_after)
                    retry_delay = min(
                        retry_delay * 2, self.max_retry_delay
                    )  # Exponential backoff
                    attempts += 1
                    if attempts >= self.retry_limit:
                        raise requests.exceptions.HTTPError(
                            "Rate limit exceeded after maximum retries",
                            response=response,
                        )
                    continue

                response.raise_for_status()

                # Reset retry delay on successful request
                retry_delay = self.initial_retry_delay

                # Process the successful response
                self.pages.append(page)

                for row in response.json()["rows"]:
                    # TODO: Consider taking this as a parameter instead.
                    # For now we either get the "text" column from fineweb or "content" from falcon.
                    if self.name == FINEWEB_EDU_SCORE_2_NAME:
                        content = row["row"]["text"]
                        self.buffer.append(content)
                    elif self.name == FALCON_NAME:
                        content = row["row"]["content"]
                        self.buffer.append(content)
                    elif self.name == SYNTHETIC_1_SFT_NAME:
                        # Rows have keys: 'response_id', 'problem_id', 'task_type', 'score', 'messages'
                        # We have to extract metadata because of task-specific parsing
                        messages = row["row"]["messages"]
                        task_type = row["row"].get("task_type", "unknown")
                        problem_id = row["row"].get("problem_id", "")
                        score = row["row"].get("score", 0)

                        # Store both content and metadata
                        self.buffer.append(
                            {
                                "messages": messages,
                                "task_type": task_type,
                                "problem_id": problem_id,
                                "score": score,
                            }
                        )
                    elif self.name == CODEFORCES_COTS_NAME:
                        messages = row["row"]["messages"]
                        problem_id = row["row"].get("id", "")
                        contest_type = row["row"].get("contest_type", "")
                        self.buffer.append(
                            {
                                "messages": messages,
                                "problem_id": problem_id,
                                "contest_type": contest_type,
                            }
                        )
                    else:
                        raise NotImplementedError(
                            f"Unable to parse rows from hugging face dataset: {self.name}"
                        )

                response.close()

            except requests.exceptions.RequestException as e:
                response.close()
                attempts += 1
                if attempts >= self.retry_limit:
                    logging.error("Maximum retry limit reached. Unable to fetch data.")
                    raise

                logging.warning(
                    f"Failed to fetch data, retrying with a newly sampled page. Attempt {attempts}/{self.retry_limit}"
                )
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, self.max_retry_delay)

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
                logging.warning(
                    f"Failed to fetch dataset configs, retrying. Attempt {attempt}/{self.retry_limit}"
                )
                if attempt < self.retry_limit:
                    time.sleep(self.retry_delay)  # Wait before the next retry
                else:
                    logging.error("Maximum retry limit reached. Unable to fetch data.")
                    raise

    def tokenize(
        self,
        tokenizer: PreTrainedTokenizerBase,
        sequence_length: int,
    ) -> typing.Union[
        typing.List[np.ndarray],
        typing.Dict[str, typing.List],
        typing.List[typing.Tuple[np.array, np.array]],
    ]:
        """Tokenize the dataset.

        Args:
            tokenizer: The tokenizer to use
            sequence_length: Maximum sequence length for tokenization

        Returns:
            Tokenized data in appropriate format
        """
        result = []
        for text in self.buffer:
            tokens = tokenizer.encode(
                text,
                max_length=sequence_length,
                padding="max_length",
                truncation=True,
                return_tensors="np",
            )
            result.append(tokens)

        return result

    def get_sample(self) -> typing.Union[str, dict]:
        return random.choice(self.buffer)

    def __iter__(self):
        return self.buffer.__iter__()

    def __len__(self):
        return len(self.buffer)


class BaseReasoningLoader(HuggingFaceLoader):
    """Base class for loaders that handle reasoning traces with questions and answers."""

    def __init__(
        self,
        name: str,
        num_pages: int = 1,
        num_rows_per_page: int = 100,
        random_seed: typing.Optional[int] = None,
        max_sequence_length: typing.Optional[int] = None,
        chars_per_token: int = 4,  # Heuristic: 4 characters per token
    ):
        self.max_sequence_length = max_sequence_length
        self.chars_per_token = chars_per_token
        self.questions = []
        self.traces = []
        self.buffer_parsed_indices = set()
        super().__init__(
            name=name,
            num_pages=num_pages,
            num_rows_per_page=num_rows_per_page,
            random_seed=random_seed,
        )
        self._parse_samples()

    def _parse_samples(self):
        self._parse_additional_samples()
        logging.info(f"Successfully parsed {len(self.questions)} samples")
        if len(self.questions) == 0:
            logging.warning("No samples successfully parsed.")

    def _parse_additional_samples(self):
        filtered_count = 0
        for i, sample in enumerate(self.buffer):
            if i in self.buffer_parsed_indices:
                continue
            self.buffer_parsed_indices.add(i)
            try:
                question, trace = self._extract_messages(sample)
                if not self._fits_sequence_length(question, trace):
                    filtered_count += 1
                    continue
                self.questions.append(question)
                self.traces.append(trace)
            except Exception as e:
                logging.warning(f"Error parsing sample {i}: {e}")
                continue
        if filtered_count > 0 and self.max_sequence_length is not None:
            logging.info(
                f"Filtered out {filtered_count} samples exceeding the max sequence length of {self.max_sequence_length} tokens"
            )

    def _extract_messages(self, sample):
        raise NotImplementedError

    def _fits_sequence_length(self, question: str, trace: str) -> bool:
        if self.max_sequence_length is None:
            return True
        total_length = self._estimate_token_length(question) + self._estimate_token_length(trace)
        total_length += 10
        return total_length <= self.max_sequence_length

    def _estimate_token_length(self, text: str) -> int:
        if not text:
            return 0
        return len(text) // self.chars_per_token

    def tokenize(
        self,
        tokenizer: PreTrainedTokenizerBase,
        sequence_length: int,
    ) -> typing.List[typing.Tuple[np.array, np.array]]:
        result = []
        if len(self.questions) == 0:
            logging.warning("No samples available for tokenization")
            return result
        for q, t in zip(self.questions, self.traces):
            formatted_question = tokenizer.apply_chat_template(
                [
                    {
                        "role": "system",
                        "content": "You are a reflective AI capable of using extended chains of thought to consider the problem thoroughly and deliberate through systematic reasoning to arrive at a correct solution before answering. Enclose your internal monologue in <think> ... </think> tags, then provide your final answer in the format that the user requests.",
                    },
                    {"role": "user", "content": q},
                ],
                add_generation_prompt=True,
                tokenize=False,
            )
            context_tokens = np.array(
                tokenizer.encode(
                    formatted_question,
                    truncation=False,
                )
            )
            reference_tokens = np.array(
                tokenizer.encode(
                    t.strip(),
                    truncation=False,
                )
            )
            result.append((context_tokens, reference_tokens))
        return result

    @property
    def samples(self):
        return [{"question": q, "trace": t} for q, t in zip(self.questions, self.traces)]


class CodeforcesCOTSLoader(BaseReasoningLoader):
    """Loader for the codeforces-cots dataset with reasoning traces."""
    def __init__(
        self,
        name: str = CODEFORCES_COTS_NAME,
        num_pages: int = 1,
        num_rows_per_page: int = 100,
        random_seed: typing.Optional[int] = None,
        max_sequence_length: typing.Optional[int] = 16384,
        chars_per_token: int = 4,
    ):
        super().__init__(
            name=name,
            num_pages=num_pages,
            num_rows_per_page=num_rows_per_page,
            random_seed=random_seed,
            max_sequence_length=max_sequence_length,
            chars_per_token=chars_per_token,
        )
    def _extract_messages(self, sample):
        messages = sample.get("messages", [])
        if len(messages) != 2:
            raise ValueError("Invalid message format")
        return messages[0]["content"], messages[1]["content"]


class Synthetic1SFTLoader(BaseReasoningLoader):
    """Loader for the synthetic-1-sft dataset with structured thinking traces."""
    supported_task_types: List[str] = ["verifiable_math", "code_output_prediction"]
    def __init__(
        self,
        name: str = SYNTHETIC_1_SFT_NAME,
        num_pages: int = 1,
        num_rows_per_page: int = 100,
        random_seed: typing.Optional[int] = None,
        target_size: typing.Optional[int] = None,
        supported_task_types: typing.Optional[List[str]] = None,
        specific_task_type: typing.Optional[str] = None,
        max_sequence_length: typing.Optional[int] = None,
        chars_per_token: int = 4,
    ):
        self.target_size = target_size
        if supported_task_types is not None:
            self.supported_task_types = supported_task_types
        self.specific_task_type = specific_task_type
        self.answers = []
        self.task_types = []
        if (
            specific_task_type
            and self.supported_task_types
            and specific_task_type not in self.supported_task_types
        ):
            raise ValueError(
                f"Task type '{specific_task_type}' is not in supported task types: {self.supported_task_types}"
            )
        super().__init__(
            name=name,
            num_pages=num_pages,
            num_rows_per_page=num_rows_per_page,
            random_seed=random_seed,
            max_sequence_length=max_sequence_length,
            chars_per_token=chars_per_token,
        )
        self._filter_buffer_by_task_type()
        self._parse_samples()
        if self.target_size is not None and len(self.questions) < self.target_size:
            self._fetch_more_data_until_target_size()

    def _filter_buffer_by_task_type(self):
        if not self.supported_task_types and not self.specific_task_type:
            return
        filtered_buffer = []
        original_size = len(self.buffer)
        for sample in self.buffer:
            if isinstance(sample, dict):
                task_type = sample.get("task_type", "unknown")
            else:
                continue
            if self.specific_task_type and task_type == self.specific_task_type:
                filtered_buffer.append(sample)
            elif (
                self.supported_task_types
                and task_type in self.supported_task_types
                and not self.specific_task_type
            ):
                filtered_buffer.append(sample)
        self.buffer = filtered_buffer
        logging.info(
            f"Filtered buffer from {original_size} to {len(self.buffer)} samples based on task type criteria"
        )

    def _fetch_more_data_until_target_size(self):
        original_num_pages = self.num_pages
        additional_pages = 1
        max_attempts = 10
        attempts = 0
        while len(self.questions) < self.target_size and attempts < max_attempts:
            attempts += 1
            logging.info(
                f"Fetching {additional_pages} more page(s) to reach target size {self.target_size}. Current size: {len(self.questions)}"
            )
            self._fetch_data_to_buffer(additional_pages)
            self._filter_buffer_by_task_type()
            self._parse_samples()
            self.num_pages = original_num_pages + additional_pages
            additional_pages *= 2
            if additional_pages > 16:
                additional_pages = 16
        if len(self.questions) < self.target_size:
            logging.warning(
                f"Could not reach target size {self.target_size} after {attempts} attempts. Stopping at {len(self.questions)} samples."
            )

    def _extract_messages(self, sample):
        if isinstance(sample, dict):
            task_type = sample.get("task_type", "unknown")
            messages = sample.get("messages", "")
        else:
            messages = sample
            task_type = "unknown"
        if task_type == "verifiable_math":
            question, trace, answer = self._parse_verifiable_math(messages)
        elif task_type == "code_output_prediction":
            question, trace, answer = self._parse_code_output_prediction(messages)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
        self.answers.append(answer)
        self.task_types.append(task_type)
        return question, trace

    def _parse_verifiable_math(self, messages):
        question = messages[0]["content"]
        assistant_message = messages[1]["content"]
        boxed_matches = re.findall(r"\\boxed\{(.*?)\}", assistant_message)
        answer = boxed_matches[-1] if boxed_matches else ""
        if boxed_matches and answer:
            last_boxed_index = assistant_message.rfind(f"\\boxed{{{answer}}}")
            if last_boxed_index > -1:
                trace = assistant_message[:last_boxed_index].strip()
            else:
                trace = assistant_message
        else:
            trace = assistant_message
        return question, trace, answer

    def _parse_code_output_prediction(self, messages):
        question = messages[0]["content"]
        assistant_message = messages[1]["content"]
        output_matches = re.findall(r'{"output": "(.*?)"}', assistant_message)
        answer = output_matches[-1] if output_matches else ""
        if output_matches and answer:
            last_output = f'{{"output": "{answer}"}}'
            trace = assistant_message.replace(last_output, "", 1)
        else:
            trace = assistant_message
        return question, trace, answer

    @property
    def samples(self):
        return [
            {"question": q, "trace": t, "answer": a, "task_type": tt}
            for q, t, a, tt in zip(
                self.questions, self.traces, self.answers, self.task_types
            )
        ]
