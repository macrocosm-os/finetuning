import json
import random
import re
import time
import typing
from typing import List
import inspect

import numpy as np
import requests
import taoverse.utilities.logging as logging
import torch
from transformers import PreTrainedTokenizerBase

from finetune.datasets.loader import DatasetLoader

FINEWEB_EDU_SCORE_2_NAME = "HuggingFaceFW/fineweb-edu-score-2"
FALCON_NAME = "tiiuae/falcon-refinedweb"
SYNTHETIC_1_SFT_NAME = "PrimeIntellect/SYNTHETIC-1-SFT-Data"


class HuggingFaceLoader(DatasetLoader):
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

        # Choose a consistent initial offset for the random pages so we do not overlap on each page get.
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

                response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code

                # Add the page since the request was successful
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
                    else:
                        raise NotImplementedError(
                            f"Unable to parse rows from hugging face dataset: {self.name}"
                        )

                response.close()

            except requests.exceptions.RequestException:
                response.close()
                attempts += 1
                logging.warning(
                    f"Failed to fetch data, retrying with a newly sampled page. Attempt {attempts}/{self.retry_limit * num_pages}"
                )
                if attempts < num_pages * self.retry_limit:
                    pass

                else:
                    logging.error("Maximum retry limit reached. Unable to fetch data.")
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
        eval_method: typing.Optional[str] = "text_loss"
    ) -> typing.Union[typing.List[np.ndarray], typing.Dict[str, typing.List], typing.List[typing.Tuple[np.array, np.array]]]:
        """Tokenize the dataset based on what's needed.
        
        Args:
            tokenizer: The tokenizer to use
            sequence_length: Maximum sequence length for tokenization
            eval_method: The evaluation method to use
        
        Returns:
            Tokenized data in the format appropriate for the eval_method
        """
        if eval_method.lower() == "reference_loss":
            return self.tokenize_for_reference_loss(tokenizer, sequence_length)
        
        # Default to standard tokenization for TEXT_LOSS
        result = []
        for sample in self.samples:
            prompt = sample["question"]
            trace = sample["trace"]
            answer = sample["answer"]
            
            # For text loss, we tokenize the full sequence: question + trace + answer
            full_text = f"{prompt}\n{trace}\n{answer}"
            tokens = tokenizer.encode(
                full_text,
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


class Synthetic1SFTLoader(HuggingFaceLoader):
    """Loader for the synthetic-1-sft dataset with structured thinking traces."""

    supported_task_types: List[str] = ["verifiable_math", "code_output_prediction"]

    def __init__(
        self,
        name: str = SYNTHETIC_1_SFT_NAME,
        num_pages: int = 1,
        num_rows_per_page: int = 100,
        random_seed: typing.Optional[int] = None,
        target_size: typing.Optional[int] = None,
        filter_to_supported_task_types: bool = True,
        specific_task_type: typing.Optional[str] = None,
        max_sequence_length: typing.Optional[int] = None,
        chars_per_token: int = 4,  # Heuristic: 4 characters per token
    ):
        """Initialize the reasoning dataset loader.

        Args:
            name: Name of the dataset from Hugging Face
            num_pages: Number of pages to fetch
            num_rows_per_page: Number of rows per page
            random_seed: Random seed for reproducibility
            target_size: Target number of samples to keep (None for all valid samples)
            filter_to_supported_task_types: Whether to filter out unsupported task types
            specific_task_type: If specified, only load samples of this task type
            max_sequence_length: Maximum sequence length to allow (None for no limit)
            chars_per_token: Characters per token heuristic for length estimation
        """
        self.target_size = target_size
        self.filter_to_supported_task_types = filter_to_supported_task_types
        self.specific_task_type = specific_task_type
        self.max_sequence_length = max_sequence_length
        self.chars_per_token = chars_per_token

        if specific_task_type and specific_task_type not in self.supported_task_types:
            raise ValueError(
                f"Task type '{specific_task_type}' is not in supported task types: {self.supported_task_types}"
            )

        # Load initial data
        super().__init__(
            name=name,
            num_pages=num_pages,
            num_rows_per_page=num_rows_per_page,
            random_seed=random_seed,
        )

        # First, filter the buffer for supported task types to avoid unnecessary parsing
        self._filter_buffer_by_task_type()

        # Parse the filtered buffer samples
        self._parse_samples()

        # If we need more data to reach target_size, fetch more pages
        if self.target_size is not None and len(self.questions) < self.target_size:
            self._fetch_more_data_until_target_size()

    def _filter_buffer_by_task_type(self):
        """Filter the buffer to keep only samples with supported task types before parsing."""
        if not self.filter_to_supported_task_types and not self.specific_task_type:
            # No filtering needed
            return

        filtered_buffer = []
        original_size = len(self.buffer)

        for sample in self.buffer:
            # Extract task type without parsing the entire sample
            if isinstance(sample, dict):
                task_type = sample.get("task_type", "unknown")
            else:
                # Skip legacy string format if filtering
                continue

            # Keep only samples with supported task types
            if self.specific_task_type and task_type == self.specific_task_type:
                filtered_buffer.append(sample)
            elif (
                self.filter_to_supported_task_types
                and task_type in self.supported_task_types
                and not self.specific_task_type
            ):
                filtered_buffer.append(sample)

        # Replace the buffer with the filtered version
        self.buffer = filtered_buffer
        logging.info(
            f"Filtered buffer from {original_size} to {len(self.buffer)} samples based on task type criteria"
        )

    def _fetch_more_data_until_target_size(self):
        """Fetch more data pages until reaching the target size."""
        original_num_pages = self.num_pages
        additional_pages = 1
        max_attempts = 10  # Limit the number of fetch attempts
        attempts = 0

        while len(self.questions) < self.target_size and attempts < max_attempts:
            attempts += 1
            logging.info(
                f"Fetching {additional_pages} more page(s) to reach target size {self.target_size}. Current size: {len(self.questions)}"
            )

            # Fetch additional pages
            self._fetch_data_to_buffer(additional_pages)

            # Filter the newly fetched data
            self._filter_buffer_by_task_type()

            # Update the number of pages for logging purposes
            self.num_pages = original_num_pages + additional_pages

            # Parse the newly added samples
            self._parse_additional_samples()

            # Increase the number of pages to fetch next time if needed
            additional_pages *= 2

            # Safety check: don't fetch too many pages in a single request
            if additional_pages > 16:
                additional_pages = 16

        if len(self.questions) < self.target_size:
            logging.warning(
                f"Could not reach target size {self.target_size} after {attempts} attempts. Stopping at {len(self.questions)} samples."
            )

    def _parse_samples(self):
        """Parse all samples to extract question, reasoning trace, and answer based on task type."""
        self.questions = []
        self.traces = []
        self.answers = []
        self.task_types = []
        self.buffer_parsed_indices = set()  # Keep track of which indices we've already parsed

        # Parse all samples in the buffer
        self._parse_additional_samples()

        # If we have a target size, trim the dataset if needed
        if self.target_size is not None and len(self.questions) > self.target_size:
            # Randomly select target_size indices
            selected_indices = random.sample(
                range(len(self.questions)), self.target_size
            )

            # Keep only the selected samples
            self.questions = [self.questions[i] for i in selected_indices]
            self.traces = [self.traces[i] for i in selected_indices]
            self.answers = [self.answers[i] for i in selected_indices]
            self.task_types = [self.task_types[i] for i in selected_indices]

    def _parse_additional_samples(self):
        """Parse newly added samples that haven't been parsed yet."""
        filtered_count = 0
        for i, sample in enumerate(self.buffer):
            # Skip already parsed samples
            if i in self.buffer_parsed_indices:
                continue

            # Mark this index as parsed
            self.buffer_parsed_indices.add(i)

            # Extract the content and task type from the sample
            if isinstance(sample, dict):
                task_type = sample.get("task_type", "unknown")
                messages = sample.get("messages", "")
            else:
                # Handle legacy string format
                messages = sample
                task_type = "unknown"

            # We already filtered by task type, so we can directly parse
            try:
                # Determine the parsing method based on task type
                if task_type == "verifiable_math":
                    question, trace, answer = self._parse_verifiable_math(messages)
                elif task_type == "code_output_prediction":
                    question, trace, answer = self._parse_code_output_prediction(
                        messages
                    )
                else:
                    # This should not happen after filtering
                    logging.warning(
                        f"Unexpected task type after filtering: {task_type}"
                    )
                    continue
                
                # Check if the sample fits within the sequence length limit
                if not self._fits_sequence_length(question, trace, answer):
                    filtered_count += 1
                    continue

                # Add the parsed components
                self.questions.append(question)
                self.traces.append(trace)
                self.answers.append(answer)
                self.task_types.append(task_type)

                # If we've reached the target size, stop parsing
                if (
                    self.target_size is not None
                    and len(self.questions) >= self.target_size
                ):
                    break

            except Exception as e:
                # Log parsing errors but continue with next sample
                logging.warning(
                    f"Error parsing sample {i} with task type {task_type}: {e}"
                )
                continue
        
        if filtered_count > 0 and self.max_sequence_length is not None:
            logging.info(
                f"Filtered out {filtered_count} samples exceeding the max sequence length of {self.max_sequence_length} tokens"
            )

    def _parse_verifiable_math(self, messages):
        """Parse verifiable math tasks with <think> tags and \boxed{} format."""
        question = messages[0]["content"]
        assistant_message = messages[1]["content"]

        # Extract the answer from the last \boxed{} occurrence
        boxed_matches = re.findall(r"\\boxed\{(.*?)\}", assistant_message)
        answer = boxed_matches[-1] if boxed_matches else ""

        # The trace is everything before the last occurrence of \boxed{answer}
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

        # Extract the answer from the last {"output": "..."} occurrence
        output_matches = re.findall(r'{"output": "(.*?)"}', assistant_message)
        answer = output_matches[-1] if output_matches else ""

        # The trace is everything except the answer
        if output_matches and answer:
            # Replace the last occurrence of {"output": "answer"} with an empty string to get the trace
            last_output = f'{{"output": "{answer}"}}'
            trace = assistant_message.replace(last_output, "", 1)
        else:
            trace = assistant_message

        return question, trace, answer

    def get_supervised_batch(self, n: int = None) -> dict:
        """Get a batch of n samples in supervised learning format.

        Args:
            n: Number of samples to return (None for all)

        Returns:
            dict: {
                'x': list of questions/prompts,
                'y': list of answers,
                'traces': list of reasoning traces,
                'task_types': list of task types
            }
        """
        if n is None:
            return {
                "x": self.questions,
                "y": self.answers,
                "traces": self.traces,
                "task_types": self.task_types,
            }

        # Sample n indices without replacement
        indices = random.sample(range(len(self.questions)), min(n, len(self.questions)))

        return {
            "x": [self.questions[i] for i in indices],
            "y": [self.answers[i] for i in indices],
            "traces": [self.traces[i] for i in indices],
            "task_types": [self.task_types[i] for i in indices],
        }

    def get_sample_with_components(self) -> dict:
        """Get a random sample with its components separated.

        Returns:
            dict: {
                'question': The question/prompt,
                'trace': The reasoning trace,
                'answer': The final answer,
                'task_type': The task type
            }
        """
        idx = random.randint(0, len(self.questions) - 1)
        return {
            "question": self.questions[idx],
            "trace": self.traces[idx],
            "answer": self.answers[idx],
            "task_type": self.task_types[idx],
        }

    def tokenize_supervised(
        self, tokenizer: PreTrainedTokenizerBase, sequence_length: int
    ) -> typing.Dict[str, typing.List[np.ndarray]]:
        """Tokenize samples for supervised learning.

        Args:
            tokenizer: The tokenizer to use
            sequence_length: Maximum sequence length

        Returns:
            dict: {
                'x': tokenized questions,
                'y': tokenized answers,
                'traces': tokenized reasoning traces,
                'task_types': list of task types
            }
        """
        tokenized_questions = []
        tokenized_answers = []
        tokenized_traces = []

        for question, trace, answer in zip(self.questions, self.traces, self.answers):
            # Tokenize question
            q_input_ids = tokenizer(
                question, max_length=sequence_length, truncation=True
            )["input_ids"]
            tokenized_questions.append(
                np.array([q_input_ids + [tokenizer.eos_token_id]])
            )

            # Tokenize answer
            a_input_ids = tokenizer(
                answer, max_length=sequence_length, truncation=True
            )["input_ids"]
            tokenized_answers.append(np.array([a_input_ids + [tokenizer.eos_token_id]]))

            # Tokenize trace
            t_input_ids = tokenizer(trace, max_length=sequence_length, truncation=True)[
                "input_ids"
            ]
            tokenized_traces.append(np.array([t_input_ids + [tokenizer.eos_token_id]]))

        return {
            "x": tokenized_questions,
            "y": tokenized_answers,
            "traces": tokenized_traces,
            "task_types": self.task_types,
        }


    def tokenize_for_reference_loss(
        self, tokenizer: PreTrainedTokenizerBase, sequence_length: int
    ) -> typing.List[typing.Tuple[np.array, np.array]]:
        """Tokenize samples for reference loss evaluation.
        
        Returns a list of (context, reference) tuples where:
        - context is the question formatted with chat template
        - reference is the trace+answer
        """
        result = []
        
        # question, trace, answer, task_type
        for q, t, a, tt in zip(self.questions, self.traces, self.answers, self.task_types):
            # Only include samples from supported task types
            if tt not in self.supported_task_types:
                continue
            
            # Apply chat template to question
            formatted_question = tokenizer.apply_chat_template(
                [{"role": "user", "content": q}],
                add_generation_prompt=False,
                tokenize=False
            )
            
            # Combine trace and answer for the reference
            trace_with_answer = t + a
            
            # Tokenize question (context) and trace+answer (reference)
            context_tokens = np.array(tokenizer.encode(
                formatted_question,
                truncation=True,
            ))
            
            reference_tokens = np.array(tokenizer.encode(
                trace_with_answer, 
                truncation=True,
            ))
            
            result.append((context_tokens, reference_tokens))
        
        return result

    @property
    def samples(self):
        """
        Return samples in the format expected by the parent tokenize method.
        """
        return [
            {"question": q, "trace": t, "answer": a, "task_type": tt}
            for q, t, a, tt in zip(self.questions, self.traces, self.answers, self.task_types)
        ]

    def _estimate_token_length(self, text: str) -> int:
        """Estimate the number of tokens in a text using the characters per token heuristic.
        
        Args:
            text: Text to estimate token length for
            
        Returns:
            Estimated number of tokens
        """
        if not text:
            return 0
        return len(text) // self.chars_per_token

    def _fits_sequence_length(self, question: str, trace: str, answer: str) -> bool:
        """Check if the combined length of question, trace, and answer fits within the sequence length.
        
        Args:
            question: The question text
            trace: The reasoning trace text
            answer: The answer text
            
        Returns:
            True if the combined estimated token length fits within max_sequence_length
        """
        if self.max_sequence_length is None:
            return True
            
        # Estimate token lengths
        question_length = self._estimate_token_length(question)
        trace_length = self._estimate_token_length(trace)
        answer_length = self._estimate_token_length(answer)
        
        # We need to account for the combined length
        total_length = question_length + trace_length + answer_length
        
        # Add a small buffer for special tokens and tokenization differences
        total_length += 10
        
        return total_length <= self.max_sequence_length
