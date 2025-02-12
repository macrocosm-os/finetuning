import datetime as dt
import random
import typing

import taoverse.utilities.logging as logging
import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

from finetune.datasets.loader import DatasetLoader

# Multiple choice answers for the prompting subnet.
PROMPTING_SUBNET_CHOICES = ["A", "B", "C", "D"]
CHALLENGE_PREFIX = "[Example 1]\nWhat is the capital of Texas?\nA. Paris\nB. London\nC. Austin\nD. Houston\nAnswer: C\n\n[Input Question]\n"


class MacrocosmosDatasetLoader(DatasetLoader):
    """Loads MMLU data from Macrocosmos' hugging face dataset, produced by subnet 1."""

    dataset_name = "macrocosm-os/macrobench-bittensor-01"
    split = "train"

    @staticmethod
    def _create_row_filter(
        validator_hotkeys: typing.List[str],
        oldest_sample_timestamp: typing.Optional[dt.datetime] = None,
        newest_sample_timestamp: typing.Optional[dt.datetime] = None,
    ) -> typing.Callable[typing.Dict[str, typing.Any], bool]:
        def _func(row: typing.Dict[str, typing.Any]) -> bool:
            timestamp = row["timestamp"].astimezone(dt.timezone.utc)
            if oldest_sample_timestamp:
                if timestamp < oldest_sample_timestamp:
                    return False
            if newest_sample_timestamp:
                if timestamp > newest_sample_timestamp:
                    return False
            if validator_hotkeys:
                return row["hotkey"] in validator_hotkeys
            if row["reference"] not in PROMPTING_SUBNET_CHOICES:
                logging.warning(f"Found invalid reference answer in dataset: {row}")
                return False

            return True

        return _func

    def __init__(
        self,
        random_seed: typing.Optional[int] = None,
        max_samples: int = 100,
        oldest_sample_timestamp: typing.Optional[dt.datetime] = None,
        newest_sample_timestamp: typing.Optional[dt.datetime] = None,
        validator_hotkeys: typing.Optional[typing.Set[str]] = None,
    ):
        """Loads prompt/response data from Subnet 1.

        Args:
            random_seed (typing.Optional[int], optional): The random seed to use for shuffling the data.
            max_samples (int, optional): The number of prompt/response samples to load.
            oldest_sample_timestamp (typing.Optional[dt.datetime], optional): If set, only considers data that was created after this timestamp. Must be in UTC.
            newest_sample_timestamp (typing.Optional[dt.datetime], optional): If set, only considers data that was before this timestamp. Must be in UTC.
            validator_hotkeys (typing.Optional[typing.Set[str]], optional): If provided, only considers data from one of these validators.
        """
        if oldest_sample_timestamp:
            oldest_sample_timestamp = oldest_sample_timestamp.astimezone(
                dt.timezone.utc
            )
        if newest_sample_timestamp:
            newest_sample_timestamp = newest_sample_timestamp.astimezone(
                dt.timezone.utc
            )

        logging.trace(
            f"Fetching samples after {oldest_sample_timestamp} and before {newest_sample_timestamp}"
        )

        # Get the runs, oldest first.
        dataset = load_dataset(
            MacrocosmosDatasetLoader.dataset_name,
            split=MacrocosmosDatasetLoader.split,
            streaming=True,
        )
        if random_seed is not None:
            dataset = dataset.shuffle(seed=random_seed)

        dataset = dataset.filter(
            MacrocosmosDatasetLoader._create_row_filter(
                validator_hotkeys, oldest_sample_timestamp, newest_sample_timestamp
            )
        )

        all_samples: typing.Set[str] = set()
        self.selected_samples: typing.Set[str] = set()

        for row in dataset:
            challenge = f"{CHALLENGE_PREFIX}{row['challenge']}"
            reference = row["reference"]
            id = row["id"]

            all_samples.add((challenge, reference))
            self.selected_samples.add(id)
            if len(all_samples) >= max_samples:
                break

        self.buffer = list(all_samples)
        if len(self.buffer) < max_samples:
            logging.debug(f"Did not collect {max_samples}, only got {len(self.buffer)}")
        else:
            logging.trace(f"Collected {max_samples} samples")

    def tokenize(
        self, tokenizer: PreTrainedTokenizerBase, sequence_length: int
    ) -> typing.List[typing.Tuple[torch.Tensor, typing.List[str], str]]:
        # Each batch is a tokenized question + the choices + the correct choice.
        batches = []
        # If truncation is necessary, truncate from the left to avoid cutting off the answer part.
        tokenizer.truncation_side = "left"

        for challenge, reference in self:
            conversation = [
                {"role": "user", "content": challenge},
            ]
            ids = tokenizer.apply_chat_template(
                conversation,
                truncation=True,
                max_length=sequence_length,
                add_generation_prompt=True,
            )

            batches.append(
                (
                    torch.stack([torch.tensor(ids)]),
                    PROMPTING_SUBNET_CHOICES,
                    reference,
                )
            )
        return batches

    def get_sample(self) -> typing.Tuple[str, str]:
        return random.choice(self.buffer)

    def get_selected_sample_ids(self) -> typing.Set[str]:
        """Returns the set of row ids that data was selected from."""
        return self.selected_samples

    def __iter__(self):
        return self.buffer.__iter__()

    def __len__(self):
        return len(self.buffer)
