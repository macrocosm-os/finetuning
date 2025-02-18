import datetime as dt
import random
import typing

import requests
import taoverse.utilities.logging as logging
import torch
from datasets import load_dataset
from pytz import timezone
from retry import retry
from transformers import PreTrainedTokenizerBase

from finetune.datasets.loader import DatasetLoader

# Multiple choice answers for the prompting subnet.
PROMPTING_SUBNET_CHOICES = ["A", "B", "C", "D"]
CHALLENGE_PREFIX = "[Example 1]\nWhat is the capital of Texas?\nA. Paris\nB. London\nC. Austin\nD. Houston\nAnswer: C\n\n[Input Question]\n"


class MacrocosmosDatasetLoader(DatasetLoader):
    """Loads MMLU data from Macrocosmos' hugging face dataset, produced by subnet 1."""

    DATASET_NAME = "macrocosm-os/macrobench-bittensor-01"
    INFO_URL = "https://datasets-server.huggingface.co/info?dataset={dataset}&config={date}"

    @staticmethod
    def _create_row_filter(
        validator_hotkeys: typing.List[str],
        oldest_sample_timestamp: typing.Optional[dt.datetime] = None,
        newest_sample_timestamp: typing.Optional[dt.datetime] = None,
    ) -> typing.Callable[typing.Dict[str, typing.Any], bool]:
        def _func(row: typing.Dict[str, typing.Any]) -> bool:
            timestamp = timezone("US/Pacific").localize(row["timestamp"])
            if oldest_sample_timestamp and timestamp < oldest_sample_timestamp:
                return False
            if newest_sample_timestamp and timestamp > newest_sample_timestamp:
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
                timezone("US/Pacific")
            )
        if newest_sample_timestamp:
            newest_sample_timestamp = newest_sample_timestamp.astimezone(
                timezone("US/Pacific")
            )

        logging.trace(
            f"Fetching samples after {oldest_sample_timestamp} and before {newest_sample_timestamp}"
        )

        oldest_date = (
            oldest_sample_timestamp.date() if oldest_sample_timestamp else dt.date.min
        )
        newest_date = (
            newest_sample_timestamp.date() if newest_sample_timestamp else dt.date.max
        )

        def _need_split(split_name: str) -> bool:
            """Returns True if the split falls between the oldest and newest sample timestamps."""
            d = dt.datetime.strptime(split_name, "%Y%m%d").date()
            return oldest_date <= d <= newest_date

        all_splits = self._get_splits(newest_date.strftime("%Y%m%d"))
        print(all_splits)
        needed_splits = sorted([s for s in all_splits if _need_split(s)])

        print(f"Need splits: {needed_splits}")

        if not needed_splits:
            raise ValueError(
                f"No splits found for samples between {oldest_sample_timestamp} and {newest_sample_timestamp}."
            )

        all_samples: typing.Set[str] = set()

        # Fetch all relevant samples from the needed splits.
        for split in needed_splits:
            print(f"Loading split {split}")
            dataset = load_dataset(
                MacrocosmosDatasetLoader.DATASET_NAME,
                split=split,
                streaming=True,
                # Make sure the latest data is fetched.
                # download_mode="force_redownload",
            )
            print(f"Loaded split {split}")

            dataset = dataset.filter(
                MacrocosmosDatasetLoader._create_row_filter(
                    validator_hotkeys, oldest_sample_timestamp, newest_sample_timestamp
                )
            )

            print(f"Filtered split {split}")

            for row in dataset:
                challenge = f"{CHALLENGE_PREFIX}{row['challenge']}"
                reference = row["reference"]
                id = row["id"]

                all_samples.add((id, (challenge, reference)))

            print(f"Added {len(dataset)} samples from split {split}")

        # All samples collected. Now shuffle and filter to the number we want.
        if random_seed:
            random.seed(random_seed)

        all_samples = sorted(list(all_samples))
        random.shuffle(all_samples)

        self.buffer = [c_and_r for _, c_and_r in all_samples[:max_samples]]
        self.selected_samples = {id for id, _ in all_samples[:max_samples]}
        if len(self.buffer) < max_samples:
            logging.debug(f"Did not collect {max_samples}, only got {len(self.buffer)}")
        else:
            logging.trace(f"Collected {max_samples} samples")

    @retry(tries=5, delay=1, backoff=2)
    def _get_splits(self, date: str) -> typing.Set[str]:
        """Returns the splits available in the dataset."""
        response = requests.get(MacrocosmosDatasetLoader.INFO_URL.format(date=date, dataset=MacrocosmosDatasetLoader.DATASET_NAME), timeout=10)
        response.raise_for_status()
        return set(response.json()["dataset_info"]["splits"].keys())

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
