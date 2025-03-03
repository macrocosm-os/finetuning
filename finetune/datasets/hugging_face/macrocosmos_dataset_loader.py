import datetime as dt
import random
import typing
import numpy as np

import taoverse.utilities.logging as logging
import torch
from datasets import get_dataset_config_names, load_dataset
from pytz import timezone
from retry import retry
from transformers import PreTrainedTokenizerBase

from constants import NUM_CONFIGS_TO_SAMPLE
from finetune.datasets.loader import DatasetLoader

# Multiple choice answers for the prompting subnet.
PROMPTING_SUBNET_CHOICES = ["A", "B", "C", "D"]
CHALLENGE_PREFIX = "[Example 1]\nWhat is the capital of Texas?\nA. Paris\nB. London\nC. Austin\nD. Houston\nAnswer: C\n\n[Input Question]\n"


class MacrocosmosDatasetLoader(DatasetLoader):
    """Loads MMLU data from Macrocosmos' hugging face dataset, produced by subnet 1."""

    DATASET_NAME = "macrocosm-os/macrobench-bittensor-01"

    @staticmethod
    def _create_row_filter(
        validator_hotkeys: typing.List[str],
        oldest_sample_timestamp: typing.Optional[dt.datetime] = None,
        newest_sample_timestamp: typing.Optional[dt.datetime] = None,
    ) -> typing.Callable[[typing.Dict[str, typing.Any]], bool]:
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

    @retry(tries=5, delay=1, backoff=2)
    def _get_all_configs(self) -> typing.List[str]:
        """Returns all available configs in the dataset."""
        try:
            configs = get_dataset_config_names(MacrocosmosDatasetLoader.DATASET_NAME)
            logging.debug(f"Found {len(configs)} configs in the dataset")
            return configs
        except Exception as e:
            logging.error(f"Error getting dataset configs: {e}")
            return []

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
        # Set the random seed if provided
        if random_seed:
            random.seed(random_seed)

        # Get all available configs from the dataset
        all_configs = self._get_all_configs()

        if not all_configs:
            raise ValueError("No configs found in the dataset.")

        logging.info(f"Found {len(all_configs)} total configs in the dataset")

        # Sample NUM_CONFIGS_TO_SAMPLE configs randomly
        sampled_configs = []
        if len(all_configs) <= NUM_CONFIGS_TO_SAMPLE:
            sampled_configs = all_configs
        else:
            sampled_configs = random.sample(all_configs, NUM_CONFIGS_TO_SAMPLE)

        logging.info(f"Sampled {len(sampled_configs)} configs: {sampled_configs}")

        all_samples = []
        config_sample_counts = {}

        # Load samples from each sampled config
        for config in sampled_configs:
            logging.debug(f"Loading samples from config: {config}")
            try:
                dataset = load_dataset(
                    path=MacrocosmosDatasetLoader.DATASET_NAME,
                    name=config,
                    split=config,
                    download_mode="force_redownload",
                )

                config_samples = []

                for row in dataset:
                    challenge = row["challenge"]
                    reference = row["reference"]
                    id = row["id"]

                    # Skip invalid reference answers
                    if reference not in PROMPTING_SUBNET_CHOICES:
                        logging.warning(
                            f"Found invalid reference answer in dataset: {row}"
                        )
                        continue

                    # Skip samples not from validator hotkeys if specified
                    if validator_hotkeys and row["hotkey"] not in validator_hotkeys:
                        continue

                    config_samples.append((id, (challenge, reference)))

                all_samples.extend(config_samples)
                config_sample_counts[config] = len(config_samples)
                logging.debug(
                    f"Added {len(config_samples)} samples from config {config}"
                )

            except Exception as e:
                logging.error(f"Error loading config {config}: {e}")
                continue

        # Shuffle and select samples
        random.shuffle(all_samples)

        # Take up to max_samples
        self.buffer = [c_and_r for _, c_and_r in all_samples[:max_samples]]
        self.selected_samples = {id for id, _ in all_samples[:max_samples]}

        # Log summary information
        if len(self.buffer) < max_samples:
            logging.info(
                f"Did not collect requested {max_samples} samples, only got {len(self.buffer)}"
            )
        else:
            logging.info(
                f"Collected {max_samples} samples out of {len(all_samples)} total available"
            )

        # Log detailed config information
        logging.debug(f"Config statistics summary:")
        for config, count in config_sample_counts.items():
            logging.debug(f"  - {config}: {count} samples")

        # Store config information for potential retrieval
        self.config_statistics = config_sample_counts

    @retry(tries=5, delay=1, backoff=2)
    def _get_splits(self) -> typing.Set[str]:
        """Returns the splits available in the dataset using datasets library directly."""
        return set(self._get_all_configs())

    def tokenize(
        self, tokenizer: PreTrainedTokenizerBase, sequence_length: int
    ) -> typing.List[typing.Tuple[np.ndarray, typing.List[str], str]]:
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
                    np.array([ids]),
                    PROMPTING_SUBNET_CHOICES,
                    reference,
                )
            )

        return batches

    def get_sample(self) -> typing.Tuple[str, str]:
        if not self.buffer:
            raise IndexError("MacrocosmosDatasetLoader is empty.")
        return random.choice(self.buffer)

    def get_selected_sample_ids(self) -> typing.Set[str]:
        """Returns the set of row ids that data was selected from."""
        return self.selected_samples

    def get_config_statistics(self) -> typing.Dict[str, int]:
        """Returns statistics about the splits used and samples collected from each."""
        return self.config_statistics

    def __iter__(self):
        return iter(self.buffer)

    def __len__(self):
        return len(self.buffer)


def main():
    """Allows direct invocation of the dataset loader for testing."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--show-samples", action="store_true")
    parser.add_argument("--show-split-stats", action="store_true")
    args = parser.parse_args()

    # Create the dataset loader with the arguments.
    loader = MacrocosmosDatasetLoader(
        random_seed=args.random_seed,
        max_samples=args.max_samples,
    )

    # Print summary information.
    print(f"Loaded {len(loader)} samples.")

    # Print split statistics if requested.
    if args.show_split_stats:
        print("\nConfig statistics:")
        for config, count in loader.get_config_statistics().items():
            print(f"  - {config}: {count} samples")

    # Print a few samples
    if args.show_samples:
        print("\nSample data:")
        samples = list(loader)
        for i, (challenge, reference) in enumerate(samples[:5]):
            print(f"\nSample {i}:")
            print(f"Challenge: {challenge}")
            print(f"Reference: {reference}")


if __name__ == "__main__":
    main()
