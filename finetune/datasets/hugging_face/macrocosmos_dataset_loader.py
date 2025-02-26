import datetime as dt
import random
import typing

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

        logging.debug(
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
            try:
                d = dt.datetime.strptime(split_name, "%Y%m%d").date()
                is_needed = oldest_date <= d <= newest_date
                logging.debug(f"Split {split_name} date {d}: oldest={oldest_date} <= d <= newest={newest_date} = {is_needed}")
                return is_needed
            except ValueError as e:
                logging.debug(f"Invalid date format for split {split_name}: {e}")
                return False  # If split name is not a valid date format, ignore it

        # Get splits using the datasets library approach
        all_splits = self._get_splits()
        logging.debug(f"Found {len(all_splits)} total available splits in the dataset")

        needed_splits = sorted([s for s in all_splits if _need_split(s)])
        logging.debug(f"Selected {len(needed_splits)} splits that match the date range: {needed_splits}")

        if not needed_splits:
            raise ValueError(
                f"No splits found for samples between {oldest_sample_timestamp} and {newest_sample_timestamp}."
            )

        all_samples: typing.Set[str] = set()
        split_sample_counts = {}

        # Fetch all relevant samples from the needed splits.
        for split in needed_splits:
            logging.debug(f"Loading samples from split: {split}")
            dataset = load_dataset(
                MacrocosmosDatasetLoader.DATASET_NAME,
                split_name := split,  # Use the split as the config name
                split=split_name,     # And also as the split name
                download_mode="force_redownload",
            )
            
            # Get initial size of all_samples to calculate new samples added from this split
            initial_size = len(all_samples)
            split_samples = 0
            
            for row in dataset:
                challenge = row["challenge"]
                reference = row["reference"]
                id = row["id"]
                all_samples.add((id, (challenge, reference)))
                split_samples += 1
            
            new_samples_added = len(all_samples) - initial_size
            split_sample_counts[split] = {
                "total_samples": split_samples,
                "unique_samples_added": new_samples_added
            }
            
            logging.debug(f"Split {split}: {split_samples} total samples, added {new_samples_added} unique samples to collection")

        # Shuffle and select samples
        if random_seed:
            random.seed(random_seed)
        all_samples = sorted(list(all_samples))
        random.shuffle(all_samples)

        self.buffer = [c_and_r for _, c_and_r in all_samples[:max_samples]]
        self.selected_samples = {id for id, _ in all_samples[:max_samples]}

        # Log summary information about samples collected
        if len(self.buffer) < max_samples:
            logging.info(f"Did not collect requested {max_samples} samples, only got {len(self.buffer)}")
        else:
            logging.info(f"Collected {max_samples} samples out of {len(all_samples)} total available")
        
        # Log detailed split information
        logging.debug(f"Split statistics summary:")
        for split, counts in split_sample_counts.items():
            logging.debug(f"  - {split}: {counts['total_samples']} total samples, {counts['unique_samples_added']} unique samples")
        
        # Store split information for potential retrieval
        self.split_statistics = split_sample_counts

    @retry(tries=5, delay=1, backoff=2)
    def _get_splits(self) -> typing.Set[str]:
        """Returns the splits available in the dataset using datasets library directly."""
        from datetime import datetime, timedelta
        from datasets import get_dataset_config_names
        
        try:
            # Try to directly get config names from the dataset
            configs = get_dataset_config_names(MacrocosmosDatasetLoader.DATASET_NAME)
            valid_splits = set(configs)
            logging.debug(f"Found {len(valid_splits)} valid splits directly from dataset configs")
            
            if valid_splits:
                return valid_splits
        except Exception as e:
            logging.debug(f"Error getting dataset config names directly: {e}")
        
        # Fallback: Generate potential date-based splits for a range of dates
        # Try from 90 days ago to today
        splits = set()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y%m%d")
            splits.add(date_str)
            current_date += timedelta(days=1)
        
        # Filter to only include splits that actually exist in the dataset
        valid_splits = set()
        for split in splits:
            try:
                # Try to load the dataset with this config - this will fail if it doesn't exist
                # We're just checking if it exists, not loading the whole dataset
                from datasets import load_dataset_builder
                builder = load_dataset_builder(MacrocosmosDatasetLoader.DATASET_NAME, split)
                valid_splits.add(split)
                logging.debug(f"Found valid split: {split}")
            except Exception as e:
                logging.debug(f"Split {split} not available: {e}")
                continue
                
        logging.debug(f"Found {len(valid_splits)} valid splits using direct approach")
        return valid_splits

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
        
    def get_split_statistics(self) -> typing.Dict[str, typing.Dict[str, int]]:
        """Returns statistics about the splits used and samples collected from each."""
        return self.split_statistics

    def __iter__(self):
        return self.buffer.__iter__()

    def __len__(self):
        return len(self.buffer)

def main():
    """
    Test function to demonstrate the usage of MacrocosmosDatasetLoader.
    Run this script directly to execute this test function.
    """
    import argparse
    from transformers import AutoTokenizer
    import datetime as dt
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test MacrocosmosDatasetLoader functionality")
    parser.add_argument("--max_samples", type=int, default=5, help="Number of samples to retrieve")
    parser.add_argument("--days_ago", type=int, default=30, help="How many days back to look for samples")
    parser.add_argument("--tokenize", action="store_true", help="Test tokenization functionality")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Model name for tokenizer (if --tokenize is set)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--show_split_stats", default = True, action="store_true", help="Show detailed split statistics")
    args = parser.parse_args()
    
    # Set up logging (without using set_trace which doesn't exist)
    if args.verbose:
        logging.info("Verbose logging enabled")
    
    # Calculate date range
    now = dt.datetime.now(timezone("US/Pacific"))
    oldest_date = now - dt.timedelta(days=args.days_ago)
    
    print(f"Initializing MacrocosmosDatasetLoader with max_samples={args.max_samples}")
    print(f"Looking for samples from {oldest_date} to {now}")
    
    try:
        # Initialize the dataset loader
        loader = MacrocosmosDatasetLoader(
            max_samples=args.max_samples,
            oldest_sample_timestamp=oldest_date,
            newest_sample_timestamp=now,
        )
        
        print(f"Successfully loaded {len(loader)} samples")
        
        # Show split statistics if requested
        if args.show_split_stats:
            print("\nSplit statistics:")
            for split, stats in loader.get_split_statistics().items():
                print(f"  - {split}: {stats['total_samples']} total samples, {stats['unique_samples_added']} unique")
        
        # Print a few samples
        print("\nSample data:")
        for i, (challenge, reference) in enumerate(loader):
            if i >= 3:  # Limit to 3 samples for display
                break
            print(f"\nSample {i+1}:")
            print(f"Challenge: {challenge[:200]}..." if len(challenge) > 200 else f"Challenge: {challenge}")
            print(f"Reference answer: {reference}")
        
        # Test tokenization if requested
        if args.tokenize:
            print("\nTesting tokenization:")
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            tokenized_samples = loader.tokenize(tokenizer, sequence_length=512)
            print(f"Tokenized {len(tokenized_samples)} samples")
            
            # Show info about the first tokenized sample
            if tokenized_samples:
                first_sample = tokenized_samples[0]
                print(f"First tokenized sample:")
                print(f"  Input shape: {first_sample[0].shape}")
                print(f"  Choices: {first_sample[1]}")
                print(f"  Correct answer: {first_sample[2]}")
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Error during test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
