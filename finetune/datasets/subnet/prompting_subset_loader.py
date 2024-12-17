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
import datetime as dt
import random
import time
import traceback
import typing

import bittensor as bt
import taoverse.utilities.logging as logging
import torch
import wandb
from transformers import PreTrainedTokenizerBase

import constants
from finetune.datasets.loader import DatasetLoader
from finetune.datasets.subnet.history_scan import SampledHistoryScan

# Multiple choice answers for the prompting subnet.
PROMPTING_SUBNET_CHOICES = ["A", "B", "C", "D"]

# The date of the earliest wandb run to fetch.
EARLIEST_DATE = dt.datetime(2024, 8, 29, tzinfo=dt.timezone.utc)


class PromptingSubsetLoader(DatasetLoader):
    @staticmethod
    def _get_filters(
        validator_hotkeys: typing.List[str],
        oldest_sample_timestamp: typing.Optional[dt.datetime] = None,
        newest_sample_timestamp: typing.Optional[dt.datetime] = None,
    ) -> typing.Dict[str, typing.List[str]]:
        filters_and = []
        filters_or = []

        filters_and.append(
            {"createdAt": {"$gte": EARLIEST_DATE.strftime("%Y-%m-%d %H:%M:%S")}}
        )
        if newest_sample_timestamp:
            filters_and.append(
                {
                    "createdAt": {
                        "$lt": newest_sample_timestamp.strftime("%Y-%m-%d %H:%M:%S")
                    }
                }
            )
        if oldest_sample_timestamp:
            filters_and.append(
                {
                    "updatedAt": {
                        "$gte": oldest_sample_timestamp.strftime("%Y-%m-%d %H:%M:%S")
                    }
                }
            )
        if validator_hotkeys:
            # 'IN' is not supported in the query language so we add a series of 'OR'.
            for hotkey in validator_hotkeys:
                filters_or.append({"config.HOTKEY_SS58": hotkey})

        # Compose the complete dictionary of filters for the wandb call.
        filters = {"$and": filters_and}
        if len(filters_or) > 0:
            filters["$or"] = filters_or

        return filters

    def __init__(
        self,
        random_seed: typing.Optional[int] = None,
        max_samples: int = 100,
        prompting_project: str = constants.PROMPTING_WANDB_PROJECT,
        oldest_sample_timestamp: typing.Optional[dt.datetime] = None,
        newest_sample_timestamp: typing.Optional[dt.datetime] = None,
        validator_hotkeys: typing.Optional[typing.Set[str]] = None,
    ):
        """Loads prompt/response data from Subnet 1.

        Please note: this loader assumes that it's going to fetch recent data (past few hours) and will likely perform poorly if you try to fetch data from a long time ago.

        Args:
            max_samples (int, optional): The number of prompt/response samples to load.
            steps (int, optional): Within a run, how many steps to look for samples.
            prompting_project (_type_, optional): The wandb project used for subnet 1. Defaults to constants.PROMPTING_WANDB_PROJECT.
            oldest_sample_timestamp (typing.Optional[dt.datetime], optional): If set, only considers data that was created after this timestamp. Must be in UTC.
            newest_sample_timestamp (typing.Optional[dt.datetime], optional): If set, only considers data that was before this timestamp. Must be in UTC.
            validator_hotkeys (typing.Optional[typing.Set[str]], optional): If provided, only considers data from one of these validators.
        """
        api = wandb.Api(timeout=100)

        if oldest_sample_timestamp:
            oldest_sample_timestamp = oldest_sample_timestamp.astimezone(
                dt.timezone.utc
            )
        if newest_sample_timestamp:
            newest_sample_timestamp = newest_sample_timestamp.astimezone(
                dt.timezone.utc
            )

        filters = PromptingSubsetLoader._get_filters(
            validator_hotkeys, oldest_sample_timestamp, newest_sample_timestamp
        )

        logging.trace(f"Fetching runs using filters {filters}")

        # Get the runs, oldest first.
        runs = list(api.runs(prompting_project, filters, order="+created_at"))
        logging.trace(f"Found {len(runs)} runs")

        all_samples: typing.Set[str] = set()
        self.buffer: typing.List[typing.Tuple[str, str]] = []
        self.selected_samples: typing.Set[str] = set()

        def _collect_samples(run: wandb.apis.public.Run) -> bool:
            """Collects samples from the provided run into all_samples.

            Args:
                run (wandb.apis.public.Run): The run to collect samples from.

            Returns:
                bool: True only if all_samples now contains the desired number of samples.
            """
            # Validator hotkeys are used to ensure the authenticity of the run.
            if validator_hotkeys:
                hotkey = run.config.get("HOTKEY_SS58", None)
                # First check that the hotkey is in fact a desired validator hotkey.
                if hotkey not in validator_hotkeys:
                    logging.trace(
                        f"Hotkey: {hotkey} does not match an expected validator for {run.id}."
                    )
                    return False

                signature = run.config.get("SIGNATURE", None)
                # Then verify the signature using the hotkey.
                if not signature or not bt.Keypair(ss58_address=hotkey).verify(
                    run.id, bytes.fromhex(signature)
                ):
                    logging.trace(
                        f"Failed Signature: {signature} is not valid for {run.id}."
                    )
                    return False

            max_step = run.lastHistoryStep + 1
            # Dynamically compute how far to look back based on oldest_sample_timestamp.
            if oldest_sample_timestamp:
                delta = dt.datetime.now(dt.timezone.utc) - oldest_sample_timestamp
                # On average, 20 seconds per step, then include a 50 step buffer.
                steps = int(delta.total_seconds() // 20) + 50
            else:
                steps = 1000
            min_step = max(0, max_step - steps)

            samples = SampledHistoryScan(
                run.client,
                run,
                [
                    "_timestamp",
                    "_step",
                    "task",
                    "challenge",
                    "reference",
                ],
                min_step,
                max_step,
                page_size=steps,
            )
            for sample in samples:
                # Skip any samples that occurred before the oldest allowed sample.
                if (
                    oldest_sample_timestamp
                    and dt.datetime.fromtimestamp(sample["_timestamp"]).astimezone(
                        dt.timezone.utc
                    )
                    < oldest_sample_timestamp
                ):
                    continue
                # Skip any samples that occurred after the newest allowed sample.
                if (
                    newest_sample_timestamp
                    and dt.datetime.fromtimestamp(sample["_timestamp"]).astimezone(
                        dt.timezone.utc
                    )
                    > newest_sample_timestamp
                ):
                    # Since samples are processed in time order, we can break here since the remaining samples will also be too new.
                    break

                # Only check samples that are multiple choice based.
                if sample.get("task", None) != "multi_choice":
                    continue

                challenge = sample.get("challenge", None)
                reference = sample.get("reference", None)

                if (
                    isinstance(challenge, str)
                    and isinstance(reference, str)
                    and reference in PROMPTING_SUBNET_CHOICES
                ):
                    step = sample.get("_step", "Unknown")
                    run_step = f"{run.id}_{step}"
                    all_samples.add((challenge, reference))
                    self.selected_samples.add(run_step)
                    if len(all_samples) >= max_samples:
                        return True
            return False

        if random_seed is not None:
            random.seed(random_seed)
        random.shuffle(list(runs))

        done = False
        for run in runs:
            attempt = 1
            max_attempts = 3
            # Try a max of 3 times per run.
            while attempt <= max_attempts:
                try:
                    done = _collect_samples(run)
                    break
                except Exception:
                    attempt += 1
                    logging.trace(
                        f"Failed to fetch data. {traceback.format_exc()}, retrying. Attempt {attempt}/{max_attempts}"
                    )
                    if attempt < max_attempts:
                        time.sleep(5)

            if done:
                break

        self.buffer = list(all_samples)
        if len(self.buffer) < max_samples:
            logging.debug(f"Did not collect {max_samples}, only got {len(self.buffer)}")
        else:
            logging.trace(f"Collected {max_samples} samples")

    def tokenize(
        self, tokenizer: PreTrainedTokenizerBase, sequence_length: int
    ) -> typing.List[typing.Tuple[torch.Tensor, typing.List[str], str]]:
        # Each batch is a tokenized question + the chocies + the correct choice.
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
        """Returns the set of run_id steps that data was selected from."""
        return self.selected_samples

    def __iter__(self):
        return self.buffer.__iter__()

    def __len__(self):
        return len(self.buffer)
