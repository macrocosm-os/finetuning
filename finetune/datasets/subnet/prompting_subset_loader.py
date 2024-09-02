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
import torch
import wandb
from taoverse.utilities import utils
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase, BatchEncoding
from wandb.apis.public.history import HistoryScan

import constants

# Multiple choice answers for the prompting subnet.
PROMPTING_SUBNET_CHOICES = ["A", "B", "C", "D"]

# The date of the earliest wandb run to fetch.
EARLIEST_DATE = dt.datetime(2024, 8, 29, tzinfo=dt.timezone.utc)


class PromptingSubsetLoader:
    def _get_filters(
        self, use_latest_data, random_seed, validator_hotkeys
    ) -> typing.Dict[str, typing.List[str]]:
        filters_and = []
        filters_or = []

        if use_latest_data:
            filters_and.append({"state": "running"})
        else:
            # If we're not fetching the latest data, then pick a random timepoint and iterate through runs
            # from that timepoint. We do this instead of randomly picking runs across time because wandb's
            # library processes runs serially. i.e., if you ask for run[N] it has to first fetch all N-1 runs.
            # createdAt is is in UTC, so make sure we use UTC tz-aware datetimes.
            random_date = utils.random_date(
                EARLIEST_DATE,
                dt.datetime.now(tz=dt.timezone.utc) - dt.timedelta(days=1),
                random_seed,
            )
            filters_and.append(
                {"createdAt": {"$gte": random_date.strftime("%Y-%m-%d %H:%M:%S")}}
            )
        if validator_hotkeys:
            # 'IN' is not supported in the query language so we add a series of 'OR'.
            for hotkey in validator_hotkeys:
                filters_or.append({"config.hotkey": hotkey})

        # Compose the complete dictionary of filters for the wandb call.
        filters = {"$and": filters_and}
        if len(filters_or) > 0:
            filters["$or"] = filters_or

        return filters

    def __init__(
        self,
        use_latest_data: bool = False,
        random_seed: typing.Optional[int] = None,
        max_samples: int = 100,
        steps: int = 300,
        progress: bool = False,
        retry_limit: int = 10,
        page_size: int = 300,
        prompting_project: str = constants.PROMPTING_WANDB_PROJECT,
        max_run_age: typing.Optional[dt.timedelta] = None,
        validator_hotkeys: typing.Optional[typing.Set[str]] = None,
    ):
        """Loads prompt/response data from Subnet 1.

        Args:
            use_latest_data (bool, optional): When true, loads data from actively running runs and gets data from the run's latest step.
            random_seed (typing.Optional[int], optional): Seed to use for all random operations.
            max_samples (int, optional): The number of prompt/response samples to load.
            steps (int, optional): Within a run, how many steps to look for samples.
            progress (bool, optional): Whether to log progress of the data loading.
            retry_limit (int, optional): How many times to retry, given any failure.
            page_size (int, optional): The number of steps to fetch from a run at a time. Recommended to be >= steps.
            prompting_project (_type_, optional): The wandb project used for subnet 1. Defaults to constants.PROMPTING_WANDB_PROJECT.
            max_run_age (typing.Optional[dt.timedelta], optional): If set, only considers data from runs that were created within the past `max_run_age`
            validator_hotkeys (typing.Optional[typing.Set[str]], optional): If provided, only considers data from one of these validators.
        """
        api = wandb.Api(timeout=100)

        filters = self._get_filters(use_latest_data, random_seed, validator_hotkeys)

        bt.logging.debug(f"Fetching runs using filters {filters}")

        # Get the runs, oldest first.
        runs = api.runs(prompting_project, filters, order="+created_at")

        if random_seed is not None:
            random.seed(random_seed)

        retry_delay = 5  # Seconds to wait between retries
        attempt = 0

        while attempt < retry_limit:
            try:
                run_order = list(range(len(runs)))

                # If we're only using the latest data, there are a small enough set of runs
                # that it's okay for us to randomize the order. For cases where we're not using
                # the latest data, we've already randomly picked data by choosing a random
                # timestamp to get runs from.
                if use_latest_data:
                    random.shuffle(run_order)

                self.buffer: typing.List[typing.Tuple[str, str]] = []
                self.selected_samples: typing.Set[str] = set()

                for run_index in tqdm(
                    run_order, desc="Run", leave=False, disable=not progress
                ):
                    run = runs[run_index]
                    # TODO: Re-enable the hotkey check once subnet 1 adds the signature.
                    # # Validator hotkeys are used to ensure the authenticity of the run.
                    # if validator_hotkeys:
                    #     hotkey = run.config["hotkey"]
                    #     # First check that the hotkey is in fact a desired validator hotkey.
                    #     if hotkey not in validator_hotkeys:
                    #         bt.logging.debug(
                    #             f"Hotkey: {hotkey} does not match an expected validator for {run.id}."
                    #         )
                    #         continue

                    #     signature = run.config["signature"]
                    #     # Then verify the signature using the hotkey.
                    #     if not bt.Keypair(ss58_address=hotkey).verify(
                    #         run.id, bytes.fromhex(signature)
                    #     ):
                    #         bt.logging.debug(
                    #             f"Failed Signature: {signature} is not valid for {run.id}."
                    #         )
                    #         continue

                    if use_latest_data:
                        last_step: int = run.lastHistoryStep
                    else:
                        last_step = random.randint(
                            min(steps, run.lastHistoryStep), run.lastHistoryStep
                        )
                    max_step = last_step + 1
                    min_step = max(0, max_step - steps)

                    history_scan = HistoryScan(
                        run.client, run, min_step, max_step, page_size=page_size
                    )
                    while True:
                        try:
                            sample = next(history_scan)

                            # Skip any samples older than max_run_age.
                            if (
                                max_run_age
                                and dt.datetime.now()
                                - dt.datetime.fromtimestamp(sample["_timestamp"])
                                > max_run_age
                            ):
                                continue

                            # Only check samples that are multiple choice based.
                            if sample.get("task", "none") == "multi_choice":
                                try:
                                    # TODO: Consider only using questions that some threshold of miners got correct.
                                    # TODO: consider adding additional instructions to the challenge.
                                    # If not found these get caught in the KeyError catch below.
                                    challenge = sample["challenge"]
                                    reference = sample["reference"]

                                    if (
                                        isinstance(challenge, str)
                                        and isinstance(reference, str)
                                        and reference in PROMPTING_SUBNET_CHOICES
                                    ):
                                        self.buffer.append((challenge, reference))
                                        step = sample.get("_step", "Unknown")
                                        self.selected_samples.add(f"{run.id}_{step}")
                                        if len(self.buffer) == max_samples:
                                            return

                                except KeyError:
                                    pass
                        except StopIteration:
                            break

                bt.logging.warning(
                    f"Did not collect {max_samples}, only got {len(self.buffer)}"
                )
                return
            except Exception:
                attempt += 1
                print(
                    f"Failed to fetch data. {traceback.format_exc()}, retrying. Attempt {attempt}/{retry_limit}"
                )
                if attempt < retry_limit:
                    time.sleep(retry_delay)  # Wait before the next retry
                else:
                    bt.logging.error(
                        "Maximum retry limit reached. Unable to fetch data."
                    )
                    raise

    def tokenize(
        self, tokenizer: PreTrainedTokenizerBase, sequence_length: int
    ) -> typing.List[typing.Tuple[torch.Tensor, int]]:
        # Each batch is a tokenized question + the chocies + the correct choice.
        batches = []
        # If truncation is necessary, truncate from the left to avoid cutting off the answer part.
        tokenizer.truncation_side = "left"

        for challenge, reference in self:
            # TODO remove extra logging.
            print(f"Challenge: {challenge} added to batches.")
            conversation = [
                {"role": "user", "content": challenge},
            ]
            ids = tokenizer.apply_chat_template(
                conversation,
                truncation=True,
                max_length=sequence_length,
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
        return self.buffer[random.randint(0, len(self.buffer))]

    def get_selected_sample_ids(self) -> typing.Set[str]:
        """Returns the set of run_id steps that data was selected from."""
        return self.selected_samples

    def __iter__(self):
        return self.buffer.__iter__()
