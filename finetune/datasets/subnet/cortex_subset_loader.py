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
from transformers import PreTrainedTokenizerBase
from wandb.apis.public.history import HistoryScan

import constants

UNWANTED_PHRASES = [
    "text-based AI language model",
    "please refrain",
    "it is never okay",
    "It is important to",
    "It's important to",
    "real-world consequences",
    "responsible AI",
    "AI principles",
    "AI assistant",
    "an AI language",
    "as a language model",
    "as an AI language model",
    "As a large language model",
    "As an AI",
    "ethical principles",
    "it is not appropriate",
    "it's not appropriate",
    "I cannot fulfill your request",
    "ethical guidelines",
    "my guidelines",
    "prioritize user safety",
    "cannot provide guidance",
    "cannot provide information",
    "unable to offer assistance",
    "cannot engage in discussions",
    "programming prohibits",
    "follow ethical guidelines",
    "cannot support or promote",
    "against my programming",
    "not able to provide",
    "cannot provide any information",
    "an AI language model you don't have",
    "As an AI language model, I cannot",
    "As an AI language model, I do not",
    "As an AI language model, I am not able",
    "As an AI language model, I don't have personal",
    "I am an AI language model and do not",
    "However, it is important to use any code or information provided responsibly and within legal and ethical boundaries.",
    "As an AI language model, I don't have",
    "As an AI language model, I am only able",
    "AI language model and I do not",
    "As an AI language model, I cannot modify",
    "As an AI language model, I do not",
    "I know as an AI language model you don't have",
    "as an AI language model, you cannot",
    "I'm sorry, but as an AI language model",
    "As an AI language model, I don't have",
    "Unfortunately, I cannot provide",
    "I'm sorry, I cannot",
    "I'm sorry, I cannot generate",
    "AI cannot create or program",
    "I'm afraid I cannot create",
    "I cannot assist",
    "I'm sorry,",
    "I'm an AI",
    "I am an AI",
    "my purpose",
    "entertainment purposes",
    "purely hypothetical",
    "not a human",
    "I am an AI",
    "cannot provide",
    "can't provide",
    "won't provide",
    "not provide",
    "a language model",
    "As a machine",
    "I don't have the ability",
    "I am here to assist",
    "my purpose is to ",
    "my knowledge cutoff",
    "my knowledge cut off",
    "September 2021",
    "I apologize, but",
    "It is not possible",
    "Please note",
    "not acceptable",
    "*This chat conversation is shared from",
    "*This conversation is shared from",
    "<|endoftext|>",
    "Я разработчик",
    "I'm sorry, I cannot",
    "breach of",
    "privacy policy",
    "I am programmed to",
    "As a helpful assistant",
    "I don't have beliefs",
    "I don't have personal",
    "I don't have a personal",
    "I don't have emotions",
    "I don't have the ability to feel",
    "I don't have a physical",
    "I don't have physical",
    "I don't have the ability to remember",
    "I don't have access to real-time",
    "I don't have sensors or a physical body",
    "I don't have sensory input",
    "I don't have a sense",
    "I don't have the capability to perceive",
    "I don't have the capability to feel",
    "I am an artificial intelligence",
    "I don't have access to real-time",
    "I don't have beliefs or disagreements",
    "I do not have a sense of",
    "I do not have beliefs",
    "I do not have personal",
    "I do not have a personal",
    "I do not have emotions",
    "I do not have the ability to feel",
    "I do not have a physical",
    "I do not have physical",
    "I do not have the ability to remember",
    "I do not have access to real-time",
    "I do not have sensors or a physical body",
    "I do not have sensory input",
    "I do not have a sense",
    "I do not have the capability to perceive",
    "I do not have the capability to feel",
    "I am an artificial intelligence",
    "I do not have access to real-time",
    "I do not have beliefs or disagreements",
    "I do not have a sense of",
    "September 2021",
    "as a language model",
    "ethical guidelines",
    "as an AI language model",
    "my guidelines",
    "As an AI",
    "cannot provide guidance",
    "cannot provide information",
    "unable to offer assistance",
    "cannot engage in discussions",
    "programming prohibits",
    "cannot support or promote",
    "activities that could harm",
    "against my programming",
    "activities that could undermine",
    "not within the scope",
    "designed to prioritize safety",
    "not able to provide",
    "maintain user safety",
    "adhere to safety guidelines",
    "dangerous or harmful",
    "cannot provide any information",
    "focus on promoting safety",
    "maintain user safety",
    "focus on promoting safety",
    "it is never okay",
    "September 2021",
    "as a language model",
    "ethical guidelines",
    "as an AI language model",
    "my guidelines",
    "As an AI",
    "prioritize user safety",
    "adhere to ethical guidelines",
    "promote safety",
    "responsible information sharing",
    "jeopardize the safety",
    "safe information",
    "cannot provide guidance",
    "cannot provide information",
    "unable to offer assistance",
    "cannot engage in discussions",
    "programming prohibits",
    "prioritize safety",
    "cannot support or promote",
    "activities that could harm",
    "against my programming",
    "potentially dangerous",
    "not within the scope",
    "not able to provide",
    "cannot provide any information",
    "I don't have beliefs",
    "I don't have personal",
    "gpt",
    "gpT",
    "gPt",
    "Gpt",
    "gPT",
    "GpT",
    "GPt",
    "GPT",
]

# The date of the earliest wandb run to fetch.
EARLIEST_DATE = dt.datetime(2024, 2, 1, tzinfo=dt.timezone.utc)


class CortexSubsetLoader:
    def _get_filters(
        self, use_latest_data, random_seed, validator_hotkeys
    ) -> typing.Dict[str, typing.List[str]]:
        filters_and = [{"config.type": constants.CORTEX_WANDB_TYPE}]
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
        max_samples: int = 300,
        steps: int = 5,
        progress: bool = False,
        retry_limit: int = 10,
        page_size: int = 100,
        cortex_project: str = constants.CORTEX_WANDB_PROJECT,
        max_run_age: typing.Optional[dt.timedelta] = None,
        min_score: typing.Optional[float] = None,
        validator_hotkeys: typing.Optional[typing.Set[str]] = None,
    ):
        """Loads prompt/response data from Subnet 18.

        Args:
            use_latest_data (bool, optional): When true, loads data from actively running runs and gets data from the run's latest step.
            random_seed (typing.Optional[int], optional): Seed to use for all random operations.
            max_samples (int, optional): The number of prompt/response samples to load.
            steps (int, optional): Within a run, how many steps to look for samples.
            progress (bool, optional): Whether to log progress of the data loading.
            retry_limit (int, optional): How many times to retry, given any failure.
            page_size (int, optional): The number of steps to fetch from a run at a time. Recommended to be >= steps.
            cortex_project (_type_, optional): The wandb project used for subnet 18. Defaults to constants.CORTEX_WANDB_PROJECT.
            max_run_age (typing.Optional[dt.timedelta], optional): If set, only considers data from runs that were created within the past `max_run_age`
            min_score (typing.Optional[float], optional): If set, only prompt/responses that were scored (by a subbnet 18 validator) higher than 'min_score' are included in the dataset.
            validator_hotkeys (typing.Optional[typing.Set[str]], optional): If provided, only considers data from one of these validators.
        """
        api = wandb.Api(timeout=100)

        filters = self._get_filters(use_latest_data, random_seed, validator_hotkeys)

        bt.logging.debug(f"Fetching runs using filters {filters}")

        # Get the runs, oldest first.
        runs = api.runs(cortex_project, filters, order="+created_at")

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

                    # Validator hotkeys are used to ensure the authenticity of the run.
                    if validator_hotkeys:
                        hotkey = run.config["hotkey"]
                        # First check that the hotkey is in fact a desired validator hotkey.
                        if hotkey not in validator_hotkeys:
                            bt.logging.debug(
                                f"Hotkey: {hotkey} does not match an expected validator for {run.id}."
                            )
                            continue

                        signature = run.config["signature"]
                        # Then verify the signature using the hotkey.
                        if not bt.Keypair(ss58_address=hotkey).verify(
                            run.id, bytes.fromhex(signature)
                        ):
                            bt.logging.debug(
                                f"Failed Signature: {signature} is not valid for {run.id}."
                            )
                            continue

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

                            # Skip any samples that are not text based.
                            if "modality" not in sample or sample["modality"] != "text":
                                continue

                            for uid in range(constants.CORTEX_MAX_UIDS):
                                try:
                                    # If not found these get caught in the KeyError catch below.
                                    prompt: typing.Optional[str] = sample[
                                        f"prompts.{uid}"
                                    ]
                                    response: typing.Optional[str] = sample[
                                        f"responses.{uid}"
                                    ]
                                    # Skip any uids below min_score if specified.
                                    if min_score:
                                        score: typing.Optional[float] = sample[
                                            f"scores.{uid}"
                                        ]
                                        if (
                                            not isinstance(score, float)
                                            or score < min_score
                                        ):
                                            continue

                                    if isinstance(prompt, str) and isinstance(
                                        response, str
                                    ):
                                        prompt = prompt.strip()
                                        response = response.strip()
                                        if len(prompt) > 0 and len(response) > 0:
                                            if not any(
                                                x in response for x in UNWANTED_PHRASES
                                            ):
                                                self.buffer.append((prompt, response))
                                                step = sample.get("_step", "Unknown")
                                                self.selected_samples.add(
                                                    f"{run.id}_{step}"
                                                )
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
        batches = []
        for prompt, response in self:
            conversation = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            prompt_ids = tokenizer.apply_chat_template(
                [conversation[0]],
                truncation=True,
                max_length=sequence_length,
                add_generation_prompt=True,
            )
            ids = tokenizer.apply_chat_template(
                conversation,
                truncation=True,
                max_length=sequence_length,
            )
            batches.append((torch.stack([torch.tensor(ids)]), len(prompt_ids)))
        return batches

    def get_sample(self) -> typing.Tuple[str, str]:
        return self.buffer[random.randint(0, len(self.buffer))]

    def get_selected_sample_ids(self) -> typing.Set[str]:
        """Returns the set of run_id steps that data was selected from."""
        return self.selected_samples

    def __iter__(self):
        return self.buffer.__iter__()
