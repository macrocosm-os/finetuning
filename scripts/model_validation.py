"""This script evaluates a local model using the same process as a Validator.

It can be used to estimate the performance of a model before submitting it."""

import argparse
import datetime as dt
import gc
import random
import sys
from typing import List

import bittensor as bt
import nltk
from taoverse.metagraph import utils as metagraph_utils
from taoverse.model.competition import utils as competition_utils
from taoverse.model.eval.task import EvalTask
from taoverse.model.model_updater import ModelUpdater
from taoverse.utilities.enum_action import IntEnumAction
import taoverse.utilities.logging as logging
import torch
from transformers import AutoTokenizer

import constants
import finetune as ft
from competitions.data import CompetitionId
from finetune.datasets.factory import DatasetLoaderFactory
from finetune.datasets.ids import DatasetId
from finetune.datasets.subnet.prompting_subset_loader import PromptingSubsetLoader
from finetune.eval.sample import EvalSample
from finetune.eval.method import EvalMethodId


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, help="Local path to your model", required=True
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device name.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="Random seed to use while loading data. If 0 then randomize.",
    )
    parser.add_argument(
        "--competition_id",
        type=CompetitionId,
        default=CompetitionId.INSTRUCT_8B.value,
        action=IntEnumAction,
        help="competition to mine for (use --list-competitions to get all competitions)",
    )
    parser.add_argument(
        "--list_competitions", action="store_true", help="Print out all competitions"
    )
    parser.add_argument(
        "--comp_block",
        type=int,
        default=9999999999,
        help="Block to lookup competition id from.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        help="HuggingFace tokenizer name to download and use instead of model's default tokenizer",
    )
    
    
    
    
    args = parser.parse_args()
    if args.list_competitions:
        logging.info(
            competition_utils.get_competition_schedule_for_block(
                args.comp_block, constants.COMPETITION_SCHEDULE_BY_BLOCK
            )
        )
        return

    competition = competition_utils.get_competition_for_block(
        args.competition_id,
        args.comp_block,
        constants.COMPETITION_SCHEDULE_BY_BLOCK,
    )

    if not competition:
        logging.info(f"Competition {args.competition_id} not found.")
        return

    kwargs = competition.constraints.kwargs.copy()
    kwargs["use_cache"] = True

    logging.info(f"Loading tokenizer and model from {args.model_path}")
    model = ft.mining.load_local_model(args.model_path, args.competition_id, kwargs)

    # Override tokenizer if specified
    if args.tokenizer_name:
        logging.info(f"Loading custom tokenizer: {args.tokenizer_name}")
        model.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        if model.tokenizer.pad_token is None:
            model.tokenizer.pad_token = model.tokenizer.eos_token
            logging.info("Set pad_token to eos_token for custom tokenizer")

    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
        logging.info("Set pad_token to eos_token for tokenizer")

    # TODO: Uncomment this and test
    # Commented out to test diff models through pipeline.
    # if competition.constraints.tokenizer:
    #     model.tokenizer = ft.model.load_tokenizer(competition.constraints)
    #     if model.tokenizer.pad_token is None:
    #         model.tokenizer.pad_token = model.tokenizer.eos_token
    #         logging.info("Set pad_token to eos_token for tokenizer")

    # if not ModelUpdater.verify_model_satisfies_parameters(
    #     model, competition.constraints
    # ):
    #     logging.info("Model does not satisfy competition parameters!!!")
    #     return

    seed = args.random_seed if args.random_seed else random.randint(0, sys.maxsize)

    logging.info("Loading evaluation tasks")
    eval_tasks: List[EvalTask] = []
    samples: List[List[EvalSample]] = []

    # Load data based on the competition.
    metagraph = bt.metagraph(constants.PROMPTING_SUBNET_UID)
    vali_uids = metagraph_utils.get_high_stake_validators(
        metagraph, constants.SAMPLE_VALI_MIN_STAKE
    )
    vali_hotkeys = set([metagraph.hotkeys[uid] for uid in vali_uids])

    for eval_task in competition.eval_tasks:
        if eval_task.dataset_id == DatasetId.SYNTHETIC_MMLU:
            data_loader = PromptingSubsetLoader(
                random_seed=seed,
                oldest_sample_timestamp=dt.datetime.now(dt.timezone.utc)
                - dt.timedelta(hours=6),
                validator_hotkeys=vali_hotkeys,
            )
        else:
            data_loader = DatasetLoaderFactory.get_loader(
                dataset_id=eval_task.dataset_id,
                dataset_kwargs=eval_task.dataset_kwargs,
                seed=seed,
                validator_hotkeys=vali_hotkeys,
                competition_id=competition.id,
            )

        if data_loader:
            eval_tasks.append(eval_task)
            logging.info(f"Loaded {len(data_loader)} samples for task {eval_task.name}")
            if eval_task.method_id == EvalMethodId.VERIFIABLE_REASONING:
                samples.append(
                    data_loader.tokenize_for_verifiable_reasoning(
                        model.tokenizer, competition.constraints.sequence_length
                    )
                )
            else:
                samples.append(
                    data_loader.tokenize(
                        model.tokenizer, competition.constraints.sequence_length
                    )
                )

    logging.info(f"Scoring model on tasks {eval_tasks}")
    # Run each computation in a subprocess so that the GPU is reset between each model.
    score, score_details = ft.validation.score_model(
        model,
        eval_tasks,
        samples,
        competition,
        args.device,
    )

    logging.info(f"Computed score: {score}. Details: {score_details}")


if __name__ == "__main__":
    # Clean GPU memory and cache before continuing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        logging.info("Cleaned GPU memory and cache")
    
    # Make sure we can download the needed ntlk modules
    nltk_modules = {
        "words",
        "punkt",
        "punkt_tab",
        "averaged_perceptron_tagger_eng",
    }
    for module in nltk_modules:
        nltk.download(module, raise_on_error=True)

    logging.reinitialize()
    logging.set_verbosity_trace()

    main()
