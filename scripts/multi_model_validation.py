"""""This script evaluates two local models sequentially using the same process as a Validator.

It can be used to estimate the performance of models before submitting them and comparing their scores."""

import argparse
import datetime as dt
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

import constants
import finetune as ft
from competitions.data import CompetitionId
from finetune.datasets.factory import DatasetLoaderFactory
from finetune.datasets.ids import DatasetId
from finetune.datasets.subnet.prompting_subset_loader import PromptingSubsetLoader
from finetune.eval.sample import EvalSample

def evaluate_model(model_path, competition, seed, vali_hotkeys, device):
    logging.info(f"Loading tokenizer and model from {model_path}")
    kwargs = competition.constraints.kwargs.copy()
    kwargs["use_cache"] = True
    
    model = ft.mining.load_local_model(model_path, competition.id, kwargs)

    if competition.constraints.tokenizer:
        model.tokenizer = ft.model.load_tokenizer(competition.constraints)

    if not ModelUpdater.verify_model_satisfies_parameters(model, competition.constraints):
        logging.info("Model does not satisfy competition parameters!!!")
        return None, None

    logging.info("Loading evaluation tasks")
    eval_tasks: List[EvalTask] = []
    samples: List[List[EvalSample]] = []

    for eval_task in competition.eval_tasks:
        if eval_task.dataset_id == DatasetId.SYNTHETIC_MMLU:
            data_loader = PromptingSubsetLoader(
                random_seed=seed,
                oldest_sample_timestamp=dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=6),
                validator_hotkeys=vali_hotkeys,
            )
        else:
            data_loader = DatasetLoaderFactory.get_loader(
                dataset_id=eval_task.dataset_id,
                dataset_kwargs=eval_task.dataset_kwargs,
                seed=seed,
                validator_hotkeys=vali_hotkeys,
            )

        if data_loader:
            eval_tasks.append(eval_task)
            logging.info(f"Loaded {len(data_loader)} samples for task {eval_task.name}")
            samples.append(
                data_loader.tokenize(model.tokenizer, competition.constraints.sequence_length)
            )

    logging.info(f"Scoring model {model_path} on tasks {eval_tasks}")
    score, score_details = ft.validation.score_model(
        model,
        eval_tasks,
        samples,
        competition,
        device,
    )

    logging.info(f"Computed score for {model_path}: {score}. Details: {score_details}")
    return score, score_details

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path1", type=str, required=True, help="Local path to the first model")
    parser.add_argument("--model_path2", type=str, required=True, help="Local path to the second model")
    parser.add_argument("--device", type=str, default="cuda", help="Device name.")
    parser.add_argument("--random_seed", type=int, default=0, help="Random seed for data loading.")
    parser.add_argument("--competition_id", type=CompetitionId, default=CompetitionId.DISTILLED_REASONING_3B.value, action=IntEnumAction, help="Competition ID")
    parser.add_argument("--comp_block", type=int, default=9999999999, help="Block to lookup competition ID from.")
    args = parser.parse_args()
    
    competition = competition_utils.get_competition_for_block(
        args.competition_id,
        args.comp_block,
        constants.COMPETITION_SCHEDULE_BY_BLOCK,
    )

    if not competition:
        logging.info(f"Competition {args.competition_id} not found.")
        return

    seed = args.random_seed if args.random_seed else random.randint(0, sys.maxsize)
    metagraph = bt.metagraph(constants.PROMPTING_SUBNET_UID)
    vali_uids = metagraph_utils.get_high_stake_validators(metagraph, constants.SAMPLE_VALI_MIN_STAKE)
    vali_hotkeys = set([metagraph.hotkeys[uid] for uid in vali_uids])

    score1, details1 = evaluate_model(args.model_path1, competition, seed, vali_hotkeys, args.device)
    score2, details2 = evaluate_model(args.model_path2, competition, seed, vali_hotkeys, args.device)
    
    if score1 is not None and score2 is not None:
        logging.info(f"Comparison - Model 1: {score1}, Model 2: {score2}")
        logging.info(f"Model 1 Details: {details1}")
        logging.info(f"Model 2 Details: {details2}")
    
if __name__ == "__main__":
    nltk_modules = {"words", "punkt", "punkt_tab", "averaged_perceptron_tagger_eng"}
    for module in nltk_modules:
        nltk.download(module, raise_on_error=True)

    logging.reinitialize()
    logging.set_verbosity_trace()
    main()
