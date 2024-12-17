import argparse
import asyncio
import dataclasses
import json
import os
import pickle
import shutil
import time
import traceback
from typing import Any, Dict, Tuple

import bittensor as bt
import dotenv
import lm_eval
import wandb
from huggingface_hub import login
from lm_eval.models.huggingface import HFLM
from taoverse.model import utils as model_utils
from taoverse.model.competition import utils as competition_utils
from taoverse.model.competition.data import Competition
from taoverse.model.data import ModelMetadata
from taoverse.model.storage.chain.chain_model_metadata_store import (
    ChainModelMetadataStore,
)
from taoverse.model.storage.hugging_face.hugging_face_model_store import (
    HuggingFaceModelStore,
)
from transformers import AutoTokenizer

import constants
from utils import benchmark_helpers


class CompletedEvalStore:
    """A store that tracks which models have been benchmarked.

    Stores state to a file in json format."""

    @dataclasses.dataclass
    class State:
        # The HF repo (project/entity) of the model
        repo: str

        # The HF commit.
        commit: str

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.evaluated_models = []

    def save(self):
        with open(self.filepath, "w+", encoding="utf-8") as f:
            json.dump([dataclasses.asdict(model) for model in self.evaluated_models], f)

    def load(self):
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                self.evaluated_models = [self.State(**model) for model in json.load(f)]
        except FileNotFoundError:
            self.evaluated_models = []

    def contains(self, state: State) -> bool:
        return state in self.evaluated_models

    def add(self, state: State):
        self.evaluated_models.append(state)


def _get_top_model_metadata(
    subtensor: bt.subtensor, competition: Competition
) -> Tuple[int, ModelMetadata]:
    """Returns the top model uid and metadata for a given competition."""
    chain_store = ChainModelMetadataStore(subtensor, subnet_uid=constants.SUBNET_UID)
    metagraph = subtensor.metagraph(netuid=constants.SUBNET_UID)

    biggest_incentive = 0
    top_uid = None
    ret_val = None
    for uid, incentive in enumerate(metagraph.I):
        if incentive > 0:
            metadata = asyncio.run(
                chain_store.retrieve_model_metadata(metagraph.hotkeys[uid])
            )
            if (
                metadata
                and metadata.id.competition_id == competition.id
                and incentive > biggest_incentive
            ):
                biggest_incentive = incentive
                ret_val = metadata
                top_uid = uid
    return top_uid, ret_val


def _run_benchmarks(
    competition: Competition,
    model_metadata: ModelMetadata,
    hf_dir: str = None,
) -> Dict[str, Any]:
    """Runs a benchmark on a given model."""

    store = HuggingFaceModelStore()
    model = asyncio.run(
        store.download_model(model_metadata.id, hf_dir, competition.constraints)
    )
    # Download the tokenizer and model.
    tokenizer = (
        model.tokenizer
        if model.tokenizer
        else AutoTokenizer.from_pretrained(
            competition.constraints.tokenizer, cache_dir=hf_dir
        )
    )
    pretrained = model.pt_model
    pretrained.to("cuda")
    print("Model device is: {}".format(pretrained.device))
    hf_model = HFLM(model.pt_model, tokenizer=tokenizer)

    return lm_eval.simple_evaluate(
        model=hf_model,
        tasks=[
            "leaderboard_mmlu_pro",
            "leaderboard_bbh",
            "leaderboard_gpqa",
            "leaderboard_ifeval",
            "leaderboard_musr",
            "mmlu",
            "agieval_en",
            "arc_challenge",
            "gsm8k_cot",
        ],
        verbosity="DEBUG",
        batch_size="auto",
        log_samples=False,
    )


def save_state(state: Dict[int, CompletedEvalStore.State], filepath: str):
    with open(filepath, "wb") as f:
        pickle.dump(state, f)


def load_state(filepath: str) -> Dict[int, CompletedEvalStore.State]:
    with open(filepath, "rb") as f:
        return pickle.load(f)


def _model_result_filepath(model_metadata: ModelMetadata, dir: str) -> str:
    return f"{dir}/{model_metadata.id.namespace}_{model_metadata.id.name}_{model_metadata.id.commit}.json"


def save_model_results(
    model_metadata: ModelMetadata, results: Dict[str, Any], dir: str
):
    with open(
        _model_result_filepath(model_metadata, dir),
        "w+",
        encoding="utf-8",
    ) as f:
        json.dump(results, f)


def load_model_results(model_metadata: ModelMetadata, dir: str) -> Dict[str, Any]:
    path = _model_result_filepath(model_metadata, dir)
    print(f"Loading results from {path}")
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def delete_dir_contents(dir: str):
    try:
        shutil.rmtree(dir)
    except:
        print(f"Failed to delete {dir}. {traceback.format_exc()}")

    if not os.path.exists(dir):
        os.makedirs(dir)


def main(args: argparse.Namespace):
    dotenv.load_dotenv()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    token = os.getenv("WANDB_ACCESS_TOKEN")
    if not token:
        raise ValueError("WANDB_ACCESS_TOKEN not set in environment.")

    # Make sure we can login to wandb.
    wandb.login(key=token)

    # Make sure we have a HF access token
    HuggingFaceModelStore.assert_access_token_exists()
    login(token=os.getenv("HF_ACCESS_TOKEN"))

    # Set the HF cache directory if provided.
    if args.hf_dir:
        os.environ["HF_HOME"] = args.hf_dir

    step = 0

    # Load state from previous runs.
    last_model_per_comp = {}
    try:
        last_model_per_comp = load_state(args.file)
    except FileNotFoundError:
        pass

    results_dir = args.results_dir
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    while True:
        try:
            print("Sleeping for 15 mins.")
            time.sleep(900)
            step += 1
            # Figure out which competition we should check next.
            subtensor = bt.subtensor()
            competition_schedule = competition_utils.get_competition_schedule_for_block(
                block=subtensor.block,
                schedule_by_block=constants.COMPETITION_SCHEDULE_BY_BLOCK,
            )
            competition = competition_schedule[step % len(competition_schedule)]
            print(f"Checking if a benchmark is needed for competition {competition.id}")

            uid, model_metadata = _get_top_model_metadata(subtensor, competition)
            if not model_metadata:
                print(f"Didn't find a top model for competition {competition.id}.")
                continue

            state = CompletedEvalStore.State(
                repo=f"{model_metadata.id.namespace}/{model_metadata.id.name}",
                commit=model_metadata.id.commit,
            )
            if state == last_model_per_comp.get(competition.id, None):
                print(
                    f"Model {state.repo} at commit {state.commit} has already been benchmarked."
                )
                continue

            try:
                results = load_model_results(model_metadata, results_dir)
                print(
                    f"Model {state.repo} at commit {state.commit} has already been benchmarked. Using previous results"
                )
            except FileNotFoundError:
                print(f"Did not find previous results for {state.repo}/{state.commit}.")
                results = None

            if not results:
                print(f"Running benchmarks for {state.repo}/{state.commit}.")
                results = _run_benchmarks(competition, model_metadata, args.hf_dir)
                results = results["results"]
                print(f"Finished running benchmarks for {state.repo}/{state.commit}.")

                save_model_results(model_metadata, results, results_dir)

            lb_results = benchmark_helpers.get_leaderboard_scores(results)
            print(f"Leaderboard results: {lb_results}")

            # Log to wandb.
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                config={
                    "uid": uid,
                    "model": model_utils.get_hf_url(model_metadata),
                    "block": model_metadata.block,
                    "competition_id": competition.id,
                },
                allow_val_change=True,
            )
            wandb_run.log(results | lb_results)
            wandb_run.finish()

            last_model_per_comp[competition.id] = state
            save_state(last_model_per_comp, args.file)

            if step % 50:
                print("Deleting HF cache.")
                delete_dir_contents(args.hf_dir)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Caught error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run benchmarks and track completed evaluations."
    )
    parser.add_argument(
        "--file",
        type=str,
        default="last_model.pickle",
        help="Path to the JSON file for storing completed evaluations.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="test-benchmarks",
        help="Wandb project to log results to.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="rusticluftig",
        help="Wandb entity to log results to.",
    )
    parser.add_argument(
        "--hf_dir",
        type=str,
        default=None,
        help="Directory to load models into",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="../results",
        help="Directory to store results in.",
    )
    args = parser.parse_args()

    main(args)
