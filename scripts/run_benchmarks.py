import argparse
import asyncio
import dataclasses
import json
import os
from typing import Any, Dict, Tuple

import bittensor as bt
import dotenv
import lm_eval
import wandb
from lm_eval.models.huggingface import HFLM
from taoverse.model.competition.data import Competition
from taoverse.model.data import ModelMetadata
from taoverse.model.storage.chain.chain_model_metadata_store import (
    ChainModelMetadataStore,
)
from taoverse.model.storage.hugging_face.hugging_face_model_store import (
    HuggingFaceModelStore,
)
from huggingface_hub import login

import constants


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

    # Download the tokenizer and model.
    # tokenizer = AutoTokenizer.from_pretrained(
    #     competition.constraints.tokenizer, cache_dir=hf_dir
    # )
    # store = HuggingFaceModelStore()
    # model = asyncio.run(
    #     store.download_model(model_metadata.id, hf_dir, competition.constraints)
    # )
    # pretrained = model.pt_model
    # pretrained.to("cuda")
    # print("Model device is: {}".format(pretrained.device))
    # hf_model = HFLM(model.pt_model, tokenizer=tokenizer)

    # return lm_eval.simple_evaluate(
    #     model=hf_model,
    #     tasks=[
    #         "leaderboard_mmlu_pro",
    #         "leaderboard_bbh",
    #         "leaderboard_gpqa",
    #         "leaderboard_ifeval",
    #         # "mmlu_pro",
    #         "mmlu",
    #     ],
    #     verbosity="DEBUG",
    #     batch_size="auto",
    #     log_samples=False,
    # )

    return lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained=rwh/bigone,tokenizer=Xenova/gpt-4,dtype=bfloat16",
        tasks=[
            # "leaderboard_mmlu_pro",
            # "leaderboard_bbh",
            # "leaderboard_gpqa",
            # "leaderboard_ifeval",
            # "mmlu_pro",
            "mmlu",
        ],
        verbosity="DEBUG",
        batch_size="auto",
        log_samples=False,
    )


def run_mmlu(model: str):
    return lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model}",
        tasks=[
            # "leaderboard_mmlu_pro",
            # "leaderboard_bbh",
            # "leaderboard_gpqa",
            # "leaderboard_ifeval",
            # "mmlu_pro",
            "mmlu",
        ],
        verbosity="DEBUG",
        batch_size="auto",
        log_samples=False,
    )


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

    for model in [
        "meta-llama/Llama-3.1-8B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "google/gemma-2-9b-it",
    ]:
        print(f"Running benchmarks for {model}.")
        results = run_mmlu(model)

        # Log to wandb.
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config={
                "model": model,
            },
            allow_val_change=True,
        )
        wandb_run.log(results["results"])
        wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run benchmarks and track completed evaluations."
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="test-benchmark",
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
    args = parser.parse_args()

    main(args)
