import argparse
import math
import random
import sys

import torch
from transformers import AutoModelForCausalLM

import constants
import finetune as ft
from competitions import utils as competition_utils
from competitions.data import Competition, CompetitionId
from finetune.dataset import CortexSubsetLoader
from finetune.validation import compute_losses
from model.data import Model, ModelId
from model.model_updater import ModelUpdater
from utilities.perf_monitor import PerfMonitor


def load_model(model_path, competition: Competition) -> Model:
    model_id = ModelId(
        namespace="namespace", name="name", competition_id=competition.id
    )
    pt_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        local_files_only=True,
        use_safetensors=True,
        **competition.constraints.kwargs,
    )

    return Model(id=model_id, pt_model=pt_model)


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
        "--latest_cortex_steps",
        type=int,
        default=5,
        help="Number of most recent Cortex steps to sample data from",
    )
    parser.add_argument(
        "--latest_cortex_samples",
        type=int,
        default=400,
        help="Number of most recent Cortex samples to eval against",
    )
    parser.add_argument(
        "--competition_id",
        type=CompetitionId,
        default=CompetitionId.SN9_MODEL,
        help="competition to validate against (use --list-competitions to get all competitions)",
    )
    parser.add_argument(
        "--list_competitions", action="store_true", help="Print out all competitions"
    )
    args = parser.parse_args()
    if args.list_competitions:
        print(constants.COMPETITION_SCHEDULE)
        return

    competition = competition_utils.get_competition(args.competition_id)
    if competition is None:
        raise AssertionError("Competition for {args.competition_id} not found")

    competition.constraints.kwargs["use_cache"] = True

    print(f"Loading model for competition {args.competition_id}")
    load_model_perf = PerfMonitor("Eval: Load model")
    with load_model_perf.sample():
        model = load_model(args.model_path, competition)
    print(load_model_perf.summary_str())

    if not ModelUpdater.verify_model_satisfies_parameters(model):
        print("Model does not satisfy competition parameters!!!")
        return

    print("Getting latest Cortex data")
    pull_data_perf = PerfMonitor("Eval: Pull data")

    with pull_data_perf.sample():
        cortex_data = CortexSubsetLoader(
            use_latest_data=True,
            random_seed=random.randint(0, sys.maxsize),
            max_samples=args.latest_cortex_samples,
            steps=args.latest_cortex_steps,
            page_size=args.latest_cortex_steps,
        )
    print(pull_data_perf.summary_str())

    print("Tokenizing cortex data")
    tokenizer = ft.model.load_tokenizer(competition)
    batches = cortex_data.tokenize(tokenizer, competition.constraints.sequence_length)

    print("Calculating losses")
    compute_loss_perf = PerfMonitor("Eval: Compute loss")
    with compute_loss_perf.sample():
        losses = compute_losses(model.pt_model, batches, device=args.device)
    print(compute_loss_perf.summary_str())

    average_model_loss = sum(losses) / len(losses) if len(losses) > 0 else math.inf

    print(f"The average model loss for {args.model_path} is {average_model_loss}")


if __name__ == "__main__":
    main()
