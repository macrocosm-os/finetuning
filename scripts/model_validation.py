"""This script evaluates a local model using the same process as a Validator.

It can be used to estimate the performance of a model before submitting it."""

import argparse
import math
import random
import sys

from taoverse.model.competition import utils as competition_utils
from taoverse.model.data import Model, ModelId
from taoverse.model.model_updater import ModelUpdater
from taoverse.utilities.enum_action import IntEnumAction
from taoverse.utilities.perf_monitor import PerfMonitor
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

import constants
import finetune as ft
from competitions.data import CompetitionId
from finetune.datasets.subnet.cortex_subset_loader import CortexSubsetLoader
from finetune.datasets.subnet.prompting_subset_loader import PromptingSubsetLoader
from finetune.validation import compute_losses, compute_multiple_choice_deviation


def load_model(model_path, competition_id, allow_remote_code, kwargs) -> Model:
    model_id = ModelId(
        namespace="namespace", name="name", competition_id=competition_id
    )
    if allow_remote_code:
        pt_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_path,
            trust_remote_code=True,
            use_safetensors=True,
            **kwargs,
        )
    else:
        pt_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_path,
            local_files_only=True,
            use_safetensors=True,
            **kwargs,
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
        "--random_seed",
        type=int,
        default=0,
        help="Random seed to use while loading data. If 0 then randomize.",
    )
    parser.add_argument(
        "--latest_cortex_steps",
        type=int,
        default=5,
        help="Number of most recent cortex steps to sample data from",
    )
    parser.add_argument(
        "--latest_cortex_samples",
        type=int,
        default=400,
        help="Number of most recent cortex samples to eval against",
    )
    parser.add_argument(
        "--latest_prompting_steps",
        type=int,
        default=500,
        help="Number of most recent prompting steps to sample data from",
    )
    parser.add_argument(
        "--latest_prompting_samples",
        type=int,
        default=100,
        help="Number of most recent prompting samples to eval against",
    )
    parser.add_argument(
        "--competition_id",
        type=CompetitionId,
        default=CompetitionId.SN9_MODEL.value,
        action=IntEnumAction,
        help="competition to mine for (use --list-competitions to get all competitions)",
    )
    parser.add_argument(
        "--allow_remote_code",
        action="store_true",
        help="If a remote code should be allowed",
    )
    parser.add_argument(
        "--skip_constraints_check",
        action="store_true",
        help="If the competition constraints check should be skipped",
    )
    parser.add_argument(
        "--list_competitions", action="store_true", help="Print out all competitions"
    )
    parser.add_argument(
        "--tokenizer_override",
        action="store_true",
        help="If a custom tokenizer should be used rather than the competition one",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="Xenova/gpt-4",
        help="Tokenizer",
    )
    parser.add_argument(
        "--comp_block",
        type=int,
        default=9999999999,
        help="Block to lookup competition id from.",
    )
    args = parser.parse_args()
    if args.list_competitions:
        print(constants.COMPETITION_SCHEDULE_BY_BLOCK)
        return

    competition = competition_utils.get_competition_for_block(
        args.competition_id,
        args.comp_block,
        constants.COMPETITION_SCHEDULE_BY_BLOCK,
    )

    kwargs = competition.constraints.kwargs.copy()
    kwargs["use_cache"] = True

    print(f"Loading model for competition {args.competition_id}")
    load_model_perf = PerfMonitor("Eval: Load model")
    with load_model_perf.sample():
        model = load_model(
            args.model_path, competition.id, args.allow_remote_code, kwargs
        )
    print(load_model_perf.summary_str())

    if not args.skip_constraints_check:
        if not ModelUpdater.verify_model_satisfies_parameters(
            model, competition.constraints
        ):
            print("Model does not satisfy competition parameters!!!")
            return

    pull_data_perf = PerfMonitor("Eval: Pull data")
    sample_data = None

    seed = args.random_seed if args.random_seed else random.randint(0, sys.maxsize)

    if args.competition_id == CompetitionId.SN9_MODEL:
        print("Getting latest sample data from cortex.")
        with pull_data_perf.sample():
            sample_data = CortexSubsetLoader(
                use_latest_data=True,
                random_seed=seed,
                max_samples=args.latest_cortex_samples,
                steps=args.latest_cortex_steps,
                page_size=args.latest_cortex_steps,
            )
    elif args.competition_id == CompetitionId.B7_MULTI_CHOICE:
        print("Getting latest sample data from prompting.")
        with pull_data_perf.sample():
            sample_data = PromptingSubsetLoader(
                use_latest_data=True,
                random_seed=seed,
                max_samples=args.latest_prompting_samples,
                steps=args.latest_prompting_steps,
                page_size=args.latest_prompting_steps,
                min_percent_correct=constants.PROMPTING_MIN_CORRECT_MINERS,
            )
    else:
        print(
            f"Competition id: {args.competition_id} has no sample loading logic specified."
        )
        return
    print(pull_data_perf.summary_str())

    print("Tokenizing sample data")
    if args.tokenizer_override:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, trust_remote_code=args.allow_remote_code
        )
    else:
        tokenizer = ft.model.load_tokenizer(competition.constraints)
    batches = sample_data.tokenize(tokenizer, competition.constraints.sequence_length)

    print("Calculating deviations")
    compute_deviation_perf = PerfMonitor("Eval: Compute deviation")

    if args.competition_id == CompetitionId.SN9_MODEL:
        with compute_deviation_perf.sample():
            deviations = compute_losses(model.pt_model, batches, device=args.device)
    elif args.competition_id == CompetitionId.B7_MULTI_CHOICE:
        generation_config = GenerationConfig(
            max_new_tokens=20,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            top_k=40,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
        with compute_deviation_perf.sample():
            deviations = compute_multiple_choice_deviation(
                model.pt_model,
                tokenizer,
                generation_config,
                batches,
                device=args.device,
            )
    else:
        print(
            f"Competition id: {args.competition_id} has no evaluation logic specified."
        )
        return

    print(compute_deviation_perf.summary_str())

    average_model_deviation = (
        sum(deviations) / len(deviations) if len(deviations) > 0 else math.inf
    )

    print(f"The average deviation for {args.model_path} is {average_model_deviation}")


if __name__ == "__main__":
    main()
