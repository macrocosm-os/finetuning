import datetime as dt
import math
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from taoverse.model.competition.data import (
    Competition,
    ModelConstraints,
    NormValidationConstraints,
)
from taoverse.model.competition.epsilon import LinearDecay
from taoverse.model.eval.normalization import NormalizationId
from taoverse.model.eval.task import EvalTask
from transformers import (
    BartForCausalLM,
    FalconForCausalLM,
    Gemma2ForCausalLM,
    GemmaForCausalLM,
    GPTNeoXForCausalLM,
    LlamaForCausalLM,
    MistralForCausalLM,
    Phi3ForCausalLM,
    PhiForCausalLM,
    Qwen2ForCausalLM,
)

from competitions.data import CompetitionId
from finetune.datasets.ids import DatasetId
from finetune.eval.method import EvalMethodId

# ---------------------------------
# Project Constants.
# ---------------------------------

__version__ = "2.6.0"
version_split = __version__.split(".")
__spec_version__ = (
    (1000 * int(version_split[0]))
    + (10 * int(version_split[1]))
    + (1 * int(version_split[2]))
)

# The version of the validator state. When incremented, causes validators
# to start from a fresh state.
VALIDATOR_STATE_VERSION = 5

# Block the subnet was registered.
GENESIS_BLOCK = 3138611
# Define the number of blocks per vali "sync". This cadence is used to align validator behavior for better vtrust.
SYNC_BLOCK_CADENCE = 270
# Rough estimate of the number of seconds per block.
SECONDS_PER_BLOCK = 12
# Validator weight moving average term.
# At 0.85 a model will go from 0 -> 0.278 in 2 cycles and from 0 -> 0.833 in 11 cycles.
ALPHA = 0.85
# Any miners with a combined competition weight below this threshold will instead receive 0 weight.
# This is intended to help vtrust in conjunction with a low alpha by handling the tail ends.
# At 1 eval per 270 blocks, newly winning models will start recieving weight after ~540 blocks.
# Previously winning models will phase out after ~2970 blocks, at which point only the new winner will have weight.
MIN_WEIGHT_THRESHOLD = 0.18

# The validator WANDB project.
WANDB_PROJECT = "finetuning"
WANDB_ENTITY = "rusticluftig"
# The uid for this subnet.
SUBNET_UID = 37
# Minimum stake to get sample data from a validator.
SAMPLE_VALI_MIN_STAKE = 100_000
# The uid for the Prompting subnet.
PROMPTING_SUBNET_UID = 1
# The Prompting validator WANDB project and filters
PROMPTING_WANDB_PROJECT = "macrocosmos/prompting-validators"
PROMPTING_MAX_AGE = dt.timedelta(hours=4)
# Minimum number of samples allowed to consider MMLU as an eval task.
MIN_ALLOWED_SAMPLES = 50
# Minimum stake to consider a validator when checking for miners with weights.
WEIGHT_SYNC_VALI_MIN_STAKE = 100_000
# Minimum percent of weight on a vali for a miner to be considered a top miner.
# Since there can be multiple competitions at different reward percentages we can't just check biggest.
# Since we only set weights per competition with a threshold of 0.18 we can just take any percent here.
WEIGHT_SYNC_MINER_MIN_PERCENT = 0.01
# The root directory of this project.
ROOT_DIR = Path(__file__).parent.parent
# The maximum bytes for the hugging face repo.
MAX_HUGGING_FACE_BYTES: int = 15 * 1024 * 1024 * 1024
# Defined model constraints by competition id to ensure they are constant across blocks.
MODEL_CONSTRAINTS_BY_COMPETITION_ID: Dict[CompetitionId, ModelConstraints] = {
    CompetitionId.B7_MULTI_CHOICE: ModelConstraints(
        max_model_parameter_size=6_900_000_000,
        sequence_length=4096,
        allowed_architectures=[
            MistralForCausalLM,
            LlamaForCausalLM,
            BartForCausalLM,
            FalconForCausalLM,
            GPTNeoXForCausalLM,
            PhiForCausalLM,
            GemmaForCausalLM,
        ],
        tokenizer="Xenova/gpt-4",
        kwargs={
            "torch_dtype": torch.bfloat16,
        },
        eval_block_delay=1600,  # ~5 hours.
        norm_validation_constraints=NormValidationConstraints(
            norm_eps_soft=200,
            norm_eps_soft_percent_threshold=0.15,
            norm_eps_hard=1000,
        ),
        epsilon_func=LinearDecay(0.05, 0.01, 7200 * 5),  # Decay over ~5 days.
        max_bytes=15 * 1024 * 1024 * 1024,
    ),
    CompetitionId.INSTRUCT_8B: ModelConstraints(
        max_model_parameter_size=8_100_000_000,
        sequence_length=4096,
        allowed_architectures=[
            BartForCausalLM,
            FalconForCausalLM,
            Gemma2ForCausalLM,
            GemmaForCausalLM,
            GPTNeoXForCausalLM,
            LlamaForCausalLM,
            MistralForCausalLM,
            Phi3ForCausalLM,
            PhiForCausalLM,
        ],
        tokenizer=None,  # Any tokenizer can be used.
        kwargs={
            "torch_dtype": torch.bfloat16,
        },
        eval_block_delay=1600,  # ~5 hours.
        norm_validation_constraints=NormValidationConstraints(
            norm_eps_soft=200,
            norm_eps_soft_percent_threshold=0.15,
            norm_eps_hard=1000,
        ),
        epsilon_func=LinearDecay(0.05, 0.01, 7200 * 5),  # Decay over ~5 days.
        max_bytes=20 * (1024**3),
    ),
}

INSTRUCT_8B_BLOCK = 4_423_335

# Schedule of competitions by block.
COMPETITION_SCHEDULE_BY_BLOCK: List[Tuple[int, List[Competition]]] = [
    (
        0,
        [
            Competition(
                CompetitionId.B7_MULTI_CHOICE,
                MODEL_CONSTRAINTS_BY_COMPETITION_ID[CompetitionId.B7_MULTI_CHOICE],
                1.0,
                eval_tasks=[
                    EvalTask(
                        name="SYNTHETIC_MMLU",
                        method_id=EvalMethodId.MULTIPLE_CHOICE,
                        dataset_id=DatasetId.SYNTHETIC_MMLU,
                        normalization_id=NormalizationId.NONE,
                        weight=0.85,
                    ),
                    EvalTask(
                        name="WORD_SORTING",
                        method_id=EvalMethodId.REFERENCE_LOSS,
                        dataset_id=DatasetId.WORD_SORTING,
                        normalization_id=NormalizationId.INVERSE_EXPONENTIAL,
                        normalization_kwargs={"ceiling": 40.0},
                        weight=0.05,
                    ),
                    EvalTask(
                        name="FINEWEB",
                        method_id=EvalMethodId.TEXT_LOSS,
                        dataset_id=DatasetId.FINEWEB,
                        normalization_id=NormalizationId.INVERSE_EXPONENTIAL,
                        normalization_kwargs={"ceiling": 20.0},
                        weight=0.05,
                    ),
                    EvalTask(
                        name="IF_EVAL_V1",
                        method_id=EvalMethodId.IF_EVAL,
                        dataset_id=DatasetId.SYNTHETIC_IF_EVAL,
                        normalization_id=NormalizationId.NONE,
                        weight=0.05,
                    ),
                ],
            ),
        ],
    ),
    (
        INSTRUCT_8B_BLOCK,
        [
            Competition(
                CompetitionId.B7_MULTI_CHOICE,
                MODEL_CONSTRAINTS_BY_COMPETITION_ID[CompetitionId.B7_MULTI_CHOICE],
                0.9,
                eval_tasks=[
                    EvalTask(
                        name="SYNTHETIC_MMLU",
                        method_id=EvalMethodId.MULTIPLE_CHOICE,
                        dataset_id=DatasetId.SYNTHETIC_MMLU,
                        normalization_id=NormalizationId.NONE,
                        weight=0.8,
                    ),
                    EvalTask(
                        name="WORD_SORTING",
                        method_id=EvalMethodId.REFERENCE_LOSS,
                        dataset_id=DatasetId.WORD_SORTING,
                        normalization_id=NormalizationId.INVERSE_EXPONENTIAL,
                        normalization_kwargs={"ceiling": 40.0},
                        weight=0.05,
                    ),
                    EvalTask(
                        name="FINEWEB",
                        method_id=EvalMethodId.TEXT_LOSS,
                        dataset_id=DatasetId.FINEWEB,
                        normalization_id=NormalizationId.INVERSE_EXPONENTIAL,
                        normalization_kwargs={"ceiling": 20.0},
                        weight=0.1,
                    ),
                    EvalTask(
                        name="IF_EVAL_V1",
                        method_id=EvalMethodId.IF_EVAL,
                        dataset_id=DatasetId.SYNTHETIC_IF_EVAL,
                        normalization_id=NormalizationId.NONE,
                        weight=0.05,
                    ),
                ],
            ),
            Competition(
                CompetitionId.INSTRUCT_8B,
                MODEL_CONSTRAINTS_BY_COMPETITION_ID[CompetitionId.INSTRUCT_8B],
                0.1,
                eval_tasks=[
                    EvalTask(
                        name="SYNTHETIC_MMLU",
                        method_id=EvalMethodId.MULTIPLE_CHOICE,
                        dataset_id=DatasetId.SYNTHETIC_MMLU,
                        normalization_id=NormalizationId.NONE,
                        weight=0.8,
                    ),
                    EvalTask(
                        name="WORD_SORTING",
                        method_id=EvalMethodId.REFERENCE_LOSS,
                        dataset_id=DatasetId.WORD_SORTING,
                        normalization_id=NormalizationId.INVERSE_EXPONENTIAL,
                        normalization_kwargs={"ceiling": 40.0},
                        weight=0.05,
                    ),
                    EvalTask(
                        name="FINEWEB",
                        method_id=EvalMethodId.TEXT_LOSS,
                        dataset_id=DatasetId.FINEWEB,
                        normalization_id=NormalizationId.INVERSE_EXPONENTIAL,
                        normalization_kwargs={"ceiling": 20.0},
                        weight=0.1,
                    ),
                    EvalTask(
                        name="IF_EVAL_V1",
                        method_id=EvalMethodId.IF_EVAL,
                        dataset_id=DatasetId.SYNTHETIC_IF_EVAL,
                        normalization_id=NormalizationId.NONE,
                        weight=0.05,
                    ),
                ],
            ),
        ],
    ),
]

for block_and_competitions in COMPETITION_SCHEDULE_BY_BLOCK:
    assert math.isclose(
        sum(competition.reward_percentage for competition in block_and_competitions[1]),
        1.0,
    )
    for comp in block_and_competitions[1]:
        assert math.isclose(
            sum(task.weight for task in comp.eval_tasks),
            1.0,
        )

# ---------------------------------
# Miner/Validator Model parameters.
# ---------------------------------

weights_version_key = __spec_version__

# time required between updates to the chain.
chain_update_cadence = dt.timedelta(minutes=20)
# Number of blocks required between retrying evaluation of a model.
model_retry_cadence = 300  # Roughly 1 hour
# How frequently to check the models given weights by other large validators.
scan_top_model_cadence = dt.timedelta(minutes=30)
# validator eval batch min to keep for next loop.
sample_min = 3
# We allow the sample_min per competition + 7 additional models to be held at any one time.
updated_models_limit = sample_min * len(MODEL_CONSTRAINTS_BY_COMPETITION_ID) + 7
