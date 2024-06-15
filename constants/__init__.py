import datetime as dt
import math
from pathlib import Path
from typing import List

from transformers import (
    MistralForCausalLM,
    LlamaForCausalLM,
    BartForCausalLM,
    FalconForCausalLM,
    GPTNeoXForCausalLM,
    PhiForCausalLM,
    GemmaForCausalLM,
)
from competitions.data import CompetitionId, Competition, ModelConstraints

# ---------------------------------
# Project Constants.
# ---------------------------------

__version__ = "0.2.7"
version_split = __version__.split(".")
__spec_version__ = (
    (1000 * int(version_split[0]))
    + (10 * int(version_split[1]))
    + (1 * int(version_split[2]))
)

# The validator WANDB project.
# TODO: Update these.
WANDB_PROJECT = "finetuning"
WANDB_ENTITY = "rusticluftig"
# The uid for this subnet.
SUBNET_UID = 6
# The uid for the Cortex subnet.
CORTEX_SUBNET_UID = 18
# The Cortex.t validator WANDB project and filters
CORTEX_WANDB_PROJECT = "cortex-t/multi-modality"
CORTEX_WANDB_TYPE = "validator"
CORTEX_MAX_UIDS = 256
CORTEX_MAX_AGE = dt.timedelta(days=1)
CORTEX_MIN_SCORE = 0.85
# Minimum stake to get data from a cortex validator.
CORTEX_MIN_STAKE = 100_000
# Minimum stake to consider a validator when checking for miners with weights.
WEIGHT_SYNC_VALI_MIN_STAKE = 100_000
# Minimum percent of weight on a vali for a miner to be considered a top miner.
# Since there can be multiple competitions at different reward percentages we can't just check biggest.
WEIGHT_SYNC_MINER_MIN_PERCENT = 0.10
# The root directory of this project.
ROOT_DIR = Path(__file__).parent.parent
# The maximum bytes for the hugging face repo
MAX_HUGGING_FACE_BYTES: int = 15 * 1024 * 1024 * 1024
# TODO: Adjust below to be done by block instead as in 9 with helpers.
# Schedule of model architectures
COMPETITION_SCHEDULE: List[Competition] = [
    Competition(
        id=CompetitionId.SN9_MODEL,
        constraints=ModelConstraints(
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
                "torch_dtype": "bfloat16",
                "attn_implementation": "flash_attention_2",
            },
        ),
        reward_percentage=1.0,
    )
]

assert math.isclose(sum(x.reward_percentage for x in COMPETITION_SCHEDULE), 1.0)

# ---------------------------------
# Miner/Validator Model parameters.
# ---------------------------------

weights_version_key = __spec_version__

# validator weight moving average term
alpha = 0.5
# validator scoring exponential temperature
# 0.01 gives ~96% to best model with only ~3 receiving any weights.
temperature = 0.01
# validator score boosting for earlier models.
timestamp_epsilon = 0.005

# norm validation values
norm_eps_soft = 200
norm_eps_soft_percent_threshold = 0.15
norm_eps_hard = 1000
# time required between updates to the chain.
chain_update_cadence = dt.timedelta(minutes=20)
# time required between retrying evaluation of a stale model. (First retry will be immediate).
model_retry_cadence = dt.timedelta(hours=4)
