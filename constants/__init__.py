from pathlib import Path
from transformers import (
    LlamaForCausalLM,
    StableLmForCausalLM,
)
from typing import List
import datetime as dt
import math
from competitions.data import CompetitionId, CompetitionParameters


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
WANDB_PROJECT = "finetuning-subnet"
# The uid for this subnet.
SUBNET_UID = 6
# The start block of this subnet
SUBNET_START_BLOCK = 2225782
# The uid for the Cortex subnet.
CORTEX_SUBNET_UID = 18
# The Cortex.t validator WANDB project and filters
CORTEX_WANDB_PROJECT = "cortex-t/multi-modality"
CORTEX_WANDB_TYPE = "validator"
CORTEX_MAX_UIDS = 256
CORTEX_MAX_AGE = dt.timedelta(days=1)
CORTEX_MIN_SCORE = 0.85
# The root directory of this project.
ROOT_DIR = Path(__file__).parent.parent
# The maximum bytes for the hugging face repo
MAX_HUGGING_FACE_BYTES: int = 15 * 1024 * 1024 * 1024
# TODO: Adjust below to be done by block instead as in 9 with helpers.
# Schedule of model architectures
COMPETITION_SCHEDULE: List[CompetitionParameters] = [
    CompetitionParameters(
        max_model_parameter_size=8 * 1024 * 1024 * 1024,
        architecture=LlamaForCausalLM,
        kwargs={},
        tokenizer="NousResearch/Meta-Llama-3-8B-Instruct",
        reward_percentage=0.6,
        competition_id="l3",
        competition_enum=CompetitionId.COMPETITION_1,
    ),
    CompetitionParameters(
        max_model_parameter_size=2 * 1024 * 1024 * 1024,
        architecture=StableLmForCausalLM,
        kwargs={},
        tokenizer="stabilityai/stablelm-2-zephyr-1_6b",
        reward_percentage=0.4,
        competition_id="s1",
        competition_enum=CompetitionId.COMPETITION_2,
    ),
]
ORIGINAL_COMPETITION_ID = "m1"


assert math.isclose(sum(x.reward_percentage for x in COMPETITION_SCHEDULE), 1.0)
assert all(
    len(x.competition_id) > 0 and len(x.competition_id) <= 2
    for x in COMPETITION_SCHEDULE
)

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
# validator eval sequence length.
sequence_length = 2048

# norm validation values
norm_eps_soft = 200
norm_eps_soft_percent_threshold = 0.15
norm_eps_hard = 1000
