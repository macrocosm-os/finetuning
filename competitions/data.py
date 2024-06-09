from enum import IntEnum
from typing import Type, Optional, Any
from transformers import (
    PreTrainedModel,
)
from dataclasses import dataclass


class CompetitionId(IntEnum):
    """Unique identifiers for each competition."""

    COMPETITION_1 = 1
    COMPETITION_2 = 2


@dataclass
class CompetitionParameters:
    """Class defining model parameters"""

    # The maximum parameter size allowed for models
    max_model_parameter_size: int
    # Architecture class of model
    architecture: Type[PreTrainedModel]
    # Any additional arguments to from_pretrained
    kwargs: Any
    # Fixed tokenizer
    tokenizer: Optional[str]
    # Reward percentage
    reward_percentage: float
    # Competition id
    competition_id: str
    # Competition enum (#TODO replace id above soon)
    competition_enum: CompetitionId
