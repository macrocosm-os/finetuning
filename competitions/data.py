from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, List, Type

from transformers import PreTrainedModel


class CompetitionId(IntEnum):
    """Unique identifiers for each competition."""

    SN9_MODEL = 1

    # Defined for tests. Will be repurposed later.
    COMPETITION_2 = 2

    # Overwrite the default __repr__, which doesn't work with
    # bt.logging for some unknown reason.
    def __repr__(self) -> str:
        return f"{self.value}"


@dataclass
class ModelConstraints:
    """Defines the constraints for models submitted to a specific competition."""

    # The maximum parameter size allowed for models
    max_model_parameter_size: int

    # Architecture class of model
    allowed_architectures: List[Type[PreTrainedModel]]

    # The model's sequence length.
    sequence_length: int

    # The Pretrained tokenizer to use.
    tokenizer: str

    # Block delay before evaluating uploaded models. Based on look-back period for eval data collection.
    eval_block_delay: int

    # Any additional arguments to pass to from_pretrained
    kwargs: Any = field(default_factory=dict)


@dataclass
class Competition:
    """Defines a competition."""

    # Unique ID for this competition.
    id: CompetitionId

    # All restrictions on models allowed in this competition.
    constraints: ModelConstraints

    # Percentage of emissions dedicated to this competition.
    reward_percentage: float
