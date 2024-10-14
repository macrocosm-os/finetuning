import dataclasses
from enum import IntEnum
from typing import List, Tuple

import torch

EvalSample = Tuple[torch.Tensor, List[str], str] | Tuple[torch.Tensor, str]


class EvalMethodId(IntEnum):
    """Enumeration of evaluation methods."""

    NONE = 0
    MULTIPLE_CHOICE = 1
    LOSS = 2


class NormalizationId(IntEnum):
    """Enumeration of normalization methods."""

    NONE = 0
    MAX = 1
    MEAN = 2


@dataclasses.dataclass
class EvalTask:
    name: str
    samples: List[EvalSample]
    method: EvalMethodId
    normalization_id: NormalizationId
    normalization_kwargs: dict = dataclasses.field(default_factory=dict)
    weight: float = 1.0
