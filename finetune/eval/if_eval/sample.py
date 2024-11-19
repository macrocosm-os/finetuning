import dataclasses
from typing import List

import torch

from finetune.eval.if_eval.rule import IFEvalRule


@dataclasses.dataclass
class IFEvalSample:
    """Represents a single untokenized sample from the IfEval loader."""

    # The first (untokenized) prompt
    prompt_1: str

    # The second (untokenized) prompt.
    prompt_2: str

    # List of rules.
    rules: List[IFEvalRule]


@dataclasses.dataclass
class IFEvalTokenizedSample:
    """Represents a single tokenized sample from the IfEval loader."""

    # The first (tokenized) prompt
    prompt_1: torch.Tensor

    # The second (tokenized) prompt.
    prompt_2: torch.Tensor

    # List of rules.
    rules: List[IFEvalRule]
