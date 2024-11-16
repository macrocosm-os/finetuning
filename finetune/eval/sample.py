from typing import List, Tuple

import torch

from finetune.eval.if_eval.sample import IFEvalTokenizedSample

# A sample to evaluate.
EvalSample = (
    Tuple[torch.Tensor, List[str], str]
    | Tuple[torch.Tensor, torch.Tensor]
    | List[torch.Tensor]
    | List[IFEvalTokenizedSample]
)
