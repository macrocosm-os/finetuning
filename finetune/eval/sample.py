from typing import List, Tuple

import torch

# A sample to evaluate.
EvalSample = Tuple[torch.Tensor, List[str], str] | Tuple[torch.Tensor, torch.Tensor]
