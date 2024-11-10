import random
from typing import List, Tuple
import torch
from transformers import PreTrainedTokenizerBase

from finetune.eval.if_eval.rule import IFEvalRule


class IfEvalLoader:
    def __init__(self, random_seed: int = None, samples: int = 100):
        self.buffer: List[List[str], List[IFEvalRule]] = []

        if random_seed:
            random.seed(random_seed)

        # TODO: Load samples from the PromptingSubsetLoader
        # Filter the samples that are too short
        # Generate samples using N random rules

    def tokenize(
        self, tokenizer: PreTrainedTokenizerBase, sequence_length: int
    ) -> List[Tuple[torch.Tensor, List[IFEvalRule]]]:
        # Each batch is a tokenized list of prompts + list of rules.
        batches = []
        # If truncation is necessary, truncate from the left to avoid cutting off the answer part.
        tokenizer.truncation_side = "left"

        for prompts, rules in self:
            prompt_ids = []
            for prompt in prompts:
                conversation = [
                    {"role": "user", "content": prompt},
                ]
                ids = tokenizer.apply_chat_template(
                    conversation,
                    max_length=sequence_length,
                    add_generation_prompt=True,
                )
                prompt_ids.append(torch.tensor(ids))

            batches.append(
                (
                    torch.stack(prompt_ids),
                    rules,
                )
            )
        return batches

    def __iter__(self):
        return self.buffer.__iter__()

    def __len__(self):
        return len(self.buffer)
