import datetime as dt
import math
import random
from typing import List, Set

import taoverse.utilities.logging as logging
import torch
from transformers import PreTrainedTokenizerBase

from finetune.datasets.generated.mmlu_parser import extract_q_and_a_text
from finetune.datasets.hugging_face.macrocosmos_dataset_loader import (
    MacrocosmosDatasetLoader,
)
from finetune.datasets.loader import DatasetLoader
from finetune.eval.if_eval import rule_factory
from finetune.eval.if_eval.sample import IFEvalTokenizedSample
from finetune.eval.if_eval.version import IfEvalVersion


class IFEvalLoader(DatasetLoader):
    """Generates samples for the IfEval task."""

    # The min/max number of rules per sample per version.
    VERSION_TO_RULE_COUNTS = {IfEvalVersion.V1: (1, 4), IfEvalVersion.V2: (2, 5)}

    def __init__(
        self,
        random_seed: int = None,
        max_samples: int = 20,
        validator_hotkeys: Set[str] = None,
        if_eval_version: IfEvalVersion = IfEvalVersion.V1,
    ):
        if random_seed:
            random.seed(random_seed)

        questions = list(
            MacrocosmosDatasetLoader(
                random_seed=random_seed,
                max_samples=max_samples * 2,
                validator_hotkeys=validator_hotkeys,
            )
        )

        logging.trace(f"Loaded {len(questions)} raw samples")

        # Parse the question and answer text from the raw text.
        parsed_q_and_a = [
            extract_q_and_a_text(prompt, answer) for prompt, answer in questions
        ]
        parsed_q_and_a = [qa for qa in parsed_q_and_a if qa is not None]
        logging.trace(
            f"Extracted {len(parsed_q_and_a)} questions and answers from raw samples"
        )

        # Filter out any Q&As that don't meet the bar.
        parsed_q_and_a = [
            (q, a) for q, a in parsed_q_and_a if not self._should_filter_question(q, a)
        ]

        # Zip the samples together, offset by 1, to create the pairs
        # of challenges to use for IFEval.
        n_samples = min(max_samples, len(parsed_q_and_a) - 1)
        self.buffer = []
        for qa1, qa2 in zip(
            parsed_q_and_a[:n_samples], parsed_q_and_a[1 : n_samples + 1]
        ):
            self.buffer.append(
                rule_factory.generate_if_eval_sample(
                    qa1,
                    qa2,
                    IFEvalLoader.VERSION_TO_RULE_COUNTS[if_eval_version][0],
                    IFEvalLoader.VERSION_TO_RULE_COUNTS[if_eval_version][1],
                    if_eval_version,
                )
            )

        logging.trace(f"Generated {len(self.buffer)} IFEval samples")

    def _should_filter_question(self, question: str, answer: str) -> bool:
        # For now, just filter out 1 word answers.
        return len(answer.split()) < 2

    def tokenize(
        self, tokenizer: PreTrainedTokenizerBase, sequence_length: int
    ) -> List[IFEvalTokenizedSample]:
        # Each batch is a tokenized list of prompts + list of rules.
        batches = []
        # If truncation is necessary, truncate from the left to avoid cutting off the answer part.
        tokenizer.truncation_side = "left"

        for sample in self:

            def _tokenize_prompt(prompt: str) -> torch.Tensor:
                ids = tokenizer.apply_chat_template(
                    conversation=[
                        {"role": "user", "content": prompt},
                    ],
                    max_length=sequence_length,
                    add_generation_prompt=True,
                )
                return torch.stack([torch.tensor(ids)])

            batches.append(
                IFEvalTokenizedSample(
                    prompt_1=_tokenize_prompt(sample.prompt_1),
                    prompt_2=_tokenize_prompt(sample.prompt_2),
                    rules=sample.rules,
                )
            )

        return batches

    def __iter__(self):
        return self.buffer.__iter__()

    def __len__(self):
        return len(self.buffer)
