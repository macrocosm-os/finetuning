# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 const

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# Tools for performing validation over models.

import typing

import torch
import dataclasses
import bittensor as bt
import transformers
from taoverse.model.competition.epsilon import EpsilonFunc
from transformers import GenerationConfig
from taoverse.model.competition.data import Competition
from finetune.eval.method import (
    compute_multiple_choice_deviation,
    compute_reference_loss,
)
from finetune.eval.normalization import normalize_score


from finetune.eval.task import EvalMethodId, EvalTask


def _is_win(
    loss_i: float,
    loss_j: float,
    block_i: int,
    block_j: int,
    epsilon_func: EpsilonFunc,
    current_block: int,
) -> bool:
    """
    Determines the winner between two models based on the epsilon adjusted loss.

    Parameters:
        loss_i (float): Loss of uid i on batch
        loss_j (float): Loss of uid j on batch.
        block_i (int): Block of uid i.
        block_j (int): Block of uid j.
        epsilon_func (EpsilonFunc): Function that determines how much advantage to give to the earlier block.
        current_block: The current block.

    Returns:
        bool: True if loss i is better, False otherwise.
    """
    # Adjust loss based on timestamp and epsilon.
    loss_i = (
        (1 - epsilon_func.compute_epsilon(current_block, block_i)) * loss_i
        if block_i < block_j
        else loss_i
    )
    loss_j = (
        (1 - epsilon_func.compute_epsilon(current_block, block_j)) * loss_j
        if block_j < block_i
        else loss_j
    )
    return loss_i < loss_j


def compute_wins(
    uids: typing.List[int],
    uid_to_score: typing.Dict[int, float],
    uid_to_block: typing.Dict[int, int],
    epsilon_func: EpsilonFunc,
    current_block: int,
) -> typing.Tuple[typing.Dict[int, int], typing.Dict[int, float]]:
    """
    Computes the wins and win rate for each model based on loss comparison.

    Parameters:
        uids (list): A list of uids to compare.
        uid_to_score (dict): A dictionary of score for each uid
        uid_to_block (dict): A dictionary of blocks for each uid.
        epsilon_func (EpsilonFunc): Function that determines how much advantage to give to the earlier block.
        current_block: The current block.

    Returns:
        tuple: A tuple containing two dictionaries, one for wins and one for win rates.
    """
    wins = {uid: 0 for uid in uids}
    win_rate = {uid: 0 for uid in uids}
    for uid_i in uids:
        total_matches = 0
        for uid_j in uids:
            if uid_i == uid_j:
                continue
            loss_i = uid_to_score[uid_i]
            loss_j = uid_to_score[uid_j]
            wins[uid_i] += (
                1
                if _is_win(
                    loss_i,
                    loss_j,
                    uid_to_block[uid_i],
                    uid_to_block[uid_j],
                    epsilon_func,
                    current_block,
                )
                else 0
            )
            total_matches += 1
        # Calculate win rate for uid i
        win_rate[uid_i] = wins[uid_i] / total_matches if total_matches > 0 else 0

    return wins, win_rate


@dataclasses.dataclass
class ScoreDetails:
    """Details of the score for a model."""

    raw_score: typing.Optional[float] = None
    norm_score: typing.Optional[float] = None
    weighted_norm_score: typing.Optional[float] = None


def score_model(
    model,
    tokenizer: transformers.PreTrainedTokenizer,
    evals: typing.List[EvalTask],
    competition: Competition,
    device: str,
) -> typing.Tuple[float, dict]:
    """Scores a model based on the provided eval tasks.

    Returns:
        tuple: A tuple containing the score and a dictionary of score details."""

    with torch.inference_mode():
        model.to(device)
        model.eval()

        score = 0
        score_details = {task.name: ScoreDetails() for task in evals}

        for task in evals:
            bt.logging.info(f"Scoring model on task: {task.name}")
            match task.method_id:
                case EvalMethodId.MULTIPLE_CHOICE:
                    compute_generation_config = GenerationConfig(
                        max_new_tokens=20,
                        max_length=competition.constraints.sequence_length,
                        do_sample=False,
                        repetition_penalty=1.1,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    raw_score = compute_multiple_choice_deviation(
                        model=model,
                        tokenizer=tokenizer,
                        generation_config=compute_generation_config,
                        batches=task.samples,
                        device=device,
                    )
                case EvalMethodId.REFERENCE_LOSS:
                    raw_score = compute_reference_loss(
                        model=model,
                        batches=task.samples,
                        device=device,
                    )
                case _:
                    raise ValueError(f"Unhandled evaluation method {task.method_id}.")
            # Normalize score
            normalized_score = normalize_score(
                raw_score, task.normalization_id, task.normalization_kwargs
            )
            weighted_norm_score = normalized_score * task.weight

            score += weighted_norm_score
            score_details[task.name] = ScoreDetails(
                raw_score=raw_score,
                norm_score=normalized_score,
                weighted_norm_score=weighted_norm_score,
            )

    return score, score_details
