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

import dataclasses
import typing

import bittensor as bt
import torch
import transformers
from taoverse.model.competition.data import Competition
from taoverse.model.competition.epsilon import EpsilonFunc
from taoverse.model.eval.normalization import normalize_score
from taoverse.model.eval.task import EvalTask
from transformers import GenerationConfig

from finetune.eval.method import (
    EvalMethodId,
    compute_multiple_choice_deviation,
    compute_reference_loss,
    compute_text_loss,
    compute_if_eval,
)
from finetune.eval.sample import EvalSample


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
    samples: typing.List[typing.List[EvalSample]],
    competition: Competition,
    device: str,
) -> typing.Tuple[float, dict]:
    """Scores a model based on the provided eval tasks.

    Args:
        model (torch.nn.Module): The model to score.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for tokenization.
        evals (list): A list of EvalTasks to score the model on.
        samples (list): A list of samples to use for scoring for the eval tasks. Must be the same length as evals.
        competition (Competition): The competition to score the model for.
        device (str): The device to use for computation (e.g., 'cpu', 'gpu').

    Returns:
        tuple: A tuple containing the score and a dictionary of score details."""

    if len(evals) != len(samples):
        raise ValueError("Number of eval tasks and samples must match.")

    with torch.inference_mode():
        model.to(device)
        model.eval()

        score = 0
        score_details = {task.name: ScoreDetails() for task in evals}

        for task, samples in zip(evals, samples):
            bt.logging.trace(f"Scoring model on task: {task.name}")
            match task.method_id:
                case EvalMethodId.MULTIPLE_CHOICE:
                    compute_mc_generation_config = GenerationConfig(
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
                        generation_config=compute_mc_generation_config,
                        batches=samples,
                        device=device,
                    )
                case EvalMethodId.REFERENCE_LOSS:
                    raw_score = compute_reference_loss(
                        model=model,
                        batches=samples,
                        device=device,
                    )
                case EvalMethodId.TEXT_LOSS:
                    raw_score = compute_text_loss(
                        model=model,
                        batches=samples,
                        device=device,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                case EvalMethodId.IF_EVAL:
                    # TODO: Figure out the right config to use.
                    compute_if_generation_config = GenerationConfig(
                        max_length=competition.constraints.sequence_length,
                        do_sample=False,
                        repetition_penalty=1.2,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    raw_score = compute_if_eval(
                        model=model,
                        tokenizer=tokenizer,
                        generation_config=compute_if_generation_config,
                        batches=samples,
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
