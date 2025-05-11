import dataclasses
import typing
import numpy as np
from typing import List, Union, Dict, Tuple
import math

import taoverse.utilities.logging as logging
import torch
from taoverse.model.competition.data import Competition
from taoverse.model.competition.epsilon import EpsilonFunc
from taoverse.model.data import Model
from taoverse.model.eval.normalization import normalize_score
from taoverse.model.eval.task import EvalTask
from transformers import GenerationConfig

from finetune.eval.method import (
    EvalMethodId,
    compute_if_eval,
    compute_multiple_choice_deviation,
    compute_reference_loss,
    compute_text_loss,
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
    model: Model,
    eval_tasks: List[EvalTask],
    samples: List[Union[List[np.ndarray], Dict[str, List]]],
    competition: Competition,
    device: str,
) -> Tuple[float, Dict[str, ScoreDetails]]:
    """Score a model on a set of evaluation tasks.

    This function evaluates a model on multiple evaluation tasks, computes raw scores,
    normalizes them according to task-specific normalization methods, and returns
    both the combined weighted score and detailed scores for each task.

    Args:
        model (Model): The model to evaluate, containing a PyTorch model and tokenizer.
        eval_tasks (List[EvalTask]): List of evaluation task definitions.
        samples (List[Union[List[np.ndarray], Dict[str, List]]]): List of tokenized samples
            corresponding to each eval task.
        competition (Competition): Competition configuration containing constraints.
        device (str): Device to run evaluation on (e.g., 'cuda', 'cpu').

    Returns:
        Tuple[float, Dict[str, ScoreDetails]]: A tuple containing:
            - float: The combined weighted score across all tasks.
            - Dict[str, ScoreDetails]: Detailed scoring information for each task.

    Raises:
        ValueError: If the number of eval tasks doesn't match the number of samples,
                   if the model doesn't have a tokenizer, or if an unsupported
                   evaluation method is specified.
        RuntimeError: If all evaluation tasks failed to load data, indicating the model
                     should remain in the queue for retry.
    """
    if len(eval_tasks) != len(samples):
        raise ValueError("Number of eval tasks and samples must match.")

    if not model.tokenizer:
        raise ValueError("Model does not have a tokenizer")

    # If we have no eval tasks left (all failed), raise an exception to signal retry
    if len(eval_tasks) == 0:
        raise RuntimeError("All evaluation tasks failed to load data. Model will be retried later.")

    # Calculate the sum of weights for the remaining tasks to normalize
    total_weight = sum(task.weight for task in eval_tasks)
    if not math.isclose(total_weight, 1.0) and total_weight > 0:
        logging.info(f"Renormalizing weights of {len(eval_tasks)} remaining eval tasks. Original sum: {total_weight}")

    with torch.inference_mode():
        model.pt_model.to(device)
        model.pt_model.eval()

        score = 0
        score_details = {task.name: ScoreDetails() for task in eval_tasks}
        tokenizer = model.tokenizer

        for task, task_samples in zip(eval_tasks, samples):
            logging.trace(f"Scoring model on task: {task.name}")
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
                        model=model.pt_model,
                        tokenizer=tokenizer,
                        generation_config=compute_mc_generation_config,
                        batches=task_samples,
                        device=device,
                    )
                case EvalMethodId.REFERENCE_LOSS:
                    raw_score = compute_reference_loss(
                        model=model.pt_model,
                        batches=task_samples,
                        device=device,
                    )
                case EvalMethodId.TEXT_LOSS:
                    raw_score = compute_text_loss(
                        model=model.pt_model,
                        batches=task_samples,
                        device=device,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                case EvalMethodId.IF_EVAL:
                    compute_if_generation_config = GenerationConfig(
                        max_new_tokens=200,
                        repetition_penalty=1.2,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.eos_token_id,
                        do_sample=False,
                        max_time=5.0,
                    )
                    raw_score = compute_if_eval(
                        model=model.pt_model,
                        tokenizer=tokenizer,
                        generation_config=compute_if_generation_config,
                        batches=task_samples,
                        device=device,
                    )
                case _:
                    raise ValueError(f"Unhandled evaluation method {task.method_id}.")
            # Normalize score
            normalized_score = normalize_score(
                raw_score, task.normalization_id, task.normalization_kwargs
            )

            # Apply renormalized weight if necessary
            task_weight = task.weight / total_weight if not math.isclose(total_weight, 1.0) and total_weight > 0 else task.weight
            weighted_norm_score = normalized_score * task_weight

            score += weighted_norm_score
            score_details[task.name] = ScoreDetails(
                raw_score=raw_score,
                norm_score=normalized_score,
                weighted_norm_score=weighted_norm_score,
            )

    return score, score_details
