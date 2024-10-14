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

import math
import re
import traceback
import typing

import bittensor as bt
import torch
import transformers
from taoverse.model.competition.epsilon import EpsilonFunc
from transformers import GenerationConfig
from taoverse.model.competition.data import Competition


from finetune.eval.task import EvalMethodId, EvalTask, NormalizationId


def iswin(
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
    score_per_uid: typing.Dict[int, float],
    uid_to_block: typing.Dict[int, int],
    epsilon_func: EpsilonFunc,
    current_block: int,
) -> typing.Tuple[typing.Dict[int, int], typing.Dict[int, float]]:
    """
    Computes the wins and win rate for each model based on loss comparison.

    Parameters:
        uids (list): A list of uids to compare.
        score_per_uid (dict): A dictionary of score for each uid
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
            loss_i = score_per_uid[uid_i]
            loss_j = score_per_uid[uid_j]
            wins[uid_i] += (
                1
                if iswin(
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


def compute_losses(
    model, batches: typing.List[typing.Tuple[torch.Tensor, int]], device: str
) -> typing.List[float]:
    """
    Computes the losses for a given model on provided batches.

    Parameters:
        model (torch.nn.Module): The model for which losses are to be computed.
        batches (dict): A list of batches and the associated lengths of the "prompt" section.
        device (str): The device to use for computation (e.g., 'cpu', 'gpu').

    Returns:
        dict: A dictionary with page indices as keys and lists of loss values as values.
    """
    # Iterate over each page and corresponding batches
    losses = []
    with torch.inference_mode():
        model.to(device)
        model.eval()
        for inputs, prompt_len in batches:
            try:
                inputs = inputs.to(device)
                labels = inputs.clone()
                labels[:, :prompt_len] = -100  # Only calculate loss on response
                outputs = model(inputs, labels=labels)
                loss = outputs.loss.item()  # Extract scalar loss value
                losses.append(loss)
            except Exception as e:
                bt.logging.error(f"Exception occurred in loss computation: {e}")
                traceback.print_exc()  # Print the stack trace
                losses.append(math.inf)  # Use infinity to indicate failure

    return losses


def compute_multiple_choice_deviation(
    model,
    tokenizer: transformers.PreTrainedTokenizer,
    generation_config: transformers.GenerationConfig,
    batches: typing.List[typing.Tuple[torch.Tensor, typing.List[str], str]],
    device: str,
) -> float:
    """
    Computes the incorrectness of multiple choice answers for a given model on provided batches.

    Parameters:
        model (torch.nn.Module): The model for which multiple choice deviations are to be computed.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to tokenize the output with before returning.
        generation_config (transformers.GenerationConfig): Configuration parameters for generating output.
        batches (dict): A list of batches, choices, and the correct answer.
        device (str): The device to use for computation (e.g., 'cpu', 'gpu').

    Returns:
        dict: A dictionary with page indices as keys and lists of multiple choice deviations as values.
    """
    # Iterate over each page and corresponding batches
    multiple_choice_deviations = []

    for (
        inputs,
        choices,
        answer,
    ) in batches:
        try:
            response = generate_output(
                model=model,
                input_ids=inputs,
                generation_config=generation_config,
                device=device,
                tokenizer=tokenizer,
            )

            # Find words which match one of the choices.
            matches = [
                word for word in re.sub(r"\W", " ", response).split() if word in choices
            ]

            # Give credit if the first matched word in the response is correct.
            if matches and matches[0] == answer:
                multiple_choice_deviations.append(0)
            else:
                multiple_choice_deviations.append(1)
        except Exception as e:
            bt.logging.error(
                f"Exception occurred in multiple choice deviation computation: {e}"
            )
            traceback.print_exc()  # Print the stack trace
            multiple_choice_deviations.append(1)  # Use 1 to indicate failure

    # For multiple choice, return a single deviation, which is the ratio of incorrect answers.
    return (
        sum(multiple_choice_deviations) / len(multiple_choice_deviations)
        if multiple_choice_deviations
        else 1
    )


def generate_output(
    model,
    input_ids: torch.Tensor,
    generation_config: transformers.GenerationConfig,
    device: str,
    tokenizer: transformers.PreTrainedTokenizer,
) -> str:
    """
    Generates the tokenized output for a model given a tokenized input and generation config.

    Args:
        model (torch.nn.Module): The model for which losses are to be computed.
        input_ids (torch.Tensor): Input tokens to generate a response to.
        generation_config (transformers.GenerationConfig): Configuration parameters for generating output.
        device (str): The device to use for computation (e.g., 'cpu', 'gpu').
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to tokenize the output with before returning.

    Returns:
        str: Generated tokenized output from the model.
    """
    input_ids = input_ids.to(device)
    output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
    )
    response = tokenizer.decode(
        output[0][len(input_ids[0]) :], skip_special_tokens=True
    )
    return response


def score_model(
    model,
    tokenizer: transformers.PreTrainedTokenizer,
    evals: typing.List[EvalTask],
    competition: Competition,
    device: str,
) -> typing.Tuple[float, dict]:
    """Scores a model based on the provided eval tasks."""

    with torch.inference_mode():
        model.to(device)
        model.eval()

        score = 0
        score_details = {}

        for eval in evals:
            match eval.method:
                case EvalMethodId.MULTIPLE_CHOICE:
                    compute_generation_config = GenerationConfig(
                        max_new_tokens=20,
                        max_length=competition.constraints.sequence_length,
                        do_sample=False,
                        repetition_penalty=1.1,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    score = compute_multiple_choice_deviation(
                        model=model,
                        tokenizer=tokenizer,
                        generation_config=compute_generation_config,
                        batches=eval.samples,
                        device=device,
                    )

                case _:
                    raise ValueError(f"Unhandled evaluation method {eval.method}.")
            # Normalize score
            normalized_score = normalize_score(
                score, eval.normalization_id, eval.normalization_kwargs
            )
            weighted_norm_score = normalized_score * eval.weight

            score += weighted_norm_score
            score_details[eval.name] = {
                "raw_score": score,
                "norm_score": normalized_score,
                "weighted_norm_score": weighted_norm_score,
            }

    return score, score_details


def normalize_score(
    score: float,
    normalization_id: NormalizationId,
    norm_kwargs: dict,
) -> float:
    match normalization_id:
        case NormalizationId.NONE:
            return _normalize_none(score, norm_kwargs)
        case _:
            raise ValueError(f"Unhandled normalization method {normalization_id}.")


def _normalize_none(score: float, _: dict) -> float:
    return score
