import difflib
import math
import re
import traceback
import typing
from enum import IntEnum

import numpy as np
import taoverse.utilities.logging as logging
import torch
import transformers
from transformers import DynamicCache, PreTrainedModel

from finetune.eval.if_eval.sample import IFEvalTokenizedSample


class EvalMethodId(IntEnum):
    """Enumeration of evaluation methods."""

    NONE = 0

    # Evaluates the model's performance on multiple choice questions.
    MULTIPLE_CHOICE = 1

    # Evaluates the model's performance on a text generation task by computing average cross entropy loss
    # on the reference portition of the text.
    REFERENCE_LOSS = 2

    # Evaluates the model's performance on a text generation task by computing average cross entropy loss
    # on the entirety of the provided text.
    TEXT_LOSS = 3

    # Evaluates the model's performance on a prompt response task that contains a set of rules that the response
    # must satisfy.
    IF_EVAL = 4


def check_for_reasonable_output(
    model, input1: torch.Tensor, input2: torch.Tensor, pad_token_id: int
) -> bool:
    """Checks that a model generates reasonable outputs for two given inputs.

    Args:
        model (torch.nn.Module): The model for which outputs are to be checked. Already loaded to device.
        input1 (np.array): Tokenized input1 to check. Already loaded to device.
        input2 (np.array): Tokenized input2 to check. Already loaded to device.
        pad_token_id (int): Pad token id for the tokenizer used to generate inputs 1 and 2.

    Returns:
        bool: True if the model generates reasonable outputs, False otherwise.
    """
    # Generate 20 tokens of output from the model for each prompt.
    output_length = 20
    # Only take the last 20 tokens since otherwise we also get the prompt ids.
    generate_id1s = model.generate(
        input1,
        min_new_tokens=output_length,
        max_new_tokens=output_length,
        pad_token_id=pad_token_id,
    )[:, -output_length:]
    generate_id2s = model.generate(
        input2,
        min_new_tokens=output_length,
        max_new_tokens=output_length,
        pad_token_id=pad_token_id,
    )[:, -output_length:]

    # Check if too many of the generated ids are the same between the two outputs.
    if torch.sum(torch.eq(generate_id1s, generate_id2s)).item() >= output_length / 2:
        logging.info(f"Model had too much overlap between generated outputs.")
        return False

    # Check if internally both responses are too repetitive.
    most_common_counts = []
    for tensor in [generate_id1s, generate_id2s]:
        # Find unique elements and their counts
        _, counts = torch.unique(tensor, return_counts=True)
        # Find the index of the maximum count
        max_count_index = torch.argmax(counts)
        # Extract the count of the most common element
        most_common_counts.append(counts[max_count_index].item())

    if all(count > output_length / 2 for count in most_common_counts):
        logging.info(
            f"Model with config {model.config} had too much repetition in generated outputs."
        )
        return False

    # Passed all the checks, return True.
    return True


def compute_text_loss(
    model: PreTrainedModel,
    batches: typing.List[np.array],
    device: str,
    pad_token_id: int,
) -> float:
    """Computes the losses for a given model on provided text batches.

    Args:
        model (PreTrainedModel): The model to eval
        batches (typing.List[np.array]): List of tokenized texts.
        device (str): The device to run the evaluation on.
        pad_token_id int: Pad token id for the tokenizer used to tokenize the batches.

    Returns:
        float: The average loss across all batches.
    """
    # First check that model generates reasonable looking outputs.
    # Grab 100 tokens from the first two batches as 'prompts'. (1 x Seq Length tensors.)
    try:
        prompt_length = 100
        token_inputs_1 = torch.tensor(batches[0][:prompt_length]).to(device)
        token_inputs_2 = torch.tensor(batches[1][:prompt_length]).to(device)

        if not check_for_reasonable_output(
            model, token_inputs_1, token_inputs_2, pad_token_id
        ):
            return math.inf
    except Exception as e:
        logging.error(
            f"Exception occurred in checking for reasonable output: {traceback.format_exc()}"
        )
        return math.inf

    # Everything looks good! Continue to computing actual losses.

    losses = []
    with torch.no_grad():
        for batch in batches:
            try:
                # Convert numpy array to torch tensor first, then move to device
                inputs = torch.tensor(batch).to(device)

                # Prepare a cache class and pass it to the model's forward.
                past_key_values = DynamicCache()
                logits = model(inputs, past_key_values=past_key_values).logits

                # Shift the logits and labels to compute the loss.
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = inputs[..., 1:].contiguous()

                # Compute loss.
                loss_fct = torch.nn.CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, model.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                losses.append(loss_fct(shift_logits, shift_labels).item())
            except Exception as e:
                logging.error(
                    f"Exception occurred in reference loss computation: {traceback.format_exc()}"
                )
                return math.inf
    return sum(losses) / len(losses) if losses else math.inf


def compute_reference_loss(
    model: PreTrainedModel,
    batches: typing.List[typing.Tuple[np.array, np.array]],
    device: str,
) -> float:
    """Given batches of [context, ref] pairs, computes the average loss on the reference portion.

    Args:
        model (PreTrainedModel): The model to eval
        batches (typing.List[typing.Tuple[np.array, np.array]]): List of [context, ref] pairs
        device (str): The device to run the evaluation on.

    Returns:
        float: The average loss across all batches.
    """
    losses = []
    with torch.no_grad():
        for context, ref in batches:
            try:
                # Convert numpy arrays to tensors
                context_tensor = torch.tensor(context)
                ref_tensor = torch.tensor(ref)

                # Check if context already has a batch dimension
                if len(context_tensor.shape) == 1:
                    # Add batch dimension if it doesn't exist
                    inputs = (
                        torch.tensor(np.concatenate([context, ref]))
                        .unsqueeze(0)
                        .to(device)
                    )
                else:
                    # If it already has a batch dimension, just concatenate along sequence dimension
                    inputs = torch.cat([context_tensor, ref_tensor], dim=1).to(device)

                # Prepare a cache class and pass it to the model's forward
                past_key_values = DynamicCache()
                logits = model(inputs, past_key_values=past_key_values).logits

                # Shift the logits and labels to compute the loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = inputs[..., 1:].contiguous()

                # Only take the reference portion
                ref_length = ref_tensor.size(-1)  # Use the last dimension size
                ref_logits = shift_logits[..., -ref_length:, :]
                ref_labels = shift_labels[..., -ref_length:]

                # Compute loss
                loss_fct = torch.nn.CrossEntropyLoss()
                ref_logits = ref_logits.view(-1, model.config.vocab_size)
                ref_labels = ref_labels.view(-1)
                losses.append(loss_fct(ref_logits, ref_labels).item())
            except Exception as e:
                logging.error(
                    f"Exception occurred in reference loss computation: {traceback.format_exc(e)}"
                )
                return math.inf
    return sum(losses) / len(losses) if losses else math.inf


def compute_multiple_choice_deviation(
    model,
    tokenizer: transformers.PreTrainedTokenizer,
    generation_config: transformers.GenerationConfig,
    batches: typing.List[typing.Tuple[np.array, typing.List[str], str]],
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
            inputs = torch.tensor(inputs).to(device)
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
            logging.error(
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


def compute_if_eval(
    model: PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    generation_config: transformers.GenerationConfig,
    batches: typing.List[IFEvalTokenizedSample],
    device: str,
) -> float:
    """Computes the IFEval score for a given model on provided batches.

    Args:
        model (PreTrainedModel): The model for which losses are to be computed.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to tokenize the output with before returning.
        generation_config (transformers.GenerationConfig): Configuration parameters for generating output.
        batches (list): A list of batches containing prompts and rules.
        device (str): The device to use for computation (e.g., 'cpu', 'gpu')."""
    scores = []
    duplicate_count = 0

    for sample in batches:
        try:
            # Convert NumPy arrays to PyTorch tensors and send to device
            prompt_1_tensor = torch.tensor(sample.prompt_1).to(device)
            prompt_2_tensor = torch.tensor(sample.prompt_2).to(device)

            response_1 = generate_output(
                model=model,
                input_ids=prompt_1_tensor,
                generation_config=generation_config,
                device=device,
                tokenizer=tokenizer,
            )
            response_2 = generate_output(
                model=model,
                input_ids=prompt_2_tensor,
                generation_config=generation_config,
                device=device,
                tokenizer=tokenizer,
            )

            if compute_similarity_score(response_1, response_2) > 0.6:
                duplicate_count += 1

            for index, response in enumerate([response_1, response_2]):
                correct = 0
                for rule in sample.rules:
                    if rule.matches(response, index):
                        correct += 1

                # The fraction correct is squared to give more reward to getting more/all of the rules correct.
                correct_ratio = correct / (len(sample.rules))
                response_score = 1 - (correct_ratio**2)

                # Append this response score one time per rule to weight scores linearly with the number of rules.
                scores.extend([response_score] * len(sample.rules))

        except Exception as e:
            logging.error(
                f"Exception occurred in multiple choice deviation computation: {e}"
            )
            traceback.print_exc()
            for _ in range(len(sample.rules) * 2):
                # Fail all rules in this sample.
                scores.append(1)

    # Penalize models that are generating too many duplicated responses for different prompts.
    if duplicate_count > len(batches) * 0.1:
        logging.trace(
            f"Model had too many duplicated responses ({duplicate_count}/{len(batches)}). Setting score to 1."
        )
        return 1

    # Return the % of rules that were not satisfied.
    return sum(scores) / len(scores) if scores else 1


def generate_output(
    model,
    input_ids,
    generation_config: transformers.GenerationConfig,
    device: str,
    tokenizer: transformers.PreTrainedTokenizer,
) -> str:
    """
    Generates the tokenized output for a model given a tokenized input and generation config.

    Args:
        model (torch.nn.Module): The model for which losses are to be computed.
        input_ids: Input tokens to generate a response to (torch.Tensor).
        generation_config (transformers.GenerationConfig): Configuration parameters for generating output.
        device (str): The device to use for computation (e.g., 'cpu', 'gpu').
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to tokenize the output with before returning.

    Returns:
        str: Generated tokenized output from the model.
    """
    # Make sure tensor is on the correct device
    target_device = torch.device(device)
    if input_ids.device != target_device:
        input_ids = input_ids.to(target_device)

    output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
    )
    response = tokenizer.decode(
        output[0][len(input_ids[0]) :], skip_special_tokens=True
    )
    return response


def compute_similarity_score(a: str, b: str) -> float:
    """Returns a similarity score between [0,1] for the two provided strings."""
    # Use difflib to compute the similarity score, ignoring whitespace deltas.
    return difflib.SequenceMatcher(lambda x: x in " \t", a, b).ratio()


