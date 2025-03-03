import difflib
import math
import re
import traceback
import typing
from enum import IntEnum

import taoverse.utilities.logging as logging
import torch
import transformers
from transformers import DynamicCache, PreTrainedModel
import numpy as np

from finetune.eval.if_eval.sample import IFEvalTokenizedSample


class EvalMethodId(IntEnum):
    """Enumeration of evaluation methods."""

    NONE = 0

    # Evaluates the model's performance on multiple choice questions.
    MULTIPLE_CHOICE = 1

    # Evaluates the model's performance on a text generation task by computing average cross entropy loss
    # on the reference portition of the text.
    REFERENCE_LOSS = 2

    # Evalutes the model's performance on a text generation task by computing average cross entropy loss
    # on the entirety of the provided text.
    TEXT_LOSS = 3

    # Evalutes the model's performance on a prompt response task that contains a set of rules that the response
    # must satisfy.
    IF_EVAL = 4


def check_for_reasonable_output(
    model, input1: torch.Tensor, input2: torch.Tensor, pad_token_id: int
) -> bool:
    """Checks that a model generates reasonable outputs for two given inputs.

    Args:
        model (torch.nn.Module): The model for which outputs are to be checked. Already loaded to device.
        input1 (torch.Tensor]): Tokenized input1 to check. Already loaded to device.
        input2 (torch.Tensor]): Tokenized input2 to check. Already loaded to device.
        pad_token_id (int): Pad token id for the tokenizer used to generate inputs 1 and 2.

    Returns:
        bool: If the model generates reasonable outputs.
    """
    # Make sure inputs are on the correct device
    device = next(model.parameters()).device
    input1 = input1.to(device)
    input2 = input2.to(device)
    
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
    batches: typing.List[np.ndarray],
    device: str,
    pad_token_id: int,
) -> float:
    """
    Computes the loss for text generation for a given model on provided batches.

    Parameters:
        model (torch.nn.Module): The model for which text loss is to be computed.
        batches (dict): A dictionary of batches with page indices as keys.
        device (str): The device to use for computation (e.g., 'cpu', 'gpu').
        pad_token_id (int): The id of the pad token.

    Returns:
        float: The average loss across all batches.
    """

    # First check that model generates reasonable looking outputs.
    # Grab 100 tokens from the first two batches as 'prompts'. (1 x Seq Length tensors.)
    if len(batches) >= 2:
        batch_0 = batches[0]
        batch_1 = batches[1]
        # Convert numpy arrays to PyTorch tensors if needed
        if isinstance(batch_0, np.ndarray):
            batch_0 = torch.tensor(batch_0)
        if isinstance(batch_1, np.ndarray):
            batch_1 = torch.tensor(batch_1)
            
        if len(batch_0.shape) == 1:
            batch_0 = batch_0.unsqueeze(0)
        if len(batch_1.shape) == 1:
            batch_1 = batch_1.unsqueeze(0)
            
        input1 = batch_0[0:1, : min(batch_0.shape[1], 100)]
        input2 = batch_1[0:1, : min(batch_1.shape[1], 100)]
        
        device = next(model.parameters()).device
        input1 = input1.to(device)
        input2 = input2.to(device)
        
        if not check_for_reasonable_output(model, input1, input2, pad_token_id):
            logging.warning(
                "Model does not generate reasonable looking outputs. Score will be 0."
            )
            return 0.0

    total_loss = 0.0
    total_tokens = 0

    for batch in batches:
        # Convert numpy array to PyTorch tensor if needed
        if isinstance(batch, np.ndarray):
            batch = torch.tensor(batch)
            
        if len(batch.shape) == 1:
            batch = batch.unsqueeze(0)
            
        batch = batch.to(device)
        tokens_to_keep = (batch != pad_token_id).int().sum().item()
        total_tokens += tokens_to_keep

        try:
            outputs = model(batch, labels=batch)
        except Exception as e:
            logging.warning(f"Error computing loss: {e}")
            # Treat this as a max loss entry.
            return 0.0

        if not isinstance(outputs, typing.Mapping):
            logging.warning(
                f"Model returned non-mapping output type {type(outputs)}. Treating as max loss (0%)."
            )
            return 0.0

        try:
            loss = outputs["loss"]
        except KeyError:
            logging.warning("Model did not return loss. Treating as max loss (0%).")
            return 0.0

        total_loss += loss.item() * tokens_to_keep

    if total_tokens == 0:
        logging.warning("No tokens to compute loss on. Treating as max loss (0%).")
        return 0.0

    return total_loss / total_tokens


def compute_reference_loss(
    model: PreTrainedModel,
    batches: typing.List[typing.Tuple[np.ndarray, np.ndarray]],
    device: str,
) -> float:
    """
    Computes the loss for reference answers for a given model on provided batches.

    Parameters:
        model (torch.nn.Module): The model for which reference loss is to be computed.
        batches (dict): A list of tuples (context, reference) 
        device (str): The device to use for computation (e.g., 'cpu', 'gpu').

    Returns:
        float: The average loss across all batches.
    """
    losses = []
    try:
        with torch.no_grad():
            for (context, ref) in batches:
                try:
                    # Convert numpy arrays to PyTorch tensors if needed
                    if isinstance(context, np.ndarray):
                        context = torch.tensor(context)
                    if isinstance(ref, np.ndarray):
                        ref = torch.tensor(ref)
                        
                    # Context and ref are 1 dimensional tensors.
                    context = context.to(device)
                    ref = ref.to(device)
                    
                    # Create the full input by concatenating context and reference
                    inputs = torch.stack([torch.cat([context, ref])])

                    # Prepare a cache class and pass it to the model's forward.
                    past_key_values = DynamicCache()
                    logits = model(inputs, past_key_values=past_key_values).logits

                    # Shift the logits and labels to compute the loss.
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = inputs[..., 1:].contiguous()

                    # Only take the reference portion.
                    ref_start = context.shape[-1] - 1  # -1 because the shifting

                    ref_shift_logits = shift_logits[:, ref_start:, :]
                    ref_shift_labels = shift_labels[:, ref_start:]

                    # Compute loss.
                    loss_fct = torch.nn.CrossEntropyLoss()
                    ref_shift_logits = ref_shift_logits.view(-1, model.config.vocab_size)
                    ref_shift_labels = ref_shift_labels.view(-1)
                    losses.append(loss_fct(ref_shift_logits, ref_shift_labels).item())
                except Exception as e:
                    logging.warning(f"Error computing reference loss: {e}")
                    return 0.0
    except Exception as e:
        logging.warning(f"Error in reference loss computation: {e}")
        return 0.0
        
    return sum(losses) / len(losses) if losses else 0.0


def compute_multiple_choice_deviation(
    model,
    tokenizer: transformers.PreTrainedTokenizer,
    generation_config: transformers.GenerationConfig,
    batches: typing.List[typing.Tuple[np.ndarray, typing.List[str], str]],
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
            # Convert numpy array to PyTorch tensor if needed
            if isinstance(inputs, np.ndarray):
                inputs = torch.tensor(inputs)

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
    """
    Computes the score for the IfEval task for a given model on provided batches.

    Parameters:
        model (torch.nn.Module): The model for which IfEval score is to be computed.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to tokenize the output with before returning.
        generation_config (transformers.GenerationConfig): Configuration parameters for generating output.
        batches (typing.List[IFEvalTokenizedSample]): A list of batches.
        device (str): The device to use for computation (e.g., 'cpu', 'gpu').

    Returns:
        float: The average score across all batches.
    """
    scores = []
    duplicate_count = 0

    for sample in batches:
        # Convert numpy arrays to PyTorch tensors if needed
        prompt_1 = sample.prompt_1
        prompt_2 = sample.prompt_2
        
        try:
            # Generate outputs for prompt 1 and prompt 2
            response_1 = generate_output(
                model=model,
                input_ids=prompt_1,
                generation_config=generation_config,
                device=device,
                tokenizer=tokenizer,
            )
            response_2 = generate_output(
                model=model,
                input_ids=prompt_2,
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
            logging.warning(f"Error in IF eval computation: {e}")
            for _ in range(len(sample.rules) * 2):
                # Fail all rules in this sample.
                scores.append(1)

    # Penalize models that are generating too many duplicated responses for different prompts.
    if duplicate_count > len(batches) * 0.1:
        logging.info(
            f"Model had too many duplicated responses ({duplicate_count}/{len(batches)}). Setting score to 1."
        )
        return 1

    # Return the % of rules that were not satisfied.
    return sum(scores) / len(scores) if scores else 1


def generate_output(
    model,
    input_ids: typing.Union[torch.Tensor, np.ndarray],
    generation_config: transformers.GenerationConfig,
    device: str,
    tokenizer: transformers.PreTrainedTokenizer,
) -> str:
    """
    Generates the tokenized output for a model given a tokenized input and generation config.

    Args:
        model (torch.nn.Module): The model for which losses are to be computed.
        input_ids (torch.Tensor or np.ndarray): Input tokens to generate a response to.
        generation_config (transformers.GenerationConfig): Configuration parameters for generating output.
        device (str): The device to use for computation (e.g., 'cpu', 'gpu').
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to tokenize the output with before returning.

    Returns:
        str: Generated tokenized output from the model.
    """
    # Convert numpy array to PyTorch tensor if needed
    if isinstance(input_ids, np.ndarray):
        input_ids = torch.tensor(input_ids)
        
    input_ids = input_ids.to(device)
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
