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
    
    VERIFIABLE_REASONING = 5


def check_for_reasonable_output(
    model, input1: torch.Tensor, input2: torch.Tensor, pad_token_id: int
) -> bool:
    """
    Performs basic validation that the model can generate reasonable output.
    """
    try:
        # Ensure input tensors have the correct shape [batch_size, sequence_length]
        if input1.dim() == 1:
            input1 = input1.unsqueeze(0)  # Add batch dimension
        if input2.dim() == 1:
            input2 = input2.unsqueeze(0)  # Add batch dimension
            
        # Check reasonable output on first input
        generate_id1s = model.generate(
            input1,
            max_new_tokens=5,
            pad_token_id=pad_token_id,
            do_sample=False,
        )
        
        # Check reasonable output on second input
        generate_id2s = model.generate(
            input2,
            max_new_tokens=5,
            pad_token_id=pad_token_id,
            do_sample=False,
        )
        
        # Check shapes
        if generate_id1s.shape[0] != input1.shape[0] or generate_id2s.shape[0] != input2.shape[0]:
            logging.error(f"Model generate output shape mismatch: {generate_id1s.shape} vs {input1.shape}")
            return False
            
        # Check that generated outputs differ between inputs
        return not torch.equal(generate_id1s, generate_id2s)
    except Exception as e:
        logging.error(f"Exception in check_for_reasonable_output: {str(e)}")
        return False


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
    """Generate text from the model and decode it.
    
    Args:
        model: The model to generate from
        input_ids: Input token IDs
        generation_config: Configuration for generation
        device: Device to run on
        tokenizer: Tokenizer for decoding
        
    Returns:
        str: Generated text
    """
    try:
        # Ensure input has batch dimension [batch_size, sequence_length]
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            
        # Generate output
        output = model.generate(
            input_ids,
            **generation_config.to_dict(),
        )
        
        # Remove batch dimension if present
        if output.dim() > 1:
            # Take the first (and possibly only) sequence in the batch
            output = output[0]
            
        # Decode the output
        return tokenizer.decode(output, skip_special_tokens=True)
    except Exception as e:
        logging.error(f"Error in generate_output: {str(e)}")
        return ""


def compute_similarity_score(a: str, b: str) -> float:
    """Returns a similarity score between [0,1] for the two provided strings."""
    # Use difflib to compute the similarity score, ignoring whitespace deltas.
    return difflib.SequenceMatcher(lambda x: x in " \t", a, b).ratio()


def compute_verifiable_reasoning(
    model: PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    generation_config: transformers.GenerationConfig,
    batches: typing.Dict[str, typing.List],
    device: str,
    trace_weight: float = 0.4,
    answer_weight: float = 0.6,
) -> float:
    """Computes a combined score based on reasoning trace perplexity and exact answer matching."""
    # Validate input batches
    required_keys = ["questions", "traces", "answers", "task_types"]
    for key in required_keys:
        if key not in batches or not batches[key]:
            logging.error(f"Missing required key in batches: {key}")
            return math.inf
    
    if len(batches["questions"]) < 2:  # Need at least 2 for reasonable output check
        logging.error("Need at least 2 samples for evaluation")
        return math.inf
        
    # # Check that model generates reasonable output
    # try:
    #     prompt_length = min(100, batches["questions"][0].shape[0], batches["questions"][1].shape[0])
        
    #     # Extract prompt portion and ensure correct shape
    #     token_inputs_1 = torch.tensor(batches["questions"][0][:prompt_length]).to(device)
    #     token_inputs_2 = torch.tensor(batches["questions"][1][:prompt_length]).to(device)
        
    #     if not check_for_reasonable_output(
    #         model, token_inputs_1, token_inputs_2, tokenizer.pad_token_id
    #     ):
    #         logging.error("Model did not generate reasonable output")
    #         return math.inf
    # except Exception as e:
    #     logging.error(f"Exception in reasonable output check: {traceback.format_exc()}")
    #     return math.inf
    
    # Process each question-trace-answer triplet
    trace_losses = []
    answer_scores = []
    
    for i in range(len(batches["questions"])):
        try:
            # 1. Generate model response from question
            question_tensor = torch.tensor(batches["questions"][i]).to(device)
            
            # Ensure question tensor has right shape
            if question_tensor.dim() == 1:
                question_tensor = question_tensor.unsqueeze(0)
            
            response = generate_output(
                model=model,
                input_ids=question_tensor,
                generation_config=generation_config,
                device=device,
                tokenizer=tokenizer,
            )
            
            # 2. Calculate perplexity on the trace
            with torch.no_grad():
                trace_tensor = torch.tensor(batches["traces"][i]).to(device)
                
                # Ensure trace tensor has batch dimension
                if trace_tensor.dim() == 1:
                    trace_tensor = trace_tensor.unsqueeze(0)
                
                # Create attention mask (1 for tokens, 0 for padding)
                attention_mask = (trace_tensor != tokenizer.pad_token_id).long()
                
                # Forward pass with the trace
                outputs = model(trace_tensor, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Remove batch dimension for loss calculation
                if logits.dim() > 2:
                    logits = logits.squeeze(0)
                if trace_tensor.dim() > 1:
                    trace_tensor = trace_tensor.squeeze(0)
                    attention_mask = attention_mask.squeeze(0)
                
                # Shift for next-token prediction loss
                shift_logits = logits[:-1, :].contiguous()
                shift_labels = trace_tensor[1:].contiguous()
                shift_attention_mask = attention_mask[1:].contiguous()
                
                # Only compute loss on non-padding tokens
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                
                # Apply mask to ignore padding tokens
                masked_loss = loss * shift_attention_mask.view(-1)
                # Compute mean loss only on non-padding tokens
                denom = shift_attention_mask.sum().item()
                if denom > 0:
                    trace_loss = masked_loss.sum().item() / denom
                else:
                    trace_loss = 0.0
                trace_losses.append(trace_loss)
            
            # 3. Extract and check answer
            task_type = batches["task_types"][i]
            expected_answer = batches["answers"][i]
            
            if task_type == "verifiable_math":
                # Extract answer from \boxed{}
                boxed_matches = re.findall(r"\\boxed\{(.*?)\}", response)
                model_answer = boxed_matches[-1] if boxed_matches else ""
                # Normalize answer for comparison
                model_answer = model_answer.strip()
                expected_answer = expected_answer.strip()
            elif task_type == "code_output_prediction":
                # Extract answer from {"output": "..."}
                output_matches = re.findall(r'{"output": "(.*?)"}', response)
                model_answer = output_matches[-1] if output_matches else ""
                # Normalize answer for comparison
                model_answer = model_answer.strip()
                expected_answer = expected_answer.strip()
            else:
                model_answer = ""
                
            # 4. Check exact match
            is_correct = model_answer == expected_answer
            answer_scores.append(0.0 if is_correct else 1.0)  # 0 for correct, 1 for incorrect
            
        except Exception as e:
            logging.error(f"Exception in reasoning evaluation: {traceback.format_exc()}")
            trace_losses.append(math.inf)
            answer_scores.append(1.0)
    
    # Calculate weighted score
    if not trace_losses or not answer_scores:
        return math.inf
        
    avg_trace_loss = sum(trace_losses) / len(trace_losses)
    avg_answer_score = sum(answer_scores) / len(answer_scores)
    
    # Normalize trace loss to [0, 1] range using exponential normalization
    # This uses the same approach as the INVERSE_EXPONENTIAL normalization
    ceiling = 20.0  # Same as used in the current REASONING_3B competition
    normalized_trace_loss = 1.0 - math.exp(-avg_trace_loss / ceiling)
    
    # Combine scores (lower is better for both components)
    combined_score = (trace_weight * normalized_trace_loss) + (answer_weight * avg_answer_score)
    
    return combined_score
