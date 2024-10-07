# The MIT License (MIT)

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
import random
import typing

import torch
from transformers import PreTrainedTokenizerBase

# Characters to use in the dycks.
DYCK_CHARACTER_PAIRS = [("<", ">"), ("[", "]"), ("{", "}"), ("(", ")")]
DYCK_ENDING_CHARS = [x[1] for x in DYCK_CHARACTER_PAIRS]
DYCK_CHALLENGE_PROMPT = "Complete the rest of the sequence, making sure that the parentheses are closed properly. Input: "


def generate_dyck(
    dyck_character_pairs: typing.List[typing.Tuple[str, str]],
    pair_count: int,
    ending_close_count: int,
) -> str:
    """Generate a Dyck word (a balanced string of brackets).

    Args:
        dyck_character_pairs (int): Pairs of open/close characters to use.
        pair_count (int): Number of pairs to generate when creating the Dyck word.
        ending_close_count (int): Number of close characters this Dyck word must end with.

    Returns:
        str: A randomly generated Dyck word.
    """
    open_count = 0
    remaining_closes = pair_count
    unused_close_characters_stack = []
    dyck_word = ""

    # Generate the word.
    for _ in range(pair_count * 2):
        # Randomly choose to open or close unless we have nothing open or already have every pair opened.
        if open_count < pair_count and (
            len(unused_close_characters_stack) == 0
            or (remaining_closes == ending_close_count)
            or (random.choice([True, False]))
        ):
            # Pick a random character pair
            dyck_char_pair = random.choice(dyck_character_pairs)
            # Add the open character to the word now.
            dyck_word += dyck_char_pair[0]
            open_count += 1
            # Add the close character to the to be used close character list
            unused_close_characters_stack.append(dyck_char_pair[1])
        else:
            # Pop off the stack
            close_char = unused_close_characters_stack.pop()
            dyck_word += close_char
            remaining_closes -= 1

    return dyck_word


class DyckLoader:
    def __init__(
        self,
        dyck_character_pairs: typing.List[
            typing.Tuple[str, str]
        ] = DYCK_CHARACTER_PAIRS,
        min_length_pairs: int = 2,
        max_length_pairs: int = 20,
        min_length_answer: int = 1,
        max_length_answer: int = 3,
        random_seed: typing.Optional[int] = None,
        samples: int = 100,
    ):
        """Loads prompt/response data from generated dyck words.

        Args:
            dyck_character_pairs (typing.List[ typing.Tuple[str, str] ], optional): Pairs of open/close characters to use. Defaults to DYCK_CHARACTER_PAIRS.
            min_length_pairs (int, optional): Minimum number of pairs to create the dyck word from. Defaults to 2.
            max_length_pairs (int, optional): Maximum number of pairs to create the dyck word from. Defaults to 20.
            min_length_answer (int, optional): Minimum length of the reference answer. Defaults to 1.
            max_length_answer (int, optional): Maximum length of the reference answer. Defaults to 3.
            random_seed (typing.Optional[int], optional): Seed to use for all random operations if set.
            samples (int, optional): Number of samples to generate. Defaults to 100.
        """
        if min_length_answer > max_length_answer:
            raise ValueError(
                "min_length_answer: {min_length_answer} is greater than max_length_answer {max_length_answer}."
            )

        if min_length_pairs > max_length_pairs:
            raise ValueError(
                "min_length_pairs: {min_length_pairs} is greater than max_length_pairs {max_length_pairs}."
            )

        if max_length_answer > max_length_pairs:
            raise ValueError(
                "max_length_answer: {max_length_answer} is greater than max_length_pairs {max_length_pairs}."
            )

        self.buffer: typing.List[typing.Tuple[str, str]] = []

        if random_seed is not None:
            random.seed(random_seed)

        for _ in range(samples):
            # Decide the length the reference should be.
            reference_length = random.randint(min_length_answer, max_length_answer)
            # We need at least as many pairs as the reference length or the min length.
            adjusted_min_length_pairs = max(reference_length, min_length_pairs)
            # Decide total number of pairs.
            pair_count = random.randint(adjusted_min_length_pairs, max_length_pairs)

            # Generate a dyck of the minimum pair length.
            full_dyck = generate_dyck(
                dyck_character_pairs, pair_count, reference_length
            )

            # Generate the reference answer by completing the now incomplete dyck.
            reference = " ".join(full_dyck[-reference_length:])

            # Generate the challenge by prepending the prompt to the now incomplete dyck.
            challenge = DYCK_CHALLENGE_PROMPT + " ".join(full_dyck[:-reference_length])

            self.buffer.append((challenge, reference))

    def tokenize(
        self, tokenizer: PreTrainedTokenizerBase, sequence_length: int
    ) -> typing.List[typing.Tuple[torch.Tensor, int]]:
        # Each batch is a tokenized question + reference answer.
        batches = []
        # If truncation is necessary, truncate from the left to avoid cutting off the answer part.
        tokenizer.truncation_side = "left"

        for challenge, reference in self:
            conversation = [
                {"role": "user", "content": challenge},
            ]
            ids = tokenizer.apply_chat_template(
                conversation,
                truncation=True,
                max_length=sequence_length,
                add_generation_prompt=True,
            )

            batches.append(
                (
                    torch.stack([torch.tensor(ids)]),
                    reference,
                )
            )
        return batches

    def get_sample(self) -> typing.Tuple[str, str]:
        return random.choice(self.buffer)

    def __iter__(self):
        return self.buffer.__iter__()

    def __len__(self):
        return len(self.buffer)
