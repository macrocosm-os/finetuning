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
import random
import typing

import torch
from transformers import PreTrainedTokenizerBase

# Characters to use in the dycks.
DYCK_CHARACTER_PAIRS = [("<", ">"), ("[", "]"), ("{", "}"), ("(", ")")]
DYCK_CHALLENGE_PROMPT = "Complete the rest of the sequence, making sure that the parentheses are closed properly. Input: "


class DyckLoader:
    def _generate_dyck(
        self, dyck_character_pairs: typing.List[typing.Tuple[str, str]], pair_count: int
    ) -> str:
        """Generate a Dyck word (a balanced string of brackets).

        Args:
            dyck_character_pairs (int): Pairs of open/close characters to use.
            pair_count (int): Number of pairs to generate when creating the Dyck word.

        Returns:
            str: A randomly generated Dyck word.
        """
        open_count = [0] * len(dyck_character_pairs)
        close_count = [0] * len(dyck_character_pairs)
        dyck_word = ""

        # Generate the word.
        for _ in range(pair_count * 2):
            # Randomly choose to open or close unless we have nothing open or already have every pair opened.
            if sum(open_count) - sum(close_count) == 0 or (
                sum(open_count) < pair_count and random.choice([True, False])
            ):
                index = random.randint(0, len(dyck_character_pairs) - 1)
                dyck_word += dyck_character_pairs[index][0]
                open_count[index] += 1
            else:
                remaining_close = [a - b for a, b in zip(open_count, close_count)]
                # Generate weights based on the values in the array.
                weights = weights = [
                    value / sum(remaining_close) for value in remaining_close
                ]
                # Generate the indexes to choose from.
                indexes = list(range(len(dyck_character_pairs)))
                # Choose a random index based on the weights (generating 1 choice and taking the first).
                index = random.choices(indexes, weights=weights, k=1)[0]
                dyck_word += dyck_character_pairs[index][1]
                close_count[index] += 1

        return dyck_word

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
        self.buffer: typing.List[typing.Tuple[str, str]] = []

        if random_seed is not None:
            random.seed(random_seed)

        for _ in range(samples):
            pair_count = random.randint(min_length_pairs, max_length_pairs)
            complete_dyck = self._generate_dyck(dyck_character_pairs, pair_count)

            reference_length = random.randint(min_length_answer, max_length_answer)
            # Ensure we don't make the reference longer than the number of pairs.
            if reference_length > pair_count:
                reference_length = pair_count

            # Randomly remove close characters into the reference answer up to length.
            # Get index of all the close characters.
            close_chars = {pair[1] for pair in DYCK_CHARACTER_PAIRS}
            close_char_indexes = []
            for index, char in enumerate(complete_dyck):
                if char in close_chars:
                    close_char_indexes.append(index)

            chosen_indexes = random.sample(close_char_indexes, reference_length)
            # Sort the chosen indexes so the reference answer has them in the right order.
            chosen_indexes.sort()

            # Generate the reference answer by getting the characters at each ordered index.
            reference = " ".join(
                char for i, char in enumerate(complete_dyck) if i in chosen_indexes
            )

            # Generate the challenge by prepending the prompt and removing the reference characters.
            challenge = DYCK_CHALLENGE_PROMPT + " ".join(
                char for i, char in enumerate(complete_dyck) if i not in chosen_indexes
            )

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
        return self.buffer[random.randint(0, len(self.buffer))]

    def __iter__(self):
        return self.buffer.__iter__()

    def __len__(self):
        return len(self.buffer)
