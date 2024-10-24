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

import nltk
import torch
from transformers import PreTrainedTokenizerBase

try:
    from nltk.corpus import words
except:
    nltk.download("words", raise_on_error=True)

WORD_SORTING_CHALLENGE_PROMPT = "Sort the following words alphabetically: "


class WordSortingLoader:
    def __init__(
        self,
        min_word_count: int = 2,
        max_word_count: int = 20,
        min_word_length: int = 3,
        random_seed: typing.Optional[int] = None,
        samples: int = 100,
    ):
        """Loads prompt/response data from generated word sorting tasks.

        Args:
            min_word_count (int, optional): Minimum number of words to generate. Defaults to 2.
            max_word_count (int, optional): Maximum number of words to generate. Defaults to 20.
            max_word_length (int, optional): Minimum length of the words to use. Defaults to 3.
            random_seed (typing.Optional[int], optional): Seed to use for all random operations if set.
            samples (int, optional): Number of samples to generate. Defaults to 100.
        """
        if min_word_count > max_word_count:
            raise ValueError(
                "min_word_count: {min_word_count} is greater than max_word_count {max_word_count}."
            )

        self.buffer: typing.List[typing.Tuple[str, str]] = []

        # Only take words from the corpus of min length or greater.
        # en-basic: 850 English words: C.K. Ogden in The ABC of Basic English (1932)
        self.words = [
            w.lower()  # These should all already be lowercase, but also lower() to be sure.
            for w in words.words(fileids=["en-basic"])
            if len(w) >= min_word_length
        ]

        if random_seed is not None:
            random.seed(random_seed)

        for _ in range(samples):
            # Decide the number of words to use.
            word_count = random.randint(min_word_count, max_word_count)

            # Generate that number of words.
            word_list_unsorted = random.sample(self.words, word_count)

            # Prepend the prompt to the challenge.
            challenge = (
                WORD_SORTING_CHALLENGE_PROMPT + " ".join(word_list_unsorted) + ". "
            )

            # Sort the list for the reference.
            reference = " ".join(sorted(word_list_unsorted))

            self.buffer.append((challenge, reference))

    def tokenize(
        self, tokenizer: PreTrainedTokenizerBase, sequence_length: int
    ) -> typing.List[typing.Tuple[torch.Tensor, torch.Tensor]]:
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
            ref_ids = tokenizer.encode(reference)

            batches.append(
                (
                    torch.tensor(ids),
                    torch.tensor(ref_ids),
                )
            )
        return batches

    def get_sample(self) -> typing.Tuple[str, str]:
        return random.choice(self.buffer)

    def __iter__(self):
        return self.buffer.__iter__()

    def __len__(self):
        return len(self.buffer)
