import abc
from typing import List

from transformers import PreTrainedTokenizerBase

from finetune.eval.sample import EvalSample


class DatasetLoader(abc.ABC):
    """Base class for dataset loaders."""

    @abc.abstractmethod
    def tokenize(
        self, tokenizer: PreTrainedTokenizerBase, sequence_length: int
    ) -> List[EvalSample]:
        pass

    @abc.abstractmethod
    def __iter__(self):
        pass

    @abc.abstractmethod
    def __len__(self):
        pass
