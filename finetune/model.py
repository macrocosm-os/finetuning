from taoverse.model.competition.data import ModelConstraints
from transformers import AutoTokenizer, PreTrainedTokenizer


def load_tokenizer(
    model_constraints: ModelConstraints, cache_dir: str = None
) -> PreTrainedTokenizer:
    """Returns the fixed tokenizer for the given model constraints."""
    return AutoTokenizer.from_pretrained(
        model_constraints.tokenizer, cache_dir=cache_dir
    )
