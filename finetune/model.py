from taoverse.model.competition.data import ModelConstraints
from transformers import AutoTokenizer, PreTrainedTokenizer


def load_tokenizer(
    model_constraints: ModelConstraints, cache_dir: str = None
) -> PreTrainedTokenizer:
    """Returns the fixed tokenizer for the given model constraints."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_constraints.tokenizer, cache_dir=cache_dir
    )
    # Overwrite the pad token id to eos token if it doesn't exist.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer
