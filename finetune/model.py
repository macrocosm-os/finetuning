from transformers import AutoTokenizer, PreTrainedTokenizer

from competitions.data import Competition


def load_tokenizer(
    competition: Competition, cache_dir: str = None
) -> PreTrainedTokenizer:
    """Returns the fixed tokenizer for the given competition."""
    return AutoTokenizer.from_pretrained(
        competition.constraints.tokenizer, cache_dir=cache_dir
    )
