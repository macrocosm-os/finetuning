import unittest

from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedModel


def get_test_model() -> PreTrainedModel:
    """Gets a test model that is small enough to load and store quickly.

    Returns:
        PreTrainedModel: Tiny test model.
    """
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=2000,
            hidden_size=256,
            intermediate_size=688,
            num_hidden_layers=2,
            num_attention_heads=2,
        )
    )
