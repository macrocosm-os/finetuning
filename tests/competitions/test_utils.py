import unittest

import torch
from transformers import (
    BartForCausalLM,
    FalconForCausalLM,
    GemmaForCausalLM,
    GPTNeoXForCausalLM,
    LlamaForCausalLM,
    MistralForCausalLM,
    PhiForCausalLM,
)

from competitions.data import Competition, CompetitionId, ModelConstraints
from competitions.utils import (
    get_model_constraints,
    get_competition_for_block,
    get_competition_schedule_for_block,
)

unittest.util._MAX_LENGTH = 2000


class TestUtils(unittest.TestCase):
    def test_get_model_constraints_valid_competition(self):
        expected_constraints = ModelConstraints(
            max_model_parameter_size=6_900_000_000,
            sequence_length=4096,
            allowed_architectures=[
                MistralForCausalLM,
                LlamaForCausalLM,
                BartForCausalLM,
                FalconForCausalLM,
                GPTNeoXForCausalLM,
                PhiForCausalLM,
                GemmaForCausalLM,
            ],
            tokenizer="Xenova/gpt-4",
            eval_block_delay=7200,
            kwargs={
                "torch_dtype": torch.bfloat16,
                "attn_implementation": "flash_attention_2",
            },
        )

        model_constraints = get_model_constraints(CompetitionId.SN9_MODEL)
        self.assertEqual(model_constraints, expected_constraints)

    def test_get_model_constraints_invalid_competition(self):
        model_constraints = get_model_constraints(CompetitionId.COMPETITION_2)
        self.assertIsNone(model_constraints)

    def test_get_competition_for_block_valid_competition(self):
        expected_competition = Competition(
            id=CompetitionId.SN9_MODEL,
            constraints=ModelConstraints(
                max_model_parameter_size=6_900_000_000,
                sequence_length=4096,
                allowed_architectures=[
                    MistralForCausalLM,
                    LlamaForCausalLM,
                    BartForCausalLM,
                    FalconForCausalLM,
                    GPTNeoXForCausalLM,
                    PhiForCausalLM,
                    GemmaForCausalLM,
                ],
                tokenizer="Xenova/gpt-4",
                eval_block_delay=7200,
                kwargs={
                    "torch_dtype": torch.bfloat16,
                    "attn_implementation": "flash_attention_2",
                },
            ),
            reward_percentage=1.0,
        )

        competition = get_competition_for_block(CompetitionId.SN9_MODEL, 0)
        self.assertEqual(competition, expected_competition)

    def test_get_competition_for_block_invalid_competition(self):
        competition = get_competition_for_block(CompetitionId.COMPETITION_2, 0)
        self.assertIsNone(competition)

    def test_get_competition_for_block_invalid_block(self):
        with self.assertRaises(Exception):
            _ = get_competition_for_block(CompetitionId.SN9_MODEL, -1)

    def test_get_competition_schedule_for_block_valid_block(self):
        expected_competition_schedule = [
            Competition(
                id=CompetitionId.SN9_MODEL,
                constraints=ModelConstraints(
                    max_model_parameter_size=6_900_000_000,
                    sequence_length=4096,
                    allowed_architectures=[
                        MistralForCausalLM,
                        LlamaForCausalLM,
                        BartForCausalLM,
                        FalconForCausalLM,
                        GPTNeoXForCausalLM,
                        PhiForCausalLM,
                        GemmaForCausalLM,
                    ],
                    tokenizer="Xenova/gpt-4",
                    eval_block_delay=7200,
                    kwargs={
                        "torch_dtype": torch.bfloat16,
                        "attn_implementation": "flash_attention_2",
                    },
                ),
                reward_percentage=1.0,
            ),
        ]

        competition_schedule = get_competition_schedule_for_block(0)
        self.assertEqual(competition_schedule, expected_competition_schedule)

    def test_get_competition_schedule_for_block_invalid_block(self):
        with self.assertRaises(Exception):
            _ = get_competition_schedule_for_block(-1)


if __name__ == "__main__":
    unittest.main()
