import unittest

from transformers import LlamaForCausalLM

from competitions.data import Competition, CompetitionId, ModelConstraints
from competitions.utils import get_competition

unittest.util._MAX_LENGTH = 2000


class TestUtils(unittest.TestCase):
    def test_get_competition_valid_competition(self):
        expected_competition = Competition(
            id=CompetitionId.SN9_MODEL,
            constraints=ModelConstraints(
                max_model_parameter_size=6_900_000_000,
                sequence_length=4096,
                allowed_architectures=[LlamaForCausalLM],
                tokenizer="Xenova/gpt-4",
                kwargs={"torch_dtype": "bfloat16"},
            ),
            reward_percentage=1.0,
        )

        competition = get_competition(CompetitionId.SN9_MODEL)
        self.assertEqual(competition, expected_competition)

    def test_get_competition_invalid_competition(self):
        competition = get_competition(CompetitionId.COMPETITION_2)
        self.assertIsNone(competition)


if __name__ == "__main__":
    unittest.main()
