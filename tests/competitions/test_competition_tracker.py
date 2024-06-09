import unittest

import torch
from transformers import LlamaForCausalLM

from competitions.competition_tracker import CompetitionTracker
from competitions.data import CompetitionId, CompetitionParameters


class TestCompetitionTracker(unittest.TestCase):
    COMPETITION_1_PARAMETERS = CompetitionParameters(
        max_model_parameter_size=8 * 1024 * 1024 * 1024,
        architecture=LlamaForCausalLM,
        kwargs={},
        tokenizer="Xenova/gpt-4",
        reward_percentage=0.6,
        competition_id="comp_1",
        competition_enum=CompetitionId.COMPTETITION_1,
    )
    COMPETITION_2_PARAMETERS = CompetitionParameters(
        max_model_parameter_size=2 * 1024 * 1024 * 1024,
        architecture=LlamaForCausalLM,
        kwargs={},
        tokenizer="Xenova/gpt-4",
        reward_percentage=0.4,
        competition_id="comp_2",
        competition_enum=CompetitionId.COMPTETITION_2,
    )

    def setUp(self):
        self.num_neurons = 4
        self.alpha = 0.5
        self.competition_tracker = CompetitionTracker(self.num_neurons, self.alpha)

    def test_record_competition_weights_new(self):
        weights = torch.Tensor([0.1, 0.2, 0.3, 0.4])
        self.competition_tracker.record_competition_weights(
            CompetitionId.COMPTETITION_1, weights
        )

        # Since this is a net new competition, check that weights go immediately to the new values.
        self.assertTrue(
            torch.equal(
                self.competition_tracker.weights_by_competition[
                    CompetitionId.COMPTETITION_1
                ],
                weights,
            )
        )

    def test_record_competition_weights_normalized(self):
        weights = torch.Tensor([0.2, 0.4, 0.6, 0.8])
        self.competition_tracker.record_competition_weights(
            CompetitionId.COMPTETITION_1, weights
        )

        normalized_weights = weights / weights.sum()
        self.assertTrue(
            torch.equal(
                self.competition_tracker.weights_by_competition[
                    CompetitionId.COMPTETITION_1
                ],
                normalized_weights,
            )
        )

    def test_record_competition_weights_moving_average(self):
        initial_weights = torch.Tensor([1, 0, 0, 0])
        new_weights = torch.Tensor([0, 0, 0, 1])
        self.competition_tracker.record_competition_weights(
            CompetitionId.COMPTETITION_1, initial_weights
        )
        self.competition_tracker.record_competition_weights(
            CompetitionId.COMPTETITION_1, new_weights
        )

        expected_weights = torch.Tensor([0.5, 0, 0, 0.5])
        self.assertTrue(
            torch.equal(
                self.competition_tracker.weights_by_competition[
                    CompetitionId.COMPTETITION_1
                ],
                expected_weights,
            )
        )

    def test_get_competition_weights(self):
        expected_weights = torch.Tensor([0.1, 0.2, 0.3, 0.4])
        self.competition_tracker.record_competition_weights(
            CompetitionId.COMPTETITION_1, expected_weights
        )

        weights = self.competition_tracker.get_competition_weights(
            CompetitionId.COMPTETITION_1
        )

        self.assertTrue(torch.equal(weights, expected_weights))

    def test_get_subnet_weights_one_competition(self):
        expected_weights = torch.Tensor([0.1, 0.2, 0.3, 0.4])
        self.competition_tracker.record_competition_weights(
            CompetitionId.COMPTETITION_1, expected_weights
        )

        weights = self.competition_tracker.get_subnet_weights(
            [self.COMPETITION_1_PARAMETERS]
        )

        self.assertTrue(torch.equal(weights, expected_weights))

    def test_get_subnet_weights_two_competitions(self):
        comp1_weights = torch.Tensor([1, 0, 0, 0])
        comp2_weights = torch.Tensor([0, 0, 0, 1])

        self.competition_tracker.record_competition_weights(
            CompetitionId.COMPTETITION_1, comp1_weights
        )
        self.competition_tracker.record_competition_weights(
            CompetitionId.COMPTETITION_2, comp2_weights
        )

        weights = self.competition_tracker.get_subnet_weights(
            [self.COMPETITION_1_PARAMETERS, self.COMPETITION_2_PARAMETERS]
        )

        expected_weights = (
            comp1_weights * self.COMPETITION_1_PARAMETERS.reward_percentage
            + comp2_weights * self.COMPETITION_2_PARAMETERS.reward_percentage
        )
        self.assertTrue(torch.equal(weights, expected_weights))

    def test_resize_one_competition(self):
        weights = torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        self.competition_tracker.record_competition_weights(
            CompetitionId.COMPTETITION_1, weights
        )

        self.assertTrue(
            torch.equal(
                self.competition_tracker.weights_by_competition[
                    CompetitionId.COMPTETITION_1
                ],
                weights,
            )
        )

    def test_resize_two_competition(self):
        comp1_weights = torch.Tensor([0.1, 0.2, 0.3, 0.4])
        comp2_weights = torch.Tensor([0.1, 0.2, 0.3, 0.2, 0.2])

        self.competition_tracker.record_competition_weights(
            CompetitionId.COMPTETITION_1, comp1_weights
        )
        self.competition_tracker.record_competition_weights(
            CompetitionId.COMPTETITION_2, comp2_weights
        )

        expanded_comp1_weights = torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.0])
        self.assertTrue(
            torch.equal(
                self.competition_tracker.weights_by_competition[
                    CompetitionId.COMPTETITION_1
                ],
                expanded_comp1_weights,
            )
        )
        self.assertTrue(
            torch.equal(
                self.competition_tracker.weights_by_competition[
                    CompetitionId.COMPTETITION_2
                ],
                comp2_weights,
            )
        )


if __name__ == "__main__":
    unittest.main()
