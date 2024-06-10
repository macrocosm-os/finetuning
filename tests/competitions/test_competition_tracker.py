import unittest

import torch
from transformers import LlamaForCausalLM

from competitions.competition_tracker import CompetitionTracker
from competitions.data import CompetitionId, Competition, ModelConstraints


class TestCompetitionTracker(unittest.TestCase):
    COMPETITION_1_PARAMETERS = Competition(
        id=CompetitionId.SN9_MODEL,
        constraints=ModelConstraints(
            max_model_parameter_size=8 * 1024 * 1024 * 1024,
            sequence_length=4096,
            allowed_architectures=[LlamaForCausalLM],
            allowed_tokenizers=["Xenova/gpt-4"],
            kwargs={},
        ),
        reward_percentage=0.6,
    )
    COMPETITION_2_PARAMETERS = Competition(
        id=CompetitionId.COMPETITION_2,
        constraints=ModelConstraints(
            max_model_parameter_size=2 * 1024 * 1024 * 1024,
            sequence_length=2048,
            allowed_architectures=[LlamaForCausalLM],
            allowed_tokenizers=["Xenova/gpt-4"],
            kwargs={},
        ),
        reward_percentage=0.4,
    )

    def setUp(self):
        self.num_neurons = 4
        self.alpha = 0.5
        self.competition_tracker = CompetitionTracker(self.num_neurons, self.alpha)

    def test_record_competition_weights_new(self):
        self.competition_tracker.record_competition_weights(
            CompetitionId.SN9_MODEL, torch.Tensor([0.1, 0.2, 0.3, 0.4])
        )

        # Since this is a net new competition, check that weights go immediately to the new values.
        self.assertTrue(
            torch.equal(
                self.competition_tracker.weights_by_competition[
                    CompetitionId.SN9_MODEL
                ],
                torch.Tensor([0.1, 0.2, 0.3, 0.4]),
            )
        )

    def test_record_competition_weights_normalized(self):
        self.competition_tracker.record_competition_weights(
            CompetitionId.SN9_MODEL, torch.Tensor([0.2, 0.4, 0.6, 0.8])
        )

        # Check that the weights are normalized to sum to 1.
        self.assertTrue(
            torch.equal(
                self.competition_tracker.weights_by_competition[
                    CompetitionId.SN9_MODEL
                ],
                torch.Tensor([0.1, 0.2, 0.3, 0.4]),
            )
        )

    def test_record_competition_weights_moving_average(self):
        self.competition_tracker.record_competition_weights(
            CompetitionId.SN9_MODEL, torch.Tensor([1, 0, 0, 0])
        )
        self.competition_tracker.record_competition_weights(
            CompetitionId.SN9_MODEL, torch.Tensor([0, 0, 0, 1])
        )

        # Check that the weights are a moving average according to the alpha of 0.5.
        self.assertTrue(
            torch.equal(
                self.competition_tracker.weights_by_competition[
                    CompetitionId.SN9_MODEL
                ],
                torch.Tensor([0.5, 0, 0, 0.5]),
            )
        )

    def test_get_competition_weights(self):
        self.competition_tracker.record_competition_weights(
            CompetitionId.SN9_MODEL, torch.Tensor([0.1, 0.2, 0.3, 0.4])
        )

        weights = self.competition_tracker.get_competition_weights(
            CompetitionId.SN9_MODEL
        )

        self.assertTrue(torch.equal(weights, torch.Tensor([0.1, 0.2, 0.3, 0.4])))

    def test_get_subnet_weights_one_competition(self):
        self.competition_tracker.record_competition_weights(
            CompetitionId.SN9_MODEL, torch.Tensor([0.1, 0.2, 0.3, 0.4])
        )

        weights = self.competition_tracker.get_subnet_weights(
            [self.COMPETITION_1_PARAMETERS]
        )

        self.assertTrue(torch.equal(weights, torch.Tensor([0.1, 0.2, 0.3, 0.4])))

    def test_get_subnet_weights_two_competitions(self):
        self.competition_tracker.record_competition_weights(
            CompetitionId.SN9_MODEL, torch.Tensor([1, 1, 0, 0])
        )
        self.competition_tracker.record_competition_weights(
            CompetitionId.COMPETITION_2, torch.Tensor([0, 0, 5, 5])
        )

        weights = self.competition_tracker.get_subnet_weights(
            [self.COMPETITION_1_PARAMETERS, self.COMPETITION_2_PARAMETERS]
        )

        # Check that the weights are both normalized and rewarded according to competition reward percent.
        self.assertTrue(torch.equal(weights, torch.Tensor([0.3, 0.3, 0.2, 0.2])))

    def test_resize_one_competition(self):
        self.competition_tracker.record_competition_weights(
            CompetitionId.SN9_MODEL, torch.Tensor([0.1, 0.2, 0.3, 0.2, 0.2])
        )

        # Check that the internal state immediately expands to 5 neurons.
        self.assertTrue(
            torch.equal(
                self.competition_tracker.weights_by_competition[
                    CompetitionId.SN9_MODEL
                ],
                torch.Tensor(
                    [0.1, 0.2, 0.3, 0.2, 0.2],
                ),
            )
        )

    def test_resize_two_competition(self):
        self.competition_tracker.record_competition_weights(
            CompetitionId.SN9_MODEL, torch.Tensor([0.1, 0.2, 0.3, 0.4])
        )
        self.competition_tracker.record_competition_weights(
            CompetitionId.COMPETITION_2, torch.Tensor([0.1, 0.2, 0.3, 0.2, 0.2])
        )

        # Check that the internal state of the first competition is expanded with 0s.
        self.assertTrue(
            torch.equal(
                self.competition_tracker.weights_by_competition[
                    CompetitionId.SN9_MODEL
                ],
                torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.0]),
            )
        )
        self.assertTrue(
            torch.equal(
                self.competition_tracker.weights_by_competition[
                    CompetitionId.COMPETITION_2
                ],
                torch.Tensor([0.1, 0.2, 0.3, 0.2, 0.2]),
            )
        )

    def test_reset_competitions(self):
        self.competition_tracker.record_competition_weights(
            CompetitionId.SN9_MODEL, torch.Tensor([1, 0, 0, 0])
        )
        self.competition_tracker.record_competition_weights(
            CompetitionId.COMPETITION_2, torch.Tensor([0, 0, 0, 1])
        )

        self.competition_tracker.reset_competitions({CompetitionId.SN9_MODEL})

        # Check that the weights for competition 2 are no longer tracked.
        self.assertTrue(
            CompetitionId.SN9_MODEL in self.competition_tracker.weights_by_competition
        )
        self.assertFalse(
            CompetitionId.COMPETITION_2
            in self.competition_tracker.weights_by_competition
        )


if __name__ == "__main__":
    unittest.main()
