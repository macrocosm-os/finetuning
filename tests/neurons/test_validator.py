import unittest
import torch
import typing
from neurons.validator import Validator

class MockValidatorConfig:
    def __init__(self, sample_min: int):
        self.sample_min = sample_min

class TestValidatorCalculateModelsToKeep(unittest.TestCase):

    def setUp(self):
        self.uid_winner = 69
        self.uid_loser_with_historical_weight = 21
        self.uid_loser_no_historical_weight = 37
        self.validator_logic_host = Validator.__new__(Validator)

    def test_buggy_calculate_models_to_keep_drops_winner(self):
        """
        Tests the production 'calculate_models_to_keep' method.
        This test should FAIL with the current buggy code and PASS with fixed code.
        """
        current_sample_min = 1
        self.validator_logic_host.config = MockValidatorConfig(sample_min=current_sample_min)

        win_rate = {
            self.uid_winner: 1.0,
            self.uid_loser_with_historical_weight: 0.2,
            self.uid_loser_no_historical_weight: 0.1,
        }

        all_test_uids = list(win_rate.keys())
        max_uid_in_test = max(all_test_uids)

        tracker_competition_weights_tensor = torch.zeros(max_uid_in_test + 1, dtype=torch.float32)
        tracker_competition_weights_tensor[self.uid_winner] = 0.0
        tracker_competition_weights_tensor[self.uid_loser_with_historical_weight] = 0.001
        tracker_competition_weights_tensor[self.uid_loser_no_historical_weight] = 0.0

        # ---- Call the Production Code ----
        models_kept = self.validator_logic_host._calculate_models_to_keep(
            win_rate,
            tracker_competition_weights_tensor
        )

        self.assertIn(
            self.uid_winner,
            models_kept,
            f"BUG BEHAVIOR: Actual winner (UID {self.uid_winner}) was dropped. "
            f"Its priority should be 1.0."
        )


if __name__ == "__main__":
    unittest.main()