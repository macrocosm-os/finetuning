import typing

import bittensor as bt
import torch

import constants
from competitions.data import CompetitionId, Competition


class CompetitionTracker:
    """Tracks weights of each miner (by UID) per competition."""

    def __init__(
        self,
        num_neurons: int,
        alpha: float = constants.alpha,
    ):
        self.weights_by_competition: typing.Dict[CompetitionId, torch.Tensor] = {}
        self.num_neurons = num_neurons
        self.alpha = alpha

    # TODO: Consider a record_competition_results which then does additional processing into weights.

    def record_competition_weights(
        self, competition_id: CompetitionId, new_weights: torch.Tensor
    ):
        """Records the weights from a new run of a competition, updating a moving average.

        Args:
            competition_id (CompetitionId): Which competition the weights are for.
            new_weights (torch.Tensor): Weights from the new run of the competition.
        """
        # Check that the weights are the appropriate length along the first dimension.
        if new_weights.size(0) != self.num_neurons:
            self._resize(new_weights.size(0))

        # Normalize the weights.
        new_weights /= new_weights.sum()
        new_weights = new_weights.nan_to_num(0.0)

        # If we haven't recorded weights for this competition yet, start from scratch.
        if competition_id not in self.weights_by_competition:
            self.weights_by_competition[competition_id] = new_weights
        # Otherwise compute a moving average using alpha.
        else:
            moving_average_weights = (
                self.alpha * self.weights_by_competition[competition_id]
                + (1 - self.alpha) * new_weights
            )
            self.weights_by_competition[competition_id] = moving_average_weights

    def reset_competitions(self, competition_ids: typing.Set[CompetitionId]):
        """Resets tracked competitions to only those identified.

        Args:
            competition_ids (typing.Set[CompetitionId]): Competition ids to continue tracking.
        """
        # Make a list to avoid issues deleting from within the dictionary iterator.
        for key in list(self.weights_by_competition.keys()):
            if key not in competition_ids:
                del self.weights_by_competition[key]

    def get_subnet_weights(
        self, competitions: typing.List[Competition]
    ) -> torch.Tensor:
        """Gets the weights that should be set on the subnet level based on all competitions.

        Args:
            competitions (typing.List[Competition]): Competitions to calculate weights across.

        Returns:
            torch.Tensor: Weights calculated across all specified competitions.
        """
        # Return a copy to ensure outside code can't modify the scores.
        # Start each uid at 0.
        subnet_weights = torch.zeros(self.num_neurons, dtype=torch.float32)

        # For each competition, add the relative competition weight
        for competition in competitions:
            if competition.id in self.weights_by_competition:
                comp_weights = self.weights_by_competition[competition.id]
                for i in range(self.num_neurons):
                    # Today uids can only participate in one competition so += and = would be equivalent.
                    subnet_weights[i] += (
                        comp_weights[i] * competition.reward_percentage
                    )

        # Normalize weights again in case a competition hasn't run yet.
        # In the case that in Block X we have Competition 1 at 100%, but Block X+1 we have
        # Competition 1 at 80% and Competition 2 at 20%, but we haven't run Competition 2 yet.
        # This ensures we still return 1 for the winner of competition 1 instead of 0.8.
        subnet_weights /= subnet_weights.sum()
        subnet_weights = subnet_weights.nan_to_num(0.0)

        return subnet_weights

    def get_competition_weights(self, competition_id: CompetitionId) -> torch.Tensor:
        """Returns the current weights for a single competition"""
        # Return a copy to ensure outside code can't modify the scores.
        return self.weights_by_competition[competition_id].clone()

    def _resize(self, new_num_neurons: int) -> None:
        """Resizes the score tensor to the new number of neurons.

        The new size must be greater than or equal to the current size.
        """
        assert (
            new_num_neurons >= self.num_neurons
        ), f"Tried to downsize the number of neurons from {self.num_neurons} to {new_num_neurons}"

        bt.logging.trace(
            f"Resizing CompetitionTracker from {self.num_neurons} to {new_num_neurons}"
        )

        # Compute additional number of neurons to add to the end of each competition weight tensor.
        to_add = new_num_neurons - self.num_neurons
        for competition in self.weights_by_competition:
            self.weights_by_competition[competition] = torch.cat(
                [
                    self.weights_by_competition[competition],
                    torch.zeros(to_add, dtype=torch.float32),
                ]
            )

        # Keep track of the new size of each competition weight tensor.
        self.num_neurons = new_num_neurons

    def save_state(self, filepath):
        """Save the current state to the provided filepath."""
        torch.save(
            {
                "weights_by_competition": self.weights_by_competition,
                "num_neurons": self.num_neurons,
                "alpha": self.alpha,
            },
            filepath,
        )

    def load_state(self, filepath):
        """Load the state from the provided filepath."""
        state = torch.load(filepath)
        self.weights_by_competition = state["weights_by_competition"]
        self.num_neurons = state["num_neurons"]
        self.alpha = state["alpha"]
