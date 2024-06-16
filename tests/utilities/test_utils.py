import functools
import time
import unittest
from tempfile import NamedTemporaryFile
from typing import List, Tuple
from unittest import mock

import bittensor as bt
import torch
import constants

from utilities import utils
from utilities.utils import run_in_subprocess, run_in_thread


class TestUtils(unittest.TestCase):
    def test_run_in_subprocess(self):
        def test_func(a: int, b: int):
            return a + b

        partial = functools.partial(test_func, 1, 2)

        result = run_in_subprocess(func=partial, ttl=5)
        self.assertEqual(3, result)

    def test_run_in_subprocess_timeout(self):
        def test_func(a: int, b: int):
            time.sleep(3)
            return a + b

        partial = functools.partial(test_func, 1, 2)

        with self.assertRaises(TimeoutError):
            result = run_in_subprocess(func=partial, ttl=1)

    def test_run_in_subprocess_no_return(self):
        def test_func(a: int, b: int):
            pass

        partial = functools.partial(test_func, 1, 2)

        result = run_in_subprocess(func=partial, ttl=5)
        self.assertIsNone(result)

    def test_run_in_subprocess_tuple_return(self):
        def test_func(a: int, b: int):
            return a, b

        partial = functools.partial(test_func, 1, 2)

        result = run_in_subprocess(func=partial, ttl=5)
        self.assertEqual((1, 2), result)

    def test_run_in_subprocess_exception(self):
        def test_func(a: int, b: int):
            raise ValueError()

        partial = functools.partial(test_func, 1, 2)

        with self.assertRaises(ValueError):
            result = run_in_subprocess(func=partial, ttl=5)

    def test_validate_hf_repo_id_too_long(self):
        with self.assertRaises(ValueError) as ve:
            # Max allowed length is 41 characters
            utils.validate_hf_repo_id("my-org/" + "a" * 40)

        self.assertRegex(
            str(ve.exception),
            "Hugging Face repo id must be between 3 and 38 characters",
        )

    def test_validate_hf_repo_id_incorrect_format(self):
        with self.assertRaises(ValueError) as ve:
            utils.validate_hf_repo_id("my-repo-name-without-a-namespace")

        self.assertRegex(
            str(ve.exception), "must be in the format <org or user name>/<repo_name>"
        )

    def test_validate_hf_repo_id_valid(self):
        namespace, name = utils.validate_hf_repo_id("my-org/my-repo-name")
        self.assertEqual("my-org", namespace)
        self.assertEqual("my-repo-name", name)

    def test_run_in_thread(self):
        def test_func(a: int, b: int):
            return a + b

        partial = functools.partial(test_func, 1, 2)

        result = run_in_thread(func=partial, ttl=5)
        self.assertEqual(3, result)

    def test_run_in_thread_timeout(self):
        def test_func(a: int, b: int):
            time.sleep(3)
            return a + b

        partial = functools.partial(test_func, 1, 2)

        with self.assertRaises(TimeoutError):
            result = run_in_thread(func=partial, ttl=1)

    def test_run_in_thread_no_return(self):
        def test_func(a: int, b: int):
            pass

        partial = functools.partial(test_func, 1, 2)

        result = run_in_thread(func=partial, ttl=5)
        self.assertIsNone(result)

    def test_run_in_thread_tuple_return(self):
        def test_func(a: int, b: int):
            return a, b

        partial = functools.partial(test_func, 1, 2)

        result = run_in_thread(func=partial, ttl=5)
        self.assertEqual((1, 2), result)

    def test_run_in_thread_exception(self):
        def test_func(a: int, b: int):
            raise ValueError()

        partial = functools.partial(test_func, 1, 2)

        with self.assertRaises(ValueError):
            result = run_in_thread(func=partial, ttl=5)

    def test_get_high_stake_validators(self):
        # Create a metagraph with 10 neurons of varying stake with the top 4 having a validator permit.
        mock_metagraph = mock.MagicMock()
        mock_metagraph.S = torch.tensor(
            [0, 1, 2, 300, 4, 5, 600, 7, 8, 9], dtype=torch.float32
        )
        mock_metagraph.validator_permit = torch.tensor(
            [
                False,
                False,
                False,
                True,
                False,
                False,
                True,
                False,
                True,
                True,
            ],
            dtype=torch.bool,
        )

        # Check 300 or above stake.
        self.assertEqual(utils.get_high_stake_validators(mock_metagraph, 300), {3, 6})

        # Check 0 or above stake gets only those with permits.
        self.assertEqual(
            utils.get_high_stake_validators(mock_metagraph, 0), {3, 6, 8, 9}
        )

    def _create_metagraph(self):
        """Returns a mocked metagraph with 2 miners and 2 valis."""
        mock_metagraph = mock.MagicMock()
        stakes = torch.tensor([0, 200, 2, 30], dtype=torch.float32)
        mock_metagraph.S = stakes
        mock_metagraph.validator_permit = stakes >= 30
        return mock_metagraph

    def _neuron_info_with_weights(
        self, uid: int, weights: List[Tuple[int, float]]
    ) -> bt.NeuronInfo:
        return bt.NeuronInfo(
            uid=uid,
            netuid=0,
            active=0,
            stake=bt.Balance.from_rao(0),
            stake_dict={},
            total_stake=bt.Balance.from_rao(0),
            rank=0,
            emission=0,
            incentive=0,
            consensus=0,
            trust=0,
            validator_trust=0,
            dividends=0,
            last_update=0,
            validator_permit=False,
            weights=weights,
            bonds=[],
            prometheus_info=None,
            axon_info=None,
            is_null=True,
            coldkey="000000000000000000000000000000000000000000000000",
            hotkey="000000000000000000000000000000000000000000000000",
            pruning_score=0,
        )

    def test_get_top_miners_deduplicated(self):
        """Tests get_top_miners, when validators agree on the top miners."""
        metagraph = self._create_metagraph()

        # Set validator weights such that they agree on miner 0 as the top miner.
        metagraph.neurons = [
            self._neuron_info_with_weights(uid=0, weights=[]),
            self._neuron_info_with_weights(uid=1, weights=[(0, 1)]),
            self._neuron_info_with_weights(uid=2, weights=[]),
            self._neuron_info_with_weights(uid=3, weights=[(0, 1)]),
        ]

        # Verify the miner UID is deduped.
        self.assertSequenceEqual(
            utils.get_top_miners(
                metagraph, min_vali_stake=30, min_miner_weight_percent=0.1
            ),
            {0},
        )

    def test_get_top_miners_multiple_miners(self):
        """Tests get_top_miners, when validators disagree on the top miner."""
        metagraph = self._create_metagraph()

        metagraph.neurons = [
            self._neuron_info_with_weights(uid=0, weights=[]),
            self._neuron_info_with_weights(uid=1, weights=[(0, 1)]),
            self._neuron_info_with_weights(uid=2, weights=[]),
            self._neuron_info_with_weights(uid=3, weights=[(2, 1)]),
        ]
        top_miners = utils.get_top_miners(
            metagraph, min_vali_stake=30, min_miner_weight_percent=0.1
        )
        self.assertEqual(len(top_miners), 2)
        self.assertEqual(top_miners, {0, 2})

    def test_get_top_miners_multiple_weights_set(self):
        """Tests get_top_miners, when validators assign multiple weights"""
        metagraph = self._create_metagraph()

        # Have vali 1 set multiple weights, ensuring some are above and below the threshold.
        # Note that although uid 1 has 0.1 weight, this is only 5% of the weight.
        metagraph.neurons = [
            self._neuron_info_with_weights(uid=0, weights=[]),
            self._neuron_info_with_weights(uid=1, weights=[(0, 1), (1, 0.1), (2, 0.5)]),
            self._neuron_info_with_weights(uid=2, weights=[]),
            self._neuron_info_with_weights(uid=3, weights=[]),
        ]
        self.assertEqual(
            utils.get_top_miners(
                metagraph, min_vali_stake=30, min_miner_weight_percent=0.1
            ),
            {0, 2},
        )

    def test_save_and_load_version(self):
        version = constants.__spec_version__
        with NamedTemporaryFile() as f:
            self.assertIsNone(utils.get_version(f.name))

            utils.save_version(f.name, version)
            self.assertEqual(utils.get_version(f.name), version)


if __name__ == "__main__":
    unittest.main()
