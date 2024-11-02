import unittest
from unittest.mock import MagicMock

import bittensor as bt
import substrateinterface as si

from finetune.utils import (
    get_hash_of_block,
    get_next_sync_block,
    get_sync_block,
)


class TestUtils(unittest.TestCase):

    def setUp(self):
        self.subtensor = MagicMock(spec=bt.subtensor)
        self.subtensor.substrate = MagicMock(spec=si.SubstrateInterface)

    def test_get_hash_of_block(self):
        block_number = 12345
        self.subtensor.get_block_hash.return_value = "some_hash"

        result = get_hash_of_block(self.subtensor, block_number)

        self.assertEqual(result, hash("some_hash"))
        self.subtensor.get_block_hash.assert_called_once_with(block_number)

    def test_get_sync_block(self):
        sync_cadence = 100
        genesis = 1_111_111
        for block in range(1_234_511, 1_234_611):
            self.assertEqual(get_sync_block(block, sync_cadence, genesis), 1_234_511)

        # Now check the next block is the start of a new sync block.
        self.assertEqual(get_sync_block(1_234_611, sync_cadence, genesis), 1_234_611)

    def test_get_next_sync_block(self):
        sync_cadence = 100
        genesis = 1_111_111
        for block in range(1_234_511, 1_234_611):
            self.assertEqual(
                get_next_sync_block(block, sync_cadence, genesis), 1_234_611
            )

        # Now check the next block is the start of a new sync block.
        self.assertEqual(
            get_next_sync_block(1_234_611, sync_cadence, genesis), 1_234_711
        )


if __name__ == "__main__":
    unittest.main()
