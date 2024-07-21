
import asyncio
import unittest
from unittest import mock

import bittensor as bt
import torch
from taoverse.model.data import ModelId

from competitions.data import CompetitionId
from finetune import graph
from tests.model.storage.fake_model_metadata_store import \
    FakeModelMetadataStore


class TestMining(unittest.TestCase):
    
    def _create_metagraph(self) -> bt.metagraph:
        mock_metagraph = mock.MagicMock()
        mock_metagraph.n = 10
        mock_metagraph.I = torch.tensor(
            [0, 0, 5, 10, 20, 0, 10, 1, 2, 3], dtype=torch.float32
        )
        mock_metagraph.hotkeys = [f"hotkey_{i}" for i in range(10)]
        return mock_metagraph
    
    def test_best_uid(self):
        """Tests that the best UID for the matching competition is returned."""
        metadata_store = FakeModelMetadataStore()
        metagraph = self._create_metagraph()
        
        # The top miners by incentive are: 4, 3, 6, 2, 7
        # Put miners 3 and 2 in the right competition.
        asyncio.run(metadata_store.store_model_metadata("hotkey_2", ModelId(namespace="hf", name="model2", competition_id=CompetitionId.SN9_MODEL)))
        asyncio.run(metadata_store.store_model_metadata("hotkey_3", ModelId(namespace="hf", name="model3", competition_id=CompetitionId.SN9_MODEL)))
        
        # Put model 4 (the top miner) in a different competition
        asyncio.run(metadata_store.store_model_metadata("hotkey_4", ModelId(namespace="hf", name="model2", competition_id=CompetitionId.COMPETITION_2)))

        self.assertEqual(3, graph.best_uid(CompetitionId.SN9_MODEL, metagraph=metagraph, metadata_store=metadata_store))

    
    def test_best_uid_no_matching_miners(self):
        """Verifies that None is returned if no miner matches the requested competition."""
        metadata_store = FakeModelMetadataStore()
        metagraph = self._create_metagraph()
        
        # The top miners by incentive are: 4, 3, 6, 2, 7
        # Put miners 3 and 2 in the wrong competition.
        asyncio.run(metadata_store.store_model_metadata("hotkey_2", ModelId(namespace="hf", name="model2", competition_id=CompetitionId.COMPETITION_2)))
        asyncio.run(metadata_store.store_model_metadata("hotkey_3", ModelId(namespace="hf", name="model3", competition_id=CompetitionId.COMPETITION_2)))
        

        self.assertIsNone(graph.best_uid(CompetitionId.SN9_MODEL, metagraph=metagraph, metadata_store=metadata_store))
    


