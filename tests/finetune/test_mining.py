import asyncio
import os
import shutil
import unittest
from tempfile import TemporaryDirectory
from unittest import mock

import bittensor as bt
import torch
from taoverse.model.data import Model, ModelId

import constants
import finetune as ft
from competitions.data import CompetitionId
from tests.model.storage.fake_model_metadata_store import FakeModelMetadataStore
from tests.model.storage.fake_remote_model_store import FakeRemoteModelStore
from tests.utils import assert_model_equality, get_test_model


class TestMining(unittest.TestCase):
    def _create_metagraph(self) -> bt.metagraph:
        mock_metagraph = mock.MagicMock()
        mock_metagraph.n = 3
        mock_metagraph.I = torch.tensor([0, 5, 10], dtype=torch.float32)
        mock_metagraph.hotkeys = [f"hotkey_{i}" for i in range(3)]
        return mock_metagraph

    def setUp(self):
        self.remote_store = FakeRemoteModelStore()
        self.metadata_store = FakeModelMetadataStore()
        self.wallet = bt.wallet("unit_test", "mining_actions")
        self.wallet.create_if_non_existent(
            coldkey_use_password=False, hotkey_use_password=False
        )
        self.tiny_model = get_test_model()

        self.model_dir = "test-models/test-mining"
        os.makedirs(name=self.model_dir, exist_ok=True)

    def tearDown(self):
        # Clean up the model directory.
        shutil.rmtree(self.model_dir)

    def test_model_to_disk_roundtrip(self):
        """Tests that saving a model to disk and loading it gets the same model."""

        ft.mining.save(model=self.tiny_model, model_dir=self.model_dir)
        model = ft.mining.load_local_model(model_dir=self.model_dir, kwargs={})

        assert_model_equality(self, self.tiny_model, model)

    def _test_push(
        self, min_expected_block: int = 1, competition_id=CompetitionId.SN9_MODEL
    ):
        asyncio.run(
            ft.mining.push(
                model=self.tiny_model,
                wallet=self.wallet,
                competition_id=competition_id,
                repo="namespace/name",
                retry_delay_secs=1,
                update_repo_visibility=False,
                metadata_store=self.metadata_store,
                remote_model_store=self.remote_store,
            )
        )

        # Check that the model was uploaded to hugging face.
        model: Model = self.remote_store.get_only_model()
        assert_model_equality(self, self.tiny_model, model.pt_model)

        # Check that the model ID was published on the chain.
        model_metadata = asyncio.run(
            self.metadata_store.retrieve_model_metadata(self.wallet.hotkey.ss58_address)
        )
        self.assertGreaterEqual(model_metadata.block, min_expected_block)

        # Check certain properties of the model metadata.
        self.assertEqual(model_metadata.id.commit, model.id.commit)
        self.assertEqual(model_metadata.id.name, model.id.name)
        self.assertEqual(model_metadata.id.namespace, model.id.namespace)
        self.assertEqual(model_metadata.id.competition_id, model.id.competition_id)

        self.metadata_store.reset()
        self.remote_store.reset()

    def test_push_success(self):
        """Tests that pushing a model to the chain is successful."""
        self._test_push()

    def test_push_model_chain_failure(self):
        """Tests that pushing a model is eventually successful even if pushes to the chain fail."""

        self.metadata_store.inject_store_errors(
            [TimeoutError("Time out"), Exception("Unknown error")]
        )

        self._test_push()

    def test_push_metadata_read_is_old(self):
        """Tests that pushing a model to the chain is successful even if the metadata read back is stale."""

        # Inject an empty response when push tries to read back the metadata commit.
        self.metadata_store.inject_model_metadata(
            self.wallet.hotkey.ss58_address, metadata=None
        )

        self._test_push(min_expected_block=2)

    async def test_get_repo_no_metadata(self):
        """Tests that get_repo raises a ValueError if the miner hasn't uploaded a model yet."""
        hotkey = "hotkey"
        metagraph = mock.MagicMock(spec=bt.metagraph)
        metagraph.hotkeys.return_value = [hotkey]

        # The miner hasn't uploaded a model yet, so expect a ValueError.
        with self.assertRaises(ValueError):
            await ft.mining.get_repo(
                0, metagraph=metagraph, metadata_store=self.metadata_store
            )

    async def test_get_repo(self):
        """Tests that get_repo raises a ValueError if the miner hasn't uploaded a model yet."""
        hotkey = "hotkey"
        metagraph = mock.MagicMock(spec=bt.metagraph)
        metagraph.hotkeys.return_value = [hotkey]

        model_id = ModelId(
            namespace="namespace",
            name="name",
            hash="hash",
            commit="commit",
            competition_id=CompetitionId.SN9_MODEL,
        )
        self.metadata_store.store_model_metadata(hotkey, model_id)

        self.assertEqual(
            await ft.mining.get_repo(
                0, metagraph=metagraph, metadata_store=self.metadata_store
            ),
            "https://huggingface.co/namespace/name/tree/commit",
        )

    async def test_load_best_model(self):
        metagraph = self._create_metagraph()

        # Submit some metadata for the miners with incentive.
        metadata_store = FakeModelMetadataStore()
        miner_1_model_id = (
            ModelId(
                namespace="hf", name="model1", competition_id=CompetitionId.SN9_MODEL
            ),
        )
        await metadata_store.store_model_metadata(
            "hotkey_1",
            miner_1_model_id,
        )
        await metadata_store.store_model_metadata(
            "hotkey_2",
            ModelId(
                namespace="hf",
                name="model2",
                competition_id=CompetitionId.B7_MULTI_CHOICE,
            ),
        )

        # Upload the model for miner 1.
        model_store = FakeRemoteModelStore()
        model = self._get_model()
        await model_store.upload_model(
            Model(
                id=miner_1_model_id,
                pt_model=model,
            ),
            model_constraints=constants.MODEL_CONSTRAINTS_BY_COMPETITION_ID.get(
                1, None
            ),
        )

        # Verify that miner 1's model is loaded.
        with TemporaryDirectory() as model_dir:
            loaded_model = await ft.mining.load_best_model(
                model_dir,
                # Choose the competition that doesn't have the most incentive.
                competition_id=CompetitionId.SN9_MODEL,
                metagraph=metagraph,
                metadata_store=metadata_store,
                remote_model_store=model_store,
            )
            self.assertEqual(loaded_model, model)


if __name__ == "__main__":
    unittest.main()
