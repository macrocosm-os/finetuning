import asyncio
import os
import shutil
import unittest
from unittest import mock

import bittensor as bt

from competitions.data import CompetitionId
import finetune as ft
from model.data import Model, ModelId
from tests.model.storage.fake_model_metadata_store import FakeModelMetadataStore
from tests.model.storage.fake_remote_model_store import FakeRemoteModelStore
from tests.utils import assert_model_equality
from transformers import PreTrainedModel, LlamaForCausalLM, LlamaConfig


class TestMining(unittest.TestCase):
    def _get_model(self) -> PreTrainedModel:
        return LlamaForCausalLM(LlamaConfig(num_hidden_layers=2))

    def setUp(self):
        self.remote_store = FakeRemoteModelStore()
        self.metadata_store = FakeModelMetadataStore()
        self.wallet = bt.wallet("unit_test", "mining_actions")
        self.wallet.create_if_non_existent(
            coldkey_use_password=False, hotkey_use_password=False
        )
        self.tiny_model = self._get_model()

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


if __name__ == "__main__":
    unittest.main()
