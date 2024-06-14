# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from dataclasses import replace
import os
import time
from typing import Any, Dict, Optional

import bittensor as bt
import huggingface_hub
from transformers import AutoModelForCausalLM, PreTrainedModel

import constants
import finetune as ft
from competitions import utils as competition_utils
from competitions.data import CompetitionId
from model.data import Model, ModelId
from model.storage.chain.chain_model_metadata_store import ChainModelMetadataStore
from model.storage.hugging_face.hugging_face_model_store import HuggingFaceModelStore
from model.storage.model_metadata_store import ModelMetadataStore
from model.storage.remote_model_store import RemoteModelStore
from model.utils import get_hash_of_two_strings
from utilities import utils


def model_path(base_dir: str, run_id: str) -> str:
    """
    Constructs a file path for storing the model relating to a training run.
    """
    return os.path.join(base_dir, "training", run_id)


async def push(
    model: PreTrainedModel,
    repo: str,
    competition_id: CompetitionId,
    wallet: bt.wallet,
    retry_delay_secs: int = 60,
    update_repo_visibility: bool = False,
    metadata_store: Optional[ModelMetadataStore] = None,
    remote_model_store: Optional[RemoteModelStore] = None,
):
    """Pushes the model to Hugging Face and publishes it on the chain for evaluation by validators.

    Args:
        model (PreTrainedModel): The model to push.
        repo (str): The repo to push to. Must be in format "namespace/name".
        wallet (bt.wallet): The wallet of the Miner uploading the model.
        retry_delay_secs (int): The number of seconds to wait before retrying to push the model to the chain.
        update_repo_visibility (bool): Whether to make the repo public after pushing the model.
        metadata_store (Optional[ModelMetadataStore]): The metadata store. If None, defaults to writing to the
            chain.
        remote_model_store (Optional[RemoteModelStore]): The remote model store. If None, defaults to writing to HuggingFace
    """
    bt.logging.info("Pushing model")

    if metadata_store is None:
        metadata_store = ChainModelMetadataStore(bt.subtensor(), wallet)

    if remote_model_store is None:
        remote_model_store = HuggingFaceModelStore()

    competition = competition_utils.get_competition(competition_id)
    if not competition:
        raise ValueError("Invalid competition_id")

    # First upload the model to HuggingFace.
    namespace, name = utils.validate_hf_repo_id(repo)
    model_id = ModelId(namespace=namespace, name=name, competition_id=competition_id)
    model_id = await remote_model_store.upload_model(
        Model(id=model_id, pt_model=model), competition
    )

    bt.logging.success("Uploaded model to hugging face.")

    secure_hash = get_hash_of_two_strings(model_id.hash, wallet.hotkey.ss58_address)
    model_id = replace(model_id, secure_hash=secure_hash)

    bt.logging.success(f"Now committing to the chain with model_id: {model_id}")

    # We can only commit to the chain every 20 minutes, so run this in a loop, until
    # successful.
    while True:
        try:
            await metadata_store.store_model_metadata(
                wallet.hotkey.ss58_address, model_id
            )

            bt.logging.info(
                "Wrote model metadata to the chain. Checking we can read it back..."
            )

            model_metadata = await metadata_store.retrieve_model_metadata(
                wallet.hotkey.ss58_address
            )

            if not model_metadata or model_metadata.id != model_id:
                bt.logging.error(
                    f"Failed to read back model metadata from the chain. Expected: {model_id}, got: {model_metadata}"
                )
                raise ValueError(
                    f"Failed to read back model metadata from the chain. Expected: {model_id}, got: {model_metadata}"
                )

            bt.logging.success("Committed model to the chain.")
            if update_repo_visibility:
                bt.logging.debug("Making repo public.")
                huggingface_hub.update_repo_visibility(repo, private=False)
                bt.logging.success("Model set to public")
            break
        except Exception as e:
            if update_repo_visibility:
                huggingface_hub.update_repo_visibility(repo, private=False)
            bt.logging.error(f"Failed to advertise model on the chain: {e}")
            bt.logging.error(f"Retrying in {retry_delay_secs} seconds...")
            time.sleep(retry_delay_secs)


def save(model: PreTrainedModel, model_dir: str):
    """Saves a model to the provided directory"""
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    # Save the model state to the specified path.
    model.save_pretrained(
        save_directory=model_dir,
        safe_serialization=True,
    )


async def get_repo(
    uid: int,
    metagraph: Optional[bt.metagraph] = None,
    metadata_store: Optional[ModelMetadataStore] = None,
) -> str:
    """Returns a URL to the HuggingFace repo of the Miner with the given UID."""
    if metadata_store is None:
        metadata_store = ChainModelMetadataStore(bt.subtensor(), constants.SUBNET_UID)
    if metagraph is None:
        metagraph = bt.metagraph(netuid=constants.SUBNET_UID)

    hotkey = metagraph.hotkeys[uid]
    model_metadata = await metadata_store.retrieve_model_metadata(hotkey)

    if not model_metadata:
        raise ValueError(f"No model metadata found for miner {uid}")

    return utils.get_hf_url(model_metadata)


def load_local_model(model_dir: str, kwargs: Dict[str, Any]) -> PreTrainedModel:
    """Loads a model from a directory."""
    return AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_dir,
        local_files_only=True,
        use_safetensors=True,
        **kwargs,
    )


async def load_remote_model(
    uid: int,
    download_dir: str,
    metagraph: Optional[bt.metagraph] = None,
    metadata_store: Optional[ModelMetadataStore] = None,
    remote_model_store: Optional[RemoteModelStore] = None,
) -> PreTrainedModel:
    """Loads the model currently being advertised by the Miner with the given UID.

    Args:
        uid (int): The UID of the Miner who's model should be downloaded.
        download_dir (str): The directory to download the model to.
        metagraph (Optional[bt.metagraph]): The metagraph of the subnet.
        metadata_store (Optional[ModelMetadataStore]): The metadata store. If None, defaults to reading from the
        remote_model_store (Optional[RemoteModelStore]): The remote model store. If None, defaults to reading from HuggingFace
    """

    if metagraph is None:
        metagraph = bt.metagraph(netuid=constants.SUBNET_UID)

    if metadata_store is None:
        metadata_store = ChainModelMetadataStore(subtensor=bt.subtensor())

    if remote_model_store is None:
        remote_model_store = HuggingFaceModelStore()

    hotkey = metagraph.hotkeys[uid]
    model_metadata = await metadata_store.retrieve_model_metadata(hotkey)
    if not model_metadata:
        raise ValueError(f"No model metadata found for miner {uid}")

    competition = competition_utils.get_competition(model_metadata.id.competition_id)
    if not competition:
        raise ValueError("Invalid competition_id")

    bt.logging.success(f"Fetched model metadata: {model_metadata}")
    model: Model = await remote_model_store.download_model(
        model_metadata.id, download_dir, competition
    )
    return model.pt_model


async def load_best_model(
    download_dir: str,
    competition_id: CompetitionId,
    metagraph: Optional[bt.metagraph] = None,
    metadata_store: Optional[ModelMetadataStore] = None,
    remote_model_store: Optional[RemoteModelStore] = None,
) -> PreTrainedModel:
    """Loads the model from the best performing miner to download_dir"""
    best_uid = ft.graph.best_uid(competition_id=competition_id)
    if best_uid is None:
        raise ValueError(f"No best models found for {competition_id}")

    return await load_remote_model(
        best_uid,
        download_dir,
        metagraph,
        metadata_store,
        remote_model_store,
    )
