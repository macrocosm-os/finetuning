# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 const

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

import asyncio
import datetime as dt
import math
import os
import typing

import bittensor as bt
import torch
import wandb
from dotenv import load_dotenv
from taoverse.metagraph import utils as metagraph_utils
from taoverse.model.data import Model
from taoverse.model.storage.chain.chain_model_metadata_store import (
    ChainModelMetadataStore,
)
from taoverse.model.storage.hugging_face.hugging_face_model_store import (
    HuggingFaceModelStore,
)
from taoverse.model.storage.model_metadata_store import ModelMetadataStore
from taoverse.utilities import utils as taoverse_utils
from taoverse.utilities import wandb as wandb_utils

import constants
import finetune as ft
from neurons import config as neuron_config
import taoverse.utilities.logging as logging


load_dotenv()  # take environment variables from .env.

os.environ["TOKENIZERS_PARALLELISM"] = "true"


async def load_starting_model(
    config: bt.config,
    metagraph: bt.metagraph,
    metadata_store: ModelMetadataStore,
    kwargs: typing.Dict[str, typing.Any],
) -> Model:
    """Loads the model to train based on the provided config."""

    # Initialize the model based on the best on the network.
    if config.load_best:
        model = await ft.mining.load_best_model(
            download_dir=config.model_dir,
            competition_id=config.competition_id,
            metagraph=metagraph,
            metadata_store=metadata_store,
        )
        logging.info(
            f"Training with best model from competition: {config.competition_id}. Model={str(model)}"
        )
        return model

    # Initialize the model based on a passed uid.
    if config.load_uid is not None:
        # Sync the state from the passed uid.
        model = await ft.mining.load_remote_model(
            config.load_uid,
            config.model_dir,
            metagraph=metagraph,
            metadata_store=metadata_store,
        )
        logging.info(
            f"Training with model from uid: {config.load_uid}. Model={str(model)}"
        )
        return model

    # Check if we should load a model from a local directory.
    if config.load_model_dir:
        model = ft.mining.load_local_model(config.load_model_dir, kwargs)
        logging.info(f"Training with model from disk. Model={str(model)}")
        return model

    raise RuntimeError(
        "No starting model specified, pass either --load_best, --load_uid, or --load_model_dir"
    )


async def main(config: bt.config):
    raise NotImplementedError("You must implement your own training logic in miner.py")

    # Create bittensor objects.
    bt.logging(config=config)
    taoverse_utils.configure_logging(config)

    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)
    metagraph = subtensor.metagraph(config.netuid)
    chain_metadata_store = ChainModelMetadataStore(
        subtensor=subtensor,
        subnet_uid=config.netuid,
        wallet=wallet,
    )

    # If running online, make sure the miner is registered, has a hugging face access token, and has provided a repo id.
    my_uid = None
    if not config.offline:
        my_uid = metagraph_utils.assert_registered(wallet, metagraph)
        HuggingFaceModelStore.assert_access_token_exists()

    # Data comes from Subnet 1's wandb project. Make sure we're logged in
    wandb_utils.login()

    # Create a unique run id for this run.
    run_id = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_dir = ft.mining.model_path(config.model_dir, run_id)
    os.makedirs(model_dir, exist_ok=True)

    use_wandb = False
    if not config.offline:
        if config.wandb_project is None or config.wandb_entity is None:
            logging.warning(
                "Wandb project or entity not specified. This run will not be logged to wandb"
            )
        else:
            use_wandb = True

    model_constraints = constants.MODEL_CONSTRAINTS_BY_COMPETITION_ID.get(
        config.competition_id, None
    )

    if not model_constraints:
        raise RuntimeError(f"No competition found for {config.competition_id}")
    kwargs = model_constraints.kwargs.copy()
    kwargs["torch_dtype"] = (
        torch.bfloat16 if config.dtype == "bfloat16" else torch.float16
    )

    # Init model.
    tokenizer = ft.model.load_tokenizer(model_constraints, cache_dir=config.model_dir)
    model = await load_starting_model(config, metagraph, chain_metadata_store, kwargs)
    model = model.train()
    model = model.to(config.device)

    logging.info(f"Saving model to path: {model_dir}.")
    ft.mining.save(model, model_dir)

    # Build optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    wandb_run = None

    # If using wandb, start a new run.
    if use_wandb:
        token = os.getenv("WANDB_API_KEY")
        if not token:
            raise ValueError(
                "To use Wandb, you must set WANDB_API_KEY in your .env file"
            )

        wandb.login(key=token)

        wandb_run = wandb.init(
            name=run_id,
            entity=config.wandb_entity,
            project=config.wandb_project,
            config={
                "uid": my_uid,
                "hotkey": wallet.hotkey.ss58_address,
                "run_name": run_id,
                "version": constants.__version__,
                "type": "miner",
            },
            allow_val_change=True,
        )
    else:
        logging.warning(
            "Not posting run to wandb. Either --offline is specified or the wandb settings are missing."
        )

    # Start the training loop
    epoch_step = 0
    global_step = 0
    n_acc_steps = 0
    best_avg_deviation = math.inf
    accumulation_steps = config.accumulation_steps

    try:
        while epoch_step < config.num_epochs or config.num_epochs == -1:
            # Implement your training logic here.
            avg_deviation = math.inf

            # Check if the average deviation of this epoch is the best we've seen so far
            if avg_deviation < best_avg_deviation:
                best_avg_deviation = avg_deviation  # Update the best average deviation

                logging.info(f"New best average deviation: {best_avg_deviation}.")

                # Save the model to your mining dir.
                logging.info(f"Saving model to path: {model_dir}.")
                ft.mining.save(model, model_dir)

        logging.info("Finished training")
        # Push the model to your run.
        if not config.offline:
            if best_avg_deviation < config.avg_loss_upload_threshold:
                logging.info(
                    f"Trained model had a best_avg_deviation of {best_avg_deviation} which is below the threshold of {config.avg_loss_upload_threshold}. Uploading to hugging face. "
                )

                # First, reload the best model from the training run.
                model_to_upload = ft.mining.load_local_model(
                    model_dir, model_constraints.kwargs
                )
                await ft.mining.push(
                    model_to_upload,
                    config.hf_repo_id,
                    config.competition_id,
                    wallet,
                    update_repo_visibility=config.update_repo_visibility,
                    metadata_store=chain_metadata_store,
                )
            else:
                logging.info(
                    f"This training run achieved a best_avg_deviation={best_avg_deviation}, which did not meet the upload threshold. Not uploading to hugging face."
                )
        else:
            logging.info(
                "Not uploading to hugging face because --offline was specified."
            )

    finally:
        # Important step.
        if wandb_run:
            wandb_run.finish()


if __name__ == "__main__":
    # Parse and print configuration
    config = neuron_config.miner_config()

    if config.list_competitions:
        print(constants.COMPETITION_SCHEDULE_BY_BLOCK)
    else:
        print(config)
        asyncio.run(main(config))
