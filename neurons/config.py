import argparse
import os

import bittensor as bt
import torch

from competitions.data import CompetitionId
import constants
from utilities.enum_action import IntEnumAction


def validator_config():
    """Returns the config for the validator."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device name.",
    )
    parser.add_argument(
        "--wandb_project",
        default=constants.WANDB_PROJECT,
        help="Turn on wandb logging (and log to this project)",
    )
    parser.add_argument(
        "--wandb_entity",
        default=constants.WANDB_ENTITY,
        help="wandb entity for logging (if --wandb_project set)",
    )
    parser.add_argument(
        "--wandb_max_steps_per_run",
        default=50,
        type=int,
        help="number of steps before creating a new wandb run",
    )
    parser.add_argument(
        "--blocks_per_epoch",
        type=int,
        default=50,
        help="Number of blocks to wait before setting weights.",
    )
    parser.add_argument(
        "--latest_cortex_steps",
        type=int,
        default=5,
        help="Number of most recent Cortex steps to sample data from",
    )
    parser.add_argument(
        "--latest_cortex_samples",
        type=int,
        default=400,
        help="Number of most recent Cortex samples to eval against",
    )
    parser.add_argument(
        "--sample_min",
        type=int,
        default=5,
        help="Number of uids to keep after evaluating a competition.",
    )
    parser.add_argument(
        "--updated_models_limit",
        type=int,
        default=15,
        help="Max number of uids that can be either pending eval or currently being evaluated.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Does not launch a wandb run, does not set weights, does not check that your key is registered.",
    )
    parser.add_argument(
        "--model_dir",
        default=os.path.join(constants.ROOT_DIR, "model-store/"),
        help="Where to store downloaded models",
    )
    parser.add_argument(
        "--netuid", type=str, default=constants.SUBNET_UID, help="The subnet UID."
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Sample a response from each model (for leaderboard)",
    )

    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    bt.axon.add_args(parser)
    config = bt.config(parser)
    return config


def miner_config():
    """
    Returns the miner configuration.
    """

    # Initialize an argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--offline",
        action="store_true",
        help="Does not launch a wandb run, does not send model to wandb, does not check if registered",
    )
    parser.add_argument(
        "--wandb_project", type=str, help="The wandb project to log to."
    )
    parser.add_argument("--wandb_entity", type=str, help="The wandb entity to log to.")
    parser.add_argument(
        "--hf_repo_id",
        type=str,
        help="The hugging face repo id, which should include the org or user and repo name. E.g. jdoe/finetuned",
    )
    parser.add_argument(
        "--update_repo_visibility",
        action="store_false",
        help="If true, the repo will be made public after uploading.",
    )
    parser.add_argument(
        "--avg_loss_upload_threshold",
        type=float,
        default=0,  # Default to never uploading.
        help="The threshold for avg_loss the model must achieve to upload it to hugging face. A miner can only advertise one model, so it should be the best one.",
    )
    parser.add_argument(
        "--model_dir",
        default=os.path.join(constants.ROOT_DIR, "local-models/"),
        help="Where to download/save models for training",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="The device on which to run. cpu or cuda",
    )
    parser.add_argument(
        "--load_best",
        action="store_true",
        help="If set, the miner loads the best model from wandb to train off.",
    )
    parser.add_argument(
        "--load_uid",
        type=int,
        default=None,
        help="If passed loads the model under the specified uid.",
    )
    parser.add_argument(
        "--load_model_dir",
        type=str,
        default=None,
        help="If provided, loads a previously trained HF model from the specified directory",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=-1,
        help="Number of training epochs (-1 is infinite)",
    )
    parser.add_argument("--lr", type=float, default=0.00001, help="Learning rate.")
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=32,
        help="The number of training accumulation steps.",
    )
    parser.add_argument(
        "--cortex_steps",
        type=int,
        default=5,
        help="Number of Cortex steps to sample data from",
    )
    parser.add_argument(
        "--cortex_samples_per_epoch",
        type=int,
        default=4096,
        help="Number of samples trained on per epoch",
    )
    parser.add_argument(
        "--attn_implementation",
        default="flash_attention_2",
        help="Implementation of attention to use",
    )
    parser.add_argument(
        "--netuid",
        type=str,
        default=constants.SUBNET_UID,
        help="The subnet UID.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="datatype to load model in, either bfloat16 or float16",
    )
    parser.add_argument(
        "--competition_id",
        type=CompetitionId,
        default=CompetitionId.SN9_MODEL.value,
        action=IntEnumAction,
        help="competition to mine for (use --list-competitions to get all competitions)",
    )
    parser.add_argument(
        "--list_competitions", action="store_true", help="Print out all competitions"
    )

    # Include wallet and logging arguments from bittensor
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)

    # Parse the arguments and create a configuration namespace
    config = bt.config(parser)

    return config
