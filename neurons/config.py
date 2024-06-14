import argparse
import os

import bittensor as bt

import constants


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
        default=100,
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
    # TODO: Consider having a per competition limit instead of sharing across competitions.
    # As it is now, with 2 competitions, each will have 5 reserved slots but one could have all 20 new slots.
    # TODO: Also consider starting at 30 and reducing by sample min per competition. Less 'correct' at 1 or 6+.
    parser.add_argument(
        "--updated_models_limit",
        type=int,
        default=15 * len(constants.COMPETITION_SCHEDULE),
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
