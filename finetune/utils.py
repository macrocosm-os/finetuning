import math
import bittensor as bt
import datetime as dt


def get_block_timestamp(subtensor: bt.subtensor, block_number: int) -> dt.datetime:
    """Returns a timezone-aware datetime object for the timestamp of the block."""
    block_data = subtensor.substrate.get_block(block_number=block_number)
    timestamp = block_data["extrinsics"][0]["call"]["call_args"][0]["value"]
    timestamp_seconds = timestamp.value / 1000
    return dt.datetime.fromtimestamp(timestamp_seconds).astimezone(dt.timezone.utc)


def get_hash_of_block(subtensor: bt.subtensor, block_number: int) -> int:
    """Returns the hash of the block at the given block number."""
    return hash(subtensor.get_block_hash(block_number))


def get_sync_block(block: int, sync_cadence: int, genesis: int = 0) -> int:
    """Returns the most recent sync block that is on or before `block`.

    Args:
        block (int): The block number.
        sync_cadence (int): The cadence of blocks to sync on.
        genesis (int, optional): The genesis block number. Defaults to 0. This can be used to synchronize on a subnet's epoch.
    """
    sync_block = (block - genesis) // sync_cadence * sync_cadence + genesis
    return sync_block


def get_next_sync_block(block: int, sync_cadence: int, genesis: int = 0) -> int:
    """Returns the next sync block that is after "block"

    Args:
        block (int): The block number.
        sync_cadence (int): The cadence of blocks to sync on.
        genesis (int, optional): The genesis block number. Defaults to 0. This can be used to synchronize on a subnet's epoch.
    """
    sync_block = (
        int(math.ceil((block - genesis) / sync_cadence)) * sync_cadence + genesis
    )
    # Make sure the sync_block is strictly after the block.
    if sync_block == block:
        sync_block += sync_cadence
    return sync_block
