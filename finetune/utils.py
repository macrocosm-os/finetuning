import bittensor as bt
from typing import Dict
from finetune.datasets.ids import DatasetId
from finetune.eval.task import EvalTask


# TODO: Move to taoverse
def hash_of_sync_block(subtensor: bt.Subtensor, sync_cadence: int) -> int:
    """Returns the hash of the most recent block that is a multiple of the sync cadence."""
    current_block = subtensor.get_current_block()
    sync_block = current_block // sync_cadence * sync_cadence
    return hash(subtensor.get_block_hash(sync_block))
