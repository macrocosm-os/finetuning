import concurrent
import functools
import multiprocessing
import os
from typing import Any, List, Optional, Set, Tuple

import bittensor as bt

import constants
from model.data import ModelId


def assert_registered(wallet: bt.wallet, metagraph: bt.metagraph) -> int:
    """Asserts the wallet is a registered miner and returns the miner's UID.

    Raises:
        ValueError: If the wallet is not registered.
    """
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        raise ValueError(
            f"You are not registered. \nUse: \n`btcli s register --netuid {metagraph.netuid}` to register via burn \n or btcli s pow_register --netuid {metagraph.netuid} to register with a proof of work"
        )
    uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
    bt.logging.success(
        f"You are registered with address: {wallet.hotkey.ss58_address} and uid: {uid}"
    )

    return uid


def get_top_miners(
    metagraph: bt.metagraph, min_vali_stake: int, min_miner_weight_percent: float
) -> Set[int]:
    """Returns the set of top miners, chosen based on weights set on the valis above the specifed threshold.

    Args:
        metagraph (bt.metagraph): Metagraph to use. Must not be lite.
        min_vali_stake (int): Minimum stake threshold for a vali's weights to be considered.
        min_miner_weight_percent (float): Minimum weight on a vali for the miner to count as a top miner.
    """

    top_miners = set()

    # Find validators over 100k in stake.
    valis_by_stake = get_high_stake_validators(metagraph, min_vali_stake)

    # For each, find miners with at least min_miner_weight_percent of the weights.
    # Since there can be multiple competitions at different reward percentages we can't just check biggest.
    for uid in valis_by_stake:
        # Weights is a list of (uid, weight) pairs
        weights: List[Tuple[int, float]] = metagraph.neurons[uid].weights
        total_weight = sum(weight for _, weight in weights)

        threshold = total_weight * min_miner_weight_percent
        for uid, weight in weights:
            if weight >= threshold:
                top_miners.add(uid)

    return list(top_miners)


def get_high_stake_validators(metagraph: bt.metagraph, min_stake: int) -> Set[int]:
    """Returns a set of validators at or above the specified stake threshold for the subnet"""
    valis = set()

    for uid, stake in enumerate(metagraph.S):
        # Use vPermit to check for validators rather than vTrust because we'd rather
        # cast a wide net in the case that vTrust is 0 due to an unhealthy state of the
        # subnet.
        if stake >= min_stake and metagraph.validator_permit[uid]:
            valis.add(uid)

    return valis


def validate_hf_repo_id(repo_id: str) -> Tuple[str, str]:
    """Verifies a Hugging Face repo id is valid and returns it split into namespace and name.

    Raises:
        ValueError: If the repo id is invalid.
    """

    if not repo_id:
        raise ValueError("Hugging Face repo id cannot be empty.")

    if not 3 < len(repo_id) <= ModelId.MAX_REPO_ID_LENGTH:
        raise ValueError(
            f"Hugging Face repo id must be between 3 and {ModelId.MAX_REPO_ID_LENGTH} characters."
        )

    parts = repo_id.split("/")
    if len(parts) != 2:
        raise ValueError(
            "Hugging Face repo id must be in the format <org or user name>/<repo_name>."
        )

    return parts[0], parts[1]


def run_in_subprocess(func: functools.partial, ttl: int) -> Any:
    """Runs the provided function on a subprocess with 'ttl' seconds to complete.

    Args:
        func (functools.partial): Function to be run.
        ttl (int): How long to try for in seconds.

    Returns:
        Any: The value returned by 'func'
    """

    def wrapped_func(func: functools.partial, queue: multiprocessing.Queue):
        try:
            result = func()
            queue.put(result)
        except (Exception, BaseException) as e:
            # Catch exceptions here to add them to the queue.
            queue.put(e)

    # Use "fork" (the default on all POSIX except macOS), because pickling doesn't seem
    # to work on "spawn".
    ctx = multiprocessing.get_context("fork")
    queue = ctx.Queue()
    process = ctx.Process(target=wrapped_func, args=[func, queue])

    process.start()

    process.join(timeout=ttl)

    if process.is_alive():
        process.terminate()
        process.join()
        raise TimeoutError(f"Failed to {func.func.__name__} after {ttl} seconds")

    # Raises an error if the queue is empty. This is fine. It means our subprocess timed out.
    result = queue.get(block=False)

    # If we put an exception on the queue then raise instead of returning.
    if isinstance(result, Exception):
        raise result
    if isinstance(result, BaseException):
        raise Exception(f"BaseException raised in subprocess: {str(result)}")

    return result


def run_in_thread(func: functools.partial, ttl: int, name=None) -> Any:
    """Runs the provided function on a thread with 'ttl' seconds to complete.

    Args:
        func (functools.partial): Function to be run.
        ttl (int): How long to try for in seconds.

    Returns:
        Any: The value returned by 'func'
    """

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    try:
        future = executor.submit(func)
        return future.result(timeout=ttl)
    except concurrent.futures.TimeoutError as e:
        bt.logging.error(f"Failed to complete '{name}' within {ttl} seconds.")
        raise TimeoutError(f"Failed to complete '{name}' within {ttl} seconds.") from e
    finally:
        bt.logging.trace(f"Completed {name}")
        executor.shutdown(wait=False)
        bt.logging.trace(f"{name} cleaned up successfully")


def get_version(filepath: str) -> Optional[int]:
    """Loads a version from the provided filepath or None if the file does not exist.

    Args:
        filepath (str): Path to the version file."""
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            line = f.readline()
            if line:
                return int(line)
            return None
    return None


def save_version(filepath: str, version: int):
    """Saves a version to the provided filepath."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(str(version))
