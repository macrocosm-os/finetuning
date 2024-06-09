import functools
import time
import unittest
from unittest import mock

import torch

from utilities import utils
from utilities.utils import run_in_subprocess, run_in_thread


class TestUtils(unittest.TestCase):
    def test_run_in_subprocess(self):
        def test_func(a: int, b: int):
            return a + b

        partial = functools.partial(test_func, 1, 2)

        result = run_in_subprocess(func=partial, ttl=5)
        self.assertEqual(3, result)

    def test_run_in_subprocess_timeout(self):
        def test_func(a: int, b: int):
            time.sleep(3)
            return a + b

        partial = functools.partial(test_func, 1, 2)

        with self.assertRaises(TimeoutError):
            result = run_in_subprocess(func=partial, ttl=1)

    def test_run_in_subprocess_no_return(self):
        def test_func(a: int, b: int):
            pass

        partial = functools.partial(test_func, 1, 2)

        result = run_in_subprocess(func=partial, ttl=5)
        self.assertIsNone(result)

    def test_run_in_subprocess_tuple_return(self):
        def test_func(a: int, b: int):
            return a, b

        partial = functools.partial(test_func, 1, 2)

        result = run_in_subprocess(func=partial, ttl=5)
        self.assertEqual((1, 2), result)

    def test_run_in_subprocess_exception(self):
        def test_func(a: int, b: int):
            raise ValueError()

        partial = functools.partial(test_func, 1, 2)

        with self.assertRaises(ValueError):
            result = run_in_subprocess(func=partial, ttl=5)

    def test_validate_hf_repo_id_too_long(self):
        with self.assertRaises(ValueError) as ve:
            # Max allowed length is 41 characters
            utils.validate_hf_repo_id("my-org/" + "a" * 40)

        self.assertRegex(
            str(ve.exception),
            "Hugging Face repo id must be between 3 and 38 characters",
        )

    def test_validate_hf_repo_id_incorrect_format(self):
        with self.assertRaises(ValueError) as ve:
            utils.validate_hf_repo_id("my-repo-name-without-a-namespace")

        self.assertRegex(
            str(ve.exception), "must be in the format <org or user name>/<repo_name>"
        )

    def test_validate_hf_repo_id_valid(self):
        namespace, name = utils.validate_hf_repo_id("my-org/my-repo-name")
        self.assertEqual("my-org", namespace)
        self.assertEqual("my-repo-name", name)

    def test_run_in_thread(self):
        def test_func(a: int, b: int):
            return a + b

        partial = functools.partial(test_func, 1, 2)

        result = run_in_thread(func=partial, ttl=5)
        self.assertEqual(3, result)

    def test_run_in_thread_timeout(self):
        def test_func(a: int, b: int):
            time.sleep(3)
            return a + b

        partial = functools.partial(test_func, 1, 2)

        with self.assertRaises(TimeoutError):
            result = run_in_thread(func=partial, ttl=1)

    def test_run_in_thread_no_return(self):
        def test_func(a: int, b: int):
            pass

        partial = functools.partial(test_func, 1, 2)

        result = run_in_thread(func=partial, ttl=5)
        self.assertIsNone(result)

    def test_run_in_thread_tuple_return(self):
        def test_func(a: int, b: int):
            return a, b

        partial = functools.partial(test_func, 1, 2)

        result = run_in_thread(func=partial, ttl=5)
        self.assertEqual((1, 2), result)

    def test_run_in_thread_exception(self):
        def test_func(a: int, b: int):
            raise ValueError()

        partial = functools.partial(test_func, 1, 2)

        with self.assertRaises(ValueError):
            result = run_in_thread(func=partial, ttl=5)

    def test_get_high_stake_validators(self):
        # Create a metagraph with 10 neurons of varying stake with the top 4 having a validator permit.
        mock_metagraph = mock.MagicMock()
        mock_metagraph.S = torch.tensor(
            [0, 1, 2, 300, 4, 5, 600, 7, 8, 9], dtype=torch.float32
        )
        mock_metagraph.validator_permit = torch.tensor(
            [
                False,
                False,
                False,
                True,
                False,
                False,
                True,
                False,
                True,
                True,
            ],
            dtype=torch.bool,
        )

        # Check 300 or above stake.
        self.assertEqual(utils.get_high_stake_validators(mock_metagraph, 300), {3, 6})

        # Check 0 or above stake gets only those with permits.
        self.assertEqual(
            utils.get_high_stake_validators(mock_metagraph, 0), {3, 6, 8, 9}
        )


if __name__ == "__main__":
    unittest.main()
