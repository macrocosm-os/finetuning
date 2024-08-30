import math
import unittest

from taoverse.model.competition.epsilon import FixedEpsilon
from taoverse.model.data import EvalResult

from model.retry import should_retry_model


class RetryTests(unittest.TestCase):
    def test_should_retry_model_empty_eval_results(self):
        """Verifies that a model is retried if it has never been evaluated."""
        self.assertTrue(should_retry_model(FixedEpsilon(0.005), 10, []))

    def test_should_retry_model_loss_worse_than_winning_model(
        self,
    ):
        """Verifies that the model is not retried if the loss is worse than the winning model."""
        eval_history = [
            EvalResult(
                block=1, score=0.2, winning_model_score=0.19, winning_model_block=1
            )
        ]
        self.assertFalse(
            should_retry_model(
                FixedEpsilon(0.005), curr_block=10, eval_history=eval_history
            )
        )

    def test_should_retry_model_loss_better_than_winning_model(self):
        """Verifies that the model is retried if the loss (accounting for epsilon) is within 0.999 of the winning model's loss."""

        test_cases = [
            # Make loss the same as the winning model and make sure it's never retried.
            (0.005, 1.0, False),
            (0.001, 1.0, False),
            (0.0001, 1.0, False),
            # Make loss better than the winning model by 50% (for easy math) and adjust epsilon to test each interesting case.
            (0.51, 0.5, False),
            (0.5004, 0.5, False),
            (0.49, 0.5, True),
        ]
        for tc in test_cases:
            epsilon, model_loss, should_retry = tc
            print(
                f"Running test with epsilon: {epsilon}, model_loss: {model_loss}, should_retry: {should_retry}"
            )
            eval_history = [
                EvalResult(
                    block=1,
                    score=model_loss,
                    winning_model_score=1.0,
                    winning_model_block=1,
                )
            ]
            self.assertEqual(
                should_retry_model(
                    FixedEpsilon(epsilon), curr_block=10, eval_history=eval_history
                ),
                should_retry,
            )

    def test_should_retry_model_uses_last_successful_eval(self):
        """Verifies that only the last successful evaluation is used to judge if the model should be retried."""

        test_cases = [
            # Test case 1: The last successful eval is worse than the winning model.
            (
                [
                    EvalResult(
                        block=2,
                        score=0.9,
                        winning_model_score=1.0,
                        winning_model_block=1,
                    ),
                    EvalResult(
                        block=4,
                        score=1.1,
                        winning_model_score=1.0,
                        winning_model_block=1,
                    ),
                ],
                False,
            ),
            # Test case 2: The last successful eval is better than the winning model.
            (
                [
                    EvalResult(
                        block=2,
                        score=1.1,
                        winning_model_score=1.0,
                        winning_model_block=1,
                    ),
                    EvalResult(
                        block=4,
                        score=0.9,
                        winning_model_score=1.0,
                        winning_model_block=1,
                    ),
                ],
                True,
            ),
        ]

        for eval_history, expected in test_cases:
            # Also inject eval failures into each position to make sure it doesn't impact the result.
            for i, result in enumerate(eval_history):
                eval_history_copy = eval_history.copy()
                # .insert() inserts at the position before the given index.
                eval_history_copy.insert(
                    i,
                    EvalResult(
                        block=result.block + 1 if i > 0 else 1,
                        score=math.inf,
                        winning_model_score=1.0,
                        winning_model_block=1,
                    ),
                )
                self.assertEqual(
                    should_retry_model(FixedEpsilon(0.005), 10, eval_history_copy),
                    expected,
                )

    def test_should_retry_model_only_failed_evals(
        self,
    ):
        """Verifies that a model is retried if it has only failed evaluations and has not been retried before."""
        test_cases = [
            (1, True),
            (2, False),
            (3, False),
        ]
        for num_failures, expected in test_cases:
            eval_history = [
                EvalResult(
                    block=i,
                    score=math.inf,
                    winning_model_score=1.0,
                    winning_model_block=1,
                )
                for i in range(1, num_failures + 1)
            ]
            self.assertEqual(
                should_retry_model(FixedEpsilon(0.005), 10, eval_history),
                expected,
            )


if __name__ == "__main__":
    unittest.main()
