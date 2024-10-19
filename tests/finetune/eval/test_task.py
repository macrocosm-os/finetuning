import unittest
from finetune.eval.task import EvalTask
from finetune.eval.method import EvalMethodId
from finetune.eval.normalization import NormalizationId


class TestEvalTask(unittest.TestCase):

    def test_eval_task_initialization(self):
        samples = [("context1", ["A", "B"], "A")]
        task = EvalTask(
            name="Test Task",
            samples=samples,
            method_id=EvalMethodId.MULTIPLE_CHOICE,
            normalization_id=NormalizationId.NONE,
        )
        self.assertEqual(task.name, "Test Task")
        self.assertEqual(task.samples, samples)
        self.assertEqual(task.method_id, EvalMethodId.MULTIPLE_CHOICE)
        self.assertEqual(task.normalization_id, NormalizationId.NONE)
        self.assertEqual(task.weight, 1.0)

    def test_eval_task_weight_validation(self):
        with self.assertRaises(ValueError):
            EvalTask(
                name="Test Task",
                samples=[],
                method_id=EvalMethodId.MULTIPLE_CHOICE,
                normalization_id=NormalizationId.NONE,
                weight=0,
            )

    def test_eval_task_normalization_kwargs_validation_none(self):
        with self.assertRaises(ValueError):
            EvalTask(
                name="Test Task",
                samples=[],
                method_id=EvalMethodId.MULTIPLE_CHOICE,
                normalization_id=NormalizationId.NONE,
                normalization_kwargs={"some_key": "some_value"},
            )

    def test_eval_task_normalization_kwargs_validation_inverse_exponential(self):
        with self.assertRaises(ValueError):
            EvalTask(
                name="Test Task",
                samples=[],
                method_id=EvalMethodId.MULTIPLE_CHOICE,
                normalization_id=NormalizationId.INVERSE_EXPONENTIAL,
            )

    def test_eval_task_sample_validation_multiple_choice(self):
        with self.assertRaises(ValueError):
            EvalTask(
                name="Test Task",
                samples=[("context1", ["choice1", "choice2"])],
                method_id=EvalMethodId.MULTIPLE_CHOICE,
                normalization_id=NormalizationId.NONE,
            )

    def test_eval_task_sample_validation_reference_loss(self):
        with self.assertRaises(ValueError):
            EvalTask(
                name="Test Task",
                samples=[("context1",)],
                method_id=EvalMethodId.REFERENCE_LOSS,
                normalization_id=NormalizationId.NONE,
            )


if __name__ == "__main__":
    unittest.main()
