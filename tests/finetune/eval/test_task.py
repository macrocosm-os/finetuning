import unittest
from finetune.eval.task import EvalTask
from finetune.eval.method import EvalMethodId
from finetune.eval.normalization import NormalizationId


class TestEvalTask(unittest.TestCase):

    def test_eval_task_initialization(self):
        task = EvalTask(
            name="Test Task",
            method_id=EvalMethodId.MULTIPLE_CHOICE,
            normalization_id=NormalizationId.NONE,
        )
        self.assertEqual(task.name, "Test Task")
        self.assertEqual(task.method_id, EvalMethodId.MULTIPLE_CHOICE)
        self.assertEqual(task.normalization_id, NormalizationId.NONE)
        self.assertEqual(task.weight, 1.0)

    def test_eval_task_weight_validation(self):
        with self.assertRaises(ValueError):
            EvalTask(
                name="Test Task",
                method_id=EvalMethodId.MULTIPLE_CHOICE,
                normalization_id=NormalizationId.NONE,
                weight=0,
            )

    def test_eval_task_normalization_kwargs_validation_none(self):
        with self.assertRaises(ValueError):
            EvalTask(
                name="Test Task",
                method_id=EvalMethodId.MULTIPLE_CHOICE,
                normalization_id=NormalizationId.NONE,
                normalization_kwargs={"some_key": "some_value"},
            )

    def test_eval_task_normalization_kwargs_validation_inverse_exponential(self):
        with self.assertRaises(ValueError):
            EvalTask(
                name="Test Task",
                method_id=EvalMethodId.MULTIPLE_CHOICE,
                normalization_id=NormalizationId.INVERSE_EXPONENTIAL,
            )


if __name__ == "__main__":
    unittest.main()
