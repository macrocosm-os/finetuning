import dataclasses

from finetune.datasets.ids import DatasetId
from finetune.eval.method import EvalMethodId
from finetune.eval.normalization import NormalizationId


@dataclasses.dataclass
class EvalTask:
    """Represents a task to evaluate a model on."""

    # Friendly name for the task.
    name: str

    # The dataset to use for this evaluation task.
    dataset_id: DatasetId

    # The identifier of the evaluation method to use.
    method_id: EvalMethodId

    # The identifier of the normalization method to use.
    normalization_id: NormalizationId

    # Additional keyword arguments to pass to the normalization method.
    normalization_kwargs: dict = dataclasses.field(default_factory=dict)

    # Weight to apply to the normalized score to provide relative weight against other EvalTasks.
    weight: float = 1.0

    def __post_init__(self):
        if self.weight <= 0:
            raise ValueError("Weight must be positive.")

        # Validate normalization kwargs.
        match self.normalization_id:
            case NormalizationId.NONE:
                if self.normalization_kwargs:
                    raise ValueError(
                        "Normalization kwargs should not be provided for NONE normalization."
                    )
            case NormalizationId.INVERSE_EXPONENTIAL:
                if "ceiling" not in self.normalization_kwargs:
                    raise ValueError(
                        "Normalization kwargs must contain a 'ceiling' value."
                    )

        # Verify the evaluation method is compatible with the dataset.
        match self.dataset_id:
            case DatasetId.MMLU:
                if self.method_id not in (EvalMethodId.MULTIPLE_CHOICE):
                    raise ValueError(
                        f"{self.method_id} is not a valid eval for {self.dataset_id}"
                    )
            case DatasetId.WORD_SORTING:
                if self.method_id not in (EvalMethodId.REFERENCE_LOSS):
                    raise ValueError(
                        f"{self.method_id} is not a valid eval for {self.dataset_id}"
                    )
            case DatasetId.DYCK_LANGUAGE:
                if self.method_id not in (EvalMethodId.REFERENCE_LOSS):
                    raise ValueError(
                        f"{self.method_id} is not a valid eval for {self.dataset_id}"
                    )
