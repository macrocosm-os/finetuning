import dataclasses
from typing import List

from finetune.eval.method import EvalMethodId
from finetune.eval.normalization import NormalizationId
from finetune.eval.sample import EvalSample


@dataclasses.dataclass
class EvalTask:
    """Represents a task to evaluate a model on."""

    # Friendly name for the task.
    name: str

    # List of samples to provide to the evaluation method.
    samples: List[EvalSample]

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

        # Validate samples based on the evaluation method.
        match self.method_id:
            case EvalMethodId.MULTIPLE_CHOICE:
                for sample in self.samples:
                    if len(sample) != 3:
                        raise ValueError(
                            "For multiple choice evaluation, each sample should be a tuple of "
                            "(context, choices, answer)."
                        )
            case EvalMethodId.REFERENCE_LOSS:
                for sample in self.samples:
                    if len(sample) != 2:
                        raise ValueError(
                            "For reference loss evaluation, each sample should be a tuple of "
                            "(context, reference)."
                        )
