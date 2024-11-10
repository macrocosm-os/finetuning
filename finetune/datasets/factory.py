from finetune.datasets.generated.dyck_loader import DyckLoader
from finetune.datasets.generated.if_eval_loader import IfEvalLoader
from finetune.datasets.generated.word_sorting_loader import WordSortingLoader
from finetune.datasets.hugging_face.hugging_face_loader import (
    HuggingFaceLoader,
    FINEWEB_EDU_SCORE_2_NAME,
)
from finetune.datasets.ids import DatasetId
from typing import Dict, Any, Set


class DatasetLoader:
    @staticmethod
    def get_loader(
        dataset_id: DatasetId,
        dataset_kwargs: Dict[str, Any],
        seed: int,
        validator_hotkeys: Set[str],
    ) -> "DatasetLoader":
        """Loads data samples from the appropriate dataset."""

        match dataset_id:
            case DatasetId.DYCK_LANGUAGE:
                return DyckLoader(random_seed=seed, **dataset_kwargs)
            case DatasetId.SYNTHETIC_MMLU:
                raise NotImplementedError(
                    "Prompting dataset is not implemented and should be loaded elsewhere."
                )
            case DatasetId.WORD_SORTING:
                return WordSortingLoader(random_seed=seed, **dataset_kwargs)
            case DatasetId.FINEWEB:
                return HuggingFaceLoader(
                    name=FINEWEB_EDU_SCORE_2_NAME, random_seed=seed
                )
            case DatasetId.SYNTHETIC_IF_EVAL:
                return IfEvalLoader(
                    random_seed=seed,
                    validator_hotkeys=validator_hotkeys,
                    **dataset_kwargs,
                )
            case _:
                raise ValueError(f"Unknown dataset_id: {dataset_id}")
