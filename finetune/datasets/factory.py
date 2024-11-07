from finetune.datasets.generated.dyck_loader import DyckLoader
from finetune.datasets.generated.word_sorting_loader import WordSortingLoader
from finetune.datasets.hugging_face.hugging_face_loader import (
    HuggingFaceLoader,
    FINEWEB_EDU_SCORE_2_NAME,
)
from finetune.datasets.ids import DatasetId
from typing import Dict, Any


class DatasetLoader:
    @staticmethod
    def get_loader(
        dataset_id: DatasetId, dataset_kwargs: Dict[str, Any], seed: int
    ) -> "DatasetLoader":
        """Loads data samples from the appropriate dataset."""

        match dataset_id:
            case DatasetId.DYCK_LANGUAGE:
                return DyckLoader(**dataset_kwargs, random_seed=seed)
            case DatasetId.SYNTHETIC_MMLU:
                raise NotImplementedError(
                    "Prompting dataset is not implemented and should be loaded elsewhere."
                )
            case DatasetId.WORD_SORTING:
                return WordSortingLoader(**dataset_kwargs, random_seed=seed)
            case DatasetId.FINEWEB:
                return HuggingFaceLoader(
                    name=FINEWEB_EDU_SCORE_2_NAME, random_seed=seed
                )
            case _:
                raise ValueError(f"Unknown dataset_id: {dataset_id}")
