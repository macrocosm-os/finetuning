from enum import IntEnum


class DatasetId(IntEnum):
    """Enumeration of the available datasets."""

    NONE = 0

    # Synthetic MMLU from SN 1.
    MMLU = 1

    # Synthetic version of BBH's word sorting task.
    WORD_SORTING = 2

    # Synthetic version of BBH's dyck language task.
    DYCK_LANGUAGE = 3
