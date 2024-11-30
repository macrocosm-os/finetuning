from enum import IntEnum


class CompetitionId(IntEnum):
    """Unique identifiers for each competition."""

    NONE = 0

    # This competition has been deprecated.
    SN9_MODEL = 1

    B7_MULTI_CHOICE = 2

    INSTRUCT_8B = 3

    # Overwrite the default __repr__, which doesn't work with
    # bt.logging for some unknown reason.
    def __repr__(self) -> str:
        return f"{self.value}"
