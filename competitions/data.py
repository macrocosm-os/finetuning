from enum import IntEnum


class CompetitionId(IntEnum):
    """Unique identifiers for each competition."""

    SN9_MODEL = 1

    # Defined for tests. Will be repurposed later.
    COMPETITION_2 = 2

    # Overwrite the default __repr__, which doesn't work with
    # bt.logging for some unknown reason.
    def __repr__(self) -> str:
        return f"{self.value}"