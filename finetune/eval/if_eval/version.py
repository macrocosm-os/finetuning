from enum import IntEnum


class IfEvalVersion(IntEnum):
    """Unique and ordered identifiers for each IfEval version."""

    NONE = 0

    # Initially release version with casing, comma, sentence count, and word count rules.
    V1 = 1

    # Adds X, X, and X rules.
    V2 = 2

    V3 = 3
