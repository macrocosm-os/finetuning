from enum import IntEnum
import math


class NormalizationId(IntEnum):
    """Enumeration of normalization methods."""

    # No normalization is applied and the raw score is used.
    NONE = 0

    # Normalizes between [0, 1] using an inverse exponential function.
    INVERSE_EXPONENTIAL = 1


def normalize_score(
    score: float,
    normalization_id: NormalizationId,
    norm_kwargs: dict = {},
) -> float:
    """Normalizes a score based on the provided normalization method.

    Args:
        score (float): The raw score to normalize.
        normalization_id (NormalizationId): The identifier of the normalization method to use.
        norm_kwargs (dict): Keyword arguments to pass to the normalization method.

    Returns:
        float: The normalized score.
    """
    match normalization_id:
        case NormalizationId.NONE:
            return _normalize_none(score)
        case NormalizationId.INVERSE_EXPONENTIAL:
            return _normalize_inverse_exponential(score, **norm_kwargs)
        case _:
            raise ValueError(f"Unhandled normalization method {normalization_id}.")


def _normalize_inverse_exponential(score: float, ceiling: float) -> float:
    """Normalizes between [0, 1] using an inverse exponential function.

    Args:
        score (float): The raw score to normalize.
        ceiling (float): The maximum desirable score. Everything above this will be clipped.

    Returns:
        float: The normalized score.
    """
    if score >= ceiling:
        return 1.0

    return (1 - math.exp(-score / ceiling)) / (1 - math.exp(-1))


def _normalize_none(score: float) -> float:
    return score
