from abc import ABC, abstractmethod
from enum import IntEnum


class RuleId(IntEnum):
    """Enumeration of IfEval rules."""

    NONE = 0

    # Word count must be at most a specified threshold.
    WORD_COUNT_AT_MOST = 1

    # Word count must be at least a specified threshold.
    WORD_COUNT_AT_LEAST = 2


class IFEvalRule(ABC):
    """Base class for all IFEval rules."""

    def __init__(self, rule_id: RuleId):
        self.rule_id = rule_id

    @abstractmethod
    def get_prompt(self) -> str:
        """Returns the prompt for this rule."""
        pass

    @abstractmethod
    def matches(self, text: str) -> bool:
        """Returns True if the text matches the rule."""
        pass
