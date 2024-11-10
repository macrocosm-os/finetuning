from abc import ABC, abstractmethod
from enum import IntEnum


class RuleId(IntEnum):
    """Enumeration of IfEval rules."""

    NONE = 0

    WORD_COUNT = 1


class RuleConstraint(IntEnum):
    """Enumeration of common constraints used by rules."""

    NONE = 0

    GREATER_THAN_EQ = 1

    LESS_THAN_EQ = 2

    EQUALS = 3


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
