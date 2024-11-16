from abc import ABC, abstractmethod
from enum import IntEnum


class RuleId(IntEnum):
    """Enumeration of IfEval rules."""

    # Word count must be at most a specified threshold.
    WORD_COUNT_AT_MOST = 1

    # Word count must be at least a specified threshold.
    WORD_COUNT_AT_LEAST = 2

    # Sentence count must be at most a specified threshold.
    SENTENCE_COUNT_AT_MOST = 3

    # Sentence count must be at least a specified threshold.
    SENTENCE_COUNT_AT_LEAST = 4

    # All letters in the output are uppercase.
    ALL_UPPER_CASE = 5

    # All letters in the output are lowercase.
    ALL_LOWER_CASE = 6

    # No commas in the output.
    NO_COMMAS = 7

    # The specified word(s) appear at least once.
    # KEYWORD_INCLUSION = 8

    # # The specified word appears a specific number of times.
    # KEYWORD_FREQUENCY = 9

    # # The specified word(s) do not appear.
    # KEYWORD_FORBIDDEN = 10

    # # There must be a specific number of bullet points in the output.
    # BULLET_COUNT_FREQUENCY = 11

    # # The output must start with the specified content.
    # STARTS_WITH = 12

    # # The output must end with the specified content.
    # ENDS_WITH = 13


class IFEvalRule(ABC):
    """Base class for all IFEval rules."""

    def __init__(self, rule_id: RuleId):
        self.rule_id = rule_id

    @abstractmethod
    def get_prompt(self, index: int) -> str:
        """Returns the prompt for this rule for the given context index."""
        pass

    @abstractmethod
    def matches(self, text: str, index: int) -> bool:
        """Returns True if the text matches the rule for the given context index."""
        pass


class DummyRule(IFEvalRule):
    """A dummy rule for tests."""

    def get_prompt(self, index: int = -1) -> str:
        return "This is a dummy rule."

    def matches(self, text: str, index: int = -1) -> bool:
        return True
