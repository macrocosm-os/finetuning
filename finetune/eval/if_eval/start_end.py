import random

from finetune.eval.if_eval.rule import IFEvalRule, RuleId

ENDS_WITH_PHRASES = [
    "Let me know if you have additional questions.",
    "Is there anything else I can help with?",
    "Hope that helps.",
]


class EndsWithRule(IFEvalRule):
    """Rule that enforces the response ends with an exact phrase."""

    def __init__(self):
        super().__init__(rule_id=RuleId.ENDS_WITH)
        self.phrase = random.choice(ENDS_WITH_PHRASES)

    def get_prompt(self, index: int = -1) -> str:
        return f'End your response with the exact phrase "{self.phrase}"'

    def matches(self, text: str, index: int = -1) -> bool:
        # Check matches while ignoring casing.
        return text.lower().endswith(self.phrase.lower())


class QuotationRule(IFEvalRule):
    """Rule that enforces the response starts and ends with a double quotation."""

    def __init__(self):
        super().__init__(rule_id=RuleId.QUOTATION)

    def get_prompt(self, index: int = -1) -> str:
        return "Wrap your entire response in double quotation marks."

    def matches(self, text: str, index: int = -1) -> bool:
        # Check matches while ignoring casing.
        return text.startswith('"') and text.endswith('"')
