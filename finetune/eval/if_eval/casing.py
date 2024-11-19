from finetune.eval.if_eval.rule import IFEvalRule, RuleId


class UppercaseRule(IFEvalRule):
    """Rule that enforces all letters are uppercase."""

    def __init__(self):
        super().__init__(rule_id=RuleId.ALL_UPPER_CASE)

    def get_prompt(self, index: int = -1) -> str:
        return "All letters in the response must be uppercase."

    def matches(self, text: str, index: int = -1) -> bool:
        return text.isupper()


class LowercaseRule(IFEvalRule):
    """Rule that enforces all letters are lowercase."""

    def __init__(self):
        super().__init__(rule_id=RuleId.ALL_LOWER_CASE)

    def get_prompt(self, index: int = -1) -> str:
        return "All letters in the response must be lowercase."

    def matches(self, text: str, index: int = -1) -> bool:
        return text.islower()
