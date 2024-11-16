from finetune.eval.if_eval.rule import IFEvalRule, RuleId


class UppercaseRule(IFEvalRule):
    """Rule that enforces all letters are uppercase."""

    def __init__(self):
        super().__init__(rule_id=RuleId.ALL_UPPER_CASE)

    def get_prompt(self, _: int) -> str:
        return "All letters in the response must be uppercase."

    def matches(self, text: str, _: int) -> bool:
        return text.isupper()


class LowercaseRule(IFEvalRule):
    """Rule that enforces all letters are lowercase."""

    def __init__(self):
        super().__init__(rule_id=RuleId.ALL_LOWER_CASE)


    def get_prompt(self, _: int) -> str:
        return "All letters in the response must be lowercase."

    def matches(self, text: str, _: int) -> bool:
        return text.islower()
