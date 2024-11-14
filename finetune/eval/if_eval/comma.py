from finetune.eval.if_eval.rule import IFEvalRule, RuleId


class NoCommaRule(IFEvalRule):
    """Rule that enforces no commas."""

    def __init__(self):
        super().__init__(rule_id=RuleId.NO_COMMAS)

    def get_prompt(self) -> str:
        return "Do not use any commas in your response."

    def matches(self, text: str) -> bool:
        return "," not in text
