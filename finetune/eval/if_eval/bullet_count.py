from finetune.eval.if_eval.rule import IFEvalRule, RuleId


class BulletFrequencyRule(IFEvalRule):
    """Rule that enforces an exact amount of * bullet points."""

    def __init__(self, count: int):
        super().__init__(rule_id=RuleId.BULLET_COUNT_FREQUENCY)

        if count < 1:
            raise ValueError(
                f"BulletFrequencyRule must expect at least 1 bullet point."
            )
        self.count = count

    def get_prompt(self, index: int = -1) -> str:
        bullet = "bullet point" if self.count == 1 else "bullet points"
        return f"The response must contain exactly {self.count} {bullet} in markdown format."

    def matches(self, text: str, index: int = -1) -> bool:
        return (
            sum(1 for line in text.splitlines() if line.lstrip().startswith("*"))
            == self.count
        )
