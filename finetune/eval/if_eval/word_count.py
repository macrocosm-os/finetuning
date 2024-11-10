from finetune.eval.if_eval.rule import IFEvalRule, RuleConstraint, RuleId


class WordCountRule(IFEvalRule):
    """Rule that enforces a certain word count."""

    def __init__(self, count: int, constraint: RuleConstraint):
        super().__init__(rule_id=RuleId.WORD_COUNT)

        if constraint not in (
            RuleConstraint.GREATER_THAN_EQ,
            RuleConstraint.LESS_THAN_EQ,
        ):
            raise ValueError(f"Invalid constraint: {constraint}")

        self.count = count
        self.constraint = constraint

    def get_prompt(self) -> str:
        constraint = (
            "at least"
            if self.constraint == RuleConstraint.GREATER_THAN_EQ
            else "no more than"
        )
        word = "word" if self.count == 1 else "words"
        return f"The response must be {constraint} {self.count} {word}."

    def matches(self, text: str) -> bool:
        words = text.split()
        match self.constraint:
            case RuleConstraint.GREATER_THAN_EQ:
                return len(words) >= self.count
            case RuleConstraint.LESS_THAN_EQ:
                return len(words) <= self.count
            case _:
                raise ValueError(f"Invalid constraint: {self.constraint}")
