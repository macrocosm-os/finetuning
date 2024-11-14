from finetune.eval.if_eval.rule import IFEvalRule, RuleId


class WordCountAtMostRule(IFEvalRule):
    """Rule that enforces word count must be at most a specified threshold."""

    def __init__(self, count: int):
        super().__init__(rule_id=RuleId.WORD_COUNT_AT_MOST)

        if count < 1:
            raise ValueError(f"WordCountAtMostRule must allow at least 1 word.")
        self.count = count

    def get_prompt(self, _: int) -> str:
        word = "word" if self.count == 1 else "words"
        return f"The response must be no more than {self.count} {word}."

    def matches(self, text: str, _: int) -> bool:
        return len(text.split()) <= self.count


class WordCountAtLeastRule(IFEvalRule):
    """Rule that enforces word count must at least a specified threshold."""

    def __init__(self, count: int):
        super().__init__(rule_id=RuleId.WORD_COUNT_AT_LEAST)

        if count < 1:
            raise ValueError(f"WordCountAtLeastRule must expect at least 1 word.")
        self.count = count

    def get_prompt(self, _: int) -> str:
        word = "word" if self.count == 1 else "words"
        return f"The response must be at least {self.count} {word}."

    def matches(self, text: str, _: int) -> bool:
        return len(text.split()) >= self.count
