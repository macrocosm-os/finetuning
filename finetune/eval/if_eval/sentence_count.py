import nltk
from finetune.eval.if_eval.rule import IFEvalRule, RuleId

try:
    from nltk.tokenize.punkt import PunktSentenceTokenizer
except:
    nltk.download("punkt", raise_on_error=True)

class SentenceCountAtMostRule(IFEvalRule):
    """Rule that enforces sentence count must be at most a specified threshold."""

    def __init__(self, count: int):
        super().__init__(rule_id=RuleId.SENTENCE_COUNT_AT_MOST)

        if count < 1:
            raise ValueError(f"SentenceCountAtMostRule must allow at least 1 sentence.")
        self.count = count

    def get_prompt(self, _: int) -> str:
        sentence = "sentence" if self.count == 1 else "sentences"
        return f"The response must be no more than {self.count} {sentence}."

    def matches(self, text: str, _: int) -> bool:
        tokenizer = PunktSentenceTokenizer()
        return len(tokenizer.tokenize(text)) <= self.count


class SentenceCountAtLeastRule(IFEvalRule):
    """Rule that enforces sentence count must at least a specified threshold."""

    def __init__(self, count: int):
        super().__init__(rule_id=RuleId.SENTENCE_COUNT_AT_LEAST)

        if count < 1:
            raise ValueError(f"SentenceCountAtLeastRule must expect at least 1 sentence.")
        self.count = count


    def get_prompt(self, _: int) -> str:
        sentence = "sentence" if self.count == 1 else "sentences"
        return f"The response must be at least {self.count} {sentence}."

    def matches(self, text: str, _: int) -> bool:
        tokenizer = PunktSentenceTokenizer()
        return len(tokenizer.tokenize(text)) >= self.count
