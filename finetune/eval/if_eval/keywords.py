import abc
import random
from typing import List, Tuple

import nltk

from finetune.eval.if_eval.rule import IFEvalRule, RuleId
from finetune.eval.if_eval.utils import get_words

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", raise_on_error=True)

try:
    nltk.data.find("taggers/averaged_perceptron_tagger_eng")
except LookupError:
    nltk.download("averaged_perceptron_tagger_eng", raise_on_error=True)

_DESIRABLE_TAGS = {
    "JJ",  # Adjective
    "JJR",  # Adjective, comparative
    "JJS",  # Adjective, superlative
    "NN",  # Noun, singular or mass
    "NNS",  # Noun, plural
    "NNP",  # Proper noun, singular
    "NNPS",  # Proper noun, plural
    "RB",  # Adverb
    "RBR",  # Adverb, comparative
    "RBS",  # Adverb, superlative
    "VB",  # Verb, base form
    "VBD",  # Verb, past tense
    "VBG",  # Verb, gerund or present participle
    "VBN",  # Verb, past participle
    "VBP",  # Verb, non-3rd person singular present
    "VBZ",  # Verb, 3rd person singular present
}


def interesting_keyword(text: str, forbidden_words: List[str]) -> str:
    """Extracts a keyword from the text, where the keyword is an interesting word, when possible.

    Args:
        text: The text to extract a keyword from.
        forbidden_words: A list of words that should not be considered as keywords.
    """

    forbidden_lower = [word.lower() for word in forbidden_words]

    # pos_tag uses the Penn Treebank tagset: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
    tags = nltk.pos_tag(nltk.word_tokenize(text.lower()))

    # Filter out words that are not interesting, including words that are 2 or fewer characters.
    def _should_use_word(word: str, tag: str) -> bool:
        return (
            tag in _DESIRABLE_TAGS
            and len(word) > 2
            and word.lower() not in forbidden_lower
        )

    interesting_words = [word for word, tag in tags if _should_use_word(word, tag)]
    if interesting_words:
        return random.choice(interesting_words)

    # No interesting words found. Choose a random word, removing punctuation first.
    filtered_words = [
        word for word in get_words(text) if word.lower() not in forbidden_lower
    ]
    if not filtered_words:
        return random.choice(["answer", "question", "response"])
    return random.choice(filtered_words)


class KeywordRuleBase(IFEvalRule):

    def __init__(self, rule_id: RuleId):
        super().__init__(rule_id=rule_id)

    @abc.abstractmethod
    def get_keywords(self) -> List[str]:
        """Returns the keywords used by the rule."""
        pass


class KeywordInclusionRule(KeywordRuleBase):
    """Rule that enforces that a specific keyword is included in the response."""

    def __init__(self, keywords: List[str]):
        super().__init__(rule_id=RuleId.KEYWORD_INCLUSION)

        self.keywords = keywords

    def get_prompt(self, index: int) -> str:
        if index >= len(self.keywords):
            raise ValueError("Index is out of range for keywords.")
        return f'The word "{self.keywords[index]}" must be included in the response.'

    def get_keywords(self) -> List[str]:
        return self.keywords

    def matches(self, text: str, index: int) -> bool:
        if index >= len(self.keywords):
            raise ValueError("Index is out of range for keywords.")
        lower_text_words = {word.lower() for word in get_words(text)}
        return self.keywords[index].lower() in lower_text_words


class KeywordFrequencyRule(KeywordRuleBase):
    """Rule that enforces that a specific keyword is included X times in the response."""

    def __init__(self, keywords_and_counts: List[Tuple[str, int]]):
        super().__init__(rule_id=RuleId.KEYWORD_FREQUENCY)

        self.keywords_and_counts = keywords_and_counts

    def get_prompt(self, index: int) -> str:
        if index >= len(self.keywords_and_counts):
            raise ValueError("Index is out of range for keywords.")
        word, count = self.keywords_and_counts[index]
        time_str = "time" if count == 1 else "times"
        return f'Include the word "{word}" {count} {time_str}.'

    def get_keywords(self) -> List[str]:
        return [word for word, _ in self.keywords_and_counts]

    def matches(self, text: str, index: int) -> bool:
        if index >= len(self.keywords_and_counts):
            raise ValueError("Index is out of range for keywords.")
        word, count = self.keywords_and_counts[index]
        lower_text_words = [word.lower() for word in get_words(text)]
        return lower_text_words.count(word.lower()) == count


class KeywordForbiddenRule(KeywordRuleBase):
    """Rule that enforces that a specific keyword is not included in the response."""

    def __init__(self, keywords: List[str]):
        super().__init__(rule_id=RuleId.KEYWORD_FORBIDDEN)

        self.keywords = keywords

    def get_prompt(self, index: int) -> str:
        if index >= len(self.keywords):
            raise ValueError("Index is out of range for keywords.")
        return f'The word "{self.keywords[index]}" must not be the response.'

    def get_keywords(self) -> List[str]:
        return self.keywords

    def matches(self, text: str, index: int) -> bool:
        if index >= len(self.keywords):
            raise ValueError("Index is out of range for keywords.")
        lower_text_words = {word.lower() for word in get_words(text)}
        return self.keywords[index].lower() not in lower_text_words
