import string
from typing import List


def get_words(text: str) -> List[str]:
    """Returns the list of words from a string."""
    no_punc = text.translate(str.maketrans("", "", string.punctuation))
    return no_punc.split()
