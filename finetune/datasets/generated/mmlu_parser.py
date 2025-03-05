import re
from typing import Tuple


def extract_q_and_a_text(full_prompt: str, answer_char: str) -> Tuple[str, str] | None:
    """Given a prompt and golden answer from the prompting subnet, returns the question and answer text.

    Args:
        full_prompt: The complete prompt from the prompting subnet.
        answer_char: The correct answer character (A, B, C, or D).

    Returns:
        A tuple containing the question text and the answer text, or None if the prompt does not match the expected format.
    """

    if answer_char not in ("A", "B", "C", "D"):
        return None

    regex = "(.*)\s+A\. (.*)\s+B\. (.*)\s+C\. (.*)\s+D\. (.*)"
    match = re.search(regex, full_prompt)
    if not match:
        return None

    # Map the correct answer letter to the matching regex group.
    ans_to_index = {"A": 2, "B": 3, "C": 4, "D": 5}
    question_text = match.group(1)
    answer_text = match.group(ans_to_index[answer_char])
    return question_text, answer_text
