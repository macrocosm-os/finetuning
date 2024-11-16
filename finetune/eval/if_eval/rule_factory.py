import random
from typing import List, Tuple

from finetune.eval.if_eval.rule import DummyRule, IFEvalRule, RuleId
from finetune.eval.if_eval.sample import IFEvalSample
from finetune.eval.if_eval.word_count import WordCountAtLeastRule, WordCountAtMostRule
from finetune.eval.if_eval.sentence_count import SentenceCountAtLeastRule, SentenceCountAtMostRule
from finetune.eval.if_eval.casing import UppercaseRule, LowercaseRule
from finetune.eval.if_eval.comma import NoCommaRule

PROMPT_FORMAT = """Please answer the question below, denoted between quotes. Your response must follow these rules:
{rules}

\"{question}\"
"""


def generate_if_eval_sample(
    qa1: Tuple[str, str], qa2: Tuple[str, str], min_rules: int, max_rules: int
) -> IFEvalSample:
    """Returns an IFEvalSample using the provided pair of Q&A.

    Args:
        qa1: The first question and answer pair.
        qa2: The second question and answer pair.
        min_rules: The minimum number of rules to generate.
        max_rules: The maximum number of rules to generate.
    """
    rule_ids = list(RuleId)
    random.shuffle(rule_ids)

    rules = []
    n_rules = random.randint(min_rules, max_rules)
    for rule_id in rule_ids:
        if len(rules) == n_rules:
            break
        if is_rule_incompatible(rule_id, rules):
            continue

        rules.append(generate_rule(rule_id, rules, qa1, qa2))

    # Now generate the sample.
    return IFEvalSample(
        prompt_1=generate_prompt(qa1[0], rules, 0),
        prompt_2=generate_prompt(qa2[0], rules, 1),
        rules=rules,
    )


def generate_prompt(
    question_text: str, rules: List[IFEvalRule], index_for_rule: int
) -> str:
    """Generates a prompt for the provided question and rules."""
    rule_prompts = "\n".join(
        f"{i+1}. {rule.get_prompt(index_for_rule)}" for i, rule in enumerate(rules)
    )
    return PROMPT_FORMAT.format(rules=rule_prompts, question=question_text)


def generate_rule(
    rule_id: RuleId,
    current_rules: List[IFEvalRule],
    qa1: Tuple[str, str],
    qa2: Tuple[str, str],
) -> IFEvalRule:
    """Generates a rule based on the provided rule_id and existing rules."""
    match rule_id:
        case RuleId.WORD_COUNT_AT_MOST:
            return WordCountAtMostRule(random.choice([x for x in range(50, 150, 5)]))
        case RuleId.WORD_COUNT_AT_LEAST:
            return WordCountAtLeastRule(random.choice([x for x in range(25, 250, 10)]))
        case RuleId.SENTENCE_COUNT_AT_MOST:
            return SentenceCountAtMostRule(random.choice([x for x in range(1, 5)]))
        case RuleId.SENTENCE_COUNT_AT_LEAST:
            return SentenceCountAtLeastRule(random.choice([x for x in range(2, 5)]))
        case RuleId.ALL_UPPER_CASE:
            return UppercaseRule()
        case RuleId.ALL_LOWER_CASE:
            return LowercaseRule()
        case RuleId.NO_COMMAS:
            return NoCommaRule()
        # case RuleId.KEYWORD_INCLUSION:
        #     return DummyRule(rule_id)
        # case RuleId.KEYWORD_FREQUENCY:
        #     return DummyRule(rule_id)
        # case RuleId.KEYWORD_FORBIDDEN:
        #     return DummyRule(rule_id)
        # case RuleId.BULLET_COUNT_FREQUENCY:
        #     return DummyRule(rule_id)
        # case RuleId.STARTS_WITH:
        #     return DummyRule(rule_id)
        # case RuleId.ENDS_WITH:
        #     return DummyRule(rule_id)
        case _:
            raise ValueError(f"RuleId {rule_id} not handled.")


def is_rule_incompatible(rule_id: RuleId, current_rules: List[IFEvalRule]) -> bool:
    """Returns whether the provided rule_id is compatible with a list of existing rules."""
    match rule_id:
        case RuleId.WORD_COUNT_AT_MOST:
            # Not compatible with other length constraints + bullet point count.
            return any(
                rule.rule_id
                in {
                    RuleId.WORD_COUNT_AT_LEAST,
                    RuleId.SENTENCE_COUNT_AT_MOST,
                    RuleId.SENTENCE_COUNT_AT_LEAST,
                    RuleId.BULLET_COUNT_FREQUENCY,
                }
                for rule in current_rules
            )
        case RuleId.WORD_COUNT_AT_LEAST:
            # Not compatible with other length constraints that limit the length.
            return any(
                rule.rule_id
                in {
                    RuleId.WORD_COUNT_AT_MOST,
                    RuleId.SENTENCE_COUNT_AT_MOST,
                }
                for rule in current_rules
            )
        case RuleId.SENTENCE_COUNT_AT_MOST:
            # Not compatible with other length constraints + bullet point count.
            return any(
                rule.rule_id
                in {
                    RuleId.WORD_COUNT_AT_LEAST,
                    RuleId.WORD_COUNT_AT_MOST,
                    RuleId.SENTENCE_COUNT_AT_LEAST,
                    RuleId.BULLET_COUNT_FREQUENCY,
                }
                for rule in current_rules
            )
        case RuleId.SENTENCE_COUNT_AT_LEAST:
            # Not compatible with other length constraints that limit the length.
            return any(
                rule.rule_id
                in {
                    RuleId.WORD_COUNT_AT_MOST,
                    RuleId.SENTENCE_COUNT_AT_MOST,
                }
                for rule in current_rules
            )
        case RuleId.ALL_UPPER_CASE:
            # Not compatible with other casing constraints.
            return any(
                rule.rule_id
                in {
                    RuleId.ALL_LOWER_CASE,
                }
                for rule in current_rules
            )
        case RuleId.ALL_LOWER_CASE:
            # Not compatible with other casing constraints.
            return any(
                rule.rule_id
                in {
                    RuleId.ALL_UPPER_CASE,
                }
                for rule in current_rules
            )
        case RuleId.NO_COMMAS:
            # Compatible with everything
            return False
        case RuleId.KEYWORD_INCLUSION:
            # Not compatible with other keyword constraints.
            return any(
                rule.rule_id in {RuleId.KEYWORD_FREQUENCY, RuleId.KEYWORD_FORBIDDEN}
                for rule in current_rules
            )
        case RuleId.KEYWORD_FREQUENCY:
            # Not compatible with other keyword constraints.
            return any(
                rule.rule_id in {RuleId.KEYWORD_INCLUSION, RuleId.KEYWORD_FORBIDDEN}
                for rule in current_rules
            )
        case RuleId.KEYWORD_FORBIDDEN:
            # Not compatible with other keyword constraints.
            return any(
                rule.rule_id in {RuleId.KEYWORD_INCLUSION, RuleId.KEYWORD_FREQUENCY}
                for rule in current_rules
            )
        case RuleId.BULLET_COUNT_FREQUENCY:
            # Not compatible with rules that limit the length.
            return any(
                rule.rule_id
                in {
                    RuleId.WORD_COUNT_AT_MOST,
                    RuleId.SENTENCE_COUNT_AT_MOST,
                }
                for rule in current_rules
            )
        case RuleId.STARTS_WITH:
            # Compatible with everything
            return False
        case RuleId.ENDS_WITH:
            # Compatible with everything
            return False

    raise ValueError(f"RuleId {rule_id} not handled.")
