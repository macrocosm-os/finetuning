import unittest
from finetune.eval.if_eval.rule import RuleId
from finetune.eval.if_eval.rule_factory import generate_rule, is_rule_incompatible


class TestRuleFactory(unittest.TestCase):
    def test_is_rule_incompatible(self):
        """Verifies that each pair-wise values of rules produce consistent compatibility rules."""
        rule_ids = list(RuleId)
        dummy_qa = ("", "")
        for i, first_id in enumerate(rule_ids[:-1]):
            for second_id in rule_ids[i + 1 :]:
                result1 = is_rule_incompatible(
                    first_id, [generate_rule(second_id, [], dummy_qa, dummy_qa)]
                )
                result2 = is_rule_incompatible(
                    second_id, [generate_rule(first_id, [], dummy_qa, dummy_qa)]
                )
                self.assertEqual(
                    result1, result2, f"Failed for pair: {first_id}, {second_id}"
                )

    def test_generate_rule(self):
        """Verifies that generate_rule creates a rule with the expected rule_id."""
        rule_ids = list(RuleId)
        dummy_qa = ("", "")
        for rule_id in rule_ids:
            rule = generate_rule(rule_id, [], dummy_qa, dummy_qa)
            self.assertEqual(rule.rule_id, rule_id, f"Failed for rule_id: {rule_id}")


if __name__ == "__main__":
    unittest.main()
