import unittest
from collections import Counter

from transformers import AutoTokenizer

from finetune.datasets.generated.if_eval_loader import IFEvalLoader
from finetune.eval.if_eval.version import IfEvalVersion


class TestIFEvalLoader(unittest.TestCase):
    def setUp(self):
        self.loader_v1 = IFEvalLoader(
            random_seed=420, max_samples=100, if_eval_version=IfEvalVersion.V1
        )
        self.loader_v2 = IFEvalLoader(
            random_seed=420, max_samples=100, if_eval_version=IfEvalVersion.V2
        )

    def test_uniform_distribution_of_rules_v1(self):
        rule_counts = [len(sample.rules) for sample in self.loader_v1]
        rule_counter = Counter(rule_counts)

        # Check if the distribution of rules is roughly uniform
        for count in range(
            IFEvalLoader.VERSION_TO_RULE_COUNTS[IfEvalVersion.V1][0],
            IFEvalLoader.VERSION_TO_RULE_COUNTS[IfEvalVersion.V1][1] + 1,
        ):
            self.assertIn(count, rule_counter)
            # Allow a large delta.
            self.assertAlmostEqual(rule_counter[count], 25, delta=15)

    def test_uniform_distribution_of_rules_v2(self):
        rule_counts = [len(sample.rules) for sample in self.loader_v2]
        rule_counter = Counter(rule_counts)

        # Check if the distribution of rules is roughly uniform
        for count in range(
            IFEvalLoader.VERSION_TO_RULE_COUNTS[IfEvalVersion.V2][0],
            IFEvalLoader.VERSION_TO_RULE_COUNTS[IfEvalVersion.V2][1] + 1,
        ):
            self.assertIn(count, rule_counter)
            # Allow a large delta.
            self.assertAlmostEqual(rule_counter[count], 25, delta=15)

    def test_unique_rule_ids(self):
        for sample in self.loader_v1:
            rule_ids = [rule.rule_id for rule in sample.rules]
            self.assertEqual(len(rule_ids), len(set(rule_ids)))

    def test_tokenize(self):
        # Basic verification that the samples can be tokenized.
        tokenizer = AutoTokenizer.from_pretrained("Xenova/gpt-4")
        tokenized_samples = self.loader_v1.tokenize(tokenizer, sequence_length=2048)
        self.assertEqual(len(tokenized_samples), len(self.loader_v1))


if __name__ == "__main__":
    unittest.main()
