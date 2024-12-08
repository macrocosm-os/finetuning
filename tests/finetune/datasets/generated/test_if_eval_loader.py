import unittest
from collections import Counter

from transformers import AutoTokenizer

from finetune.datasets.generated.if_eval_loader import IFEvalLoader


class TestIFEvalLoader(unittest.TestCase):
    def setUp(self):
        # Use the default version here until there are logic changes as opposed to just new rules across versions.
        self.loader = IFEvalLoader(random_seed=420, max_samples=100)

    def test_uniform_distribution_of_rules(self):
        rule_counts = [len(sample.rules) for sample in self.loader]
        rule_counter = Counter(rule_counts)

        # Check if the distribution of rules is roughly uniform
        for count in range(IFEvalLoader.MIN_RULES, IFEvalLoader.MAX_RULES + 1):
            self.assertIn(count, rule_counter)
            # Allow a large delta.
            self.assertAlmostEqual(rule_counter[count], 25, delta=15)

    def test_unique_rule_ids(self):
        for sample in self.loader:
            rule_ids = [rule.rule_id for rule in sample.rules]
            self.assertEqual(len(rule_ids), len(set(rule_ids)))

    def test_tokenize(self):
        # Basic verification that the samples can be tokenized.
        tokenizer = AutoTokenizer.from_pretrained("Xenova/gpt-4")
        tokenized_samples = self.loader.tokenize(tokenizer, sequence_length=2048)
        self.assertEqual(len(tokenized_samples), len(self.loader))


if __name__ == "__main__":
    unittest.main()
