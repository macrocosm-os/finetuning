import unittest
from finetune.datasets.generated.mmlu_parser import extract_q_and_a_text


class TestMMLUParser(unittest.TestCase):

    def test_valid_prompt_with_answer_A(self):
        prompt = """[Example 1] In what year was Elvis Presley born?
        A. 1935
        B. 1945
        C. 1955
        D. 1965
        answer: A
        
        [Input Question] What is the capital of France?
        A. Paris
        B. London
        C. Berlin
        D. Madrid
        """
        result = extract_q_and_a_text(prompt, "A")
        self.assertEqual(result, ("What is the capital of France?", "Paris"))

    def test_valid_prompt_with_answer_B(self):
        prompt = "[Input Question] What is the capital of France? A. Paris B. London C. Berlin D. Madrid"
        result = extract_q_and_a_text(prompt, "B")
        self.assertEqual(result, ("What is the capital of France?", "London"))

    def test_valid_prompt_with_answer_C(self):
        prompt = "[Input Question] What is the capital of France? A. Paris B. London C. Berlin D. Madrid"
        result = extract_q_and_a_text(prompt, "C")
        self.assertEqual(result, ("What is the capital of France?", "Berlin"))

    def test_valid_prompt_with_answer_D(self):
        prompt = "[Input Question] What is the capital of France? A. Paris B. London C. Berlin D. Madrid"
        result = extract_q_and_a_text(prompt, "D")
        self.assertEqual(result, ("What is the capital of France?", "Madrid"))

    def test_invalid_answer_char(self):
        prompt = "[Input Question] What is the capital of France? A. Paris B. London C. Berlin D. Madrid"
        result = extract_q_and_a_text(prompt, "E")
        self.assertIsNone(result)

    def test_invalid_prompt_format(self):
        prompt = "What is the capital of France? A. Paris B. London C. Berlin D. Madrid"
        result = extract_q_and_a_text(prompt, "A")
        self.assertIsNone(result)

    def test_empty_prompt(self):
        prompt = ""
        result = extract_q_and_a_text(prompt, "A")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
