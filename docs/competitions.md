# Live Competitions

## Competition 3: INSTRUCT_8B

### Goal

The goal of this competition is to train a SOTA instruct 8B model. This competition provides more freedom to miners than other competitions: there are no restrictions on the tokenizer used and miners are allowed to use a wider range of architectures.

### Evaluation

The evaluation tasks are the same as the B7_MULTICHOICE competition

### Definitions

[Code Link](https://github.com/macrocosm-os/finetuning/blob/c6dce9d27d1317b9c543071913ae34df09faddc7/constants/__init__.py#L114)

# Scheduled Competitions

## Competition 4: DISTILLED_REASONING_3B

### Goal

The goal of this competition is to train a 3.2-3.4B parameter model specialized in step-by-step reasoning. Submitted models must demonstrate strong capabilities in structured thinking, particularly for mathematical reasoning and code understanding tasks. This competition aims to produce efficient, smaller-scale models that maintain high-quality reasoning abilities compared to larger models.

In this first iteration of the competition we will focus on optimizing the model's perplexity score on reasoning traces. This will then allow us to calibrate the next stages of our reasoning competition.

### Evaluation

Models submitted to this competition are evaluated on the [SYNTHETIC_1_SFT](<https://huggingface.co/datasets/PrimeIntellect/SYNTHETIC-1-SFT-Data>) dataset, which contains verifiable and structured reasoning problems:

Our task evaluates the models on a dataset of verifiable math and code output prediction problems requiring step-by-step reasoning.

The evaluation uses the REFERENCE_LOSS method, which measures how well the model can generate accurate reasoning traces and answers for these problems.

### Definitions

[Competition Constraints](https://github.com/macrocosm-os/finetuning/blob/abfcba14469b1752fcc460e0caaa3460a726c81f/constants/__init__.py#L143)

[Dataset Loader](https://github.com/macrocosm-os/finetuning/blob/abfcba14469b1752fcc460e0caaa3460a726c81f/finetune/datasets/hugging_face/hugging_face_loader.py#L301)

[Evaluation Method](https://github.com/macrocosm-os/finetuning/blob/abfcba14469b1752fcc460e0caaa3460a726c81f/finetune/eval/method.py#L157)

# Deprecated Competitions

## Competition 1: SN9_MODEL

Competition 1 was the OG competition for the finetuning subnet.

### Goal

The purpose of this competition was to finetune the top models from the [pretraining subnet](https://www.macrocosmos.ai/sn9) to produce a chat bot.

### Evaluation

Models submitted to this competition were evaluated using a synthetically generated Q&A dataset from the [cortex subnet](https://github.com/Datura-ai/cortex.t). Specifically, models were evaluated based on their average loss of their generated answers.

## Competition 2: B7_MULTICHOICE

### Goal

The purpose of this competition is to finetune the top models from the [pretraining subnet](https://www.macrocosmos.ai/sn9) to produce a chat bot.

### Evaluation

Models submitted to this competition are evaluated on a set of evaluation tasks, where each task is worth a subportion of the overall score. The current evaluations are:

1) SYNTHENTIC_MMLU: In this task model is evaluated on a synthetic MMLU-like dataset from the [Text Prompting subnet](https://www.macrocosmos.ai/sn1). This dataset is a multiple choice dataset with a large array of multiple choice questions, spanning a domain of topics and difficulty levels, akin to MMLU. Currently, the dataset is generated using Wikipedia as the source-of-truth, though this will be expanded over time to include more domain-focused sources.
2) WORD_SORTING: In this task, the model is given a list of words and are required to sort them alphabetically. [Code](https://github.com/macrocosm-os/finetuning/blob/main/finetune/datasets/generated/dyck_loader.py)
3) FINEWEB: In this task, the model's cross entropy loss is computed on a small sample of the fineweb dataset: <https://hf.rst.im/datasets/HuggingFaceFW/fineweb-edu-score-2>
4) IF_EVAL: In this task, the model is evaluated on a sythentic version of the IFEval dataset (<https://hf.rst.im/datasets/google/IFEval>). The prompt contains a list of rules the response must follow. The full list of possible rules is listed in [rule.py](https://github.com/macrocosm-os/finetuning/blob/main/finetune/eval/if_eval/rule.py)

### Definitions

[Code Link](https://github.com/macrocosm-os/finetuning/blob/94e8fd92ab4158e1e4a425a9562695eebafa27b1/constants/__init__.py#L128)
