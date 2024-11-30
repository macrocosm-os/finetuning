# Live Competitions

## Competition B7_MULTICHOICE:

### Goal

The purpose of this competition is to finetune the top models from the [pretraining subnet](https://www.macrocosmos.ai/sn9) to produce a chat bot.

### Evaluation

Models submitted to this competition are evaluated on a set of evaluation tasks, where each task is worth a subportion of the overall score. The current evaluations are:
1) SYNTHENTIC_MMLU: In this task model is evaluated on a synthetic MMLU-like dataset from the [Text Prompting subnet](https://www.macrocosmos.ai/sn1). This dataset is a multiple choice dataset with a large array of multiple choice questions, spanning a domain of topics and difficulty levels, akin to MMLU. Currently, the dataset is generated using Wikipedia as the source-of-truth, though this will be expanded over time to include more domain-focused sources.
2) WORD_SORTING: In this task, the model is given a list of words and are required to sort them alphabetically. [Code](https://github.com/macrocosm-os/finetuning/blob/main/finetune/datasets/generated/dyck_loader.py)
3) FINEWEB: In this task, the model's cross entropy loss is computed on a small sample of the fineweb dataset: https://hf.rst.im/datasets/HuggingFaceFW/fineweb-edu-score-2
4) IF_EVAL: In this task, the model is evaluated on a sythentic version of the IFEval dataset (https://hf.rst.im/datasets/google/IFEval). The prompt contains a list of rules the response must follow. The full list of possible rules is listed in [rule.py](https://github.com/macrocosm-os/finetuning/blob/main/finetune/eval/if_eval/rule.py)

### Definitions

[Code Link](https://github.com/macrocosm-os/finetuning/blob/94e8fd92ab4158e1e4a425a9562695eebafa27b1/constants/__init__.py#L128)

## Competition INSTRUCT_8B:

### Goal

The goal of this competition is to train a SOTA instruct 8B model. This competition provides more freedom to miners than other competitions: there are no restrictions on the tokenizer used and miners are allowed to use a wider range of architectures.

### Evaluation

The evaluation tasks are the same as the B7_MULTICHOICE competition

### Definitions

TODO: Fill in post check-in

# Deprecated Competitions

## Competition 1: SN9_MODEL

Competition 1 was the OG competition for the finetuning subnet. 

### Goal

The purpose of this competition was to finetune the top models from the [pretraining subnet](https://www.macrocosmos.ai/sn9) to produce a chat bot.

### Evaluation

Models submitted to this competition were evaluated using a synthetically generated Q&A dataset from the [cortex subnet](https://github.com/Datura-ai/cortex.t). Specifically, models were evaluated based on their average loss of their generated answers. 