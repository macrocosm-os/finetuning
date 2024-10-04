# Competitions

## Competition B7_MULTICHOICE:

### Goal

The purpose of this competition is to finetune the top models from the [pretraining subnet](https://www.macrocosmos.ai/sn9) to produce a chat bot.

### Evaluation

Models submitted to this competition are evaluated using a synthetic MMLU-like dataset from the [Text Prompting subnet](https://www.macrocosmos.ai/sn1). This new dataset is a multiple choice dataset with a large array of multiple choice questions, spanning a domain of topics and difficulty levels, akin to MMLU. Currently, the dataset is generated using Wikipedia as the source-of-truth, though this will be expanded over time to include more domain-focused sources.

Our early testing of this dataset, shows promising correlation between this competition's evaluation function and the model performance on MMLU-pro, as shown by the graph below.

![Synthetic MMLU evaluation](b7_mc_eval.png)

### Definitions

[Code Link](https://github.com/macrocosm-os/finetuning/blob/94e8fd92ab4158e1e4a425a9562695eebafa27b1/constants/__init__.py#L128)