<picture>
    <source srcset="./assets/macrocosmos-white.png"  media="(prefers-color-scheme: dark)">
    <img src="macrocosmos-white.png">
</picture>

<picture>
    <source srcset="./assets/macrocosmos-black.png"  media="(prefers-color-scheme: light)">
    <img src="macrocosmos-black.png">
</picture>

<div align="center">

# **Finetuning Subnet** <!-- omit in toc -->
[![Bittensor](/docs/taologo.png)](https://bittensor.com/)

</div>

# Subnet 37: Finetuning Reasoning Models

## Introduction

**Subnet 37** is an LLM finetuning subnet, owned by **Macrocosmos**. Its focus lies in finetuning reasoning models—a critical aspect as these models are increasingly outperforming traditional LLMs on numerous benchmarks. Finetuning determines how an LLM presents itself and provides the polish necessary for commercial use. Subnet 37 connects businesses to affordable finetuning solutions.

## Product

Our vision is to build a **decentralized AI stack** that enables people to build and finetune LLMs. **Subnet 37** is the final step in this stack, offering decentralized and cost-effective finetuning. Users can bring pretrained models to Subnet 37 for low-cost finetuning or pre-train foundation models on **Subnet 9** (also owned by Macrocosmos) before transferring them to Subnet 37.

## Team

The key figures behind Subnet 37 are:

- **Will Squires, Co-Founder and CEO**  
  Entrepreneur and AI startup builder with a Master of Engineering from the University of Warwick.

- **Steffen Cruz, Co-Founder and CTO**  
  Former CTO at OpenTensor Foundation and original architect of Subnet 1 (Apex). Holds a PhD in Experimental Nuclear Physics.

- **Alan Aboudib, Machine Learning Lead**  
  Subnet designer with a PhD and Postdoc in Deep Learning and Computer Vision from the Collège de France.

- **Rodrigo Lopez Portillo Alcocer, Machine Learning Engineer**  
  Maintains subnet quality. Brings a decade of experience in physics, mathematics, and machine learning.

## Market

Subnet 37 operates within the **LLM finetuning market**. As of 2025, over 300 million companies are exploring LLM integration. The global LLM market is projected to reach **$259.8 million USD by 2030**.

Subnet 37 offers:
- A decentralized, low-cost alternative to closed-source models.
- Finetuned reasoning models which are often too costly via APIs like ChatGPT or Claude.

This positions Subnet 37 as a compelling option for businesses aiming to integrate LLMs without incurring long-term API costs.

## Indicators of Value Growth

- Demand for commercial LLMs is growing.
- Subnet 37 undercuts centralized services.
- Finetuning powers the best-performing reasoning models in the field.

## Long-Term Value Potential

In 2025, Subnet 37 evolved to support **3B reasoning models**. Macrocosmos actively monitors trends and updates Subnet 37 to stay ahead, ensuring businesses receive cutting-edge finetuning capabilities.

## Community & Customer Interest

- Active **miners** and **validators** contribute to model development.
- A vibrant community helps develop and benchmark models for commercial applications.

## Use-Cases and Potential

Subnet 37 supports:

- Building **affordable finetuned LLMs** for commercial use.
- Distributing reasoning models to businesses or other subnets.
- **Benchmarking open-source models** for quality and performance.

LLMs are rarely deployed without finetuning, making Subnet 37 a core component of modern AI pipelines.

## Dashboards, Tools, and Resources

- **Dashboard**: [macrocosmos.ai/sn37/dashboard](https://macrocosmos.ai/sn37/dashboard)
- **Technical Documentation**: [docs.macrocosmos.ai/subnet-37-finetuning](https://docs.macrocosmos.ai/subnet-37-finetuning)
- **GitHub Repository**: [github.com/macrocosm-os/finetuning](https://github.com/macrocosm-os/finetuning)

## Incentive Overview

Subnet 37 rewards miners through a **continuous benchmarking competition**:

1. **Miners** train models and publish them to HuggingFace with metadata on the Bittensor chain.
2. **Validators** evaluate the models using the metadata, awarding the top-performing model per competition.
3. Evaluation results are logged on WandB.

Each competition:
- Has specific emission splits.
- Defines model/tokenizer specs, sequence lengths, and evaluation tasks.
- Supports weighted multi-task evaluation and normalization.
- Allows validator-funded public and user-defined competitions.

We now use **Subnet 1** for evaluation data due to its superior quality compared to synthetic data from **Subnet 18**. Hash-based synchronization ensures secure and fair evaluation.

## Resources

- **Website**: [macrocosmos.ai/sn37](https://macrocosmos.ai/sn37)
- **Dashboard**: [macrocosmos.ai/sn37/dashboard](https://macrocosmos.ai/sn37/dashboard)
- **GitHub**: [github.com/macrocosm-os/finetuning](https://github.com/macrocosm-os/finetuning)
- **Documentation**: [docs.macrocosmos.ai/subnet-37-finetuning](https://docs.macrocosmos.ai/subnet-37-finetuning)
- **Miner Setup**: See the [miner docs](docs/miner.md#getting-started) to learn how to setup a Miner.
- **Validator Setup**: See the [validator docs](docs/validator.md#getting-started) to learn how to setup a Validator.

### Articles

- [Fine-tuning, finely tuned: How SN37 is delivering SOTA fine-tuning on Bittensor](#)
- [Fine-tuning, harmonized: Taoverse and Macrocosmos team up on SN37](#)

### Community

- **Bittensor Discord**: [Join Channel](https://discord.com/channels/1238450997848707082/1238453186768011275)
- **Macrocosmos Discord**: [Join Channel](https://discord.com/channels/799672011265015819/1253448873305964626)
- **Telegram (Cosmonauts)**: [t.me/macrocosmosai](https://t.me/macrocosmosai)
- **X (formerly Twitter)**: [@MacrocosmosAI](https://x.com/MacrocosmosAI)

--- 
## License
This repository is licensed under the MIT License.
```text
# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
```
