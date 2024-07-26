# Awesome-GenAI-Unlearning
![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green)  [![Awesome](https://awesome.re/badge.svg)](https://awesome.re) 


This repository contains a list of papers on Generative AI Machine Unlearning based on our survey paper: [**Machine Unlearning in Generative AI: A Survey**](https://arxiv.org/abs/2312.11518) (*[Zheyuan (Frank) Liu](https://franciscoliu.github.io/), [Guangyao Dou](https://guangyaodou.github.io/), [Zhaoxuan Tan](https://zhaoxuan.info/), [Yijun Tian](https://www.yijuntian.com/) and [Meng Jiang](http://www.meng-jiang.com/)*).
We categorize existing works based on their [approaches](), modality and [applications](). Additionally, we include datasets and benchmarks for various unlearning scenarios. 


## Table of Contents
- [Awesome-GenAI-Unlearning ](#awesome-genai-unlearning-)
  - [Table of Contents](#table-of-contents)
  - [Datasets, Benchmarks](#datasets-benchmarks)
    - [Datasets](#datasets)
    - [Benchmarks](#benchmarks)
  - [Generative Image Models](#generative-image-models)
  - [Large Language Models (LLMs)](#large-language-models-(LLMs))
  - [Large Multimodal Models (LMMs)](#large-multimodal-models-(LMMs))
  - [Applications](#applications)
    - [Safety Alignment](#safety-alignment-1)
    - [Copyright Protection](#copyright-protection-1)
    - [Hallucination Reduction](#hallucination-reduction-1)
    - [Privacy Compliance](#privacy-compliance-1)
    - [Bias/Unfairness Alleviation](bias/unfairness-alleviation-1)


  
## Datasets, Benchmarks:

### Datasets:
#### Safety Alignment
- **LAION** LAION-400-MILLION OPEN DATASET ([code](https://laion.ai/blog/laion-400-open-dataset/))
- **Civil Comments** [CoRR 2019] Nuanced metrics for measuring unintended bias with real data for text classification ([code](https://huggingface.co/datasets/google/civil_comments))
- **PKU-SafeRLHF** [arxiv 2310.12773] Safe RLHF: Constrained Value Alignment via Safe Reinforcement Learning from Human Feedback ([code](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF-10K))
- **Anthropic red team** [arxiv 2204.05862] Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback ([code](https://github.com/anthropics/hh-rlhf))

#### Copyrights Protection
- **Harry Potter** Copyright issue, cannot be disclosed.
- **Bookcorpus** [arxiv 1506.06724] Aligning Books and Movies: Towards Story-like Visual Explanations by Watching Movies and Reading Books ([code](https://huggingface.co/datasets/bookcorpus/bookcorpus))
- **TOFU** [arxiv 2401.06121] TOFU: A Task of Fictitious Unlearning for LLMs ([code](https://locuslab.github.io/tofu/))

#### Hallucination Reduction 
- **HaluEVAL** [EMNLP 2023] HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models ([code](https://github.com/RUCAIBox/HaluEval))
- **TruthfulQA** [ACL 2023] TruthfulQA: Measuring How Models Mimic Human Falsehoods ([code](https://github.com/sylinrl/TruthfulQA))
- **CounterFact** [NeurIPS 2022] Locating and Editing Factual Associations in GPT ([code](https://huggingface.co/datasets/azhx/counterfact))
- **ZsRE** [CoNLL 2017] Zero-Shot Relation Extraction via Reading Comprehension ([code](https://nlp.cs.washington.edu/zeroshot/))
- **MSCOCO** [arxiv 1405.0312] Microsoft COCO: Common Objects in Context ([code](https://cocodataset.org/#home))

#### Privacy Compliance
- **Pile** [arxiv 2101.00027] The Pile: An 800GB Dataset of Diverse Text for Language Modeling ([code](https://pile.eleuther.ai/))
- **Yelp/Amazon Reviews** ([code](https://jmcauley.ucsd.edu/data/amazon/))
- **SST-2** [EMNLP 2013] Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank ([code](https://gist.github.com/shagunsodhani/6ca136088f58d24f7b08056ec8b97595))
- **PersonaChat** [arxiv 1801.07243] Personalizing Dialogue Agents: I have a dog, do you have pets too? ([code](https://www.kaggle.com/datasets/atharvjairath/personachat))
- **LEDGAR** [ACL 2020] LEDGAR: A Large-Scale Multilabel Corpus for Text Classification of Legal Provisions in Contracts ([code](https://drive.switch.ch/index.php/s/j9S0GRMAbGZKa1A))
- **SAMsum** [ACL 2019] SAMSum Corpus: A Human-annotated Dialogue Dataset for Abstractive Summarization ([code](https://huggingface.co/datasets/Samsung/samsum))
- **IMDB** ([code](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews))
- **CeleA-HQ** [Neurips 2018] IntroVAE: Introspective Variational Autoencoders for Photographic Image Synthesis ([code](https://github.com/tkarras/progressive_growing_of_gans))
- **I2P** Inappropriate Image Prompts (I2P) Benchmark ([code](https://github.com/ml-research/i2p))

#### Bias/Unfairness Alleviation
- **StereoSet** [ACL 2021] StereoSet: Measuring stereotypical bias in pretrained language models ([code](https://github.com/moinnadeem/StereoSet))
- **HateXplain** [AAAI 2021] HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection ([code](https://github.com/hate-alert/HateXplain))
- **CrowS Pairs** [EMNLP 2021] CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models ([code](https://github.com/nyu-mll/crows-pairs))

### Benchmarks:
#### Generative Image Models
- **UnlearnCanvas** [arxiv 2402.11846] UnlearnCanvas: A Stylized Image Dataaset to Benchmark Machine Unlearning for Diffusion Models ([code](https://github.com/OPTML-Group/UnlearnCanvas))

#### LLMs
- **TOFU** [arxiv 2401.06121] TOFU: A Task of Fictitious Unlearning for LLMs ([code](https://locuslab.github.io/tofu/))
- **WMDP** [arxiv 2403.03218] The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning ([code](https://www.wmdp.ai/))


#### LMMs
- **Object HalBench:** [EMNLP 2018] Object Hallucination in Image Captioning ([code](https://github.com/LisaAnne/Hallucination))
- **MHumanEval:** [CVPR'24] RLHF-V: Towards Trustworthy MLLMs via Behavior Alignment from Fine-grained Correctional Human Feedback ([code](https://github.com/RLHF-V/RLHF-V))
- **LLaVA Bench:** [Neurips 2023 (oral)] Visual Instruction Tuning ([code](https://github.com/haotian-liu/LLaVA/blob/main/docs/LLaVA_Bench.md))
- **MMHal-Bench:** Aligning Large Multimodal Models with Factually Augmented RLHF ([code](https://huggingface.co/datasets/Shengcao1006/MMHal-Bench))
- **POPE:** [EMNLP 2023] POPE: Polling-based Object Probing Evaluation for Object Hallucination ([code](https://github.com/RUCAIBox/POPE))

## Generative Image Models:


## Large Language Models (LLMs):


## Large Multimodal Models (LMMs):


## Applications:
### Safety Alignment:

### Copyright Protection:

### Hallucination Reduction:

### Privacy Compliance:

### Bias/Unfairness Alleviation:




## Contributing:
👍 Contributions to this repository are welcome! 
We will try to make this list updated. If you find any error or any missed paper, please don't hesitate to open an issue or pull request.
