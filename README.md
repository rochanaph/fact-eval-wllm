# Factuality Beyond Coherence: Evaluating LLM Watermarking Methods for Medical Texts

**Research Paper**: *"Factuality Beyond Coherence: Evaluating LLM Watermarking Methods for Medical Texts"* - **Accepted at EMNLP 2025 Findings**

---

This repository contains the experimental framework and evaluation tools for our EMNLP 2025 paper on watermarking methods for medical texts. We extend the MarkLLM toolkit with medical domain-specific evaluation capabilities and introduce the Factuality-Weighted Score (FWS) for assessing watermark quality in safety-critical applications.

## 🆕 New Components (Built on MarkLLM)

### `evaluation_scripts/` - Standalone Evaluation Tools
- `text_generation.py` - Watermarked text generation with medical models
- `quality_evaluation.py` - Perplexity and SimCSE evaluation 
- `task_evaluation.py` - ROUGE, F1, and AlignScore metrics
- `detection_evaluation.py` - Detection performance analysis
- `run_evaluation.py` - Flexible evaluation runner with selective metrics

### `logs/` - Experimental Results
- `KGW/` - KGW watermarking results across medical datasets
- `SWEET/` - SWEET entropy-aware watermarking results
- `EXPEdit/` - EXPEdit distortion-free results
- `DIP/` - DiPmark distribution-preserving results
- `GPTJUDGE-Results/` - GPT-based quality assessment outputs

### Additional Research Tools
- `GPT - Judger.ipynb` - GPT-4 based text quality assessment notebook
- `judgerfunctions.py` - GPT evaluation utilities and scoring functions
- `our_utils.py` - Medical model loaders and dataset handling utilities

## Quick Start

### Medical Text Generation and Evaluation
```bash
# Text completion task with HealthQA dataset
python evaluation_scripts/text_generation.py --algorithm KGW --model jsl --dataset HQA --gamma 0.5 --delta 2

# Question-answering task with HealthQA-2 dataset  
python evaluation_scripts/text_generation.py --algorithm KGW --model jsl --dataset HQA2 --gamma 0.5 --delta 2

# Summarization task with MeQSum dataset
python evaluation_scripts/text_generation.py --algorithm SWEET --model biomistral --dataset MEQS --gamma 0.25 --delta 0.5 --entropy 0.9

# Evaluate with Factuality-Weighted Score
python evaluation_scripts/run_evaluation.py --algorithm KGW --model jsl --dataset HQA2 --gamma 0.5 --delta 2 --metrics all
```

For more details read README file at `evaluation_scripts/`

### Supported Medical Models
- **`meditron`**: Meditron-7B (`epfl-llm/meditron-7b`) - Medical domain adaptation of Llama-2
- **`jsl`**: JSL-MedLlama-3-8B-v2.0 (`johnsnowlabs/JSL-MedLlama-3-8B-v2.0`) - John Snow Labs medical LLM  
- **`biomistral`**: BioMistral-7B (`BioMistral/BioMistral-7B`) - Medical domain Mistral model

### Medical Datasets and Tasks
- **`HQA`**: HealthQA dataset for **text completion/generation tasks** (230-word medical passages)
- **`HQA2`**: HealthQA dataset for **question-answering tasks** (medical Q&A pairs) 
- **`MEQS`**: MeQSum dataset for **summarization tasks** (medical question summarization)

---

## Original MarkLLM Documentation

For complete documentation about the MarkLLM toolkit, including detailed usage examples, API references, and comprehensive guides on watermarking algorithms, please refer to the [original MarkLLM README](MarkLLM-README.md).