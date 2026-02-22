# BillSum CEFR-Level Summarization

Multi-level text summarization of US Congressional bills using reinforcement learning techniques to generate summaries at different CEFR (Common European Framework of Reference) readability levels.

## Overview

This project fine-tunes the Gemma language model to generate legislative summaries tailored to different reading proficiency levels (A, B, C). The goal is to make complex legal text accessible to readers with varying levels of English proficiency.

### CEFR Levels
- **A (A1/A2)**: Basic user - elementary understanding
- **B (B1/B2)**: Independent user - intermediate understanding
- **C (C1/C2)**: Proficient user - advanced understanding

## Experiments

### 1. Experiment Setup
- **Dataset**: BillSum (US Congressional bill summaries)
- **Base Model**: google/gemma-3-1b-it
- **Target**: Generate 3 different summary versions (A, B, C levels) for each bill
- **Evaluation Metrics**:
  - CEFR classification score (reading level accuracy)
  - BERTScore F1 (semantic similarity to reference)

### 2. Methods Tried

Results on US Test Dataset:

| Method | Training Approach | Reward Signal | Level A (BERT/CEFR) | Level B (BERT/CEFR) | Level C (BERT/CEFR) |
|--------|------------------|---------------|---------------------|---------------------|---------------------|
| Gemma OOB | Zero-shot baseline | N/A | 83.73 / 5.15 | 84.95 / 50.42 | 84.78 / 64.45 |
| Gemma OOB + DPO | Direct Preference Optimization | CEFR classifier | 83.50 / 8.43 | 84.71 / 49.60 | 84.37 / 72.95 |
| Gemma OOB + GRPO | Group Relative Policy Optimization | CEFR classifier + Length Penalty | - | - | - |
| Gemma SFT | Supervised Fine-Tuning | N/A | 89.63 / 5.78 | 89.39 / 71.61 | 89.47 / 27.23 |
| Gemma SFT + DPO | DPO on SFT checkpoint | CEFR classifier + BERTScore | 83.62 / 5.08 | 84.92 / 49.90 | 84.67 / 66.40 |
| Gemma SFT + GRPO | GRPO on SFT checkpoint | CEFR classifier + BERTScore + Length Penalty | - | - | - |

**Metrics**: BERTScore F1 / CEFR Classification Accuracy (higher is better for both)

### 3. Results
See `results.txt` for detailed experimental results and metrics.

### 4. Next Steps
- Apply CEFR + BERTScore rewards to OOB baseline
- Test CEFR + Length Penalty for SFT models
- Evaluate on larger Gemma variants (7B, 27B)

## Code Structure

### Core Modules
- **`prompts.py`**: Prompt templates and formatting for CEFR-level generation
- **`evaluator.py`**: Evaluation pipeline for CEFR and BERTScore metrics
- **`inference.py`**: vLLM-based text generation engine
- **`merge_model.py`**: Utility to merge LoRA adapters with base models

### Training Scripts
- **`train_sft.py`**: Supervised fine-tuning with CEFR-labeled data
- **`train_dpo.py`**: Direct Preference Optimization training
- **`train_grpo.py`**: Group Relative Policy Optimization training

### Execution Scripts
- **`run_inference.py`**: Generate predictions for dataset (supports rollout mode)
- **`run_metrics.py`**: Compute CEFR and BERTScore on predictions

### Utilities
- **`data_notebooks/`**: Jupyter notebooks for data exploration and analysis
- **`deprecated/`**: Legacy implementations (deprecated)

## Usage

### Installation
```bash
pip install -r requirements.txt
```

### Training Pipeline

#### 1. Supervised Fine-Tuning (Common for all methods)
```bash
python train_sft.py \
    --model_id google/gemma-3-1b-it \
    --train_dataset dataset/us_sft_train.jsonl \
    --eval_dataset dataset/us_sft_dev.jsonl \
    --output_dir models/gemma_sft/v1
```

#### 2. Reinforcement Learning Training

**Option A: GRPO (Online RL - generates rollouts during training)**
```bash
python train_grpo.py \
    --model_id models/gemma_sft/v1 \
    --train_dataset dataset/train.jsonl \
    --eval_dataset dataset/dev.jsonl \
    --output_dir models/gemma_grpo/v1
```

**Option B: DPO (Offline RL - requires pre-generated rollouts)**

First, generate rollouts and compute metrics:
```bash
# Generate multiple candidate summaries
python run_inference.py \
    --input_file dataset/train.jsonl \
    --output_file dataset/train_rollouts.jsonl \
    --model_name models/gemma_sft/v1 \
    --rollouts

# Compute CEFR and BERTScore for each candidate
python run_metrics.py \
    --input_file dataset/train_rollouts.jsonl \
    --output_file dataset/train_with_metrics.jsonl
```

Then run DPO training:
```bash
python train_dpo.py \
    --model_id models/gemma_sft/v1 \
    --train_dataset dataset/train_with_metrics.jsonl \
    --eval_dataset dataset/dev_with_metrics.jsonl \
    --output_dir models/gemma_dpo/v1
```

#### 3. Merge LoRA Adapters (Optional)
```bash
python merge_model.py \
    --input_dir models/gemma_grpo/v1 \
    --output_dir models/gemma_grpo_merged/v1 \
    --model_name google/gemma-3-1b-it
```

### Evaluation
```bash
python run_inference.py \
    --input_file dataset/test.jsonl \
    --output_file dataset/test_predictions.jsonl \
    --model_name models/gemma_grpo/v1

python run_metrics.py \
    --input_file dataset/test_predictions.jsonl \
    --output_file dataset/test_results.jsonl
```

## Requirements
- Python 3.10+
- CUDA-capable GPU (recommended: 24GB+ VRAM for training)
- See `requirements.txt` for full dependency list

## References
- **BillSum Dataset**: [arXiv:1910.00523](https://arxiv.org/abs/1910.00523)
- **CEFR Framework**: [Council of Europe](https://www.coe.int/en/web/common-european-framework-reference-languages)
- **DPO**: [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
- **GRPO**: Group Relative Policy Optimization (TRL implementation)
