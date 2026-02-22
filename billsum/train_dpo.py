"""
Direct Preference Optimization (DPO) training script for fine-tuning language models.

This script trains a language model using DPO, a technique that optimizes models
based on preference pairs (chosen vs. rejected outputs). The training data is
constructed from predictions with different reward scores based on CEFR levels.

Usage:
    python train_dpo.py --model_id google/gemma-3-1b-it \\
                        --train_dataset data/train.jsonl \\
                        --eval_dataset data/eval.jsonl \\
                        --output_dir ./dpo_out \\
                        --reward_type cefr_only \\
                        --cefr_weight 1.0 \\
                        --bert_weight 0.5 \\
                        --threshold_a 0.1 \\
                        --threshold_b 0.2 \\
                        --threshold_c 0.2
"""

from unsloth import FastLanguageModel

import os
import json
import argparse
import torch
from prompts import generate_prompt
from trl import DPOTrainer, DPOConfig
from datasets import Dataset, DatasetDict
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
MAX_SEQ_LENGTH = 4096


def format_example(ex):
    """
    Expected dataset columns:
      - prompt: str
      - chosen: str
      - rejected: str
    This returns the exact fields DPOTrainer expects:
      - prompt, chosen, rejected
    """
    return {
        "prompt": ex["prompt"],
        "chosen": ex["chosen"],
        "rejected": ex["rejected"],
    }


def compute_reward(pred, reward_type="cefr_only", cefr_weight=1.0, bert_weight=0.5):
    """
    Compute a reward score for a prediction based on various metrics.

    Args:
        pred (dict): Prediction dictionary containing metrics like 'cefr_prob' and 'bert_score'
        reward_type (str): Type of reward calculation:
            - "cefr_only": Use only CEFR probability
            - "bert_only": Use only BERT score
            - "combined": Weighted combination of CEFR and BERT scores
        cefr_weight (float): Weight for CEFR probability in combined mode
        bert_weight (float): Weight for BERT score in combined mode

    Returns:
        float: Computed reward score
    """
    if reward_type == "cefr_only":
        return pred["cefr_prob"]
    elif reward_type == "bert_only":
        return pred.get("bert_score", 0.0)
    elif reward_type == "combined":
        cefr_score = pred.get("cefr_prob", 0.0) * cefr_weight
        bert_score = pred.get("bert_score", 0.0) * bert_weight
        return (cefr_score + bert_score) / (cefr_weight + bert_weight)
    else:
        raise ValueError(f"Unknown reward_type: {reward_type}")


def load_dataset(
    file_path,
    tokenizer,
    reward_type="cefr_only",
    cefr_weight=1.0,
    bert_weight=0.5,
    threshold_a=0.1,
    threshold_b=0.2,
    threshold_c=0.2,
):
    """
    Load and process dataset for DPO training.

    Creates preference pairs (chosen/rejected) from predictions by comparing reward scores.
    Pairs are only created if the reward difference exceeds a threshold specific to each CEFR level.

    Args:
        file_path (str): Path to JSONL file containing predictions
        tokenizer: Tokenizer for generating prompts
        reward_type (str): Type of reward calculation (see compute_reward)
        cefr_weight (float): Weight for CEFR probability in combined mode
        bert_weight (float): Weight for BERT score in combined mode
        threshold_a (float): Minimum reward difference for A-level pairs
        threshold_b (float): Minimum reward difference for B-level pairs
        threshold_c (float): Minimum reward difference for C-level pairs

    Returns:
        Dataset: HuggingFace Dataset with 'prompt', 'chosen', and 'rejected' columns
    """
    dataset = [json.loads(line.strip()) for line in open(file_path)]

    dpo_dataset = {"prompt": [], "chosen": [], "rejected": []}
    a_labels, b_labels, c_labels = 0, 0, 0
    for item in dataset:
        for cefr_label, predictions in item["predictions"].items():
            prompt = generate_prompt(tokenizer, item["text"], cefr_label)

            # Select threshold based on CEFR level
            if cefr_label == "A":
                threshold = threshold_a
            elif cefr_label == "B":
                threshold = threshold_b
            elif cefr_label == "C":
                threshold = threshold_c
            else:
                raise ValueError(f"Unknown cefr level: {cefr_label}")

            for i in range(0, len(predictions)):
                for j in range(i + 1, len(predictions)):
                    r_i = compute_reward(
                        predictions[i], reward_type, cefr_weight, bert_weight
                    )
                    r_j = compute_reward(
                        predictions[j], reward_type, cefr_weight, bert_weight
                    )

                    accept, reject = None, None
                    if r_i > r_j and (r_i - r_j) > threshold:
                        accept = predictions[i]["generation"]
                        reject = predictions[j]["generation"]
                    elif r_j > r_i and (r_j - r_i) > threshold:
                        accept = predictions[j]["generation"]
                        reject = predictions[i]["generation"]

                    if accept and reject:
                        a_labels += "A1/A2" in prompt
                        b_labels += "B1/B2" in prompt
                        c_labels += "C1/C2" in prompt
                        dpo_dataset["prompt"].append(prompt)
                        dpo_dataset["chosen"].append(accept)
                        dpo_dataset["rejected"].append(reject)

    print(a_labels, b_labels, c_labels, len(dpo_dataset["prompt"]))
    return Dataset.from_dict(dpo_dataset)


def main():
    """
    Main training function.

    Parses command-line arguments, loads model and datasets, configures DPO training,
    and runs the training loop.
    """
    parser = argparse.ArgumentParser(
        description="Train a language model using Direct Preference Optimization (DPO)"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="google/gemma-3-1b-it",
        help="HuggingFace model ID to fine-tune",
    )
    parser.add_argument(
        "--train_dataset",
        type=str,
        required=True,
        help="Path to training dataset JSONL file",
    )
    parser.add_argument(
        "--eval_dataset",
        type=str,
        required=True,
        help="Path to evaluation dataset JSONL file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./dpo_out",
        help="Directory to save the fine-tuned model",
    )
    parser.add_argument(
        "--reward_type",
        type=str,
        default="cefr_only",
        choices=["cefr_only", "bert_only", "combined"],
        help="Type of reward calculation for preference pairs",
    )
    parser.add_argument(
        "--cefr_weight",
        type=float,
        default=1.0,
        help="Weight for CEFR probability in combined reward mode",
    )
    parser.add_argument(
        "--bert_weight",
        type=float,
        default=0.5,
        help="Weight for BERT score in combined reward mode",
    )
    parser.add_argument(
        "--threshold_a",
        type=float,
        default=0.1,
        help="Minimum reward difference for A-level preference pairs",
    )
    parser.add_argument(
        "--threshold_b",
        type=float,
        default=0.2,
        help="Minimum reward difference for B-level preference pairs",
    )
    parser.add_argument(
        "--threshold_c",
        type=float,
        default=0.2,
        help="Minimum reward difference for C-level preference pairs",
    )
    args = parser.parse_args()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    # For many causal LMs, pad_token may be missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_id,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=32,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=MAX_SEQ_LENGTH,
    )

    # Load and prepare datasets
    train_dataset = load_dataset(
        args.train_dataset,
        tokenizer,
        args.reward_type,
        args.cefr_weight,
        args.bert_weight,
        args.threshold_a,
        args.threshold_b,
        args.threshold_c,
    )
    eval_dataset = load_dataset(
        args.eval_dataset,
        tokenizer,
        args.reward_type,
        args.cefr_weight,
        args.bert_weight,
        args.threshold_a,
        args.threshold_b,
        args.threshold_c,
    )
    print(train_dataset[0]["prompt"])
    print(train_dataset[0]["chosen"])
    print(train_dataset[0]["rejected"])

    train_ds = train_dataset.map(format_example)
    eval_ds = eval_dataset.map(format_example)

    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        learning_rate=5e-6,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        optim="adamw_8bit",
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_total_limit=1,
        bf16=True,
        seed=42,
        gradient_checkpointing=True,
        # --- DPO specific ---
        beta=0.1,
        max_length=MAX_SEQ_LENGTH,
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
