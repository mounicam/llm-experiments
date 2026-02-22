"""
Supervised Fine-Tuning (SFT) training script for fine-tuning language models.

This script trains a language model using SFT on summarization data with CEFR
(Common European Framework of Reference) level annotations. The model learns
to generate summaries at different reading difficulty levels.

Usage:
    python train_sft.py --model_id google/gemma-3-1b-it \
                        --train_dataset dataset/org_with_cefr_labels/us_sft_train.jsonl \
                        --eval_dataset dataset/org_with_cefr_labels/us_sft_dev.jsonl \
                        --output_dir models/gemma_sft/v3

Notes:
    - completion_only_loss needs to be True to only train on the completion
    - Preprocessing combines prompt + completion for proper training
    - Long examples are filtered to prevent NaN eval_loss
"""

import os
import shutil
import argparse

import pandas as pd
from datasets import Dataset, DatasetDict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from prompts import PROMPT_TEMPLATE, SYSTEM_PROMPT

# ======================
# Config
# ======================

MAX_SEQ_LENGTH = 4096

# ======================
# Data loading
# ======================


def load_splits(train_path, dev_path) -> DatasetDict:
    """
    Load training and validation datasets from JSONL files.

    Args:
        train_path (str): Path to training JSONL file
        dev_path (str): Path to validation JSONL file

    Returns:
        DatasetDict: Dictionary containing 'train' and 'validation' datasets
    """
    train_df = pd.read_json(train_path, lines=True)
    dev_df = pd.read_json(dev_path, lines=True)

    # Drop rows with missing fields
    for df in (train_df, dev_df):
        df.dropna(subset=["text", "summary", "cefr_labels"], inplace=True)

    train = Dataset.from_pandas(train_df)
    dev = Dataset.from_pandas(dev_df)

    dataset = DatasetDict({"train": train, "validation": dev})
    print(dataset)
    return dataset


# ======================
# Preprocessing
# ======================


def build_preprocess_fn(tokenizer):
    """
    Build a preprocessing function for dataset.

    Creates prompt and completion messages for chat template format.

    Args:
        tokenizer: HuggingFace tokenizer for the model

    Returns:
        function: Preprocessing function that formats examples
    """

    def preprocess(example):
        cefr_label = example["cefr_labels"][0]["label"][:1]
        cefr_label = f"{cefr_label}1/{cefr_label}2"

        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": PROMPT_TEMPLATE.render(
                    text=example["text"], level=cefr_label
                ),
            },
        ]
        completion = [
            {"role": "assistant", "content": example["summary"]},
        ]
        return {"prompt": prompt, "completion": completion}

    return preprocess


def filter_long_examples(dataset, tokenizer, max_length=MAX_SEQ_LENGTH):
    """
    Filter out examples that exceed the maximum sequence length.

    Args:
        dataset: HuggingFace Dataset to filter
        tokenizer: Tokenizer for computing sequence lengths
        max_length (int): Maximum allowed sequence length in tokens

    Returns:
        Dataset: Filtered dataset with only examples within length limit
    """

    def is_short_enough(batch):
        texts = [
            tokenizer.apply_chat_template(
                p + c,
                tokenize=False,
                add_generation_prompt=False,
            )
            for p, c in zip(batch["prompt"], batch["completion"])
        ]

        tokenized = tokenizer(
            texts,
            truncation=False,
        )

        return [len(ids) <= max_length for ids in tokenized["input_ids"]]

    return dataset.filter(is_short_enough, batched=True)


def prepare_dataset(dataset: DatasetDict, tokenizer: AutoTokenizer) -> DatasetDict:
    """
    Prepare dataset for SFT training by preprocessing and filtering examples.

    Args:
        dataset (DatasetDict): Raw dataset with train/validation splits
        tokenizer (AutoTokenizer): Tokenizer for the model

    Returns:
        DatasetDict: Processed dataset ready for training
    """
    preprocess_fn = build_preprocess_fn(tokenizer)
    dataset = dataset.map(preprocess_fn)
    dataset["train"] = filter_long_examples(dataset["train"], tokenizer)
    dataset["validation"] = filter_long_examples(dataset["validation"], tokenizer)
    # Quick sanity check
    sample = next(iter(dataset["train"]))
    print(sample["prompt"])
    print(sample["completion"])
    return dataset


# ======================
# LoRA config
# ======================


def get_lora_config():
    """
    Create LoRA configuration for parameter-efficient fine-tuning.

    Returns:
        LoraConfig: Configuration for LoRA adapters
    """
    return LoraConfig(
        lora_alpha=32,
        lora_dropout=0.05,
        r=32,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )


# ======================
# Training
# ======================


def create_trainer(model_id: str, output_dir: str, dataset: DatasetDict) -> SFTTrainer:
    """
    Create and configure SFT trainer.

    Args:
        model_id (str): HuggingFace model ID to fine-tune
        output_dir (str): Directory to save the fine-tuned model
        dataset (DatasetDict): Prepared dataset with train/validation splits

    Returns:
        SFTTrainer: Configured trainer ready for training
    """
    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        # quantization_config=quantization_config
    )

    # Prepare dataset (adds "messages")
    dataset = prepare_dataset(dataset, tokenizer)

    # Make sure old output dir doesn't interfere (optional)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    args = SFTConfig(
        output_dir=output_dir,
        max_length=MAX_SEQ_LENGTH,
        packing=False,
        num_train_epochs=10,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        optim="adamw_torch",
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        learning_rate=1e-4,
        fp16=(model.dtype == torch.float16),
        bf16=(model.dtype == torch.bfloat16),
        warmup_ratio=0.05,
        lr_scheduler_type="linear",
        push_to_hub=False,
        load_best_model_at_end=True,
        save_total_limit=1,
        completion_only_loss=True,
    )

    lora_config = get_lora_config()

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=args,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    return trainer


def main():
    """
    Main training function.

    Parses command-line arguments, loads model and datasets, configures SFT training,
    and runs the training loop.
    """
    parser = argparse.ArgumentParser(
        description="Train a language model using Supervised Fine-Tuning (SFT)"
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
        default="dataset/org_with_cefr_labels/us_sft_train.jsonl",
        help="Path to training dataset JSONL file",
    )
    parser.add_argument(
        "--eval_dataset",
        type=str,
        default="dataset/org_with_cefr_labels/us_sft_dev.jsonl",
        help="Path to evaluation dataset JSONL file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/gemma_sft/v3",
        help="Directory to save the fine-tuned model",
    )
    args = parser.parse_args()

    # Load datasets
    dataset = load_splits(args.train_dataset, args.eval_dataset)

    # Create trainer and train
    trainer = create_trainer(args.model_id, args.output_dir, dataset)
    trainer.train()


# ======================
# Entry point
# ======================

if __name__ == "__main__":
    main()
