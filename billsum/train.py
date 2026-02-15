import os
import shutil

import pandas as pd
from datasets import Dataset, DatasetDict

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from jinja2 import Template
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from prompts import PROMPT_TEMPLATE, SYSTEM_PROMPT

"""
Command used:  accelerate launch --mixed_precision bf16 train.py

completion_only needs to be set True. Also, preprocessing needs to be prompt + completion. Otherwise model learns the input too which is not needed.

Had to filter long examples to not eval_loss as NaN. 

"""

# ======================
# Config
# ======================

MODEL_ID = "google/gemma-3-1b-it"
OUTPUT_DIR = "models/gemma_sft/v3"
TRAIN_PATH = "dataset/org_with_cefr_labels/us_sft_train.jsonl"
DEV_PATH = "dataset/org_with_cefr_labels/us_sft_dev.jsonl"
MAX_SEQ_LENGTH=4096

# ======================
# Data loading
# ======================


def load_splits() -> DatasetDict:
    """Load train/dev/test CSVs into a DatasetDict."""
    train_df = pd.read_json(TRAIN_PATH, lines=True)
    dev_df = pd.read_json(DEV_PATH, lines=True)

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
    return LoraConfig(
        lora_alpha=32,
        lora_dropout=0.05,
        r=32,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        # Make sure to save the lm_head and embed_tokens as you train special tokens
        # modules_to_save=["lm_head", "embed_tokens"],
    )


# ======================
# Training
# ======================


def create_trainer(dataset: DatasetDict) -> SFTTrainer:
    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        # quantization_config=quantization_config
    )

    # Prepare dataset (adds "messages")
    dataset = prepare_dataset(dataset, tokenizer)

    # Make sure old output dir doesn't interfere (optional)
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    # if os.path.exists(LOGGING_DIR):
    #     shutil.rmtree(LOGGING_DIR)

    args = SFTConfig(
        output_dir=OUTPUT_DIR,
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
        # report_to=["tensorboard"],
        # logging_dir=LOGGING_DIR,
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


def train():
    dataset = load_splits()
    trainer = create_trainer(dataset)
    trainer.train()


# ======================
# Entry point
# ======================

if __name__ == "__main__":
    train()
