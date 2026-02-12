import os
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS"] = "CPP"

import pandas as pd
from datasets import Dataset, DatasetDict
from prompts import PROMPT_TEMPLATE, SYSTEM_PROMPT
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_SEQ_LENGTH = 512


# ======================
# Config
# ======================

TRAIN_PATH = "dataset/org_with_cefr_labels/sft_small_train.jsonl"
DEV_PATH = "dataset/org_with_cefr_labels/sft_small_dev.jsonl"
OUTPUT_DIR = "models/gemma_sft/v1"

# ======================
# Data loading
# ======================


def load_splits(train_path, dev_path) -> DatasetDict:
    """Load train/dev/test CSVs into a DatasetDict."""
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
    def preprocess(example):
        cefr_label = example["cefr_labels"][0]["label"]

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


def filter_long_examples(dataset, tokenizer, max_length=4096):

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
# Training
# ======================

dataset_dict = load_splits(TRAIN_PATH, DEV_PATH)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-3-270m-it",
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=32,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing=False,  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)

dataset = prepare_dataset(dataset_dict, tokenizer)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    max_seq_length=MAX_SEQ_LENGTH,
    packing=False,  # Can make training 5x faster for short sequences.
    args=SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_ratio=0.05,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=10,
        optim="adamw_8bit",
        # weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        save_total_limit=1,
        output_dir=OUTPUT_DIR,
        save_strategy="epoch",
        eval_strategy="epoch",
        bf16=True,
        fp16=False,
        completion_only_loss=True,
        gradient_checkpointing=False,
    ),
)

trainer.train()
