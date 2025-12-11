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


# ======================
# Config
# ======================

MODEL_ID = "google/gemma-3-4b-it"

OUTPUT_DIR = "models/readme/gemma/lora"
LOGGING_DIR = "./logs/gemma/test"

TRAIN_PATH = "readme/readme_en_train.csv"
DEV_PATH = "readme/readme_en_val.csv"
TEST_PATH = "readme/readme_en_test.csv"


PROMPT_TEMPLATE = """
You are a language learning evaluator assessing the complexity of an English sentence given its context. Please give a rating between 1 to 6. Do not output any other text.
Context: {{ context }}
Sentence: {{ sentence }}
Rating (1-6):
"""


# ======================
# Data loading
# ======================

def load_splits() -> DatasetDict:
    """Load train/dev/test CSVs into a DatasetDict."""
    train_df = pd.read_csv(TRAIN_PATH)
    dev_df = pd.read_csv(DEV_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # Drop rows with missing fields
    for df in (train_df, dev_df, test_df):
        df.dropna(subset=["Rating", "Sentence", "Paragraph"], inplace=True)

    train = Dataset.from_pandas(train_df)
    dev = Dataset.from_pandas(dev_df)
    test = Dataset.from_pandas(test_df)

    dataset = DatasetDict({"train": train, "validation": dev, "test": test})
    print(dataset)
    return dataset


# ======================
# Preprocessing
# ======================

def build_preprocess_fn(tokenizer: AutoTokenizer):
    """Return a preprocess function that builds chat messages."""
    jinja_template = Template(PROMPT_TEMPLATE)

    def preprocess(example):
        messages = [
            # {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": jinja_template.render(
                    context=example["Paragraph"],
                    sentence=example["Sentence"],
                ),
            },
            {
                "role": "assistant",
                "content": f"The score is {int(example['Rating'])}",
            },
        ]

        # You *can* get the rendered text if needed:
        # text = tokenizer.apply_chat_template(messages, add_generation_prompt=False)

        # TRL SFTTrainer with chat-format datasets expects "messages"
        return {"messages": messages}

    return preprocess


def prepare_dataset(dataset: DatasetDict, tokenizer: AutoTokenizer) -> DatasetDict:
    preprocess_fn = build_preprocess_fn(tokenizer)
    dataset = dataset.map(preprocess_fn)
    # Quick sanity check
    sample = next(iter(dataset["train"]))
    print(sample["messages"])
    return dataset


# ======================
# LoRA config
# ======================

def get_lora_config():
    return LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=8,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        # Make sure to save the lm_head and embed_tokens as you train special tokens
        modules_to_save=["lm_head", "embed_tokens"],
    )


# ======================
# Training
# ======================

def create_trainer(dataset: DatasetDict) -> SFTTrainer:
    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        quantization_config=quantization_config,
        device_map="cuda:0",
    )

    # Prepare dataset (adds "messages")
    dataset = prepare_dataset(dataset, tokenizer)

    torch_dtype = model.dtype

    # Make sure old output dir doesn't interfere (optional)
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    if os.path.exists(LOGGING_DIR):
        shutil.rmtree(LOGGING_DIR)

    args = SFTConfig(
        output_dir=OUTPUT_DIR,
        max_length=256,
        packing=False,
        num_train_epochs=5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        gradient_checkpointing=False,
        optim="adamw_torch_fused",
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        learning_rate=1e-4,
        fp16=(torch_dtype == torch.float16),
        bf16=(torch_dtype == torch.bfloat16),
        warmup_ratio=0.05,
        lr_scheduler_type="linear",
        push_to_hub=False,
        report_to=["tensorboard"],
        logging_dir=LOGGING_DIR,
        load_best_model_at_end=True,
        save_total_limit=1,
        dataset_kwargs={
            "add_special_tokens": False,
            "append_concat_token": True,
        },
    )

    lora_config = get_lora_config()

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=args,
        processing_class=tokenizer,
        peft_config=lora_config,  # comment out if you want full fine-tune
        # compute_metrics=compute_metrics,  # plug in if you have it
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
