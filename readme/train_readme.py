"""
Train a language model for sentence complexity rating using supervised fine-tuning.

This script fine-tunes a pre-trained causal language model to rate the complexity of
English sentences on a scale from 1 to 6, given their context. It uses LoRA (Low-Rank
Adaptation) for parameter-efficient fine-tuning and 4-bit quantization for memory efficiency.

Usage:
    Basic usage with defaults:
        python train_readme.py

    Custom configuration:
        python train_readme.py \\
            --model_id google/gemma-3-4b-it \\
            --train_path readme/readme_en_train.csv \\
            --dev_path readme/readme_en_val.csv \\
            --test_path readme/readme_en_test.csv \\
            --output_dir models/readme/gemma/lora \\
            --num_epochs 5 \\
            --batch_size 2 \\
            --learning_rate 1e-4
"""

import os
import shutil
import argparse

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

# Default prompt template for language complexity evaluation
PROMPT_TEMPLATE = """
You are a language learning evaluator assessing the complexity of an English sentence given its context. Please give a rating between 1 to 6. Do not output any other text.
Context: {{ context }}
Sentence: {{ sentence }}
Rating (1-6):
"""


# ======================
# Data loading
# ======================

def load_splits(train_path: str, dev_path: str, test_path: str) -> DatasetDict:
    """
    Load train/dev/test CSVs into a DatasetDict.

    Args:
        train_path: Path to training CSV file
        dev_path: Path to validation CSV file
        test_path: Path to test CSV file

    Returns:
        DatasetDict containing train, validation, and test splits
    """
    # Load CSVs into pandas DataFrames
    train_df = pd.read_csv(train_path)
    dev_df = pd.read_csv(dev_path)
    test_df = pd.read_csv(test_path)

    # Drop rows with missing values in essential columns
    for df in (train_df, dev_df, test_df):
        df.dropna(subset=["Rating", "Sentence", "Paragraph"], inplace=True)

    # Convert pandas DataFrames to HuggingFace Dataset objects
    train = Dataset.from_pandas(train_df)
    dev = Dataset.from_pandas(dev_df)
    test = Dataset.from_pandas(test_df)

    # Create a DatasetDict for easy access to all splits
    dataset = DatasetDict({"train": train, "validation": dev, "test": test})
    print(dataset)
    return dataset


# ======================
# Preprocessing
# ======================

def build_preprocess_fn(tokenizer: AutoTokenizer):
    """
    Build a preprocessing function that converts examples into chat format.

    Args:
        tokenizer: HuggingFace tokenizer for the model

    Returns:
        Function that converts dataset examples into message format
    """
    jinja_template = Template(PROMPT_TEMPLATE)

    def preprocess(example):
        """
        Convert a single example into chat message format.

        The format includes a user message with the complexity rating task
        and an assistant response with the rating.
        """
        messages = [
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

        # TRL SFTTrainer expects "messages" key for chat-format datasets
        return {"messages": messages}

    return preprocess


def prepare_dataset(dataset: DatasetDict, tokenizer: AutoTokenizer) -> DatasetDict:
    """
    Apply preprocessing to convert dataset into chat format.

    Args:
        dataset: DatasetDict with train/validation/test splits
        tokenizer: HuggingFace tokenizer for the model

    Returns:
        Preprocessed DatasetDict with messages field
    """
    preprocess_fn = build_preprocess_fn(tokenizer)
    dataset = dataset.map(preprocess_fn)

    # Print a sample to verify the message format
    sample = next(iter(dataset["train"]))
    print(sample["messages"])
    return dataset


# ======================
# LoRA config
# ======================

def get_lora_config():
    """
    Configure LoRA (Low-Rank Adaptation) parameters for efficient fine-tuning.

    Returns:
        LoraConfig object with parameter-efficient fine-tuning settings
    """
    return LoraConfig(
        lora_alpha=16,  # Scaling factor for LoRA weights
        lora_dropout=0.05,  # Dropout probability for LoRA layers
        r=8,  # Rank of the low-rank matrices
        bias="none",  # Don't train bias parameters
        target_modules="all-linear",  # Apply LoRA to all linear layers
        task_type="CAUSAL_LM",  # Task type for causal language modeling
        # Save embedding and output layers to support special tokens
        modules_to_save=["lm_head", "embed_tokens"],
    )


# ======================
# Training
# ======================

def create_trainer(
    dataset: DatasetDict,
    model_id: str,
    output_dir: str,
    logging_dir: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
) -> SFTTrainer:
    """
    Create and configure the SFTTrainer for supervised fine-tuning.

    Args:
        dataset: DatasetDict with train/validation/test splits
        model_id: HuggingFace model identifier
        output_dir: Directory to save model checkpoints
        logging_dir: Directory to save training logs
        num_epochs: Number of training epochs
        batch_size: Per-device training batch size
        learning_rate: Learning rate for optimization

    Returns:
        Configured SFTTrainer ready for training
    """
    # Load tokenizer from pretrained model
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Configure 4-bit quantization for memory-efficient training
    # Uses NF4 (Normal Float 4) quantization with double quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,  # Enable 4-bit quantization
        bnb_4bit_use_double_quant=True,  # Quantize the quantization constants
        bnb_4bit_quant_type='nf4',  # Use NF4 quantization type
        bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in bfloat16
        bnb_4bit_quant_storage=torch.bfloat16,  # Store quantized weights in bfloat16
    )

    # Load the pre-trained causal language model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for better numerical stability
        attn_implementation="eager",  # Use eager attention implementation
        quantization_config=quantization_config,
        device_map="cuda:0",  # Map model to first GPU
    )

    # Convert dataset examples to chat message format
    dataset = prepare_dataset(dataset, tokenizer)

    torch_dtype = model.dtype

    # Clean up old output directories to start fresh
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    if os.path.exists(logging_dir):
        shutil.rmtree(logging_dir)

    # Configure supervised fine-tuning parameters
    args = SFTConfig(
        output_dir=output_dir,
        max_length=256,  # Maximum sequence length
        packing=False,  # Don't pack multiple examples into one sequence
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,  # Accumulate gradients for effective batch size
        gradient_checkpointing=False,  # Disable gradient checkpointing
        optim="adamw_torch_fused",  # Use fused AdamW optimizer
        logging_steps=10,  # Log metrics every 10 steps
        save_strategy="epoch",  # Save checkpoint after each epoch
        eval_strategy="epoch",  # Evaluate after each epoch
        learning_rate=learning_rate,
        fp16=(torch_dtype == torch.float16),  # Enable FP16 if model uses float16
        bf16=(torch_dtype == torch.bfloat16),  # Enable BF16 if model uses bfloat16
        warmup_ratio=0.05,  # Warmup for 5% of training steps
        lr_scheduler_type="linear",  # Linear learning rate schedule
        push_to_hub=False,  # Don't push to HuggingFace Hub
        report_to=["tensorboard"],  # Log to TensorBoard
        logging_dir=logging_dir,
        load_best_model_at_end=True,  # Load best checkpoint at end
        save_total_limit=1,  # Only keep the best checkpoint
        dataset_kwargs={
            "add_special_tokens": False,  # Tokenizer will handle special tokens
            "append_concat_token": True,  # Append token when concatenating
        },
    )

    # Get LoRA configuration for parameter-efficient fine-tuning
    lora_config = get_lora_config()

    # Create the trainer with all configurations
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=args,
        processing_class=tokenizer,
        peft_config=lora_config,  # Use LoRA for efficient fine-tuning
    )

    return trainer


def train():
    """
    Main training function.

    Parses command-line arguments, loads model and datasets, configures SFT training,
    and runs the training loop with LoRA for parameter-efficient fine-tuning.
    """
    parser = argparse.ArgumentParser(
        description="Train a language model for sentence complexity rating using supervised fine-tuning."
    )

    # Model arguments
    parser.add_argument(
        "--model_id",
        type=str,
        default="google/gemma-3-4b-it",
        help="HuggingFace model identifier to fine-tune",
    )

    # Data arguments
    parser.add_argument(
        "--train_path",
        type=str,
        default="readme/readme_en_train.csv",
        help="Path to training CSV file",
    )
    parser.add_argument(
        "--dev_path",
        type=str,
        default="readme/readme_en_val.csv",
        help="Path to validation CSV file",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default="readme/readme_en_test.csv",
        help="Path to test CSV file",
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/readme/gemma/lora",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="./logs/gemma/test",
        help="Directory to save training logs",
    )

    # Training arguments
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Per-device training batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for optimization",
    )

    args = parser.parse_args()

    # Load the dataset splits
    print(f"Loading datasets from {args.train_path}, {args.dev_path}, {args.test_path}")
    dataset = load_splits(args.train_path, args.dev_path, args.test_path)

    # Create and configure the trainer
    print(f"Creating trainer with model {args.model_id}")
    trainer = create_trainer(
        dataset=dataset,
        model_id=args.model_id,
        output_dir=args.output_dir,
        logging_dir=args.logging_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    # Start training
    print("Starting training...")
    trainer.train()

    print(f"Training complete! Model saved to {args.output_dir}")


# ======================
# Entry point
# ======================

if __name__ == "__main__":
    train()
