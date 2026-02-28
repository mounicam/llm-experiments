"""
Group Relative Policy Optimization (GRPO) training script for fine-tuning language models.

This script trains a language model using GRPO, a reinforcement learning technique
that optimizes models based on reward functions. The model learns to generate
text summaries at different CEFR (Common European Framework of Reference) levels.

Usage:
    python train_grpo.py --model_id google/gemma-3-1b-it \\
                         --train_dataset dataset/us_RL_train_2k.jsonl \\
                         --eval_dataset dataset/us_RL_dev_1k.jsonl \\
                         --output_dir models/grpo_out \\
"""

from unsloth import FastLanguageModel

import argparse
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer
from datasets import DatasetDict
from dataset_loader import load_splits
from prompts import generate_input_content, READABILTIY_LABELS
from reward_functions import fkgl_reward_func, bertscore_reward_func, format_reward_func, length_reward_func

# ======================
# Config
# ======================

MAX_SEQ_LENGTH = 2688
MAX_SUMMARY_LENGTH = 512
LORA_RANK = 32


# ======================
# Preprocessing
# ======================


def build_preprocess_fn(tokenizer):
    """
    Build a preprocessing function for expanding dataset by CEFR levels.

    Creates three versions of each example (A, B, C levels) to train the model
    on generating summaries at different reading difficulty levels.

    Args:
        tokenizer: HuggingFace tokenizer for the model

    Returns:
        function: Preprocessing function that expands examples by CEFR levels
    """

    def preprocess(examples):
        new_prompts = []
        new_answers = []

        # Iterate through every row in the incoming batch
        for i in range(len(examples["text"])):
            original_text = examples["text"][i]
            # Answer is the ground truth summary (acts as reference)
            original_summary = examples["summary"][i]

            # Create three versions of the prompt for each text
            for level in READABILTIY_LABELS:

                prompt = [
                    {
                        "role": "user",
                        "content": generate_input_content(level, original_text),
                    },
                ]

                new_prompts.append(prompt)
                new_answers.append(original_summary)

        # Return the expanded batch
        return {"prompt": new_prompts, "answer": new_answers}

    return preprocess


def prepare_dataset(dataset: DatasetDict, tokenizer: AutoTokenizer) -> DatasetDict:
    """
    Prepare dataset for GRPO training by expanding and filtering examples.

    Expands each example into three versions (A, B, C CEFR levels) and filters
    out examples that exceed the maximum prompt length.

    Args:
        dataset (DatasetDict): Raw dataset with train/validation splits
        tokenizer (AutoTokenizer): Tokenizer for the model

    Returns:
        DatasetDict: Processed dataset ready for training
    """
    preprocess_fn = build_preprocess_fn(tokenizer)

    # Identify original columns to remove them during expansion
    column_names = dataset["train"].column_names

    # Apply expansion to train and validation
    dataset = dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=column_names,
        desc="Expanding dataset for all readability levels",
    )

    # Sanity Check
    print(f"Dataset expanded! New training size: {len(dataset['train'])}")
    sample = dataset["train"][0]
    print(sample)
    return dataset


# ======================
# Reward functions imported from reward_functions.py
# ======================
# - fkgl_reward_func: Flesch-Kincaid Grade Level for readability
# - bertscore_reward_func: Semantic similarity with reference
# - format_reward_func: Format quality and completion checks


def main():
    """
    Main training function.

    Parses command-line arguments, loads model and datasets, configures GRPO training,
    and runs the training loop with reward-based optimization.
    """
    parser = argparse.ArgumentParser(
        description="Train a language model using Group Relative Policy Optimization (GRPO)"
    )
    parser.add_argument(
        "--model_id", type=str, required=True, help="HuggingFace model ID to fine-tune"
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
        required=True,
        help="Directory to save the fine-tuned model",
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=3, help="Number of training epochs"
    )

    args = parser.parse_args()

    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_id,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=0.6,
        use_vllm=True,
        bnb_4bit_use_double_quant=True,
    )

    # Significantly reduces time to generate
    tokenizer.pad_token = tokenizer.eos_token
    model.config.eos_token_id = tokenizer.eos_token_id

    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=LORA_RANK,
        use_gradient_checkpointing=False,
        random_state=3407,
    )

    # Load and prepare datasets
    dataset = load_splits(args.train_dataset, args.eval_dataset)
    dataset = prepare_dataset(dataset, tokenizer)

    # Configure GRPO training
    training_args = GRPOConfig(
        use_vllm=True,
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=10,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_generations=8,
        max_prompt_length=MAX_SEQ_LENGTH - MAX_SUMMARY_LENGTH,
        max_completion_length=MAX_SUMMARY_LENGTH,
        num_train_epochs=args.num_train_epochs,
        save_steps=500,
        max_grad_norm=1.0,
        output_dir=args.output_dir,
        gradient_checkpointing=True,
        bf16=True,
        save_total_limit=10,
        save_strategy="steps",
        beta=0.1,
        report_to="tensorboard",
        logging_dir=f"{args.output_dir}/logs"
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            fkgl_reward_func,
            bertscore_reward_func,
            format_reward_func,
            # length_reward_func,
        ],
        args=training_args,
        train_dataset=dataset["train"],
    )

    # Train model
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
