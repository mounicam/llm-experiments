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
import torch
import pandas as pd
from functools import partial
from tqdm import tqdm
from vllm import SamplingParams
from transformers import pipeline
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompts import PROMPT_TEMPLATE, SYSTEM_PROMPT, CEFR_LABELS

# ======================
# Config
# ======================

MAX_SEQ_LENGTH = 4096
MAX_SUMMARY_LENGTH = 512
LORA_RANK = 32


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
            for cefr_base in ["A", "B", "C"]:
                target_label = f"{cefr_base}1/{cefr_base}2"

                prompt = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": PROMPT_TEMPLATE.render(
                            text=original_text, level=target_label
                        ),
                    },
                ]

                new_prompts.append(prompt)
                new_answers.append(original_summary)

        # Return the expanded batch
        return {"prompt": new_prompts, "answer": new_answers}

    return preprocess


def filter_long_examples(
    dataset, tokenizer, max_length=MAX_SEQ_LENGTH - MAX_SUMMARY_LENGTH
):
    """
    Filter out examples that exceed the maximum prompt length.

    Args:
        dataset: HuggingFace Dataset to filter
        tokenizer: Tokenizer for computing sequence lengths
        max_length (int): Maximum allowed prompt length in tokens

    Returns:
        Dataset: Filtered dataset with only examples within length limit
    """

    def is_short_enough(batch):
        texts = [
            tokenizer.apply_chat_template(
                p,
                tokenize=False,
                add_generation_prompt=False,
            )
            for p in batch["prompt"]
        ]

        tokenized = tokenizer(
            texts,
            truncation=False,
        )

        return [len(ids) <= max_length for ids in tokenized["input_ids"]]

    return dataset.filter(is_short_enough, batched=True)


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
        desc="Expanding dataset for all CEFR levels",
    )

    # Filter for length AFTER expanding the dataset
    dataset["train"] = filter_long_examples(dataset["train"], tokenizer)
    if "validation" in dataset:
        dataset["validation"] = filter_long_examples(dataset["validation"], tokenizer)

    # Sanity Check
    print(f"Dataset expanded! New training size: {len(dataset['train'])}")
    sample = dataset["train"][0]
    print(f"Example target level: {sample['prompt'][1]['content'].split('level ')[-1]}")

    return dataset


# ======================
# Reward function
# ======================


CLASSIFIER = pipeline(
    "text-classification",
    model="AbdullahBarayan/ModernBERT-base-doc_en-Cefr",
    device=0,
    # batch_size=16,
    top_k=None,
    dtype=torch.float16,
)


def quality_reward_func(
    prompts,
    completions,
    answer,
    reward_type="cefr_only",
    cefr_weight=1.0,
    bert_weight=0.5,
    **kwargs,
) -> list[float]:
    """
    Compute quality rewards for generated completions based on various metrics.

    Evaluates completions using CEFR classification and optionally BERT scores.
    Supports multiple reward calculation strategies similar to DPO training.

    Args:
        prompts: List of conversation prompts
        completions: List of model-generated completions
        answer: Reference answers for BERT score calculation
        reward_type (str): Type of reward calculation:
            - "cefr_only": Use only CEFR probability
            - "bert_only": Use only BERT score (requires bert_score package)
            - "combined": Weighted combination of CEFR and BERT scores
        cefr_weight (float): Weight for CEFR probability in combined mode
        bert_weight (float): Weight for BERT score in combined mode
        **kwargs: Additional keyword arguments

    Returns:
        list[float]: Reward scores for each completion
    """
    responses = [completion[0]["content"] for completion in completions]

    # Compute CEFR scores if needed
    cefr_values = []
    if reward_type in ["cefr_only", "combined"]:
        cefr_probs_flat = list(
            tqdm(CLASSIFIER(responses, batch_size=16), total=len(responses))
        )
        prompt = prompts[0][-1]["content"]
        for cefr_label in CEFR_LABELS:
            label = f"{cefr_label}1/{cefr_label}2"
            if label in prompt:
                for probs in cefr_probs_flat:
                    probs = [p["score"] for p in probs if p["label"][:1] == cefr_label]
                    cefr_values.append(sum(probs))
                break

    # Compute BERT scores if needed
    bert_scores = []
    if reward_type in ["bert_only", "combined"]:
        try:
            from bert_score import score

            references = (
                [answer] * len(responses)
                if isinstance(answer, str)
                else answer * len(responses)
            )
            _, _, f1_flat = score(
                responses,
                references,
                model_type="roberta-large",
                lang="en",
                batch_size=16,
                device="cuda:0",
                verbose=False,
            )
            bert_scores = f1_flat.tolist()
        except ImportError:
            raise ImportError(
                "bert_score package required for bert_only or combined reward types"
            )

    # Calculate final rewards based on type
    if reward_type == "cefr_only":
        assert len(completions) == len(cefr_values)
        return cefr_values
    elif reward_type == "bert_only":
        assert len(completions) == len(bert_scores)
        return bert_scores
    elif reward_type == "combined":
        assert len(cefr_values) == len(bert_scores)
        return [
            (c * cefr_weight + b * bert_weight) / (cefr_weight + bert_weight)
            for c, b in zip(cefr_values, bert_scores)
        ]
    else:
        raise ValueError(f"Unknown reward_type: {reward_type}")


def format_reward_func(completions, **kwargs) -> list[float]:
    """
    Compute format-based rewards for generated completions.

    Evaluates completion quality based on proper formatting, natural endings,
    and conciseness. Penalizes empty responses and truncated text.

    Args:
        completions: List of model-generated completions
        **kwargs: Additional keyword arguments

    Returns:
        list[float]: Format quality scores for each completion
    """
    responses = [completion[0]["content"] for completion in completions]
    rewards = []

    for response in responses:
        score = 0.0

        # 1. Clean response for checking
        text = response.strip()

        if len(text) == 0:
            rewards.append(-1.0)  # Heavy penalty for empty responses
            continue

        # 2. Check for natural punctuation at the end (sentences usually end in . ! or ?)
        # If it doesn't end in punctuation, it was likely "guillotined" at 256 tokens.
        if text[-1] in [".", "!", "?", '"', "'"]:
            score += 0.2
        else:
            score -= 0.5  # Penalty for rambling/cutoff

        # 3. Completion check (EOS check)
        # In TRL, we can check if the model stopped naturally by looking at the last few chars
        # for your specific model's EOS string or just checking if it hit the cap.
        # If the generated text length is significantly less than the max (256),
        # it almost certainly hit an EOS.
        if (
            len(text.split()) < MAX_SUMMARY_LENGTH
        ):  # Assuming 256 tokens is roughly 200 words
            score += 0.3  # Bonus for being concise and finishing early

        rewards.append(score)

    return rewards


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
        "--num_train_epochs", type=int, default=1, help="Number of training epochs"
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
    args = parser.parse_args()

    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_id,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=0.7,
        use_vllm=True,
        bnb_4bit_use_double_quant=True,
    )

    print(tokenizer.eos_token, tokenizer.pad_token)
    # Significantly reduces time to generate
    tokenizer.pad_token = tokenizer.eos_token
    model.config.eos_token_id = tokenizer.eos_token_id

    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_rank,
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
        report_to="none",
        output_dir=args.output_dir,
        gradient_checkpointing=True,
        bf16=True,
        save_total_limit=10,
        save_strategy="steps",
        beta=0.1,
    )

    # Initialize trainer with reward functions
    # Bind reward_type and weights to the quality reward function
    quality_reward_with_args = partial(
        quality_reward_func,
        reward_type=args.reward_type,
        cefr_weight=args.cefr_weight,
        bert_weight=args.bert_weight,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[quality_reward_with_args, format_reward_func],
        args=training_args,
        train_dataset=dataset["train"],
    )

    # Train model
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
