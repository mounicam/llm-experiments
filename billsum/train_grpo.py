from unsloth import FastLanguageModel

import torch
import pandas as pd
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

MODEL_ID = "google/gemma-3-1b-it"
# MODEL_ID = "models/sft"
OUTPUT_DIR = "models/gemma_oob_grpo/v1"
TRAIN_PATH = "dataset/us_RL_train_2k.jsonl"
DEV_PATH = "dataset/us_RL_dev_1k.jsonl"
MAX_SEQ_LENGTH = 4096
MAX_SUMMARY_LENGTH = 512
LORA_RANK = 32


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


# def build_preprocess_fn(tokenizer):
#     def preprocess(example):
#         cefr_label = example["cefr_labels"][0]["label"][:1]
#         label = f"{cefr_label}1/{cefr_label}2"
#         prompt = [
#             {"role": "system", "content": SYSTEM_PROMPT},
#             {
#                 "role": "user",
#                 "content": PROMPT_TEMPLATE.render(text=example["text"], level=label),
#             },
#         ]
#         return {"prompt": prompt, "answer": example["summary"]}

#     return preprocess


# def prepare_dataset(dataset: DatasetDict, tokenizer: AutoTokenizer) -> DatasetDict:
#     preprocess_fn = build_preprocess_fn(tokenizer)
#     dataset = dataset.map(preprocess_fn)
#     dataset["train"] = filter_long_examples(dataset["train"], tokenizer)
#     dataset["validation"] = filter_long_examples(dataset["validation"], tokenizer)
#     # Quick sanity check
#     sample = next(iter(dataset["train"]))
#     print(sample["prompt"])
#     print(sample["answer"])
#     return dataset


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


def reward_func(prompts, completions, answer, **kwargs) -> list[float]:

    responses = [completion[0]["content"] for completion in completions]
    # for resp in responses:
    #     print(resp)
    # print("*" * 30)
    cefr_probs_flat = list(tqdm(CLASSIFIER(responses, batch_size=16), total=len(responses)))
    cefr_values = []
    prompt = prompts[0][-1]["content"]
    for cefr_label in CEFR_LABELS:
        label = f"{cefr_label}1/{cefr_label}2"
        if label in prompt:
            for probs in cefr_probs_flat:
                probs = [p["score"] for p in probs if p["label"][:1] == cefr_label]
                cefr_values.append(sum(probs))
            break

    # _, _, f1_flat = score(
    #     responses,
    #     references,
    #     model_type="roberta-large",  # "microsoft/deberta-xlarge-mnli",
    #     lang="en",
    #     batch_size=16,
    #     device="cuda:0",
    #     verbose=True,
    # )
    # bert_scores_flat = f1_flat.tolist()
    # return [(c + 0.5 * b) / 1.5 for c, b in zip(cefr_values, bert_scores_flat)]
    assert len(completions) == len(cefr_values)
    return cefr_values


def format_reward_func(completions, **kwargs) -> list[float]:
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


# ======================
# Training
# ======================

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    fast_inference=True,
    max_lora_rank=LORA_RANK,
    gpu_memory_utilization=0.7,
    use_vllm=True,
    bnb_4bit_use_double_quant=True
)

print(tokenizer.eos_token, tokenizer.pad_token)
# Significantly reduces time to generate
tokenizer.pad_token = tokenizer.eos_token
model.config.eos_token_id = tokenizer.eos_token_id

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

dataset = load_splits(TRAIN_PATH, DEV_PATH)
dataset = prepare_dataset(dataset, tokenizer)

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
    # per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    num_generations=8,
    max_prompt_length=MAX_SEQ_LENGTH - MAX_SUMMARY_LENGTH,
    max_completion_length=MAX_SUMMARY_LENGTH,
    num_train_epochs=2,
    save_steps=500,
    max_grad_norm=1.0,
    report_to="none",
    output_dir=OUTPUT_DIR,
    gradient_checkpointing=False,
    bf16=True,
    # eval_strategy="steps",
    # eval_steps=1,
    save_total_limit=10,
    save_strategy="steps",
    beta=0.1,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[reward_func, format_reward_func],
    args=training_args,
    train_dataset=dataset["train"],
    # eval_dataset=dataset["validation"],
)

trainer.train()

"""
 def masked_batch_mean(x):
            print(3542, x.shape, completion_mask.shape)
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence"
                return x.mean()
            else:
                # If shapes mismatch, slice 'x' to match the 'mask' 
                # (keeping the right-most tokens which are the completion)
                if x.shape[1] != completion_mask.shape[1]:
                    x_1 = x[:, -completion_mask.shape[1]:]
                    return (x_1 * mask).sum() / completion_token_count
                else:
                    return (x * completion_mask).sum() / completion_token_count

"""
