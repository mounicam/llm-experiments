import torch
import pandas as pd
from transformers import pipeline
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer

# ======================
# Config
# ======================

MODEL_ID = "google/gemma-3-1b-it"
OUTPUT_DIR = "models/gemma_sft_grpo/v1"
TRAIN_PATH = "experiments/gemma_1b_sft/v3/us_RL_train_4k_w_metrics.jsonl"
DEV_PATH = "experiments/gemma_1b_sft/v3/us_RL_dev_4k_w_metrics.jsonl"
MAX_SEQ_LENGTH = 256
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

        return {"prompt": prompt, "answer": example["summary"]}

    return preprocess


def filter_long_examples(dataset, tokenizer, max_length=MAX_SEQ_LENGTH):

    def is_short_enough(batch):
        texts = [
            tokenizer.apply_chat_template(
                p,
                tokenize=False,
                add_generation_prompt=False,
            )
            for p in zip(batch["prompt"])
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
# Reward function
# ======================


# CLASSIFIER = pipeline(
#     "text-classification",
#     model="AbdullahBarayan/ModernBERT-base-doc_en-Cefr",
#     device=0,
#     batch_size=16,
#     top_k=None,
# )


def reward_func(prompts, completions, answer, **kwargs) -> list[float]:

    print(f"Scoring {len(completions)} total candidates...")

    # responses = [completion[0]["content"] for completion in completions]
    # references = [answer] * len(responses)

    # cefr_probs_flat = list(tqdm(classifier(all_texts), total=len(all_texts)))
    # cefr_values = []
    # prompt = prompts[0][-1]["content"]
    # for cefr_label in CEFR_LABELS:
    #     label = f"{cefr_label}1/{cefr_label}2"
    #     if label in prompt:
    #         for probs in cefr_probs_flat:
    #             p = probs[f"{cefr_label}1"] + probs[f"{cefr_label}2"]
    #             cefr_values.append(p)

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
    return [1] * len(completions)


# ======================
# Training
# ======================

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="google/gemma-3-1b-it",
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,  # False for LoRA 16bit
    fast_inference=True,  # Enable vLLM fast inference
    max_lora_rank=LORA_RANK,
    gpu_memory_utilization=0.4,  # Reduce if out of memory
    use_vllm=True,
    vllm_enforce_eager=True
)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],  # Remove QKVO if out of memory
    lora_alpha=LORA_RANK,
    use_gradient_checkpointing="unsloth",  # Enable long context finetuning
    random_state=3407,
)



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
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,  # Increase to 4 for smoother training
    num_generations=5,  # Decrease if out of memory
    max_prompt_length=MAX_SEQ_LENGTH,
    max_completion_length=MAX_SEQ_LENGTH,
    num_train_epochs=1,  # Set to 1 for a full training run
    # max_steps=250,
    # save_steps=250,
    max_grad_norm=0.1,
    report_to="none",  # Can use Weights & Biases
    output_dir="outputs",
    gradient_checkpointing=True
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[reward_func],
    args=training_args,
    train_dataset=dataset,
)
