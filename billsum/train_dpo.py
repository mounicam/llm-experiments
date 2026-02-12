# dpo_train.py
import os
import json
import argparse
import torch
from prompts import generate_prompt
from trl import DPOTrainer, DPOConfig
from unsloth import FastLanguageModel
from datasets import Dataset, DatasetDict
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


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


def compute_reward(pred):
    return (pred["cefr_prob"] + 0.5 * pred["bert_score"]) * 100.0 / 1.5


def load_dataset(file_path, tokenizer):
    dataset = [json.loads(line.strip()) for line in open(file_path)][:5000]

    dpo_dataset = {"prompt": [], "chosen": [], "rejected": []}
    for item in dataset:
        for cefr_label, predictions in item["predictions"].items():
            prompt = generate_prompt(tokenizer, item["text"], cefr_label)

            for i in range(0, len(predictions)):
                for j in range(i + 1, len(predictions)):
                    r_i = compute_reward(predictions[i])
                    r_j = compute_reward(predictions[j])

                    accept, reject = None, None
                    if r_i > r_j and (r_i - r_j) > 1.5:
                        accept = predictions[i]["generation"]
                        reject = predictions[j]["generation"]
                    elif r_j > r_i and (r_j - r_i) > 1.5:
                        accept = predictions[j]["generation"]
                        reject = predictions[i]["generation"]

                    if accept and reject:
                        dpo_dataset["prompt"].append(prompt)
                        dpo_dataset["chosen"].append(accept)
                        dpo_dataset["rejected"].append(reject)

    return Dataset.from_dict(dpo_dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="google/gemma-3-1b-it")
    parser.add_argument("--train_dataset", type=str)
    parser.add_argument("--eval_dataset", type=str)
    parser.add_argument("--output_dir", type=str, default="./dpo_out")
    args = parser.parse_args()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    # For many causal LMs, pad_token may be missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # bnb4 = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    #     bnb_4bit_quant_storage=torch.bfloat16,
    # )

    # # Policy model (trainable)
    # policy_base = AutoModelForCausalLM.from_pretrained(
    #     args.model_id,
    #     quantization_config=bnb4,
    #     device_map="auto",
    #     attn_implementation="eager",
    # )
    # policy_base.config.use_cache = False
    # policy_base.gradient_checkpointing_enable()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_id,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
    )

    # LoRA adapters on policy
    # peft_cfg = LoraConfig(
    #     r=16,
    #     lora_alpha=32,
    #     lora_dropout=0.05,
    #     bias="none",
    #     task_type="CAUSAL_LM",
    #     target_modules=[
    #         "q_proj",
    #         "k_proj",
    #         "v_proj",
    #         "o_proj",
    #     ],  # common set; adjust if needed
    # )
    # policy_model = get_peft_model(policy_base, peft_cfg)

    # # Reference model (frozen). In DPO, ref is usually the same base model before training.
    # ref_model = AutoModelForCausalLM.from_pretrained(
    #     args.model_id,
    #     quantization_config=bnb4,  # could also do 8-bit to be safer numerically
    #     device_map="auto",
    #     attn_implementation="eager",
    # )
    # ref_model.config.use_cache = False
    # ref_model.eval()
    # for p in ref_model.parameters():
    #     p.requires_grad_(False)

    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=32,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        max_seq_length=4096,
    )

    # Dataset
    train_dataset = load_dataset(args.train_dataset, tokenizer)
    # eval_dataset = load_dataset(args.eval_dataset, tokenizer)
    print(train_dataset[0]["prompt"])
    print(train_dataset[0]["chosen"])
    print(train_dataset[0]["rejected"])

    train_ds = train_dataset.map(format_example)
    # eval_ds = eval_dataset.map(format_example)

    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        learning_rate=5e-6,
        num_train_epochs=1,
        logging_steps=10,
        save_strategy="epoch",
        optim="adamw_8bit",
        warmup_ratio=0.1,
        # eval_strategy="epoch",
        save_total_limit=1,
        bf16=True,
        seed=42,
        gradient_checkpointing=False,
        # --- DPO specific ---
        beta=0.1,  # KL temperature
        max_length=4096,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    trainer = DPOTrainer(
        model=model,
        # ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_ds,
        # eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
