from peft import PeftModel
import transformers

model_id = "google/gemma-3-1b-it"

# Load Model base model
base_model = transformers.AutoModelForCausalLM.from_pretrained(model_id)

# Merge LoRA and base model and save
peft_model = PeftModel.from_pretrained(
    base_model, "models/gemma_sft/v3/checkpoint-2391"
)
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained("models/gemma_sft/v3/merged")

processor = transformers.AutoTokenizer.from_pretrained(model_id)
processor.save_pretrained("models/gemma_sft/v3/merged")
