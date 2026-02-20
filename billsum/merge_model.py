import argparse
import transformers
from peft import PeftModel

def parse_args():
    parser = argparse.ArgumentParser(
        description="Augment dataset with rollout generations using vLLM"
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to peft dir",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to peft dir",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-3-1b-it",
        help="Model name to use for generation",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Load Model base model
    base_model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name)

    # Merge LoRA and base model and save
    peft_model = PeftModel.from_pretrained(base_model, args.input_dir)
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(args.output_dir)

    processor = transformers.AutoTokenizer.from_pretrained(args.model_name)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
