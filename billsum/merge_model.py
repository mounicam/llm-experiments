"""
Utility script for merging LoRA adapters with base models.

This script loads a PEFT (Parameter-Efficient Fine-Tuning) model with LoRA adapters
and merges them into the base model, creating a standalone model that can be used
without the PEFT library.
"""
import argparse
import transformers
from peft import PeftModel


def parse_args():
    """
    Parse command line arguments for model merging.

    Returns:
        argparse.Namespace: Parsed arguments containing input_dir, output_dir, and model_name
    """
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapters with base model and save the merged model"
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to directory containing PEFT/LoRA adapter weights",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to directory where merged model will be saved",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-3-1b-it",
        help="HuggingFace model name or path to base model",
    )

    return parser.parse_args()


def main():
    """
    Main function to merge LoRA adapters with base model.

    Loads the base model, applies PEFT/LoRA adapters from input_dir,
    merges them into a single model, and saves both the merged model
    and tokenizer to output_dir.
    """
    args = parse_args()

    # Load base model
    base_model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name)

    # Merge LoRA adapters with base model and save
    peft_model = PeftModel.from_pretrained(base_model, args.input_dir)
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(args.output_dir)

    # Save tokenizer alongside merged model
    processor = transformers.AutoTokenizer.from_pretrained(args.model_name)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
