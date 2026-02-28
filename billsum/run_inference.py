"""
Dataset augmentation script for generating model predictions using vLLM.

This script generates text summaries for dataset examples at different readability levels
using vLLM for efficient batch inference. Supports both evaluation mode (single
sampling configuration) and rollout mode (multiple diverse sampling configurations).
Optionally computes metrics (CEFR, BERTScore, FKGL) for the generated predictions.

Usage:
    # Evaluation mode (single sampling config)
    python run_inference.py --input_file dataset/input.jsonl \
                            --output_file dataset/output.jsonl \
                            --model_name google/gemma-3-1b-it

    # Rollout mode (multiple sampling configs for RL training)
    python run_inference.py --input_file dataset/input.jsonl \
                            --output_file dataset/output.jsonl \
                            --model_name google/gemma-3-1b-it \
                            --rollouts

    # With metrics computation
    python run_inference.py --input_file dataset/input.jsonl \
                            --output_file dataset/output.jsonl \
                            --model_name google/gemma-3-1b-it \
                            --metrics

    # With metrics computation and save to JSON
    python run_inference.py --input_file dataset/input.jsonl \
                            --output_file dataset/output.jsonl \
                            --model_name google/gemma-3-1b-it \
                            --metrics \
                            --metrics_output metrics.json
"""

import json
import argparse
from vllm import SamplingParams
from inference import TextGenerator
from evaluator import Evaluator


def parse_args():
    """
    Parse command-line arguments for inference script.

    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Augment dataset with rollout generations using vLLM"
    )

    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to input JSONL dataset",
    )

    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to output JSONL file",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-3-1b-it",
        help="Model name to use for generation",
    )

    parser.add_argument(
        "--rollouts",
        action="store_true",
        default=False,
        help="Sampling params for rollouts other use the ones for evaluation",
    )

    parser.add_argument(
        "--metrics",
        action="store_true",
        default=False,
        help="Compute metrics (CEFR, BERTScore, FKGL) after generating predictions",
    )

    parser.add_argument(
        "--metrics_output",
        type=str,
        required=False,
        help="Path to save metrics summary as JSON file (only used with --metrics)",
    )

    return parser.parse_args()


def main():
    """
    Main function for running inference on a dataset.

    Loads dataset, initializes text generator with appropriate sampling parameters,
    generates predictions for all readability levels, optionally computes metrics,
    and saves the augmented dataset.
    """
    args = parse_args()

    # Load dataset
    with open(args.input_file, "r") as f:
        dataset = [json.loads(line.strip()) for line in f]

    # Create sampling params
    if args.rollouts:
        # Multiple diverse sampling configs for generating varied rollouts
        sampling_params_list = [
            SamplingParams(temperature=0.3, top_p=0.85, n=1, max_tokens=1024),
            SamplingParams(temperature=0.7, top_p=0.9, n=1, max_tokens=1024),
            SamplingParams(temperature=0.9, top_p=0.95, n=1, max_tokens=1024),
            SamplingParams(
                temperature=1.1, top_p=1.0, n=1, max_tokens=1024, presence_penalty=0.6
            ),
        ]
    else:
        # Single conservative sampling config for evaluation
        sampling_params_list = [
            SamplingParams(temperature=0.3, top_p=0.85, n=1, max_tokens=1024),
        ]

    # Initialize generator
    text_generator = TextGenerator(args.model_name, sampling_params_list)

    # Generate predictions for all readability levels
    text_generator.generate_dataset(dataset)

    # Compute metrics if requested
    if args.metrics:
        print("\nComputing metrics...")
        evaluator = Evaluator(verbose=True)
        evaluator.compute_metrics(dataset)

        # Save metrics summary to JSON if output path specified
        if args.metrics_output:
            metrics_summary = evaluator.get_metrics(dataset)
            with open(args.metrics_output, "w") as f:
                json.dump(metrics_summary, f, indent=2)
            print(f"Metrics summary saved to {args.metrics_output}")

    # Save output
    with open(args.output_file, "w") as fp:
        for instance in dataset:
            fp.write(json.dumps(instance) + "\n")

    print(f"\nOutput saved to {args.output_file}")


if __name__ == "__main__":
    main()
