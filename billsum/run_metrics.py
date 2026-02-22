"""
Metrics computation script for augmenting predictions with reward scores.

This script computes metrics (CEFR classification scores and BERTScore) for
generated summaries and adds them to the predictions file. These metrics
are used as rewards for reinforcement learning training (DPO/GRPO).

Usage:
    # Compute and save metrics to output file
    python run_metrics.py --input_file dataset/predictions.jsonl \
                          --output_file dataset/predictions_with_metrics.jsonl

    # Print metrics only (no output file)
    python run_metrics.py --input_file dataset/predictions.jsonl
"""

import json
import argparse
from evaluator import Evaluator


def parse_args():
    """
    Parse command-line arguments for metrics computation.

    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Augment predictions with metrics that are used as rewards for RL."
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
        required=False,
        help="Path to output JSONL file",
    )

    return parser.parse_args()


def main():
    """
    Main function for computing and saving metrics.

    Loads predictions dataset, computes CEFR and BERT metrics, and either
    saves the augmented dataset or prints aggregate metrics.
    """
    args = parse_args()

    with open(args.input_file, "r") as f:
        dataset = [json.loads(line.strip()) for line in f]

    evaluator = Evaluator(verbose=True)

    if args.output_file is not None:
        # Compute metrics and save to output file
        evaluator.compute_metrics(dataset)
        with open(args.output_file, "w") as fp:
            for instance in dataset:
                fp.write(json.dumps(instance) + "\n")
    else:
        # Just print aggregate metrics without saving
        evaluator.print_metrics(dataset)


if __name__ == "__main__":
    main()
