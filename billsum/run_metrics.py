# Compute metrics and add them to the predictions file

import json
import argparse
from evaluator import Evaluator


def parse_args():
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
    args = parse_args()

    with open(args.input_file, "r") as f:
        dataset = [json.loads(line.strip()) for line in f]

    evaluator = Evaluator(verbose=True)

    if args.output_file is not None:
        evaluator.compute_metrics(dataset)
        with open(args.output_file, "w") as fp:
            for instance in dataset:
                fp.write(json.dumps(instance) + "\n")
    else:
        evaluator.print_metrics(dataset)


if __name__ == "__main__":
    main()
