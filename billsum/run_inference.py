# Augment dataset with rollouts


import json
from vllm import SamplingParams
from inference import TextGenerator

# Augment dataset with rollouts

import json
import argparse
from vllm import SamplingParams
from inference import TextGenerator


def parse_args():
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

    return parser.parse_args()


def main():
    args = parse_args()

    # Load dataset
    with open(args.input_file, "r") as f:
        dataset = [json.loads(line.strip()) for line in f]

    # Create sampling params
    if args.rollouts:
        sampling_params_list = [
            SamplingParams(temperature=0.3, top_p=0.85, n=1, max_tokens=1024),
            SamplingParams(temperature=0.7, top_p=0.9, n=1, max_tokens=1024),
            SamplingParams(temperature=0.9, top_p=0.95, n=1, max_tokens=1024),
            SamplingParams(temperature=1.1, top_p=1.0, n=1, max_tokens=1024, presence_penalty=0.6)
        ]
    else:
        sampling_params_list = [
            SamplingParams(temperature=0.3, top_p=0.85, n=1, max_tokens=1024),
        ]

    # Initialize generator
    text_generator = TextGenerator(args.model_name, sampling_params_list)

    # Generate
    text_generator.generate_dataset(dataset)

    # Save output
    with open(args.output_file, "w") as fp:
        for instance in dataset:
            fp.write(json.dumps(instance) + "\n")


if __name__ == "__main__":
    main()


# input_file = "dataset/org_with_cefr_labels/RL_dev_xaa.jsonl"
# output_file = "dataset/predictions/gemma/oob/RL_dev_xaa.jsonl"
# model_name = "google/gemma-3-1b-it"
