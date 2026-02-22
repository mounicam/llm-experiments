"""
Text generation module for efficient batch inference using vLLM.

This module provides the TextGenerator class for generating text summaries
at multiple CEFR levels using vLLM's optimized inference engine. It supports
prefix caching for efficient batch processing across different reading levels.
"""

from vllm import LLM
from prompts import generate_prompt
from transformers import AutoTokenizer
from prompts import CEFR_LABELS, SYSTEM_PROMPT, PROMPT_TEMPLATE


class TextGenerator:
    """
    Text generator for creating summaries at different CEFR levels using vLLM.

    This class handles efficient batch generation across multiple CEFR levels
    (A, B, C) by flattening prompts, running a single generation pass, and
    reshaping results back into the dataset structure.

    Attributes:
        sampling_params_list: List of SamplingParams for diverse generation
        tokenizer: HuggingFace tokenizer for prompt formatting
        generator: vLLM engine for efficient inference
    """

    def __init__(self, model_name, sampling_params):
        """
        Initialize the text generator with model and sampling parameters.

        Args:
            model_name (str): HuggingFace model ID or path to load
            sampling_params (list): List of vLLM SamplingParams for generation
        """
        self.sampling_params_list = sampling_params
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.generator = LLM(
            model=model_name,
            enable_prefix_caching=True,
            gpu_memory_utilization=0.8,
            tensor_parallel_size=1,
        )

    def _generate_prompts_with_cefr_labels(self, dataset, cefr_label):
        """
        Generate prompts for all dataset examples at a specific CEFR level.

        Args:
            dataset (list): List of dataset examples with 'text' field
            cefr_label (str): CEFR level (A, B, or C)

        Returns:
            list: List of formatted prompts ready for generation
        """
        input_prompts = []
        for item in dataset:
            final_msg = generate_prompt(self.tokenizer, item["text"], cefr_label)
            input_prompts.append(final_msg)
        return input_prompts

    def generate_dataset(self, dataset):
        """
        Generate predictions for all CEFR levels and augment dataset in-place.

        This method:
        1. Pre-computes all prompts for all CEFR levels (A, B, C)
        2. Runs a single batched generation call for efficiency
        3. Reshapes results back into the dataset structure

        The dataset is modified in-place, adding a 'predictions' field with
        structure: predictions[cefr_label][idx] = {"generation": text}

        Args:
            dataset (list): List of dataset examples to augment with predictions
        """

        # 1. PRE-COMPUTE ALL PROMPTS
        # We create a massive list: [Prompt1_A, Prompt1_B... PromptN_C]
        all_flattened_prompts = []
        for cefr_label in CEFR_LABELS:
            prompts = self._generate_prompts_with_cefr_labels(dataset, cefr_label)
            all_flattened_prompts.extend(prompts)
        print(prompts[0])

        # 2. RUN ONE SINGLE GENERATION CALL
        # This is where the magic happens. vLLM will parallelize across all labels automatically.
        # We do this for each of your SamplingParams settings.
        all_generations = [[] for _ in range(len(all_flattened_prompts))]

        for params in self.sampling_params_list:
            outputs = self.generator.generate(all_flattened_prompts, params)
            for idx, output in enumerate(outputs):
                all_generations[idx].extend([o.text for o in output.outputs])
        print(all_generations[0])

        # 3. RE-SHAPE BACK INTO THE DATASET
        # Since we flattened [Labels][Dataset], we unflatten carefully
        num_items = len(dataset)
        for label_idx, cefr_label in enumerate(CEFR_LABELS):
            # Calculate where this label's results start in the big flat list
            offset = label_idx * num_items
            for i in range(num_items):
                if "predictions" not in dataset[i]:
                    dataset[i]["predictions"] = {}
                dataset[i]["predictions"][cefr_label] = [
                    {"generation": gen} for gen in all_generations[offset + i]
                ]
