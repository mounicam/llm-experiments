from vllm import LLM
from transformers import AutoTokenizer
from prompts import CEFR_LABELS, SYSTEM_PROMPT, PROMPT_TEMPLATE


class TextGenerator:

    def __init__(self, model_name, sampling_params):
        self.sampling_params_list = sampling_params
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.generator = LLM(
            model=model_name,
            enable_prefix_caching=True,
            gpu_memory_utilization=0.8,
            tensor_parallel_size=1,
        )

    def _generate_prompts_with_cefr_labels(self, dataset, cefr_label):
        input_prompts = []
        for item in dataset:
            final_msg = generate_prompt(self.tokenizer, text, cefr_label)
            input_prompts.append(final_msg)
        return input_prompts

    def generate_dataset(self, dataset):

        # 1. PRE-COMPUTE ALL PROMPTS
        # We create a massive list: [Prompt1_A, Prompt1_B... PromptN_C]
        all_flattened_prompts = []
        for cefr_label in CEFR_LABELS:
            prompts = self._generate_prompts_with_cefr_labels(dataset, cefr_label)
            all_flattened_prompts.extend(prompts)

        # 2. RUN ONE SINGLE GENERATION CALL
        # This is where the magic happens. vLLM will parallelize across all labels automatically.
        # We do this for each of your SamplingParams settings.
        all_generations = [[] for _ in range(len(all_flattened_prompts))]

        for params in self.sampling_params_list:
            outputs = self.generator.generate(all_flattened_prompts, params)
            for idx, output in enumerate(outputs):
                all_generations[idx].extend([o.text for o in output.outputs])

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
