import torch
import pandas as pd
from datasets import Dataset
from tqdm import tqdm
from jinja2 import Template
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

model_id = "models/readme/gemma/lora/merged"
llm = LLM(model=model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)


# SYSTEM_PROMPT = '''You are a language learning evaluator assessing the complexity of an English sentence given its context.

# Rubric:
# 1 (A1) – Very basic words and phrases; simple self-introduction; minimal grammar.
# 2 (A2) – Simple sentences; familiar everyday expressions; limited range.
# 3 (B1) – Can write or speak in connected sentences about familiar topics; some errors.
# 4 (B2) – Generally fluent; can discuss abstract topics; good grammar control.
# 5 (C1) – Flexible, natural use of language; few errors; advanced vocabulary.
# 6 (C2) – Near-native mastery; precise, nuanced expression; fully natural flow.

# Please give a rating between 1-6 following the rubric above.
# '''

PROMPT_TEMPLATE = """
You are a language learning evaluator assessing the complexity of an English sentence given its context. Please give a rating between 1 to 6. Do not output any other text.
Context: {{ context }}
Sentence: {{ sentence }}
Rating (1-6):
"""

JINJA_PROMPT_TEMPLATE = Template(PROMPT_TEMPLATE)

def preprocess(example):
    messages = [
        # {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": JINJA_PROMPT_TEMPLATE.render(
            context=example['Paragraph'], 
            sentence=example['Sentence']
        )},
        {"role": "assistant", "content": f"The score is {example['Rating']}"}
    ]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=False)
    return {"messages": messages}


def build_prompt(messages):
    return tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
    )


def extract_completion(s: str) -> str:
    if "The score is" in s:
        return s.split("The score is", 1)[1].strip()
    return s.strip()


test_df = pd.read_csv("readme/readme_en_test.csv")
test_df = test_df.dropna(subset=["Rating", "Sentence", "Paragraph"])
test_ds = Dataset.from_pandas(test_df)
test_ds = test_ds.map(preprocess)

prompts = []
golds = []
for example in test_ds:
    prompts.append(build_prompt(example["messages"]))
    golds.append(extract_completion(example["messages"][-1]["content"]))
    
outputs = llm.generate(prompts[0])
print(outputs)

# Choose a batch size based on GPU RAM
BATCH_SIZE = 8
preds = []
for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="Evaluating"):
    batch_prompts = prompts[i:i+BATCH_SIZE]

    outputs = llm.generate(batch_prompts)

    for prompt, out in zip(batch_prompts, outputs):
        gen = out.outputs[0].text
        pred = extract_completion(gen.strip())
        preds.append(pred)

# Compute accuracy
print(preds, golds)
acc = sum(int(p) == int(float(g)) for p, g in zip(preds, golds))
print("Accuracy:", acc * 100.0 / len(golds))