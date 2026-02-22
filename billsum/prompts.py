"""
Prompt templates for CEFR-based text simplification.

This module provides utilities for generating prompts that instruct language models
to summarize legal text at different CEFR (Common European Framework of Reference)
readability levels.
"""
from jinja2 import Template

# CEFR proficiency levels: A (basic), B (independent), C (proficient)
CEFR_LABELS = ["A", "B", "C"]

SYSTEM_PROMPT = "You are helpful assistant designed to make English legal text more readable for different target audience at different CEFR readability levels."

PROMPT_TEMPLATE = Template(
    "Summarize the following text for a {{ level }} reader. {{ text }} \n Please output the summary as a paragraph."
)


def generate_prompt(tokenizer, text, cefr_label):
    """
    Generate a formatted chat prompt for text simplification at a specific CEFR level.

    Args:
        tokenizer: Tokenizer with chat template support (e.g., from transformers)
        text (str): The legal text to be summarized
        cefr_label (str): CEFR level label (e.g., "A", "B", or "C")

    Returns:
        str: Formatted prompt string with chat template applied, ready for model input

    Example:
        >>> tokenizer = AutoTokenizer.from_pretrained("model_name")
        >>> prompt = generate_prompt(tokenizer, "Legal text here...", "A")
    """
    # Expand CEFR label to include sublevels (e.g., "A" -> "A1/A2")
    cefr_label = f"{cefr_label}1/{cefr_label}2"
    msg = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": PROMPT_TEMPLATE.render(text=text, level=cefr_label),
        },
    ]
    final_msg = tokenizer.apply_chat_template(
        msg, tokenize=False, add_generation_prompt=True
    )
    return final_msg
