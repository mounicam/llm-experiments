"""
Prompt templates for target readability text simplification.

This module provides utilities for generating prompts that instruct language models
to summarize legal text at different readability levels.
"""

import re

READABILTIY_LABELS = ["beginner", "intermediate", "advanced"]
CEFR_LABELS = {"beginner": "A", "intermediate": "B", "advanced": "C"}

SYSTEM_PROMPT = """You are a Legislative Analysis Engine. Your task is to transform complex US Congressional bills into clear summaries.

Operational Rules:
No Conversational Filler: Do not use phrases like 'Sure,' 'I can help,' or 'Here is the summary.'
Structure: Use a 'Subject-Verb-Object' structure. Focus on active legislative verbs: 'Establishes,' 'Amends,' 'Directs,' 'Authorizes.'
**Formatting: Start your response immediately with the <summary> tag and end it with the </summary> tag. Please output a single paragraph. **
Readability Target: You will be provided with a target level (beginner, intermediate, or advanced). Adjust vocabulary and sentence complexity strictly to match that level.
"""

# SYSTEM_PROMPT = """You are a Legislative Analysis Engine. Your task is to transform complex US Congressional bills into clear summaries.

# Operational Rules:
# No Conversational Filler: Do not use phrases like 'Sure,' 'I can help,' or 'Here is the summary.'
# Structure: Use a 'Subject-Verb-Object' structure. Focus on active legislative verbs: 'Establishes,' 'Amends,' 'Directs,' 'Authorizes.'
# Formatting: Start your response immediately with the <summary> tag and end it with the </summary> tag.
# Readability Target: You will be provided with a target level (beginner, intermediate, or advanced). Adjust vocabulary and sentence complexity strictly to match that level.
# """


def parse_summary(text):
    """
    Extract content between <summary> and </summary> tags.

    Args:
        text (str): Generated text that may contain summary tags

    Returns:
        str: Content between summary tags, or original text if tags not found
    """
    match = re.search(r"<summary>(.*?)</summary>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def generate_input_content(level, text):
    """Formats the bill text for the prompt."""
    return f"{SYSTEM_PROMPT}\n\nTarget Level: {level}\nBill Text: {text}"


def generate_prompt(tokenizer, text, level):
    """Generate a formatted chat prompt for text simplification at a specific readability level."""
    msg = [
        {
            "role": "user",
            "content": generate_input_content(level, text),
        }
    ]
    final_msg = tokenizer.apply_chat_template(
        msg, tokenize=False, add_generation_prompt=True
    )
    return final_msg
