"""
Prompt templates for target readability text simplification.

This module provides utilities for generating prompts that instruct language models
to summarize legal text at different readability levels.
"""

from jinja2 import Template

READABILTIY_LABELS = ["beginner", "intermediate", "advanced"]
CEFR_LABELS = {"beginner": "A", "intermediate": "B", "advanced": "C"}

SYSTEM_PROMPT = """You are a Legislative Analysis Engine. Your task is to transform complex US Congressional bills into clear summaries.

Operational Rules:
No Conversational Filler: Do not use phrases like 'Sure,' 'I can help,' or 'Here is the summary.'
Structure: Use a 'Subject-Verb-Object' structure. Focus on active legislative verbs: 'Establishes,' 'Amends,' 'Directs,' 'Authorizes.'
Formatting: Start your response immediately with the <summary> tag and end it with the </summary> tag.
Readability Target: You will be provided with a target level (beginner, intermediate, or advanced). Adjust vocabulary and sentence complexity strictly to match that level.
"""


def generate_input_content(level, text):
    """Formats the bill text for the prompt."""
    return f"{SYSTEM_PROMPT}\n\nTarget Level: {level}\nBill Text: {text}"


def generate_prompt(tokenizer, text, level):
    """Generate a formatted chat prompt for text simplification at a specific readability level."""
    msg = [
        {
            "role": "user",
            "content": generate_input_content(level, text),
        },
        {"role": "assistant", "content": "<summary>"},
    ]
    final_msg = tokenizer.apply_chat_template(
        msg, tokenize=False, add_generation_prompt=True
    )
    return final_msg
