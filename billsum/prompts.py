from jinja2 import Template

CEFR_LABELS = ["A1", "A2", "B1", "B2", "C1", "C2"]

SYSTEM_PROMPT = "You are helpful assistant designed to make English legal text more readable for different target audience at different CEFR readability levels."

PROMPT_TEMPLATE = Template(
    "Summarize the following text for a {{ level }} reader. {{ text }} \n Please output the summary as a paragraph."
)


def generate_prompt(tokenizer, text, cefr_label):
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
