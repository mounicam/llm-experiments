from jinja2 import Template

CEFR_LABELS = ["A1", "A2", "B1", "B2", "C1", "C2"]

SYSTEM_PROMPT = "You are helpful assistant designed to make English legal text more readable for different target audience at different CEFR readability levels."

PROMPT_TEMPLATE = Template(
    "Summarize the following text for a {{ level }} reader. {{ text }} \n Please output the summary as a paragraph."
)
