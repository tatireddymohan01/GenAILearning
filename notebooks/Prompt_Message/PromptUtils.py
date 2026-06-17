"""
PromptUtils.py — Shared utilities for all Prompt & Message notebooks.

All notebooks use the same LLM and test inputs so you can directly compare
how each prompting technique affects output quality and structure.
"""

from dotenv import load_dotenv
import os

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

from langchain_openai import ChatOpenAI

# ─── Constants ────────────────────────────────────────────────────────────────

LLM_MODEL   = 'gpt-4o-mini'
TEMPERATURE = 0.3

# Same inputs used in every notebook — compare results across techniques
TEST_INPUTS = [
    'Explain what an API is.',
    'What is the difference between a list and a tuple in Python?',
    'How does a neural network learn?',
]

# ─── Model ────────────────────────────────────────────────────────────────────

def get_llm(temperature=TEMPERATURE):
    """Return a ChatOpenAI instance."""
    return ChatOpenAI(model=LLM_MODEL, temperature=temperature)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def run_inputs(fn, inputs=None, label='Prompt Test'):
    """Run test inputs through any prompt function and print results."""
    if inputs is None:
        inputs = TEST_INPUTS
    sep = '=' * 60
    print(f'\n{sep}')
    print(f' {label} — Test Results')
    print(sep)
    for inp in inputs:
        print(f'\nInput: {inp}')
        result = fn(inp)
        content = result.content if hasattr(result, 'content') else str(result)
        print(f'Output: {content}')
        print('-' * 40)
