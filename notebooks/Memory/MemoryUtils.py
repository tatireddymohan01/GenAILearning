"""
MemoryUtils.py — Shared utilities for all Memory notebooks (LangChain 1.x).

Modern API used:
  - ChatMessageHistory           (langchain_community.chat_message_histories)
  - RunnableWithMessageHistory   (langchain_core.runnables.history)
  - trim_messages                (langchain_core.messages)
"""

from dotenv import load_dotenv
import os

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', '')

from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

# ─── Constants ────────────────────────────────────────────────────────────────

LLM_MODEL   = 'gpt-4o-mini'
TEMPERATURE = 0.0

# Same 5-turn conversation used in every notebook
TEST_TURNS = [
    ('My name is Mohan and I am a Python developer.',
     'Nice to meet you, Mohan! What kind of projects do you work on?'),
    ('I am building a document Q&A chatbot for my company Acme Corp.',
     'That sounds great! Are you using a specific framework?'),
    ('Yes, I am using LangChain with FAISS for vector storage.',
     'Excellent! FAISS is fast and works well for local deployments.'),
    ('My company is based in Hyderabad and has about 500 employees.',
     'Good to know! Does the chatbot need to handle multiple languages?'),
    ('No, English only. We use Azure OpenAI as our LLM provider.',
     'Smart choice. Azure gives enterprise compliance out of the box.'),
]

# Queries that test memory recall across all 5 turns
TEST_QUERIES = [
    'What is my name?',               # Turn 1
    'What am I building?',            # Turn 2
    'What vector store am I using?',  # Turn 3
    'Where is my company located?',   # Turn 4
    'What LLM provider do I use?',    # Turn 5
]

# ─── Model ────────────────────────────────────────────────────────────────────

def get_llm(temperature=TEMPERATURE):
    return ChatOpenAI(model=LLM_MODEL, temperature=temperature)


# ─── History helpers ──────────────────────────────────────────────────────────

def make_history() -> ChatMessageHistory:
    """Return a new empty ChatMessageHistory."""
    return ChatMessageHistory()


def load_test_history(history: ChatMessageHistory):
    """Load TEST_TURNS into a ChatMessageHistory object."""
    for human, ai in TEST_TURNS:
        history.add_user_message(human)
        history.add_ai_message(ai)
    print(f'Loaded {len(TEST_TURNS)} turns ({len(history.messages)} messages).')


# ─── Test runner ──────────────────────────────────────────────────────────────

def run_recall_test(chat_fn, label='Memory Test'):
    """Run TEST_QUERIES through chat_fn and print how well memory recalls each fact."""
    sep = '=' * 60
    print()
    print(sep)
    print(f' {label} -- Recall Test')
    print(sep)
    for query in TEST_QUERIES:
        answer = chat_fn(query)
        print()
        print(f'Q: {query}')
        print(f'A: {answer}')
        print('-' * 40)
