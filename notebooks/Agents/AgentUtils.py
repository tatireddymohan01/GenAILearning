"""
AgentUtils.py — Shared utilities for all Agent notebooks.

All notebooks use the same 3 tools and test queries so you can directly compare
how each agent type approaches the same problems differently.
"""

from dotenv import load_dotenv
import os

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# ─── Constants ────────────────────────────────────────────────────────────────

LLM_MODEL = 'gpt-4o-mini'

# Same queries used in every notebook — compare agent behaviour across types
TEST_QUERIES = [
    'What is 144 divided by 12?',
    'Search docs for: what is a LangChain agent?',
    'What is the weather in Hyderabad?',
    'Search docs for RAG and also calculate 25 multiplied by 4',
]

# ─── Model ────────────────────────────────────────────────────────────────────

def get_llm(temperature=0.0):
    return ChatOpenAI(model=LLM_MODEL, temperature=temperature)


# ─── Shared Tools ─────────────────────────────────────────────────────────────

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. Input must be a valid Python math expression e.g. '144 / 12'."""
    try:
        result = eval(expression, {'__builtins__': {}}, {})
        return str(result)
    except Exception as e:
        return f'Error: {e}'


@tool
def search_docs(query: str) -> str:
    """Search the company documentation for information about LangChain, agents, RAG, memory, tools, and AI topics."""
    knowledge = {
        'agent':        'A LangChain agent uses an LLM to decide which actions to take and in what order. It can call tools, observe results, and reason across multiple steps.',
        'tool':         'Tools are functions agents can call to interact with external systems — e.g. search, calculator, APIs. The @tool decorator turns any Python function into a LangChain tool.',
        'rag':          'RAG (Retrieval-Augmented Generation) combines a retrieval system with an LLM. The retriever fetches relevant documents; the LLM uses them to answer grounded in real data.',
        'memory':       'LangChain memory allows agents and chains to remember past messages. MessagesPlaceholder injects history into the prompt on every turn.',
        'langchain':    'LangChain is a framework for building applications powered by LLMs. It provides abstractions for chains, agents, tools, memory, and RAG pipelines.',
        'langsmith':    'LangSmith is a platform for monitoring, debugging, and deploying LLM applications. It traces every LLM call so you can inspect inputs, outputs, and latency.',
        'chain':        'A LangChain chain is a sequence of components connected with the LCEL pipe operator (|). Example: prompt | llm | output_parser.',
        'embedding':    'Embeddings convert text into dense numerical vectors. Similar texts have similar vectors, enabling semantic search in vector databases like FAISS.',
        'vectorstore':  'A vector store (e.g. FAISS, Chroma) stores embeddings and supports similarity search — the foundation of RAG retrieval.',
        'prompt':       'A prompt template defines the structure of an LLM input with {variables} filled at runtime. ChatPromptTemplate supports typed messages (system/human/ai).',
        'react':        'ReAct (Reason + Act) is an agent pattern where the LLM writes a Thought before each Action and reads an Observation after. This makes reasoning transparent and debuggable.',
        'plan':         'Plan-and-Execute agents generate a full multi-step plan first, then execute each step. Better than ReAct for complex tasks that require upfront reasoning.',
    }
    query_lower = query.lower()
    results = [v for k, v in knowledge.items() if k in query_lower]
    if results:
        return ' '.join(results)
    return f'No documentation found for: "{query}". Try keywords like: agent, tool, rag, memory, langchain, react, plan.'


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    weather = {
        'hyderabad':  '34°C, Sunny',
        'mumbai':     '29°C, Humid',
        'delhi':      '27°C, Partly Cloudy',
        'bangalore':  '22°C, Pleasant',
        'chennai':    '31°C, Hot',
        'seattle':    '15°C, Rainy',
        'new york':   '18°C, Cloudy',
        'london':     '12°C, Foggy',
    }
    return weather.get(city.lower(), f'Weather data not available for {city}.')


TOOLS = [calculate, search_docs, get_weather]


# ─── Test Runner ──────────────────────────────────────────────────────────────

def run_queries(fn, queries=None, label='Agent Test'):
    """Run test queries through any agent function and print results."""
    if queries is None:
        queries = TEST_QUERIES
    sep = '=' * 60
    print(f'\n{sep}')
    print(f' {label} — Test Results')
    print(sep)
    for q in queries:
        print(f'\nQ: {q}')
        result = fn(q)
        print(f'A: {result}')
        print('-' * 40)
