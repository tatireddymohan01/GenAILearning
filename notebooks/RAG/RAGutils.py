"""
RAGutils.py — Shared utilities for all RAG notebooks.

All notebooks use the same data source, models, and test queries
so you can directly compare how each RAG technique behaves differently.
"""

from dotenv import load_dotenv
import os

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

# ─── Constants ────────────────────────────────────────────────────────────────

LANGSMITH_URL = "https://docs.langchain.com/langsmith/agent-server"
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200
EMBED_MODEL   = "text-embedding-3-small"
LLM_MODEL     = "gpt-4o-mini"

# Same queries used in every notebook — compare results across RAG techniques
TEST_QUERIES = [
    "What is LangSmith persistence?",
    "What are checkpoints in LangSmith?",
    "Explain LangSmith deployment",
]

# ─── Loaders ──────────────────────────────────────────────────────────────────

def load_docs(urls=None):
    """Load documents from one or more URLs. Defaults to LangSmith docs."""
    if urls is None:
        urls = [LANGSMITH_URL]
    if isinstance(urls, str):
        urls = [urls]
    all_docs = []
    for url in urls:
        loader = WebBaseLoader(url)
        all_docs.extend(loader.load())
    return all_docs


def split_docs(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)


# ─── Models & Stores ──────────────────────────────────────────────────────────

def get_embeddings():
    """Return OpenAI embeddings model."""
    return OpenAIEmbeddings(model=EMBED_MODEL)


def create_vectorstore(chunks, embeddings=None):
    """Create a FAISS vector store from document chunks."""
    if embeddings is None:
        embeddings = get_embeddings()
    return FAISS.from_documents(chunks, embeddings)


def get_llm():
    """Return ChatOpenAI LLM."""
    return ChatOpenAI(model=LLM_MODEL)


def get_prompt():
    """Standard RAG prompt used across all notebooks."""
    return ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful AI assistant.\n"
         "Answer ONLY from the provided context.\n"
         "If the answer is not in the context, say 'I don't know'.\n"
         "Be concise and clear."),
        ("human",
         "Context:\n{context}\n\n"
         "Question:\n{question}\n\n"
         "Answer:")
    ])


# ─── Helpers ──────────────────────────────────────────────────────────────────

def format_docs(docs):
    """Convert a list of Document objects into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def run_queries(fn, queries=None, label="RAG"):
    """Run a list of test queries through any RAG function and print results."""
    if queries is None:
        queries = TEST_QUERIES
    print(f"\n{'='*60}")
    print(f" {label} — Test Results")
    print(f"{'='*60}")
    for q in queries:
        print(f"\nQ: {q}")
        answer = fn(q)
        print(f"A: {answer}")
        print("-" * 40)


# ─── One-liner Setup ──────────────────────────────────────────────────────────

def setup():
    """
    Full RAG setup in one call. Returns everything needed for RAG notebooks.

    Returns:
        chunks      - split document chunks (needed for BM25, parent retriever, etc.)
        vectorstore - FAISS vector store (ready for similarity search)
        embeddings  - OpenAI embeddings model (needed for HyDE)
        llm         - ChatOpenAI LLM
        prompt      - standard RAG prompt template
    """
    print("Loading documents...")
    docs = load_docs()

    print("Splitting into chunks...")
    chunks = split_docs(docs)

    print("Creating embeddings and vector store...")
    embeddings = get_embeddings()
    vectorstore = create_vectorstore(chunks, embeddings)

    llm     = get_llm()
    prompt  = get_prompt()

    print(f"\nReady: {len(docs)} page(s) -> {len(chunks)} chunks -> vector store created")
    print(f"LLM: {LLM_MODEL} | Embeddings: {EMBED_MODEL}")

    return chunks, vectorstore, embeddings, llm, prompt
