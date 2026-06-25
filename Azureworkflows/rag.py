"""
RAG over Azure AI Search + OpenAI GPT-4o-mini.

Pipeline:
    Question → Azure AI Search → top-N content chunks → GPT-4o-mini → Answer

Usage:
    python rag.py

Requirements:
    pip install azure-search-documents langchain-openai python-dotenv

Environment variables (or .env file):
    OPENAI_API_KEY
    AZURE_SEARCH_ENDPOINT    — e.g. https://mohan-ai-search.search.windows.net
    AZURE_SEARCH_KEY         — Admin or Query key from Azure Portal
    AZURE_SEARCH_INDEX_NAME  — e.g. search-1782371635608
"""

import os
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

SEARCH_ENDPOINT  = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_KEY       = os.getenv("AZURE_SEARCH_KEY")
SEARCH_INDEX     = os.getenv("AZURE_SEARCH_INDEX_NAME")
TOP_K            = 3   # number of search results to use as context


def search(query: str) -> list[dict]:
    client = SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=SEARCH_INDEX,
        credential=AzureKeyCredential(SEARCH_KEY),
    )
    results = client.search(search_text=query, top=TOP_K, select=["content", "title"])
    return [{"title": r["title"], "content": r["content"]} for r in results]


def build_context(results: list[dict]) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"[Document {i}: {r['title']}]\n{r['content']}")
    return "\n\n---\n\n".join(parts)


def ask(question: str) -> str:
    results = search(question)
    if not results:
        return "No relevant documents found in the search index."

    context = build_context(results)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant. Answer ONLY from the provided context. "
         "If the answer is not in the context, say 'I don't know'."),
        ("human", "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"),
    ])

    llm = ChatOpenAI(model="gpt-4o-mini")
    response = llm.invoke(prompt.format_messages(context=context, question=question))
    return response.content


if __name__ == "__main__":
    question = input("Ask a question: ").strip()
    if not question:
        print("No question entered.")
    else:
        print("\nSearching...\n")
        answer = ask(question)
        print(f"Answer:\n{answer}")
