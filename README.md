# GenAI Learning

A structured learning repository for Generative AI concepts using LangChain and OpenAI.
Each topic is organized into numbered notebooks — concept by concept — so you can follow the progression clearly.

## Topics Covered

| Folder | Concepts |
|--------|----------|
| `notebooks/RAG/` | 12 RAG techniques from Simple RAG → Multi-hop RAG |
| `notebooks/Prompt_Message/` | 10 prompting techniques from PromptTemplate → Optimization |
| `notebooks/Agents/` | 9-step agent journey: tools → ReAct → tool-calling → memory → RAG → multi-agent → self-correction |
| `notebooks/Memory/` | Buffer memory with session history |

---

## Project Structure

```
GenAI_Leaning/
├── README.md
├── requirement.txt
├── .env                              # NOT in repo — add your API keys here
├── app.py
├── retrievers_examples.py
│
└── notebooks/
    ├── RAG/
    │   ├── RAG_Concepts.md           # Full concept guide (read this first)
    │   ├── RAGutils.py               # Shared utilities for all RAG notebooks
    │   ├── 1simpleRAG.ipynb
    │   ├── 2simpleRAG_Retriver.ipynb
    │   ├── 3AdvancedRAG.ipynb
    │   ├── 4HybridRAG_AgeneticRAG.ipynb
    │   ├── 5RAGFusion.ipynb
    │   ├── 6HyDE.ipynb
    │   ├── 7ParentDocumentRetriever.ipynb
    │   ├── 8ContextualCompression.ipynb
    │   ├── 9CRAG.ipynb
    │   ├── 10SelfRAG.ipynb
    │   └── 11MultiHopRAG.ipynb
    │
    ├── Prompt_Message/
    │   ├── Prompt_Message_Concepts.md  # Full concept guide (read this first)
    │   ├── PromptUtils.py              # Shared utilities for all Prompt notebooks
    │   ├── 1BasicPromptTemplate.ipynb
    │   ├── 2ChatPromptTemplate.ipynb
    │   ├── 3FewShotPrompt.ipynb
    │   ├── 4ConversationHistory.ipynb
    │   ├── 5StructuredOutput.ipynb
    │   ├── 6ToolCalling.ipynb
    │   ├── 7ReActPrompt.ipynb
    │   ├── 8PartialPrompts.ipynb
    │   ├── 9PipelinePrompts.ipynb
    │   └── 10PromptOptimization.ipynb
    │
    ├── Agents/
    │   ├── Agents_Concepts.md           # Full concept guide (read this first)
    │   ├── AgentUtils.py                # Shared utilities for all Agent notebooks
    │   ├── 0AgentLearningPath.ipynb
    │   ├── 1LLMWithTools.ipynb
    │   ├── 2ReActAgent.ipynb
    │   ├── 3ToolCallingAgent.ipynb
    │   ├── 4StructuredOutputAgent.ipynb
    │   ├── 5ConversationalAgent.ipynb
    │   ├── 6RAGAgent.ipynb
    │   ├── 7MultiAgent_Router.ipynb
    │   ├── 8PlanAndExecuteAgent.ipynb
    │   └── 9SelfCorrectingAgent.ipynb
    │
    └── Memory/
        └── 9emory.ipynb              # Buffer memory with RunnableWithMessageHistory
```

---

## Setup

### 1. Create and activate virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirement.txt
```

### 3. Create `.env` file

```
OPENAI_API_KEY=sk-your-api-key-here
```

### 4. Register the Jupyter kernel

```bash
python -m ipykernel install --user --name genai-venv --display-name "Python (GenAI venv)"
```

### 5. Open a notebook in VS Code

- Open any `.ipynb` file
- Select kernel: **Python (GenAI venv)**
- Run cells with `Shift+Enter`

---

## Models Used

| Purpose | Model |
|---------|-------|
| LLM | `gpt-4o-mini` |
| Embeddings | `text-embedding-3-small` |

---

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `langchain` | LLM framework |
| `langchain-openai` | OpenAI integration |
| `langchain-community` | Community retrievers & loaders |
| `langchain-text-splitters` | Document chunking |
| `faiss-cpu` | Vector similarity search |
| `rank_bm25` | BM25 keyword search |
| `jupyter` | Notebook environment |
| `python-dotenv` | Environment variable management |

---

## Troubleshooting

**`ModuleNotFoundError`**
```bash
pip install -r requirement.txt
```

**`OPENAI_API_KEY not found`**
- Check `.env` exists in the project root
- Restart the kernel after updating `.env`

**Kernel not showing in VS Code**
```bash
python -m ipykernel install --user --name genai-venv --display-name "Python (GenAI venv)"
```

**FAISS error**
```bash
pip install faiss-cpu --force-reinstall
```

---

## References

- [LangChain Docs](https://python.langchain.com/)
- [OpenAI API Docs](https://platform.openai.com/docs/)
- [FAISS](https://github.com/facebookresearch/faiss)

---

**Python**: 3.12 | **LangChain**: 1.2.x | **Last Updated**: June 2026
