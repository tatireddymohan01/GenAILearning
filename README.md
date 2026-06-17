# GenAI RAG Projects

A collection of Retrieval-Augmented Generation (RAG) and Agent implementations using LangChain, demonstrating various AI patterns including simple RAG, hybrid RAG, and agentic RAG with tool calling.

## Project Overview

This repository contains multiple implementations:

- **Simple RAG**: Basic retrieval-augmented generation with vector stores
- **Retriever Examples**: Various retriever patterns (BM25, vector-based)
- **LLM with Tool Calling**: Modern LangChain tool calling patterns with proper message formatting
- **Hybrid RAG**: Combined vector and keyword retrieval for better results
- **Agentic RAG**: Intelligent query refinement with quality checks

## Prerequisites

- **Python 3.10+** (Tested with Python 3.12)
- **pip** (Python package manager)
- **OpenAI API Key** (for LLM and embedding models)
- **Virtual Environment** (recommended: venv or conda)

## Environment Setup

### 1. Create Virtual Environment

#### Using venv (Recommended):
```bash
cd c:\E\GenAI\first
python -m venv .venv
```

#### Or using Conda:
```bash
conda create -n genai python=3.12
conda activate genai
```

### 2. Activate Virtual Environment

#### Windows (PowerShell):
```powershell
.\.venv\Scripts\Activate.ps1
```

#### Windows (Command Prompt):
```cmd
.venv\Scripts\activate.bat
```

#### Linux/Mac:
```bash
source .venv/bin/activate
```

### 3. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
touch .env  # Linux/Mac
# or
New-Item -Path ".\.env" -ItemType File  # PowerShell
```

Add your OpenAI API key to `.env`:

```
OPENAI_API_KEY=sk-your-api-key-here
```

**Note**: Never commit `.env` to version control. It's already in `.gitignore`.

## Installation

### 1. Install Requirements

```bash
pip install -r requirement.txt
```

### 2. Verify Installation

```bash
python -c "import langchain; print(f'LangChain version: {langchain.__version__}')"
```

## Project Structure

```
c:\E\GenAI\first\
├── README.md                          # This file
├── requirement.txt                    # Python dependencies
├── .env                               # Environment variables (NOT in repo)
├── app.py                             # Main application
├── retrievers_examples.py             # Retriever pattern examples
│
└── notebooks/
    ├── simpleRAG.ipynb               # Basic RAG implementation
    ├── simpleRAG_Retriver.ipynb      # Retriever-based RAG
    ├── LLMWithToolCalling.ipynb      # Modern tool calling patterns
    ├── HybridRAG.ipynb               # Hybrid retrieval (vector + keyword)
    ├── HybridRAG_AgeneticRAG.ipynb   # Agentic RAG with refinement
    └── test.ipynb                    # Testing and experimentation
```

## Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `langchain` | 1.2.15+ | LLM framework |
| `langchain-openai` | Latest | OpenAI integration |
| `langchain-community` | Latest | Community retrievers & loaders |
| `faiss-cpu` | Latest | Vector similarity search |
| `python-dotenv` | Latest | Environment variable management |

## Running the Notebooks

### Option 1: VS Code Jupyter Extension (Recommended)

1. Open VS Code
2. Install "Jupyter" extension
3. Open any `.ipynb` file
4. Select Python kernel
5. Run cells with Shift+Enter

### Option 2: Jupyter Lab

```bash
jupyter lab
# Navigate to notebooks/ folder
# Open desired notebook
```

### Option 3: Jupyter Notebook

```bash
jupyter notebook
# Navigate to notebooks/ folder
# Open desired notebook
```

## Usage Examples

### Simple RAG
```python
# In notebook: simpleRAG.ipynb
# Query the documentation with automatic retrieval and LLM response
query = "How does LangChain work?"
answer = rag_chain.invoke(query)
```

### Tool Calling
```python
# In notebook: LLMWithToolCalling.ipynb
# Use LLM with bound tools for calculations
run_tool_calling("What is 5 + 7?")
# Output: sum of 5 and 7 = 12
```

### Agentic RAG
```python
# In notebook: HybridRAG_AgeneticRAG.ipynb
# RAG with query refinement for better results
answer = agentic_rag("What is Persistence and types?")
# Automatically refines query if initial answer is insufficient
```

## Troubleshooting

### Common Issues

#### 1. `ModuleNotFoundError: No module named 'langchain'`
```bash
# Reinstall dependencies
pip install -r requirement.txt --upgrade
```

#### 2. `OPENAI_API_KEY not found`
- Check `.env` file exists in project root
- Verify API key is correct
- Restart kernel after updating `.env`

#### 3. Jupyter Kernel Issues
```bash
# Reinstall IPython kernel
python -m ipykernel install --user --name genai --display-name "GenAI"
```

#### 4. FAISS Not Loading
```bash
# Try CPU version
pip install faiss-cpu --force-reinstall
```

## API Usage Notes

- **Models Used**:
  - LLM: `gpt-4o-mini` (fast, cost-effective)
  - Embeddings: `text-embedding-3-small` (efficient)
  
- **Rate Limits**: OpenAI has rate limits. Space out requests if testing heavily.

- **Costs**: Monitor your OpenAI usage to avoid unexpected charges.

## Configuration

### Modify Search Settings

In any notebook, you can adjust retriever behavior:

```python
# Vector retriever - return top K results
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# BM25 retriever - keyword search results
bm25_retriever.k = 3
```

### Change LLM Model

```python
# Switch to different model
llm = ChatOpenAI(model="gpt-4")  # More powerful but slower
llm = ChatOpenAI(model="gpt-3.5-turbo")  # Faster but less capable
```

## Best Practices

1. **Always activate virtual environment** before running code
2. **Keep `.env` file secure** - never commit it
3. **Test in notebooks first** before moving to production
4. **Monitor API usage** to manage costs
5. **Use hybrid retrieval** for better accuracy
6. **Refine queries** when initial results are poor
7. **Handle errors gracefully** in production code

## Development Workflow

```bash
# 1. Activate environment
.\.venv\Scripts\Activate.ps1

# 2. Install/update packages
pip install -r requirement.txt

# 3. Launch Jupyter
jupyter lab

# 4. Work in notebooks
# ... develop and test ...

# 5. Move to production
# Copy working code to app.py or modules
```

## Next Steps

- Explore different prompts and contexts
- Experiment with hybrid retrieval weights
- Implement custom tools
- Add database integration
- Deploy as API service

## Support & Learning

- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI API Docs](https://platform.openai.com/docs/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [LangSmith Docs](https://docs.langchain.com/langsmith/)

## License

This project is for educational purposes.

---

**Last Updated**: April 2026  
**Python Version**: 3.12  
**LangChain Version**: 1.2.15+
