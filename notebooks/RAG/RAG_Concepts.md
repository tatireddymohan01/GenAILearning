# RAG Concepts — Complete Learning Guide

---

## What is RAG?

Large Language Models (LLMs) like GPT-4 are trained on data up to a certain date.  
They cannot access your private documents, internal knowledge bases, or recent information.

**The Problem:**
```
User: "What does our company policy say about remote work?"
LLM:  "I don't have access to your company documents."
```

**RAG solves this** by retrieving relevant documents first, then letting the LLM answer using those documents as context.

```
Without RAG:   User Query → LLM → Answer (from training data only)

With RAG:      User Query → Retrieve Docs → LLM + Docs → Answer (grounded in your data)
```

RAG = **Retrieval** (find relevant docs) + **Augmented** (add them to prompt) + **Generation** (LLM answers)

---

## The Core RAG Pipeline

Every RAG system — no matter how advanced — is built on this foundation:

```
INDEXING (done once, offline):
  Documents → Split into Chunks → Embed Chunks → Store in Vector DB

QUERYING (done on every user question):
  User Query → Embed Query → Search Vector DB → Retrieve Top K Chunks → LLM → Answer
```

Understanding this pipeline deeply is the key to understanding every RAG variant.

---
---

## Concept 1 — Simple RAG ✅
**Notebook: 1simpleRAG.ipynb**

### Core Concept

The most fundamental form of RAG. You build it step by step manually without any abstractions.

**Indexing Phase (run once):**
```
Raw Documents (web page, PDF, text)
       ↓
Split into small chunks (e.g., 1000 characters each)
       ↓
Each chunk → OpenAI Embeddings → Vector (list of 1536 numbers)
       ↓
Store all vectors in FAISS (vector database)
```

**Query Phase (run on every question):**
```
User Query
       ↓
Query → OpenAI Embeddings → Query Vector
       ↓
FAISS: compare query vector vs all stored vectors (cosine similarity)
       ↓
Return Top K most similar chunks
       ↓
Build prompt: "Use this context to answer: {chunks} \n\n Question: {query}"
       ↓
LLM → Answer
```

### Why Embeddings Work

Embeddings convert text into numbers that capture **meaning**, not just words.  
Texts with similar meaning end up with similar vectors, even if they use different words.

```
"capital of France"   → [0.2, 0.8, 0.1, ...]
"Paris is the seat of French government" → [0.21, 0.79, 0.12, ...]

These two vectors are very close → similarity search finds them together
```

### What You Actually Built

```python
# Manual similarity search
results = vectorstore.similarity_search(query, k=2)
context = "\n\n".join([doc.page_content for doc in results])

# Manual prompt building
response = chain.invoke({"context": context, "question": query})
```

### Limitations of Simple RAG

| Limitation | Explanation |
|------------|-------------|
| **Keyword mismatch** | If user asks "database storage" but doc says "PostgreSQL persistence", embeddings may not find it |
| **Duplicate results** | Top 3 results may all say the same thing — no diversity |
| **No quality check** | Retrieved docs may not actually answer the question |
| **Manual wiring** | Every step is manual — hard to maintain and extend |
| **Fixed chunk size** | All chunks are the same size regardless of content structure |
| **No conversation** | Cannot handle follow-up questions |

### What the Next Concept Fixes

Concept 2 fixes the **manual wiring** problem by introducing retriever abstraction and **duplicate result** problem with MMR.

---
---

## Concept 2 — LCEL RAG + Retriever Abstraction ✅
**Notebook: 2simpleRAG_Retriver.ipynb**

### Core Concept

Two improvements over Simple RAG:

**Improvement 1: Retriever Abstraction**  
Instead of manually calling `similarity_search()` and building context, LangChain's retriever handles it.

```python
# Simple RAG (manual)
results = vectorstore.similarity_search(query, k=2)
context = "\n\n".join([doc.page_content for doc in results])

# LCEL RAG (abstracted)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
rag_chain = {"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | parser
```

The chain handles retrieval automatically when you call `rag_chain.invoke(query)`.

**Improvement 2: MMR (Maximal Marginal Relevance)**

Standard similarity search returns the top K most similar docs. Problem: they may all be nearly identical.

```
Query: "What is LangSmith persistence?"

Without MMR (top 3 similar):
  Doc 1: "LangSmith stores data in PostgreSQL..."
  Doc 2: "LangSmith uses PostgreSQL for storage..."   ← almost same as Doc 1
  Doc 3: "PostgreSQL is the default database..."       ← almost same as Doc 1

With MMR (top 3 diverse + relevant):
  Doc 1: "LangSmith stores data in PostgreSQL..."
  Doc 2: "Checkpoints are short-term memory snapshots..."  ← different angle
  Doc 3: "The store enables long-term memory across threads..."  ← another angle
```

MMR balances: relevance to query + diversity from already selected docs.

**Improvement 3: Multi-Retriever**  
Query multiple data sources (vector stores) in parallel and combine results.

```
Query
  ↓
Retriever 1 (Source A) ──┐
                          ├── Merge Results → LLM → Answer
Retriever 2 (Source B) ──┘
```

### LCEL (LangChain Expression Language)

LCEL uses the `|` pipe operator to chain operations — similar to Unix pipes.

```python
chain = retriever | format_docs   # retriever output flows into format_docs
full_chain = {"context": chain, "question": RunnablePassthrough()} | prompt | llm | parser
```

This makes pipelines readable, composable, and easy to extend.

### Limitations of LCEL RAG

| Limitation | Explanation |
|------------|-------------|
| **All retrieved docs treated equally** | Doc ranked 1st and doc ranked 5th both go to LLM with same weight |
| **No quality scoring** | System doesn't know if retrieved docs are actually relevant |
| **Still single retrieval method** | Either vector OR keyword — not both together |
| **No self-correction** | If results are poor, system doesn't know or retry |

### What the Next Concept Fixes

Concept 3 fixes the **all docs treated equally** problem by adding a scoring layer — reranking.

---
---

## Concept 3 — Advanced RAG (Reranking) ✅
**Notebook: 3AdvancedRAG.ipynb**

### Core Concept

Retrieval gives you candidate docs. Reranking **scores and re-orders** them by actual relevance to the query, then keeps only the best.

Think of it as a **two-stage filter**:
- Stage 1 (Retrieval): Cast a wide net — get many candidate docs quickly
- Stage 2 (Reranking): Carefully score each candidate — keep only the best

```
Query
  ↓
Retriever 1 (Vector) → 4 docs ──┐
                                  ├── 8 docs total
Retriever 2 (Vector) → 4 docs ──┘
  ↓
Reranker: score each doc against query (1-10)
  Doc A: score 9  ← keep
  Doc B: score 8  ← keep
  Doc C: score 8  ← keep
  Doc D: score 3  ← discard
  Doc E: score 2  ← discard
  ...
  ↓
Top 3 docs only → LLM → Answer
```

### Why Reranking is Needed

Vector similarity measures geometric distance between embeddings — not actual relevance.

```
Query: "How does LangSmith handle database failures?"

Vector search returns:
  Doc 1 (score: 0.89): "LangSmith uses PostgreSQL for persistence..."
  Doc 2 (score: 0.85): "Database configuration options in LangSmith..."
  Doc 3 (score: 0.82): "LangSmith deployment uses a task queue..."  ← not about failures

After reranking:
  Doc 1 (relevance: 9): "LangSmith uses PostgreSQL for persistence..."  ← still top
  Doc 2 (relevance: 7): "Database configuration options..."
  Doc 3 (relevance: 2): "task queue..." ← dropped — not relevant to "failures"
```

### Types of Rerankers

**LLM-based reranker (what you built):**
```python
prompt = f"Query: {query}\nDocument: {doc}\nScore relevance 1-10:"
score = llm.invoke(prompt).content
```
Pros: No extra setup. Cons: Slow (one LLM call per doc), expensive.

**Cross-encoder models (e.g., Cohere Rerank, BGE Reranker):**  
Specialized ML models trained specifically for relevance scoring.  
Much faster and cheaper than using a full LLM for scoring.

### Limitations of Reranking

| Limitation | Explanation |
|------------|-------------|
| **Still single search method** | If BM25 keyword search would find better docs, you miss them |
| **LLM reranker is slow** | One LLM call per doc — 8 docs = 8 LLM calls before even answering |
| **Depends on initial retrieval quality** | Reranking can only work with what retrieval gives it — garbage in, garbage out |
| **No self-correction loop** | If final answer is still bad, pipeline doesn't know |

### What the Next Concept Fixes

Concept 4 fixes the **single search method** problem by combining vector search with keyword (BM25) search.

---
---

## Concept 4 — Hybrid RAG ✅
**Notebook: 4HybridRAG_AgeneticRAG.ipynb (Part 1)**

### Core Concept

No single retrieval method is perfect. Hybrid RAG combines two fundamentally different retrieval approaches:

| Method | How It Works | Good At | Bad At |
|--------|-------------|---------|--------|
| **Vector Search** | Semantic similarity via embeddings | Meaning, paraphrases, concepts | Exact keywords, technical terms |
| **BM25 Keyword Search** | Term frequency matching (like search engines) | Exact words, proper nouns, IDs | Synonyms, context, meaning |

By combining both, you cover each other's blind spots.

```
Query: "PostgreSQL checkpoint failure recovery"
  ↓
Vector Search:  finds docs about "database state recovery mechanisms"  (semantic match)
BM25 Search:    finds docs containing "PostgreSQL" and "checkpoint"   (keyword match)
  ↓
Merge + Deduplicate
  ↓
Result: Better coverage than either alone
```

### Real-World Example

```
Query: "LangSmith gpt-4o-mini configuration"

Vector search finds:
  "LLM configuration options for the agent server..."   (semantic: LLM config)
  "Model settings affect response quality..."            (semantic: model config)

BM25 finds:
  "...using gpt-4o-mini as the default model..."        (exact: gpt-4o-mini keyword)
  "...LangSmith supports OpenAI model configurations..."(exact: LangSmith keyword)

Combined: ALL 4 docs → LLM has complete picture
```

### How Deduplication Works

Both retrievers may return the same doc. You remove duplicates using the doc content as the key:

```python
unique_docs = list({doc.page_content: doc for doc in all_docs}.values())
```

### Limitations of Hybrid RAG

| Limitation | Explanation |
|------------|-------------|
| **Query quality still matters** | If user asks a vague or poorly worded question, even hybrid retrieval gives poor results |
| **No answer quality check** | System delivers answer even if retrieved docs don't actually answer the question |
| **Static pipeline** | Same steps run regardless of whether the question needs retrieval at all |
| **No retry mechanism** | One shot — if answer is wrong, it's wrong |

### What the Next Concept Fixes

Concept 5 fixes the **no quality check** and **no retry** problems by adding an intelligent feedback loop.

---
---

## Concept 5 — Agentic RAG ✅
**Notebook: 4HybridRAG_AgeneticRAG.ipynb (Part 2)**

### Core Concept

Agentic RAG adds **intelligence and self-correction** to the RAG pipeline. The system can evaluate its own output and decide whether to try again with a better query.

```
User Query
  ↓
Hybrid Retrieve → LLM → Answer
  ↓
Self-Check: "Is this answer sufficient?"
  ↓ YES                     ↓ NO
Return Answer          Refine the query
                            ↓
                       Hybrid Retrieve again (new query)
                            ↓
                       Merge new + original docs
                            ↓
                       LLM → Better Answer
```

### The Intelligence: Query Refinement

When the initial answer is insufficient, the LLM doesn't just retry with the same query. It **improves the query** to get better results:

```
Original query: "what is Persistence and types"
  ↓
Initial answer: vague, incomplete
  ↓
LLM refines to: "LangSmith persistence mechanisms including checkpoints store and core resources"
  ↓
New retrieval finds much more specific docs
  ↓
Final answer: detailed, accurate
```

### Why "Agentic"?

Traditional RAG: fixed sequence of steps, no decision-making.  
Agentic RAG: the system makes **decisions** (is the answer good enough? what query to use next?).

This is the first step toward autonomous AI agents.

### Limitations of Agentic RAG

| Limitation | Explanation |
|------------|-------------|
| **Single query variant** | Generates one refined query — may still miss relevant angles |
| **No fallback if knowledge base lacks answer** | If the doc doesn't exist in your store, refining the query won't help |
| **One retrieval method still** | You're using hybrid but still searching the same vector store |
| **No source evaluation** | Doesn't judge whether the retrieved docs are trustworthy/relevant before using |

### What the Next Concepts Fix

- Concept 6 (RAG Fusion) fixes **single query variant** — generates multiple query variations
- Concept 10 (CRAG) fixes **no fallback** — web search when knowledge base fails

---
---

## Concept 6 — RAG Fusion
**Notebook: to be created**

### Core Concept

A single user query is just one way to ask a question. RAG Fusion generates **multiple rephrased versions** of the query, retrieves for each, then intelligently merges all results.

```
User Query: "How does LangSmith store data?"
  ↓
LLM generates query variations:
  Q1: "LangSmith database storage mechanisms"
  Q2: "LangSmith PostgreSQL configuration"
  Q3: "LangSmith persistence layer architecture"
  Q4: "data storage options in LangSmith deployment"
  ↓
Retrieve docs for EACH query (4 separate retrieval runs)
  ↓
Merge all results (may have 12-20 docs)
  ↓
Reciprocal Rank Fusion (RRF) — smart merging
  ↓
Top docs → LLM → Answer
```

### Reciprocal Rank Fusion (RRF)

RRF is a formula that combines multiple ranked lists into one.  
A doc that appears in top positions across multiple query results gets a high final score.

```
Doc A: ranked #1 for Q1, #2 for Q2, #3 for Q4 → very high RRF score
Doc B: ranked #1 for Q3 only                   → medium RRF score
Doc C: ranked #5 for Q1 only                   → low RRF score
```

Docs that are consistently relevant across multiple query angles rise to the top.

### Why It's Better Than Single Query

```
Single Query: "LangSmith data storage"
  → Finds: PostgreSQL docs ✓
  → Misses: checkpoint docs, store docs, task queue persistence docs

RAG Fusion with 4 queries:
  → Finds: PostgreSQL ✓, checkpoints ✓, store ✓, task queue ✓
  → Complete picture!
```

### Limitations of RAG Fusion

| Limitation | Explanation |
|------------|-------------|
| **Higher cost** | 4 queries = 4x retrieval operations + 1 LLM call to generate queries |
| **Slower** | Multiple retrieval rounds before getting an answer |
| **Still same knowledge base** | If the answer isn't in your docs, more queries won't help |
| **Chunks may lack surrounding context** | Retrieved chunk may be the right topic but miss context from neighboring text |

### What the Next Concept Fixes

Concept 7 (HyDE) fixes the **query-to-document mismatch** at the embedding level — a different angle on improving retrieval quality.

---
---

## Concept 7 — HyDE (Hypothetical Document Embeddings)
**Notebook: to be created**

### Core Concept

There is a fundamental mismatch between how **questions** and **answers** are written:

```
Question: "What databases does LangSmith support?"
  → Short, interrogative, lacks context

Answer doc: "LangSmith persists core resources using PostgreSQL by default. 
             MongoDB can also be configured as an alternative..."
  → Long, declarative, full of details
```

In embedding space, short questions and detailed answers end up far apart.

**HyDE fixes this by generating a hypothetical answer first:**

```
User Query: "What databases does LangSmith support?"
  ↓
LLM generates hypothetical answer (without RAG):
  "LangSmith likely supports PostgreSQL as the primary database, 
   with possible support for MongoDB and other alternatives..."
  ↓
Embed the HYPOTHETICAL ANSWER (not the question)
  ↓
Use hypothetical answer's embedding to search vector store
  ↓
Hypothetical answer is close to real answer docs in embedding space
  ↓
Real docs retrieved → LLM → Final Answer (using real docs, not hypothetical)
```

### Why It Works

```
Without HyDE:
  Query embedding ←——————————— far ———————————→ Answer doc embedding

With HyDE:
  Query → Hypothetical Answer embedding ←— close —→ Answer doc embedding
```

The hypothetical answer "looks like" a real answer, so it's close to real answers in vector space.

### Important Note

The hypothetical answer is used **only for retrieval** — it's thrown away afterward.  
The LLM answers using the real retrieved docs, not the hypothesis.

### Limitations of HyDE

| Limitation | Explanation |
|------------|-------------|
| **Hallucination risk** | LLM may generate a wildly wrong hypothesis, retrieving irrelevant docs |
| **Extra LLM call** | One LLM call to generate hypothesis + another to answer = 2x LLM calls |
| **Doesn't help with exact keyword matches** | Hypothetical answers use different words, may miss BM25-style matches |
| **Small chunks miss surrounding context** | Retrieved chunk is precise but may lack the surrounding paragraph needed to fully understand |

### What the Next Concept Fixes

Concept 8 (Parent Document Retriever) fixes the **small chunks miss context** problem.

---
---

## Concept 8 — Parent Document Retriever
**Notebook: to be created**

### Core Concept

There is a tension in chunk sizing:

```
Small chunks (200 chars):
  ✓ Very precise matching — query finds exactly the right sentence
  ✗ No surrounding context — LLM can't understand without the full paragraph

Large chunks (2000 chars):
  ✓ Full context for LLM to understand
  ✗ Poor matching — query gets diluted by irrelevant surrounding text
```

**Parent Document Retriever has both:**
- Index **small chunks** for retrieval (precise matching)
- Store **large parent chunks** separately
- When a small chunk matches, return its **parent** to the LLM

```
Document: "LangSmith Architecture..."
  ↓
Split into small chunks (child):
  Child 1: "checkpoints are snapshots of state"       → indexed for retrieval
  Child 2: "checkpoints written at each step"         → indexed for retrieval
  Child 3: "store persists across threads"            → indexed for retrieval

Also stored as large parent chunks:
  Parent A (contains Child 1 & 2): full paragraph about checkpoints
  Parent B (contains Child 3): full paragraph about the store

Query: "what are checkpoints?"
  ↓
Small chunk matching: Child 1 matches precisely
  ↓
Retrieve its PARENT: Parent A (full paragraph about checkpoints)
  ↓
LLM gets full context → Better Answer
```

### Limitations of Parent Document Retriever

| Limitation | Explanation |
|------------|-------------|
| **More storage** | Both small and large chunks stored — doubles storage needs |
| **Parent chunks can still be noisy** | Parent may include some irrelevant surrounding text |
| **LLM context still has noise** | Even parent chunks contain some irrelevant sentences mixed in |

### What the Next Concept Fixes

Concept 9 (Contextual Compression) fixes the **noise in retrieved docs** by stripping irrelevant parts before passing to LLM.

---
---

## Concept 9 — Contextual Compression
**Notebook: to be created**

### Core Concept

Even after good retrieval, the retrieved chunk often contains irrelevant sentences mixed with the relevant ones.  
Contextual Compression **extracts only the relevant portion** before passing to the LLM.

```
Query: "What is a checkpoint in LangSmith?"

Retrieved chunk (full, 800 chars):
  "LangSmith supports multiple deployment modes including cloud and self-hosted.
   The platform uses PostgreSQL as its default database. Checkpoints are 
   short-term memory snapshots of graph execution state, written at each step,
   allowing a run to resume from the last checkpoint if interrupted. The task 
   queue handles background processing for asynchronous operations..."

After Contextual Compression:
  "Checkpoints are short-term memory snapshots of graph execution state, 
   written at each step, allowing a run to resume from the last checkpoint 
   if interrupted."
```

Now the LLM receives **only** what's relevant.

### Benefits

- **Better answers** — LLM focuses on relevant text, not distracted by noise
- **Cheaper** — fewer tokens sent to LLM
- **More docs fit in context** — with compression, you can include more sources

### Types of Compressors

**LLM-based compressor:** Ask LLM to extract relevant sentences  
**Embedding-based compressor:** Keep only sentences whose embeddings are close to the query  

### Limitations of Contextual Compression

| Limitation | Explanation |
|------------|-------------|
| **Extra processing step** | More latency before answering |
| **May over-compress** | Could strip context that seems irrelevant but is actually needed |
| **Still relies on local knowledge base** | If the answer isn't in your docs, compression doesn't help |
| **No fallback** | If retrieved docs are completely off-topic, system still tries to answer with them |

### What the Next Concept Fixes

Concept 10 (CRAG) fixes the **no fallback** problem — it evaluates doc quality and falls back to web search when your knowledge base can't help.

---
---

## Concept 10 — CRAG (Corrective RAG)
**Notebook: to be created**

### Core Concept

All previous concepts assume your knowledge base has the answer.  
CRAG adds a **quality gate** — it evaluates whether retrieved docs are actually relevant, and falls back to web search if they're not.

```
User Query
  ↓
Retrieve from your knowledge base
  ↓
Evaluator: "Are these docs relevant to the query?"
  ↓
  RELEVANT              IRRELEVANT              AMBIGUOUS
     ↓                      ↓                      ↓
  Use docs            Web Search            Use docs + Web Search
     ↓                      ↓                      ↓
  LLM → Answer       LLM → Answer           LLM → Answer
```

### The Evaluator

A separate LLM (or classifier) scores each retrieved doc:
```
Score: RELEVANT    → doc directly answers the query
Score: IRRELEVANT  → doc is off-topic
Score: AMBIGUOUS   → doc is partially related
```

### Knowledge Refinement

Before using docs, CRAG also refines them — strips irrelevant sentences (like Contextual Compression), keeping only what matters.

### Real-World Value

```
Your vector store has: LangSmith documentation (last updated 3 months ago)
User asks: "What's the latest LangSmith release?"

Without CRAG: retrieves old docs → gives outdated answer confidently
With CRAG:    evaluates docs → finds them outdated → web search → current answer
```

### Limitations of CRAG

| Limitation | Explanation |
|------------|-------------|
| **Multiple LLM calls** | Retrieval + evaluation + (optional web search) + answer = expensive |
| **Web search dependency** | Fallback requires internet access and a search API |
| **Evaluation can be wrong** | LLM evaluator may misjudge doc relevance |
| **LLM still doesn't control the full flow** | Pipeline is still externally controlled — LLM just scores, it doesn't decide strategy |

### What the Next Concept Fixes

Concept 11 (Self-RAG) gives the LLM **full control** over every decision in the pipeline.

---
---

## Concept 11 — Self-RAG
**Notebook: to be created**

### Core Concept

In all previous concepts, the **pipeline decides what to do** and the LLM just answers.  
Self-RAG flips this: the **LLM itself decides** every step of the process.

The LLM makes 4 key decisions using special tokens:

```
Decision 1 — Should I retrieve at all?
  [Retrieve] → yes, fetch docs
  [No Retrieve] → I know this already, answer directly

Decision 2 — Are these docs relevant?
  [Relevant] → use this doc
  [Irrelevant] → discard this doc

Decision 3 — Is my answer grounded in the docs?
  [Fully Supported] → my answer is backed by the docs
  [Partially Supported] → some parts aren't backed
  [No Support] → my answer contradicts the docs → retry

Decision 4 — Is my answer complete and useful?
  [Useful] → return to user
  [Not Useful] → generate again
```

### Full Flow

```
User Query
  ↓
LLM: "Do I need to retrieve?" 
  ↓ NO → answer from training → check utility → done
  ↓ YES
Retrieve K docs
  ↓
For each doc:
  LLM: "Is this doc relevant?" → keep or discard
  ↓
Generate answer using kept docs
  ↓
LLM: "Is my answer grounded in the docs?"
  ↓ YES → check utility → done
  ↓ NO  → retry with different docs
```

### Why Self-RAG is Powerful

```
Simple RAG:   always retrieves → always uses all docs → answers even when wrong
Self-RAG:     retrieves only when needed → filters irrelevant docs → 
              verifies answer is grounded → retries if not → self-validates

Result: Higher accuracy, fewer hallucinations, more efficient (skips retrieval when unnecessary)
```

### Limitations of Self-RAG

| Limitation | Explanation |
|------------|-------------|
| **Complex to implement** | Requires fine-tuned model or careful prompt engineering |
| **Slower** | Multiple LLM decisions per query |
| **Still single knowledge source** | Cannot chain multiple retrieval steps for complex reasoning |
| **Single-hop only** | Each retrieval step is independent — can't reason across multiple steps |

### What the Next Concept Fixes

Concept 12 (Multi-hop RAG) fixes the **single-hop** limitation — for questions that require chaining multiple retrievals together.

---
---

## Concept 12 — Multi-hop RAG
**Notebook: to be created**

### Core Concept

Some questions cannot be answered in a single retrieval. They require **reasoning across multiple documents** in a chain, where each retrieval informs the next.

```
Simple question (single-hop):
  "What is a LangSmith checkpoint?"
  → One retrieval → Done

Complex question (multi-hop):
  "What storage mechanism does the tool that LangSmith uses for short-term memory rely on?"
  → Hop 1: Retrieve → "LangSmith uses checkpoints for short-term memory"
  → Hop 2: Retrieve → "Checkpoints are stored in PostgreSQL"
  → Hop 3: Retrieve → "PostgreSQL uses WAL (Write-Ahead Logging) for storage"
  → Answer: "WAL (Write-Ahead Logging)"
```

### How It Works

```
Query
  ↓
Decompose into sub-questions:
  Q1: "What does LangSmith use for short-term memory?"
  Q2: "How is {answer to Q1} stored?"
  Q3: "What storage mechanism does {answer to Q2} rely on?"
  ↓
Retrieve and answer Q1
  ↓
Use answer to build Q2 → retrieve and answer Q2
  ↓
Use answer to build Q3 → retrieve and answer Q3
  ↓
Compose final answer from all hops
```

### When Multi-hop is Needed

```
"Which team built the framework used by the assistant that handles LangSmith monitoring?"
→ Step 1: What assistant handles LangSmith monitoring? → "LangSmith Agent"
→ Step 2: What framework does LangSmith Agent use? → "LangGraph"
→ Step 3: Which team built LangGraph? → "LangChain team"
→ Final: "LangChain team"
```

### Limitations of Multi-hop RAG

| Limitation | Explanation |
|------------|-------------|
| **Error propagation** | Wrong answer at Hop 1 leads to wrong retrieval at Hop 2, 3... |
| **Expensive** | Multiple LLM calls and retrieval rounds |
| **Complex to build** | Requires query decomposition, sub-answer tracking, final composition |
| **Overkill for simple questions** | Running multi-hop on a simple question wastes resources |

---
---

## How Each Concept Solves the Previous One's Limitation

```
Simple RAG
  Problem: Manual pipeline, duplicate results, no quality check
      ↓ solved by
LCEL + Retriever + MMR
  Problem: All docs treated equally, still single method
      ↓ solved by
Reranking
  Problem: Single retrieval method, no keyword matching
      ↓ solved by
Hybrid RAG (Vector + BM25)
  Problem: Vague queries still give poor results, no retry
      ↓ solved by
Agentic RAG
  Problem: Only one query variation tried
      ↓ solved by
RAG Fusion
  Problem: Question-to-document embedding mismatch
      ↓ solved by
HyDE
  Problem: Small chunks miss surrounding context
      ↓ solved by
Parent Document Retriever
  Problem: Retrieved docs contain noisy/irrelevant sentences
      ↓ solved by
Contextual Compression
  Problem: Knowledge base may not have the answer
      ↓ solved by
CRAG
  Problem: External pipeline controls flow, not the LLM
      ↓ solved by
Self-RAG
  Problem: Single-hop only, can't reason across multiple steps
      ↓ solved by
Multi-hop RAG
```

---

## Learning Order & Status

| # | Concept | Status | Key Skill Learned |
|---|---------|--------|-------------------|
| 1 | Simple RAG | ✅ Done | Load, split, embed, retrieve, generate |
| 2 | LCEL + Retriever + MMR | ✅ Done | Pipelines, diversity, multi-source |
| 3 | Reranking | ✅ Done | Quality filtering, two-stage retrieval |
| 4 | Hybrid RAG | ✅ Done | Vector + keyword combined |
| 5 | Agentic RAG | ✅ Done | Self-correction, query refinement |
| 6 | RAG Fusion | ⬅ Next | Multi-query generation, RRF merging |
| 7 | HyDE | To do | Embedding space alignment |
| 8 | Parent Doc Retriever | To do | Precision retrieval + full context |
| 9 | Contextual Compression | To do | Noise removal from docs |
| 10 | CRAG | To do | Doc quality evaluation, web fallback |
| 11 | Self-RAG | To do | LLM-controlled pipeline |
| 12 | Multi-hop RAG | To do | Chained multi-step reasoning |

---

## Quick Reference Card

| Concept | One-Line Summary |
|---------|-----------------|
| Simple RAG | Embed docs → find similar → answer |
| LCEL + MMR | Clean pipeline + diverse results |
| Reranking | Score all candidates, keep best |
| Hybrid RAG | Vector (meaning) + BM25 (keywords) |
| Agentic RAG | Check answer quality, refine and retry |
| RAG Fusion | Many query variants → merge → answer |
| HyDE | Generate fake answer → use it to search |
| Parent Doc | Small chunks for search, big chunks for LLM |
| Contextual Compression | Strip noise from retrieved docs |
| CRAG | Bad docs? Fall back to web search |
| Self-RAG | LLM decides when/whether to retrieve |
| Multi-hop | Chain multiple retrievals for complex questions |
