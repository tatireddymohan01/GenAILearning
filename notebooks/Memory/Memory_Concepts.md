# Memory Concepts — Complete Learning Guide (LangChain 1.x)

---

## What is Memory in LangChain?

LLMs are **stateless by default** — each call is independent with no knowledge of previous turns.

**The Problem:**
```
Turn 1: "My name is Mohan."   → AI: "Hello Mohan!"
Turn 2: "What is my name?"    → AI: "I don't know your name."   ← forgets everything
```

**Memory solves this** by storing conversation history and injecting relevant context into each prompt.

### The Modern LangChain 1.x Memory Pattern

The old `langchain.memory` module (ConversationBufferMemory, ConversationSummaryMemory, etc.) was
**removed in LangChain 1.x**. The modern approach separates concerns into two parts:

```
PART 1 — Storage  (where messages live)
  ChatMessageHistory         → in-memory (lost on restart)
  FileChatMessageHistory     → persisted to a JSON file
  SQLChatMessageHistory      → persisted to SQLite / PostgreSQL

PART 2 — Injection  (how history gets into the prompt)
  RunnableWithMessageHistory → auto-injects history into any LCEL chain
  trim_messages              → trims history by token count or message count before injection
  Manual summarization       → LLM compresses old messages into a SystemMessage
  FAISS semantic retrieval   → fetch only the most relevant past turns
```

The key class is `RunnableWithMessageHistory` — it wraps any chain and automatically:
- Loads history from storage before each call
- Saves the new turn to storage after each call

---

## The Core Modern Pattern

```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}  # session_id → ChatMessageHistory

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chain = prompt | llm

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# Same session_id = same conversation continued
response = chain_with_history.invoke(
    {"input": "What is my name?"},
    config={"configurable": {"session_id": "mohan"}}
)
```

---
---

## Concept 1 — ChatMessageHistory + RunnableWithMessageHistory
**Notebook: 1ChatHistoryAndRWMH.ipynb**

### Core Concept

The foundation of all modern LangChain memory. `ChatMessageHistory` is just a list of messages.
`RunnableWithMessageHistory` wraps any chain so history is loaded and saved automatically.

```
Session A (session_id="mohan"):
  Turn 1: Human: "I am Mohan." → AI: "Hello!" → stored in store["mohan"]
  Turn 2: Human: "My name?"   → store["mohan"] loaded → LLM sees Turn 1 → "Your name is Mohan."

Session B (session_id="priya"):
  Turn 1: Human: "My name?"   → store["priya"] is empty → "I don't know your name."
  (Completely isolated from Session A)
```

### What the Next Concept Introduces

Concept 2 shows how to **limit** what gets injected — the buffer grows forever without trimming.

---
---

## Concept 2 — Window Memory with trim_messages
**Notebook: 2WindowMemory.ipynb**

### Core Concept

`trim_messages` from `langchain_core.messages` trims a message list to fit within a limit.
For **window memory**, set `max_tokens` to a message count (not actual tokens) using a count function.

```python
from langchain_core.messages import trim_messages

# Keep last 3 turns (6 messages) — older ones dropped
trimmed = trim_messages(
    messages,
    max_tokens=3,
    token_counter=lambda msgs: len(msgs),   # count messages, not tokens
    strategy="last",
    start_on="human",
)
```

```
All messages (5 turns):  [H1 A1 H2 A2 H3 A3 H4 A4 H5 A5]
After trim (last 3):                     [H3 A3 H4 A4 H5 A5]
                                          ← H1, A1, H2, A2 dropped
```

### Limitation

Dropped messages are gone — important early facts lost permanently.

### What the Next Concept Introduces

Concept 3 compresses old messages into a summary instead of dropping them.

---
---

## Concept 3 — Summary Memory
**Notebook: 3SummaryMemory.ipynb**

### Core Concept

When the message list grows large, use an LLM to **compress it into a running summary**.
Inject the summary as a `SystemMessage` — compact but information-preserving.

```
After Turn 5, instead of 10 raw messages:
  SystemMessage: "User is Mohan, a Python developer at Acme Corp (Hyderabad, 500 employees).
                  Building a document Q&A chatbot using LangChain + FAISS + Azure OpenAI."

Only 1 message injected instead of 10 — context window stays small.
```

### Implementation Pattern

```python
def summarize(messages: list, llm) -> str:
    conversation = "\n".join([
        f"Human: {m.content}" if isinstance(m, HumanMessage) else f"AI: {m.content}"
        for m in messages
    ])
    return llm.invoke(f"Summarize this conversation in 2-3 sentences:\n{conversation}").content

# Running summary updated after each turn
summary = summarize(history.messages, llm)

# Injected as SystemMessage
messages_for_llm = [SystemMessage(content=f"Context: {summary}"), HumanMessage(content=query)]
```

### Limitation

Summarization is lossy — exact wording, numbers, and details can be lost in compression.
Also: summary is called every turn, adding LLM cost.

### What the Next Concept Introduces

Concept 4 keeps recent messages raw and only summarizes the older ones.

---
---

## Concept 4 — SummaryBuffer Memory
**Notebook: 4SummaryBufferMemory.ipynb**

### Core Concept

The best of both worlds: `trim_messages` keeps recent messages raw (within a token budget),
while an LLM summarizes the messages that were trimmed out.

```
All 10 messages (5 turns):
  [H1 A1 H2 A2 H3 A3 H4 A4 H5 A5]

trim_messages keeps last 200 tokens:
  Trimmed out: [H1 A1 H2 A2]    → LLM summarizes → 1 SystemMessage
  Kept raw:             [H3 A3 H4 A4 H5 A5]

Prompt = [SystemMessage(summary)] + [H3 A3 H4 A4 H5 A5] + [new question]
```

### Why This Is the Best Practical Strategy

```
Buffer Memory (inject all):         large prompt, grows unbounded
Window Memory (drop old):           precise recent + forgets old
Summary Memory (compress all):      compact but lossy, slow
SummaryBuffer (trim + summarize):   compact recent + compressed history — best balance
```

---
---

## Concept 5 — Token Buffer with trim_messages
**Notebook: 5TokenBufferMemory.ipynb**

### Core Concept

`trim_messages` with a real token counter: keep as many recent messages as fit within a token budget.
Unlike window memory (which counts turns), this counts **actual tokens** for precise context control.

```python
from langchain_core.messages import trim_messages

trimmed = trim_messages(
    messages,
    max_tokens=300,
    token_counter="approximate",   # or pass llm for exact counting
    strategy="last",
    start_on="human",
    include_system=True,
)
```

### Token vs Message Count Comparison

```
Message count (Window, k=3):
  If turn 2 = 400 tokens — still kept (ignores size)
  Unpredictable context window usage

Token count (trim_messages, max=300):
  Drops oldest messages until total is under 300 tokens
  Predictable — always fits in budget
```

---
---

## Concept 6 — VectorStore Memory
**Notebook: 6VectorStoreMemory.ipynb**

### Core Concept

Store past turns as **embeddings** in FAISS. For each new query, retrieve only the
**most semantically relevant** past turns — not the most recent ones.

```
Past turns stored as vectors:
  [Turn 1] "Mohan is a Python developer"     → vector
  [Turn 3] "Using LangChain with FAISS"      → vector
  [Turn 5] "Azure OpenAI as LLM provider"    → vector

Query: "What vector store am I using?"
  → embed query → search FAISS → retrieves Turn 3 (most similar)
  → NOT Turn 5 (most recent but about LLM, not vector store)
```

### Why This Beats Chronological Memory

```
Window Memory at Turn 100: sees only Turns 98-100
Vector Memory at Turn 100: sees Turn 3 (about FAISS) even though it was 97 turns ago
```

### Implementation Pattern

```python
vectorstore = FAISS.from_texts([], embeddings)

# Store each turn
for human, ai in turns:
    vectorstore.add_texts([f"Human: {human}\nAI: {ai}"])

# Retrieve relevant past turns
docs = vectorstore.similarity_search(query, k=2)
context = "\n".join([d.page_content for d in docs])
```

---
---

## Concept 7 — Persistent Storage Backends
**Notebook: 7PersistentStorage.ipynb**

### Core Concept

`ChatMessageHistory` (in-memory) is lost when the process restarts.
LangChain provides **persistent backends** — history survives restarts, deployments, and scaling.

```
In-memory (ChatMessageHistory):
  Process restart → all history LOST

File backend (FileChatMessageHistory):
  History saved to messages.json → survives restart

SQL backend (SQLChatMessageHistory):
  History saved to SQLite/Postgres → survives restart + scales to multi-server
```

### Available Backends

| Backend | Class | Use Case |
|---------|-------|----------|
| In-memory | `ChatMessageHistory` | Dev/testing only |
| File | `FileChatMessageHistory` | Single-user apps, prototypes |
| SQLite | `SQLChatMessageHistory(conn_string="sqlite:///...")` | Local persistence |
| PostgreSQL | `SQLChatMessageHistory(conn_string="postgresql://...")` | Production, multi-server |
| Redis | `RedisChatMessageHistory` | High-traffic, low latency |

### Pattern — Same Code, Different Backend

```python
# Swap the history object — everything else stays identical
from langchain_community.chat_message_histories import (
    ChatMessageHistory,       # in-memory
    FileChatMessageHistory,   # file
    SQLChatMessageHistory,    # SQL
)

def get_session_history(session_id: str):
    # Just change this one line to switch storage backends
    return SQLChatMessageHistory(session_id, connection_string="sqlite:///chat.db")
```

---
---

## Concept 8 — LangGraph Checkpointing
**Notebook: 8LangGraphMemory.ipynb**

### Core Concept

LangGraph treats memory as **first-class agent state**. The checkpointer automatically saves
the entire state graph after every node execution — no manual save/load needed.

```
SHORT-TERM (per thread/session):
  thread_id="mohan_1" → MemorySaver auto-accumulates all messages
  Resume same thread at any time — full history restored automatically

LONG-TERM (across sessions/threads):
  InMemoryStore → agent explicitly saves user-level facts
  user_id="mohan" → facts retrieved in any future thread
```

### Why LangGraph > RunnableWithMessageHistory for Agents

```
RunnableWithMessageHistory:
  Simple chains only — one input → one output
  No branching, no loops, no tool calls
  Manual save/load wiring

LangGraph:
  Full agent graphs — nodes, edges, conditional routing
  Tool calls, retries, human-in-the-loop all handled
  State persisted automatically at every step
  Time-travel: rewind to any previous checkpoint
```

---
---

## How Each Concept Builds on the Previous

```
ChatMessageHistory + RunnableWithMessageHistory
  Limitation: buffer grows forever, no trimming
      ↓ solved by
Window Memory (trim_messages by count)
  Limitation: drops old turns entirely, important facts lost
      ↓ solved by
Summary Memory (LLM compresses old messages)
  Limitation: even recent turns compressed, lossy
      ↓ solved by
SummaryBuffer (trim recent raw + summarize old)
  Limitation: token count unpredictable by turn count
      ↓ solved by
Token Buffer (trim_messages by token count)
  Limitation: retrieves by recency only, not relevance
      ↓ solved by
VectorStore Memory (FAISS semantic retrieval)
  Limitation: lost on process restart
      ↓ solved by
Persistent Storage (File / SQL / Redis backends)
  Limitation: manual wiring, breaks in multi-agent setups
      ↓ solved by
LangGraph Checkpointing (auto-state, multi-agent ready)
```

---

## Learning Order & Status

| # | Notebook | Concept | Status |
|---|----------|---------|--------|
| 1 | `1ChatHistoryAndRWMH.ipynb` | ChatMessageHistory + RunnableWithMessageHistory | To do |
| 2 | `2WindowMemory.ipynb` | trim_messages by message count | To do |
| 3 | `3SummaryMemory.ipynb` | LLM compression into running summary | To do |
| 4 | `4SummaryBufferMemory.ipynb` | Hybrid: trim recent raw + summarize old | To do |
| 5 | `5TokenBufferMemory.ipynb` | trim_messages by token count | To do |
| 6 | `6VectorStoreMemory.ipynb` | FAISS semantic retrieval of relevant turns | To do |
| 7 | `7PersistentStorage.ipynb` | File / SQL / Redis backends for persistence | To do |
| 8 | `8LangGraphMemory.ipynb` | Auto-state checkpointing, long-term store | To do |

---

## Quick Reference Card

| Concept | Key Class/Function | One-Line Summary |
|---------|-------------------|-----------------|
| Buffer | `ChatMessageHistory` + `RunnableWithMessageHistory` | Store all, inject all, auto-managed |
| Window | `trim_messages(token_counter=len)` | Inject last K turns by count |
| Summary | LLM + `SystemMessage` | Compress all history into one sentence |
| SummaryBuffer | `trim_messages` + LLM summarize | Recent raw + old summarized |
| Token Buffer | `trim_messages(token_counter="approximate")` | Inject all that fit in token budget |
| VectorStore | `FAISS` + `similarity_search` | Inject most semantically relevant turns |
| Persistent | `FileChatMessageHistory` / `SQLChatMessageHistory` | History survives restarts |
| LangGraph | `MemorySaver` + `InMemoryStore` | Auto-state, multi-agent, cross-session |

---

## Key Note — Why the Old Classes Were Removed

The old `langchain.memory` classes (ConversationBufferMemory, ConversationSummaryMemory, etc.)
mixed two concerns into one object: **storage + retrieval strategy**.
This made them hard to compose and impossible to swap backends without rewriting logic.

Modern LangChain separates them:
```
Storage   → ChatMessageHistory / FileChatMessageHistory / SQLChatMessageHistory
Strategy  → trim_messages / manual summarization / FAISS retrieval
Injection → RunnableWithMessageHistory (wires storage + strategy into any chain)
```
