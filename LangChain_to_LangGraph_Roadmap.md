# LangChain → LangGraph Learning Roadmap

> **Goal:** Learn LangChain concepts in the right order, then transition to LangGraph.
> Check off each item as you complete the notebook.

---

## Phase 1 — LangChain Foundations ✅ (Already Done)

These are complete. They form the base everything else builds on.

| # | Notebook | Concept | Status |
|---|----------|---------|--------|
| RAG-1 | `notebooks/RAG/1simpleRAG.ipynb` | Load → Split → Embed → Retrieve → Generate | ✅ Done |
| RAG-2 | `notebooks/RAG/2simpleRAG_Retriver.ipynb` | LCEL chains, Retriever abstraction, MMR | ✅ Done |
| RAG-3 | `notebooks/RAG/3AdvancedRAG.ipynb` | Reranking, two-stage retrieval | ✅ Done |
| RAG-4 | `notebooks/RAG/4HybridRAG_AgeneticRAG.ipynb` | Vector + BM25 hybrid retrieval | ✅ Done |
| RAG-5 | `notebooks/RAG/4HybridRAG_AgeneticRAG.ipynb` | Agentic RAG, self-correction loop | ✅ Done |

---

## Phase 2 — LangChain Essentials (Do These Before LangGraph)

These are the **critical** topics. LangGraph is built on top of all of these concepts.
Complete them in order — each one is a prerequisite for LangGraph.

---

### 2A — Messages & Prompts
> **Why:** LangGraph state IS a list of messages. You must know message types before touching LangGraph.

| # | Notebook | Concept | Status |
|---|----------|---------|--------|
| PM-1 | `notebooks/Prompt_Message/1BasicPromptTemplate.ipynb` | PromptTemplate, `{variable}` substitution | ☐ |
| PM-2 | `notebooks/Prompt_Message/2ChatPromptTemplate.ipynb` | SystemMessage, HumanMessage, AIMessage, ToolMessage | ☐ |
| PM-3 | `notebooks/Prompt_Message/3FewShotPrompt.ipynb` | Few-shot examples, pattern learning | ☐ |
| PM-4 | `notebooks/Prompt_Message/4ConversationHistory.ipynb` | MessagesPlaceholder, multi-turn history | ☐ |
| PM-5 | `notebooks/Prompt_Message/5StructuredOutput.ipynb` | Pydantic schemas, JSON output, output parsers | ☐ |
| PM-6 | `notebooks/Prompt_Message/6ToolCalling.ipynb` | `@tool`, `bind_tools`, ToolMessage loop | ☐ |

---

### 2B — Agents
> **Why:** LangGraph IS the upgraded replacement for LangChain agents. Understand what it replaces.

| # | Notebook | Concept | Status |
|---|----------|---------|--------|
| AG-1 | `notebooks/Agents/1LLMWithTools.ipynb` | LLM + tools, basic tool loop | ☐ |
| AG-2 | `notebooks/Agents/2ReActAgent.ipynb` | Thought → Action → Observation loop | ☐ |
| AG-3 | `notebooks/Agents/3ToolCallingAgent.ipynb` | Tool calling agent (modern approach) | ☐ |
| AG-4 | `notebooks/Agents/4StructuredOutputAgent.ipynb` | Agents with structured output | ☐ |
| AG-5 | `notebooks/Agents/5ConversationalAgent.ipynb` | Multi-turn agent with memory | ☐ |
| AG-6 | `notebooks/Agents/6RAGAgent.ipynb` | Agent + RAG combined | ☐ |
| AG-7 | `notebooks/Agents/7MultiAgent_Router.ipynb` | Router between multiple agents | ☐ |
| AG-8 | `notebooks/Agents/8PlanAndExecuteAgent.ipynb` | Plan-and-execute pattern | ☐ |
| AG-9 | `notebooks/Agents/9SelfCorrectingAgent.ipynb` | Self-correction, retry loops | ☐ |

---

### 2C — Memory (The Problem LangGraph Solves)
> **Why:** Do Memory 1–3 to understand the problem. Then when LangGraph replaces all of them, it makes sense.

| # | Notebook | Concept | Status |
|---|----------|---------|--------|
| MEM-1 | `notebooks/Memory/1ChatHistoryAndRWMH.ipynb` | ChatMessageHistory + RunnableWithMessageHistory | ☐ |
| MEM-2 | `notebooks/Memory/2WindowMemory.ipynb` | Sliding window via trim_messages (count) | ☐ |
| MEM-3 | `notebooks/Memory/3SummaryMemory.ipynb` | LLM compresses history into running summary | ☐ |
| MEM-4 | `notebooks/Memory/4SummaryBufferMemory.ipynb` | Hybrid: recent=raw, old=summarized | ☐ |
| MEM-5 | `notebooks/Memory/5TokenBufferMemory.ipynb` | Token-budget memory via trim_messages | ☐ |
| MEM-6 | `notebooks/Memory/6VectorStoreMemory.ipynb` | Semantic retrieval of past turns (FAISS) | ☐ |
| MEM-7 | `notebooks/Memory/7PersistentStorage.ipynb` | File + SQLite backends (survive restarts) | ☐ |

---

## Phase 3 — LangGraph (The Destination)

You are ready for LangGraph after completing Phases 1 and 2.

> **What LangGraph adds over LangChain agents:**
> - Stateful graph execution (nodes + edges instead of chains)
> - Built-in memory via Checkpointers (replaces all Memory classes)
> - Conditional branching, loops, human-in-the-loop
> - Multi-agent coordination with shared state
> - Time-travel: rewind to any checkpoint and replay

### Suggested LangGraph Learning Order

| # | Topic | Key Concept |
|---|-------|-------------|
| LG-1 | StateGraph basics | Nodes, edges, state, `compile()` |
| LG-2 | MemorySaver (short-term) | Checkpointer, thread_id, session persistence |
| LG-3 | Conditional edges | `add_conditional_edges`, routing logic |
| LG-4 | Tool-calling agent in LangGraph | LLM node + tool node + loop |
| LG-5 | Human-in-the-loop | `interrupt_before`, approvals, resume |
| LG-6 | InMemoryStore (long-term) | Cross-session user memory |
| LG-7 | Multi-agent with LangGraph | Supervisor, subgraphs, handoff |
| LG-8 | Production backends | SqliteSaver, PostgresSaver |

---

## Phase 4 — Advanced LangChain (Do After or Alongside LangGraph)

These are not prerequisites for LangGraph. Come back to them to deepen RAG and prompt skills.

| # | Notebook | Concept |
|---|----------|---------|
| RAG-6 | `notebooks/RAG/5RAGFusion.ipynb` | Multi-query generation, RRF merging |
| RAG-7 | `notebooks/RAG/6HyDE.ipynb` | Hypothetical document embeddings |
| RAG-8 | `notebooks/RAG/7ParentDocumentRetriever.ipynb` | Precision search + full context |
| RAG-9 | `notebooks/RAG/8ContextualCompression.ipynb` | Strip noise from retrieved docs |
| RAG-10 | `notebooks/RAG/9CRAG.ipynb` | Doc quality evaluation, web fallback |
| RAG-11 | `notebooks/RAG/10SelfRAG.ipynb` | LLM controls the full pipeline |
| RAG-12 | `notebooks/RAG/11MultiHopRAG.ipynb` | Chained multi-step reasoning |
| MEM-8 | `notebooks/Memory/8LangGraphMemory.ipynb` | LangGraph memory (bridges to Phase 3) |
| PM-7 | `notebooks/Prompt_Message/7ReActPrompt.ipynb` | ReAct prompt engineering |
| PM-8 | `notebooks/Prompt_Message/8PartialPrompts.ipynb` | Pre-fill variables, dynamic templates |
| PM-9 | `notebooks/Prompt_Message/9PipelinePrompts.ipynb` | Modular prompt composition |
| PM-10 | `notebooks/Prompt_Message/10PromptOptimization.ipynb` | Patterns, token efficiency, temperature |

---

## Summary — Priority Order

```
Phase 1 (Done)
  RAG 1–5 ✅

Phase 2 (Do next — in this order)
  Prompt/Message 1–6   →   Agents 1–9   →   Memory 1–7

Phase 3 (LangGraph — after Phase 2)
  LG-1 through LG-8

Phase 4 (Deepen — anytime after Phase 2)
  RAG 6–12, Memory 8, Prompt/Message 7–10
```

---

## Key Insight: Why This Order

```
Prompt/Message 1–6  ←  LangGraph state = list of messages
                        You MUST know HumanMessage, AIMessage, ToolMessage

Agents 1–9          ←  LangGraph replaces LangChain agents
                        Understand what you're upgrading from

Memory 1–7          ←  LangGraph replaces all memory classes
                        Understand the problem before seeing the solution
```
