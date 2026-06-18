# Agent Concepts — Complete Learning Guide

---

## What is an Agent?

An LLM on its own can only **generate text** — it cannot take actions, look up real data, or run code.
An **Agent** wraps the LLM with tools, memory, and a decision loop so it can act on the world.

```
Without Agent:   Query → LLM → Text answer (from training data only)

With Agent:      Query → LLM decides → call tool → observe result → LLM → Grounded answer
```

**Agent = LLM + Tools + Loop + (optionally) Memory + Planning**

---

## The Key Distinction: LLM + Tools vs Agent

This is the most important concept to understand before anything else.

```
LLM + Tools (NOT an agent):
  YOU write the loop.
  LLM outputs a tool_call → your code runs the function → your code sends result back.
  If the LLM needs 3 tool calls in sequence, you handle all 3 iterations manually.

Agent:
  The FRAMEWORK manages the loop.
  You call agent.invoke(query) and get the final answer.
  The agent decides which tools to call, calls them, observes results,
  and knows when it has enough information to stop — all autonomously.
```

**Notebook 1** (`1LLMWithTools.ipynb`) demonstrates the manual approach so you understand
what agent frameworks automate for you.

---

## Common Use Case (All Notebooks)

Every notebook uses the **same 3 tools** from `AgentUtils.py` and the **same 4 test queries**.
This lets you directly compare how each agent type handles identical problems.

**Tools:**
```python
calculate(expression)   → evaluate math: '144 / 12' → '12.0'
search_docs(query)      → search LangChain/AI documentation
get_weather(city)       → get current weather for a city
```

**Test Queries:**
```
Q1: 'What is 144 divided by 12?'                        → uses calculate
Q2: 'Search docs for: what is a LangChain agent?'       → uses search_docs
Q3: 'What is the weather in Hyderabad?'                 → uses get_weather
Q4: 'Search docs for RAG and also calculate 25 * 4'     → uses multiple tools
```

---

## Notebook Map

| Notebook | Type | Framework | Key Learning |
|----------|------|-----------|-------------|
| `1LLMWithTools.ipynb` | **Foundation** | Manual (no agent) | bind_tools, YOU run the loop |
| `2ReActAgent.ipynb` | Agent | `create_react_agent` + `AgentExecutor` | Thought/Action/Observation |
| `3ToolCallingAgent.ipynb` | Agent | `create_agent` | Autonomous loop, native tool calling |
| `4StructuredOutputAgent.ipynb` | Agent | `create_agent` + Pydantic | Typed output schema |
| `5ConversationalAgent.ipynb` | Agent | `create_agent` + history | Multi-turn memory |
| `6RAGAgent.ipynb` | Agent | `create_agent` + FAISS | Vectorstore as a tool |
| `MultiAgent_Router.ipynb` | Agent | `create_agent` × N | Router + specialized sub-agents |
| `8PlanAndExecuteAgent.ipynb` | Agent | `create_agent` executor | Plan first, execute step by step |
| `9SelfCorrectingAgent.ipynb` | Agent | `create_agent` + critic | Auto-retry on failure |

---
---

## Foundation — LLM + Tool Binding (NOT an Agent)
**Notebook: 1LLMWithTools.ipynb**

### Core Concept

`bind_tools` registers tool definitions with the LLM. When the LLM decides to use a tool,
it outputs a **structured `tool_calls` object** — it does NOT execute the tool.
YOUR code reads the object, runs the function, and sends the result back.

```python
llm_with_tools = llm.bind_tools(tools)

response = llm_with_tools.invoke('What is 144 / 12?')
response.content     # '' ← empty, LLM chose a tool
response.tool_calls  # [{'name': 'calculate', 'args': {'expression': '144/12'}, 'id': '...'}]

# YOUR code executes:
result = calculate.invoke({'expression': '144/12'})   # → '12.0'
messages.append(ToolMessage(content=result, tool_call_id=...))

# YOUR code calls LLM again:
final = llm_with_tools.invoke(messages)
final.content  # '144 divided by 12 is 12.'
```

### Why this matters
Understanding this pattern shows you exactly what agent frameworks automate.
When you see `create_agent` in later notebooks, you know what it's doing internally.

### What the next concept fixes
You write the loop for every query. For multi-step tasks this becomes complex fast.
→ Concept 2 (ReAct Agent) introduces the first real agent framework that manages the loop.

---
---

## Concept 2 — ReAct Agent
**Notebook: 2ReActAgent.ipynb**

### Core Concept

ReAct = **Re**asoning + **Act**ing. The LLM writes an explicit **Thought** before every Action.
`AgentExecutor` manages the Thought/Action/Observation loop — you don't write it.

```
Query
  → Thought: I need to calculate 144/12. I'll use the calculate tool.
  → Action: calculate
  → Action Input: 144/12
  → Observation: 12.0
  → Thought: I now know the final answer.
  → Final Answer: 144 divided by 12 is 12.
```

### How it differs from LLM + Tools
```
LLM + Tools:  tool decisions are opaque — you see what was called but not why
ReAct Agent:  Thought makes every decision transparent and debuggable
```

### Key components
```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

# ReAct requires a specific prompt with {tools}, {tool_names}, {input}, {agent_scratchpad}
prompt   = PromptTemplate.from_template(react_template)
agent    = create_react_agent(llm=llm, tools=TOOLS, prompt=prompt)
executor = AgentExecutor(agent=agent, tools=TOOLS, verbose=True, max_iterations=6)

result = executor.invoke({'input': query})   # framework manages the entire loop
```

### What the next concept fixes
The text-format parsing of Thought/Action/Observation is fragile — if the LLM skips a
line or adds extra text, parsing fails. Also requires a custom prompt template.
→ Concept 3 uses `create_agent` with native tool calling — no text parsing needed.

---
---

## Concept 3 — Tool Calling Agent (create_agent)
**Notebook: 3ToolCallingAgent.ipynb**

### Core Concept

`create_agent` is the modern LangChain agent. It uses the LLM's **native function calling API**
instead of text-format parsing. No custom prompt template needed. More robust than ReAct.

```python
from langchain.agents import create_agent

agent = create_agent(model=llm, tools=TOOLS, system_prompt='You are a helpful assistant.')

result = agent.invoke({'messages': [{'role': 'user', 'content': query}]})
answer = result['messages'][-1].content
```

### create_agent vs create_react_agent
| | `create_react_agent` | `create_agent` |
|---|---|---|
| Output parsing | Text-format (fragile) | Native tool calls (robust) |
| Prompt needed | Yes — specific ReAct template | No — just system prompt |
| Reasoning visible | Yes — Thought before each step | No — decisions are internal |
| Token use | Higher (Thoughts add tokens) | Lower |
| Best for | Debugging, education | Production, simplicity |

### Internal trace
```python
result['messages']  # shows every step the agent took:
  HumanMessage      → the query
  AIMessage         → tool call decision
  ToolMessage       → tool result
  AIMessage         → final answer
```

### What the next concept fixes
Output is a plain string. Downstream code can't reliably extract specific fields.
→ Concept 4 forces the agent to return a typed Pydantic object.

---
---

## Concept 4 — Structured Output Agent
**Notebook: 4StructuredOutputAgent.ipynb**

### Core Concept

Two-stage pattern: `create_agent` runs tools autonomously (Stage 1), then a second LLM call
with `with_structured_output` reformats the answer into a Pydantic schema (Stage 2).

```python
class AgentResponse(BaseModel):
    answer:     str
    tools_used: List[str]
    confidence: str   # 'high', 'medium', 'low'
    reasoning:  str

structured_llm = llm.with_structured_output(AgentResponse)
```

### Why it matters
```
String output:     "The weather in Hyderabad is 34°C and Sunny."
  → need regex/parsing to extract temperature

Pydantic output:
  response.answer     = "The weather in Hyderabad is 34°C and Sunny."
  response.tools_used = ["get_weather"]
  response.confidence = "high"
  → fields accessed directly, usable in APIs, databases, pipelines
```

### What the next concept fixes
Still stateless — every call starts fresh with no memory of prior turns.
→ Concept 5 passes full message history to the agent on every turn.

---
---

## Concept 5 — Conversational Agent
**Notebook: 5ConversationalAgent.ipynb**

### Core Concept

Maintain a `history` list of messages. Pass the full history to `create_agent` on every call.
The agent sees all prior Human + AI messages and can reference earlier answers.

```python
history = []

def chat(query):
    history.append({'role': 'user', 'content': query})
    result  = agent.invoke({'messages': history})
    answer  = result['messages'][-1].content
    history.append({'role': 'assistant', 'content': answer})
    return answer

chat('What is 144 / 12?')         # → "12"
chat('Multiply that result by 5.') # → "60"  ← remembers 12
chat('Summarize what we did.')     # → full summary
```

### Memory strategies
```
Full history:   pass ALL messages — perfect recall, context grows with session
Window (last N): pass only last N messages — bounded cost, forgets older turns
```

### What the next concept fixes
Tools are static (hardcoded strings/math). Cannot search real documents dynamically.
→ Concept 6 wraps a FAISS vectorstore as a tool the agent can call when needed.

---
---

## Concept 6 — RAG Agent
**Notebook: 6RAGAgent.ipynb**

### Core Concept

A `create_agent` instance where one tool is a FAISS vectorstore retriever.
The agent decides *when* to retrieve — only searching when the query actually needs it.

```python
@tool
def search_docs_rag(query: str) -> str:
    """Search documentation for LangChain and AI concepts."""
    docs = retriever.invoke(query)
    return '\n\n'.join(doc.page_content for doc in docs)

agent = create_agent(model=llm, tools=[search_docs_rag, calculate, get_weather], ...)
```

### Fixed RAG Pipeline vs RAG Agent
```
Fixed RAG Pipeline (always retrieves):
  Query → Retrieve → LLM → Answer
  Even 'What is 2+2?' retrieves docs — wasteful.

RAG Agent (agent decides when to retrieve):
  'What is 2+2?'  → calls calculate only
  'What is RAG?'  → calls search_docs_rag only
  'Explain + calc' → calls both
  Smarter, more efficient.
```

### What the next concept fixes
One agent handles all task types — no specialization per domain.
→ Concept 7 routes queries to specialized sub-agents, each tuned for one job.

---
---

## Concept 7 — Multi-Agent (Router Pattern)
**Notebook: MultiAgent_Router.ipynb**

### Core Concept

Three specialized `create_agent` instances, each with a focused system prompt and only
the tools for its domain. A router (LLM call) classifies each query and picks the right agent.

```python
math_agent    = create_agent(model=llm, tools=[calculate],   system_prompt='Math specialist...')
docs_agent    = create_agent(model=llm, tools=[search_docs], system_prompt='Docs specialist...')
weather_agent = create_agent(model=llm, tools=[get_weather], system_prompt='Weather reporter...')

def route(query): ...  # LLM classifies: 'math' | 'docs' | 'weather'

def multi_agent(query):
    selected = agent_registry[route(query)]
    return selected.invoke({'messages': [{'role': 'user', 'content': query}]})
```

### Router options
```
LLM Router:     one LLM call to classify → flexible, handles ambiguity
Keyword Router: instant pattern match → free, no LLM call, less flexible
```

### What the next concept fixes
Still reactive — no upfront plan for complex multi-step tasks.
→ Concept 8 generates a full plan before executing any step.

---
---

## Concept 8 — Plan-and-Execute Agent
**Notebook: 8PlanAndExecuteAgent.ipynb**

### Core Concept

Three roles working in sequence:
1. **Planner** — LLM chain generates a numbered step-by-step plan
2. **Executor** — `create_agent` executes each step (with prior results as context)
3. **Synthesizer** — LLM chain combines all step results into the final answer

```
Query: 'Search docs for RAG and calculate 25 * 4'

Planner →
  Step 1: Call search_docs to find what RAG means
  Step 2: Call calculate to compute 25 * 4

Executor (create_agent) →
  Step 1 → "RAG combines retrieval with LLM..."
  Step 2 → "100"

Synthesizer →
  "RAG is... and 25 × 4 = 100."
```

### vs ReAct
```
ReAct:            one step at a time, doesn't know what's coming next
Plan-and-Execute: full plan before any action, each step sees prior results
```

### What the next concept fixes
No quality check — the final answer isn't validated.
→ Concept 9 adds a critic that evaluates the answer and retries if it fails.

---
---

## Concept 9 — Self-Correcting Agent
**Notebook: 9SelfCorrectingAgent.ipynb**

### Core Concept

A `create_agent` wrapped in a critic-retry loop. After every answer, a separate Critic LLM
evaluates quality. If it fails, the agent retries with the critic's feedback — automatically.

```
Query → create_agent → Answer (attempt 1)
  → Critic: PASS → return immediately
  → Critic: FAIL 'missing X' → retry with feedback (attempt 2)
  → Critic: PASS → return
  → [max 3 attempts]
```

### Why a separate critic?
The agent LLM tends to think its own answers are correct.
A separate Critic LLM is more objective — it evaluates without bias.

### The critic prompt structure
```python
critic_prompt = """
Evaluate whether this answer fully addresses the query.
Criteria: all parts answered, tools used correctly, specific not vague.
Reply: VERDICT: PASS  or  VERDICT: FAIL\nREASON: ...
"""
```

---

## How Each Concept Builds on the Previous

```
Foundation: LLM + Tool Binding
  Problem: YOU write the loop manually
      ↓ Concept 2 (ReAct) adds
  Framework-managed loop + visible Thought before each Action
      ↓ Concept 3 (Tool Calling) replaces text parsing with
  Native tool calling — simpler, more robust
      ↓ Concept 4 (Structured Output) adds
  Typed Pydantic output instead of plain string
      ↓ Concept 5 (Conversational) adds
  Message history — agent remembers across turns
      ↓ Concept 6 (RAG) adds
  Vectorstore as a tool — dynamic document retrieval
      ↓ Concept 7 (Multi-Agent) adds
  Specialization — separate agents per task type
      ↓ Concept 8 (Plan-and-Execute) adds
  Upfront planning before any execution
      ↓ Concept 9 (Self-Correcting) adds
  Quality validation + automatic retry
```

---

## Quick Reference

| Notebook | One-line summary |
|----------|-----------------|
| 1LLMWithTools | `bind_tools` + manual loop — YOU are the agent |
| 2ReActAgent | `create_react_agent` — Thought/Action/Observation, transparent |
| 3ToolCallingAgent | `create_agent` — autonomous, native tool calling, simplest agent |
| 4StructuredOutputAgent | `create_agent` + Pydantic schema — typed output |
| 5ConversationalAgent | `create_agent` + message history — remembers across turns |
| 6RAGAgent | `create_agent` + FAISS — agent decides when to retrieve |
| MultiAgent_Router | Router + specialized `create_agent` per domain |
| 8PlanAndExecuteAgent | Planner → `create_agent` executor → Synthesizer |
| 9SelfCorrectingAgent | `create_agent` + Critic LLM + retry loop |

---

## Key Imports Reference

```python
# Foundation
from langchain_core.messages import HumanMessage, ToolMessage
llm_with_tools = llm.bind_tools(tools)

# ReAct Agent
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

# Tool Calling Agent and all others
from langchain.agents import create_agent

# Structured output
from pydantic import BaseModel, Field
structured_llm = llm.with_structured_output(MySchema)

# RAG tools
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
```
