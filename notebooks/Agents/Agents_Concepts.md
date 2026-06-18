# Agent Concepts — Complete Learning Guide

---

## What is an Agent?

An LLM on its own can only **generate text** — it cannot take actions, look up real data, or run code.
An **Agent** wraps the LLM with tools, memory, and a decision loop so it can act on the world.

**The Problem:**
```
User:  "What is the weather in Hyderabad today?"
LLM:   "I don't have access to real-time weather data."  ← can't act

Agent: → calls get_weather("Hyderabad") → "34°C, Sunny"
       → "It is currently 34°C and sunny in Hyderabad."  ← can act
```

**Agents solve this** by giving the LLM the ability to decide which tool to call, call it,
observe the result, and use that result in its answer.

```
Without Agent:   Query → LLM → Text answer (from training data only)

With Agent:      Query → LLM decides → Tool → Observation → LLM → Grounded answer
```

**Agent = LLM + Tools + Loop + (optionally) Memory + Planning**

---

## The Core Agent Loop

Every agent — no matter how advanced — runs some version of this loop:

```
AGENT LOOP:
  1. Receive user query
  2. LLM thinks: "What do I need to do?"
  3. LLM decides: call a tool OR answer directly
  4. If tool called: execute it → observe result
  5. LLM thinks again with new information
  6. Repeat until LLM has enough to answer
  7. Return final answer to user
```

The differences between agent types are **how much** the LLM reasons, plans, remembers,
and validates — not the fundamental loop.

---

## Common Use Case (All Notebooks)

Every notebook uses the **same 3 tools** and **same 4 test queries** — just like the RAG notebooks
use the same data source. This lets you see exactly how each agent type handles identical problems.

**Tools:**
```
calculate(expression)  → evaluate math: '144 / 12' → '12.0'
search_docs(query)     → search LangChain/AI documentation
get_weather(city)      → get current weather for a city
```

**Test Queries:**
```
Q1: 'What is 144 divided by 12?'                         → uses calculate
Q2: 'Search docs for: what is a LangChain agent?'        → uses search_docs
Q3: 'What is the weather in Hyderabad?'                  → uses get_weather
Q4: 'Search docs for RAG and also calculate 25 * 4'      → uses multiple tools
```

---
---

## Concept 1 — Tool Calling Agent
**Notebook: 1ToolCallingAgent.ipynb**

### Core Concept

The simplest agent pattern. The LLM is given a list of tools and decides which one to call.
There is no explicit reasoning trace — just: query → tool decision → execute → answer.

```
Query
  ↓
LLM: "I need to call calculate('144 / 12')"
  ↓
Your code executes: calculate('144 / 12') → '12.0'
  ↓
ToolMessage sent back to LLM
  ↓
LLM: "144 divided by 12 is 12."
```

### How Tool Calling Works

The LLM does not directly execute tools — it **outputs a structured tool call** that your code runs:

```python
response = llm_with_tools.invoke("What is 144 / 12?")

# LLM output is NOT "12" — it's a structured call:
response.tool_calls → [{'name': 'calculate', 'args': {'expression': '144 / 12'}, 'id': '...'}]

# Your code executes the tool and sends result back as ToolMessage
# LLM then forms the final text answer
```

### What You Actually Build

```python
llm_with_tools = llm.bind_tools(tools)

def tool_calling_agent(query):
    messages = [HumanMessage(content=query)]
    response = llm_with_tools.invoke(messages)

    if response.tool_calls:
        # Execute each tool
        for call in response.tool_calls:
            result = tool_map[call['name']].invoke(call['args'])
            messages.append(ToolMessage(content=result, tool_call_id=call['id']))
        # LLM forms final answer from tool results
        final = llm_with_tools.invoke(messages)
        return final.content
    return response.content  # answered directly
```

### Parallel Tool Calls

The LLM can call multiple tools in a single response:

```
Query: "Search docs for RAG and calculate 25 * 4"
  ↓
LLM output:
  tool_call_1: search_docs(query="RAG")
  tool_call_2: calculate(expression="25 * 4")
  ↓
Both executed in one round → ToolMessage × 2 → LLM → final answer
```

### Limitations of Tool Calling Agent

| Limitation | Explanation |
|------------|-------------|
| **No reasoning trace** | You see the decision but not *why* the LLM chose that tool |
| **One round of tool calls** | Cannot chain tools where output of one feeds into another |
| **No memory** | Each call is independent — LLM forgets previous turns |
| **No quality check** | LLM doesn't verify if its answer is actually correct |
| **Brittle on ambiguity** | Ambiguous queries may trigger the wrong tool silently |

### What the Next Concept Fixes

Concept 2 fixes the **no reasoning trace** problem — ReAct makes the LLM write its thoughts
before each action, making it transparent and debuggable.

---
---

## Concept 2 — ReAct Agent (Reason + Act)
**Notebook: 2ReActAgent.ipynb**

### Core Concept

ReAct = **Re**asoning + **Act**ing. The LLM explicitly writes its thoughts before every action.
This creates a transparent Thought → Action → Observation loop you can read and debug.

```
Query: "Search docs for RAG and calculate 25 * 4"
  ↓
Thought: I need to do two things: search docs for RAG and calculate 25*4. Let me start with the search.
Action: search_docs
Action Input: "RAG"
Observation: "RAG combines retrieval with LLM..."
  ↓
Thought: Good. Now I need to calculate 25*4.
Action: calculate
Action Input: "25 * 4"
Observation: "100"
  ↓
Thought: I now have both answers.
Final Answer: RAG is... and 25 * 4 = 100.
```

### Why Thought Matters

The Thought step forces the LLM to plan before acting. Better Thoughts → better Actions:

```
Bad Thought:  "I need to find information."
              → vague → may call wrong tool

Good Thought: "The user wants LangChain agent docs. I'll search for 'agent' specifically."
              → specific → calls right tool with right args
```

### The ReAct Prompt Structure

```
You have access to tools: {tools}

Use this format:
  Question: ...
  Thought: ...
  Action: tool_name
  Action Input: ...
  Observation: (filled by tool result)
  ... (repeat)
  Thought: I now know the final answer
  Final Answer: ...
```

The LLM follows this format strictly — your code parses Action/Action Input,
calls the tool, and injects the Observation.

### Limitations of ReAct Agent

| Limitation | Explanation |
|------------|-------------|
| **Verbose** | Thought/Action/Observation adds many tokens per step |
| **Can loop** | Without `max_iterations`, may get stuck in a reasoning loop |
| **Text parsing fragile** | If LLM deviates from format, parsing fails |
| **Returns free text** | Output is a string — hard to use programmatically |
| **No memory across calls** | Each invocation starts fresh |

### What the Next Concept Fixes

Concept 3 fixes the **returns free text** problem — Structured Output Agent forces the
LLM to return a typed Pydantic object instead of a string.

---
---

## Concept 3 — Structured Output Agent
**Notebook: 3StructuredOutputAgent.ipynb**

### Core Concept

Force the agent to return a **typed Pydantic object** instead of free text.
This makes agent output directly usable in code — no string parsing needed.

```
Without Structured Output:
  Answer: "The weather in Hyderabad is 34°C and Sunny, and 144/12 equals 12."
  → string, need regex to extract values

With Structured Output:
  AgentResponse(
      answer="The weather in Hyderabad is 34°C and Sunny, and 144/12 equals 12.",
      tools_used=["get_weather", "calculate"],
      confidence="high"
  )
  → directly access .answer, .tools_used, .confidence
```

### Two-Stage Pattern

```
Stage 1 — Tool Execution (tool-calling LLM):
  Query → call tools → collect results

Stage 2 — Structured Synthesis (structured-output LLM):
  Tool results + query → force LLM to fill Pydantic schema
```

```python
class AgentResponse(BaseModel):
    answer:     str
    tools_used: List[str]
    confidence: str  # high / medium / low

structured_llm = llm.with_structured_output(AgentResponse)
```

### When to Use

```
Downstream code needs specific fields:
  pipeline.process(agent_output.answer)    ← clean
  pipeline.process(parse(agent_text))      ← fragile

Logging structured data:
  db.insert({'answer': r.answer, 'tools': r.tools_used})  ← clean
```

### Limitations of Structured Output Agent

| Limitation | Explanation |
|------------|-------------|
| **Extra LLM call** | Two-stage means 2× LLM calls per query |
| **Schema may be over-constrained** | LLM may struggle with complex nested schemas |
| **No memory** | Still stateless — each call is independent |
| **LLM may hallucinate fields** | Schema enforces structure, not factual accuracy |

### What the Next Concept Fixes

Concept 4 fixes the **no memory** problem — Conversational Agent remembers past turns.

---
---

## Concept 4 — Conversational Agent
**Notebook: 4ConversationalAgent.ipynb**

### Core Concept

A conversational agent maintains **full message history** across turns.
The LLM sees all past Human + AI messages on every call — it can reference,
correct, and build on what was said before.

```
Turn 1: "What is 144 / 12?"
  → AI: "144 / 12 = 12."
  → history: [HumanMessage, AIMessage]

Turn 2: "Now multiply that result by 5."
  → AI sees full history → knows "that result" = 12
  → calls calculate("12 * 5") → 60
  → "12 × 5 = 60."

Turn 3: "Summarize what we calculated."
  → AI sees both turns → gives full summary
```

### Without vs With Memory

```
Without memory:
  Q1: "My name is Mohan."  → "Hello Mohan!"
  Q2: "What is my name?"   → "I don't know."  ← forgets

With memory:
  Q1: "My name is Mohan."  → "Hello Mohan!"
  Q2: "What is my name?"   → "Your name is Mohan."  ← remembers
```

### Memory Strategies

```
Full history:
  Pass ALL past messages → perfect recall, context grows unbounded

Window (last N turns):
  Pass only last N messages → bounded cost, forgets older turns

Summary:
  Summarize old messages into one SystemMessage → compact, loses detail
```

### Limitations of Conversational Agent

| Limitation | Explanation |
|------------|-------------|
| **Growing context cost** | Every turn adds tokens — long sessions get expensive |
| **No external knowledge** | Can only use tools — can't search real documents |
| **Session isolation** | Memory is in-process — lost when app restarts |
| **No self-correction** | If it answers wrong, it won't retry without being told |

### What the Next Concept Fixes

Concept 5 fixes the **no external knowledge** problem — RAG Agent dynamically
retrieves from a real document store as a tool.

---
---

## Concept 5 — RAG Agent
**Notebook: 5RAGAgent.ipynb**

### Core Concept

A RAG Agent wraps a full RAG pipeline (vectorstore + LLM) as a **tool**.
The agent decides *when* to retrieve — it doesn't always search, only when needed.
This is smarter than a fixed RAG pipeline that always retrieves.

```
Query: "What is 144 / 12?"
  ↓
Agent thinks: "This is math — I need calculate, not search."
  → calls calculate('144 / 12') → 12

Query: "Explain what an embedding is"
  ↓
Agent thinks: "This needs documentation — I'll search_docs."
  → calls search_docs("embedding") → definition from vectorstore
  → answers using retrieved text
```

### RAG Pipeline vs RAG Agent

```
RAG Pipeline (always retrieves):
  Query → Retrieve → LLM → Answer
  Even "What is 2+2?" goes through retrieval — wasteful

RAG Agent (decides whether to retrieve):
  Query → Agent decides → maybe retrieve, maybe calculate, maybe both
  More efficient, more flexible
```

### Wrapping RAG as a Tool

```python
vectorstore = FAISS.from_documents(docs, embeddings)
retriever   = vectorstore.as_retriever(search_kwargs={'k': 3})

@tool
def search_docs_rag(query: str) -> str:
    """Search the documentation for detailed information."""
    docs = retriever.invoke(query)
    return '\n\n'.join(doc.page_content for doc in docs)

# Agent now has real vectorstore search alongside calculate and weather
agent = create_agent(model=llm, tools=[search_docs_rag, calculate, get_weather])
```

### Limitations of RAG Agent

| Limitation | Explanation |
|------------|-------------|
| **Retrieval quality limits answers** | Bad chunks → bad answers, even with smart agent |
| **Single knowledge source** | One vectorstore — can't dynamically pick the right source |
| **One specialist at a time** | Agent handles everything — no specialization |
| **Context window** | Many retrieved docs + conversation history may overflow context |

### What the Next Concept Fixes

Concept 6 fixes the **no specialization** problem — Multi-Agent routes queries
to specialized sub-agents, each optimized for a different task.

---
---

## Concept 6 — Multi-Agent (Router Pattern)
**Notebook: 6MultiAgent.ipynb**

### Core Concept

No single agent is best at everything. Multi-Agent creates **specialized sub-agents**
and a **router** that decides which specialist to call for each query.

```
Query: "What is 144 / 12?"
  ↓
Router: "This is math." → Math Agent
  → Math Agent calls calculate() → 12

Query: "What is a LangChain agent?"
  ↓
Router: "This is documentation." → Docs Agent
  → Docs Agent calls search_docs() → explanation

Query: "What is the weather in Delhi?"
  ↓
Router: "This is weather." → Weather Agent
  → Weather Agent calls get_weather() → 27°C
```

### Why Specialization Helps

```
One general agent (poor):
  System prompt tries to handle everything
  Tools conflict — agent may call wrong one
  Hard to optimize for one task type

Specialized agents (better):
  Math Agent: system prompt = "You are a math expert. Only do math."
  Docs Agent: system prompt = "You are a documentation specialist."
  Weather Agent: system prompt = "You answer weather queries."
  Each agent is tuned for one job → higher accuracy
```

### Router Patterns

```
LLM Router (flexible):
  router_llm.invoke("Is this math, docs, or weather? Question: ...")
  → returns "math" / "docs" / "weather"
  Pros: handles ambiguity     Cons: extra LLM call, can misroute

Keyword Router (fast):
  if "weather" in query → weather_agent
  if any(op in query) → math_agent
  else → docs_agent
  Pros: instant, free         Cons: brittle on unusual phrasing

Embedding Router (smart):
  Embed query → compare to agent descriptions in vector space
  Pros: semantic matching     Cons: requires embedding call
```

### Limitations of Multi-Agent

| Limitation | Explanation |
|------------|-------------|
| **Router errors cascade** | Wrong routing → wrong agent → wrong answer |
| **Hard to handle hybrid queries** | "Do math AND search docs" — needs both agents |
| **No planning for multi-step** | Still reactive — no upfront planning for complex tasks |
| **Independent agents can't collaborate** | Agents work in isolation; one can't hand off to another |

### What the Next Concept Fixes

Concept 7 fixes the **no planning for multi-step** problem — Plan-and-Execute
generates a full plan first, then executes each step in order.

---
---

## Concept 7 — Plan-and-Execute Agent
**Notebook: 7PlanAndExecuteAgent.ipynb**

### Core Concept

For complex queries that require multiple ordered steps, ReAct can get confused.
Plan-and-Execute splits the work into two phases:

**Phase 1 — Planner:** Generate a complete step-by-step plan upfront.
**Phase 2 — Executor:** Execute each step, passing results forward.

```
Query: "Search docs for RAG, then calculate the number of words in the definition,
        then get weather in Hyderabad"

Phase 1 — Plan:
  Step 1: Search docs for 'RAG' to get the definition
  Step 2: Count the words in the definition from Step 1
  Step 3: Get weather in Hyderabad

Phase 2 — Execute:
  Step 1 → "RAG combines retrieval with LLM..." (40 words)
  Step 2 → calculate word count from Step 1 result → 40
  Step 3 → get_weather('Hyderabad') → 34°C, Sunny

Synthesize: RAG definition has 40 words. Weather in Hyderabad: 34°C, Sunny.
```

### Plan-and-Execute vs ReAct

```
ReAct (reactive):
  One step at a time — doesn't know what's coming next
  Good for: simple 1-2 step tasks
  Risk: can wander, lose track of the original goal

Plan-and-Execute (proactive):
  Full plan before any action
  Each step knows what previous steps produced
  Good for: multi-step tasks with dependencies
  Risk: bad plan → bad execution (no mid-plan correction)
```

### Three Roles

```
Planner LLM:
  Input:  user query
  Output: ["Step 1: ...", "Step 2: ...", "Step 3: ..."]

Executor Agent:
  Input:  one step + results from previous steps
  Output: result of this step (tool call or text)

Synthesizer LLM:
  Input:  all step results
  Output: final coherent answer to the original query
```

### Limitations of Plan-and-Execute

| Limitation | Explanation |
|------------|-------------|
| **Bad plan = bad result** | If planner misunderstands, all steps are wrong |
| **No mid-plan correction** | Plan is fixed upfront — can't adapt if step 2 fails |
| **More LLM calls** | Planner + executor × N + synthesizer = expensive |
| **No quality validation** | Doesn't check if the final answer is actually correct |

### What the Next Concept Fixes

Concept 8 fixes the **no quality validation** problem — Self-Correcting Agent
evaluates its own answer and retries if it's wrong or incomplete.

---
---

## Concept 8 — Self-Correcting Agent
**Notebook: 8SelfCorrectingAgent.ipynb**

### Core Concept

A Self-Correcting Agent adds a **critic** step after the initial answer:
a second LLM call evaluates whether the answer is correct and complete.
If it fails the check, the agent retries with the critic's feedback.

```
Query: "Search docs for RAG and calculate 25 * 4"
  ↓
Attempt 1:
  Agent answers → "RAG is a retrieval method. 25*4=100"
  Critic: "Does this answer use both search_docs and calculate? Does it explain RAG adequately?"
  Verdict: PASS → return answer immediately

Attempt 1 (bad scenario):
  Agent answers → "25*4=100"         ← forgot to search docs
  Critic: FAIL — "Missing RAG explanation"
  ↓
Attempt 2 (with feedback):
  Agent: "You forgot to search docs for RAG. Please retry."
  Agent answers → "RAG combines retrieval... 25*4=100"
  Critic: PASS → return answer
```

### The Critic Prompt

The critic is a separate LLM call — not the same agent that generated the answer.
This is important: the answering LLM tends to think its own answer is good.
An independent critic is more objective.

```python
critic_prompt = f"""
Query: {query}
Agent Answer: {answer}

Evaluate:
1. Does the answer fully address the query?
2. Are all required tools used?
3. Is the answer factually consistent with the tool results?

Reply: PASS or FAIL. If FAIL, explain what is missing.
"""
verdict = critic_llm.invoke(critic_prompt).content
```

### Self-Correction vs ReAct

```
ReAct: reasons during execution (reactive, real-time)
Self-Correcting: validates after execution (post-hoc, deliberate)

ReAct catches errors mid-thought.
Self-Correcting catches errors in the final answer.

Use both together for maximum reliability.
```

### Limitations of Self-Correcting Agent

| Limitation | Explanation |
|------------|-------------|
| **2× LLM calls minimum** | Every query = answer + critic = higher cost |
| **Critic can be wrong** | Critic LLM may incorrectly pass bad answers or fail good ones |
| **Max retry risk** | Without a limit, may retry indefinitely |
| **Doesn't fix planning** | Still reactive — no multi-step planning |

---
---

## How Each Concept Solves the Previous One's Limitation

```
Tool Calling Agent
  Problem: No reasoning trace, tool decisions are opaque
      ↓ solved by
ReAct Agent
  Problem: Returns free text — hard to use in code
      ↓ solved by
Structured Output Agent
  Problem: No memory — forgets between calls
      ↓ solved by
Conversational Agent
  Problem: No real document knowledge
      ↓ solved by
RAG Agent
  Problem: One agent handles everything — no specialization
      ↓ solved by
Multi-Agent (Router)
  Problem: No upfront planning for multi-step tasks
      ↓ solved by
Plan-and-Execute Agent
  Problem: No quality validation of the final answer
      ↓ solved by
Self-Correcting Agent
```

---

## Learning Order & Status

| # | Concept | Status | Key Skill Learned |
|---|---------|--------|-------------------|
| 1 | Tool Calling Agent | To do | bind_tools, tool execution loop, ToolMessage |
| 2 | ReAct Agent | To do | create_react_agent, Thought/Action/Observation |
| 3 | Structured Output Agent | To do | Pydantic output schema, two-stage agent |
| 4 | Conversational Agent | To do | Message history, multi-turn memory, window strategy |
| 5 | RAG Agent | To do | Vectorstore as tool, dynamic retrieval |
| 6 | Multi-Agent | To do | Router pattern, specialized sub-agents |
| 7 | Plan-and-Execute | To do | Planner + Executor + Synthesizer |
| 8 | Self-Correcting | To do | Critic LLM, retry loop, answer validation |

---

## Quick Reference Card

| Concept | One-Line Summary |
|---------|-----------------|
| Tool Calling | LLM decides which tool → your code runs it → LLM answers |
| ReAct | LLM writes Thought before every Action — transparent loop |
| Structured Output | Agent fills a Pydantic schema instead of returning text |
| Conversational | Full message history injected on every turn — LLM remembers |
| RAG Agent | Retrieval pipeline wrapped as a tool — agent decides when to search |
| Multi-Agent | Router picks the right specialist for each query type |
| Plan-and-Execute | Plan all steps first, execute in order, synthesize at end |
| Self-Correcting | Critic LLM validates the answer — retry if it fails |

---

## Key LangChain Imports for Agents

```python
# Tools
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# ReAct
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

# Modern agent (create_agent)
from langchain.agents import create_agent

# Structured output
from pydantic import BaseModel, Field
llm.with_structured_output(MySchema)

# Conversational memory
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
```
