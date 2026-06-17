# Prompt & Message Concepts — Complete Learning Guide

---

## What are Prompts and Messages?

LLMs don't receive raw text — they receive **structured inputs** called prompts.  
The way you structure a prompt directly determines the quality of the output.

**The Problem:**
```
Bad prompt:   "Tell me about AI"
LLM:          "AI is a broad field..." (generic, unfocused, unhelpful)

Good prompt:  "You are a senior ML engineer. Explain in 3 bullet points
               how transformer attention works to a junior developer."
LLM:          Precise, structured, audience-appropriate answer
```

**Messages** are the structured units inside a prompt. Every modern LLM API works with a list of messages:

```
Without structure:   raw_text → LLM → random answer

With structure:      [SystemMessage, HumanMessage, AIMessage...] → LLM → targeted answer
```

Prompt Engineering = **designing inputs** that reliably produce the outputs you need.

---

## The Core Message Structure

Every chat-based LLM interaction is a list of messages with roles:

```
ROLES:
  system    → sets context, persona, rules (invisible to user, always followed)
  human     → user's input / question
  ai        → LLM's past response (used in multi-turn conversations)
  tool      → output returned from a tool/function call

EXAMPLE CONVERSATION:
  [system]  "You are a helpful assistant. Answer concisely."
  [human]   "What is LangChain?"
  [ai]      "LangChain is a framework for building LLM applications."
  [human]   "What can I build with it?"   ← new question with full context
```

Understanding this structure is the foundation of every concept below.

---
---

## Concept 1 — Basic PromptTemplate
**Notebook: 1BasicPromptTemplate.ipynb**

### Core Concept

The simplest form of prompting. A string template with `{variables}` that get filled in at runtime.

```
Template:   "Summarize this text in {num_sentences} sentences: {text}"
           ↓
fill in:    num_sentences=2, text="LangChain is a framework..."
           ↓
Final:      "Summarize this text in 2 sentences: LangChain is a framework..."
           ↓
LLM → Answer
```

### What You Actually Build

```python
from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    input_variables=["topic", "audience"],
    template="Explain {topic} to a {audience} in simple terms."
)

# Fill in variables
prompt_text = template.format(topic="neural networks", audience="10-year-old")
# → "Explain neural networks to a 10-year-old in simple terms."

response = llm.invoke(prompt_text)
```

### When to Use

- Simple, single-turn text completion tasks
- Batch processing where you fill the same template with many inputs
- Non-chat models (completion APIs)

### Limitations of Basic PromptTemplate

| Limitation | Explanation |
|------------|-------------|
| **No role separation** | System instructions and user input are mixed in one string |
| **No conversation history** | Cannot handle multi-turn dialogue |
| **No few-shot support** | No built-in way to add examples |
| **Harder to control LLM behavior** | Without a system role, the LLM has no persona or rules |

### What the Next Concept Fixes

Concept 2 fixes the **no role separation** problem by introducing ChatPromptTemplate and typed message roles.

---
---

## Concept 2 — ChatPromptTemplate & Message Types
**Notebook: 2ChatPromptTemplate.ipynb**

### Core Concept

Chat models work with a **list of messages**, each tagged with a role.  
`ChatPromptTemplate` lets you define this structure as a reusable template.

**The 4 Message Types:**

```
SystemMessage   → Persona, rules, constraints. Set once, always active.
HumanMessage    → What the user says. Drives the conversation.
AIMessage       → What the LLM said before (used in multi-turn memory).
ToolMessage     → What a tool/function returned (used in agent workflows).
```

### What You Actually Build

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Define template with roles
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}. Always respond in {language}."),
    ("human",  "{user_input}")
])

# Fill variables and format into message list
messages = prompt.format_messages(
    role="Python expert",
    language="English",
    user_input="What is a decorator?"
)

# messages is now:
# [SystemMessage("You are a Python expert. Always respond in English."),
#  HumanMessage("What is a decorator?")]

response = llm.invoke(messages)
```

### Role Behavior

```
System role controls everything downstream:
  "You are a pirate."         → LLM speaks like a pirate
  "Always respond in JSON."   → LLM structures output as JSON
  "You are a strict critic."  → LLM gives harsh, honest feedback

Human role is the query:
  "Explain X" / "What is Y?" / "Write me Z"

AI role preserves past context:
  Used to pass conversation history so LLM knows what was already said
```

### Limitations of ChatPromptTemplate

| Limitation | Explanation |
|------------|-------------|
| **No examples in prompt** | LLM relies on instructions only — no demonstrations of what "good" looks like |
| **Static examples if added manually** | Can't dynamically select the most relevant examples per query |
| **History must be managed manually** | Injecting AIMessage history is your responsibility |

### What the Next Concept Fixes

Concept 3 fixes the **no examples** problem with Few-Shot prompting.

---
---

## Concept 3 — Few-Shot Prompting
**Notebook: 3FewShotPrompt.ipynb**

### Core Concept

Instead of just telling the LLM what to do, you **show it examples** of input → output pairs.  
LLMs learn patterns from examples far more reliably than from instructions alone.

```
Without few-shot (instruction only):
  System: "Classify sentiment as POSITIVE or NEGATIVE."
  Human:  "This product broke on day one."
  LLM:    "The sentiment is negative."   ← inconsistent format

With few-shot (examples + instruction):
  System: "Classify sentiment."
  Human:  "I love this product!" → POSITIVE
  Human:  "Terrible quality."   → NEGATIVE
  Human:  "This product broke on day one."
  LLM:    "NEGATIVE"             ← consistent format learned from examples
```

### What You Actually Build

```python
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

examples = [
    {"input": "I love this!", "output": "POSITIVE"},
    {"input": "Terrible product.", "output": "NEGATIVE"},
    {"input": "It is okay I guess.", "output": "NEUTRAL"},
]

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai",    "{output}"),
])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
)

final_prompt = ChatPromptTemplate.from_messages([
    ("system", "Classify the sentiment. Reply with only: POSITIVE, NEGATIVE, or NEUTRAL."),
    few_shot_prompt,
    ("human", "{input}"),
])
```

### Why Examples Beat Instructions

```
Instructions:  "Output exactly one word: POSITIVE, NEGATIVE, or NEUTRAL"
  LLM might still say: "The sentiment is POSITIVE." (adds extra words)

Examples show the exact pattern:
  Human: "I love this!"    AI: "POSITIVE"   ← LLM sees: one word, no extra text
  Human: "Terrible."       AI: "NEGATIVE"
  → LLM reliably outputs one word
```

### Dynamic Few-Shot (Advanced)

Select examples based on similarity to the current input:

```
Query: "This tastes awful."
  ↓
Example selector finds most similar stored examples:
  "This smells terrible." → NEGATIVE     ← similar negative food/product feedback
  "Disgusting experience." → NEGATIVE
  ↓
These 2 examples added to prompt (not all examples — saves tokens)
```

### Limitations of Few-Shot Prompting

| Limitation | Explanation |
|------------|-------------|
| **Token cost** | More examples = longer prompt = more tokens = higher cost |
| **Static examples can go stale** | Hardcoded examples may not cover new edge cases |
| **No persistent memory** | Each call is independent — LLM doesn't remember previous sessions |
| **Can't handle follow-up questions** | No mechanism to reference what was said earlier in a session |

### What the Next Concept Fixes

Concept 4 fixes the **no persistent memory** problem with MessagesPlaceholder for conversation history.

---
---

## Concept 4 — MessagesPlaceholder & Conversation History
**Notebook: 4ConversationHistory.ipynb**

### Core Concept

`MessagesPlaceholder` reserves a **slot in your prompt template** where a dynamic list of messages gets injected at runtime.  
This is how multi-turn conversations are built — past messages are passed in as context.

```
Without history:
  Turn 1: Human: "My name is Mohan." → AI: "Hello Mohan!"
  Turn 2: Human: "What is my name?"  → AI: "I don't know your name."  ← forgets!

With MessagesPlaceholder:
  Turn 2 prompt:
    [system]  "You are a helpful assistant."
    [human]   "My name is Mohan."          ← history injected here
    [ai]      "Hello Mohan!"               ← history injected here
    [human]   "What is my name?"           ← new question
  AI: "Your name is Mohan."               ← remembers!
```

### What You Actually Build

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),   # slot for past messages
    ("human", "{input}"),
])

# Maintain history list across turns
history = []

def chat(user_input):
    messages = prompt.format_messages(history=history, input=user_input)
    response = llm.invoke(messages)

    # Append this turn to history
    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=response.content))

    return response.content
```

### Memory Strategies

```
Full history:
  → Pass ALL past messages every turn
  ✓ Perfect recall    ✗ Context window fills up fast

Window memory:
  → Pass only last N messages
  ✓ Bounded cost      ✗ Forgets things outside the window

Summary memory:
  → Summarize old messages into one SystemMessage
  ✓ Compact           ✗ Loses exact details

Token-limited memory:
  → Keep passing messages until token limit, then drop oldest
  ✓ Precise control   ✗ Slightly complex to implement
```

### Limitations of MessagesPlaceholder

| Limitation | Explanation |
|------------|-------------|
| **Growing context cost** | Each turn adds tokens — long conversations get expensive |
| **No output format enforcement** | LLM may return plain text when you need JSON, lists, etc. |
| **No schema validation** | You can't enforce that the LLM returns a specific structure |

### What the Next Concept Fixes

Concept 5 fixes the **no output format enforcement** problem with Structured Output prompts.

---
---

## Concept 5 — Structured Output Prompts
**Notebook: 5StructuredOutput.ipynb**

### Core Concept

LLMs return unstructured text by default. Structured Output forces the response into a specific schema (JSON, Pydantic model, TypedDict).

```
Without structured output:
  Human: "Extract the name and age from: 'John is 30 years old.'"
  AI:    "The name is John and he is 30 years old."
         ← string, hard to parse programmatically

With structured output:
  AI:    {"name": "John", "age": 30}
         ← directly usable in code
```

### What You Actually Build

```python
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

# Define the output schema
class Person(BaseModel):
    name: str
    age: int
    occupation: str

# Bind schema to LLM — LLM will always return this structure
structured_llm = llm.with_structured_output(Person)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract the person details from the text."),
    ("human",  "{text}")
])

chain = prompt | structured_llm

result = chain.invoke({"text": "John is a 30-year-old software engineer."})
# result.name       → "John"
# result.age        → 30
# result.occupation → "software engineer"
```

### Three Ways to Get Structured Output

```
Method 1 — with_structured_output (Pydantic):
  llm.with_structured_output(MySchema)
  ✓ Cleanest, validated, IDE autocomplete
  ✗ Requires OpenAI function-calling support

Method 2 — JSON mode:
  llm.bind(response_format={"type": "json_object"})
  ✓ Any JSON structure
  ✗ No schema validation — LLM may miss fields

Method 3 — Output parsers:
  chain = prompt | llm | JsonOutputParser()
  ✓ Works with any model
  ✗ May fail if LLM deviates from format
```

### Limitations of Structured Output Prompts

| Limitation | Explanation |
|------------|-------------|
| **LLM may still hallucinate fields** | Schema enforces format but not factual accuracy |
| **No reasoning visible** | Output is just data — you don't see how LLM arrived at it |
| **Complex schemas confuse smaller models** | GPT-4 handles nested schemas; GPT-3.5 may struggle |
| **LLM can't take actions** | You get structured data but LLM can't call APIs or tools |

### What the Next Concept Fixes

Concept 6 fixes the **LLM can't take actions** problem with Tool/Function Calling prompts.

---
---

## Concept 6 — Tool & Function Calling Prompts
**Notebook: 6ToolCalling.ipynb**

### Core Concept

Tool calling allows the LLM to **decide which tool to call and with what arguments** — instead of answering directly, it outputs a structured tool invocation.

```
Without tools:
  Human: "What's the weather in Hyderabad right now?"
  AI:    "I don't have access to real-time weather data."  ← stuck

With tools:
  Human: "What's the weather in Hyderabad right now?"
  AI:    → calls get_weather(city="Hyderabad")            ← takes action
  Tool:  → returns {"temp": 34, "condition": "Sunny"}
  AI:    "It's currently 34°C and sunny in Hyderabad."    ← grounded answer
```

### What You Actually Build

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Weather in {city}: 34°C, Sunny"

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

# Bind tools to LLM
llm_with_tools = llm.bind_tools([get_weather, calculate])

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant with access to tools."),
    ("human",  "{input}"),
])

chain = prompt | llm_with_tools
response = chain.invoke({"input": "What is 25 * 48?"})
# response.tool_calls → [{"name": "calculate", "args": {"expression": "25*48"}}]
```

### The Full Tool Loop

```
Human: "What's 25 * 48 and what's the weather in Mumbai?"
  ↓
LLM decides:
  Tool call 1: calculate(expression="25*48")
  Tool call 2: get_weather(city="Mumbai")
  ↓
Tools execute:
  calculate → "1200"
  get_weather → "31°C, Cloudy"
  ↓
Results sent back as ToolMessages
  ↓
LLM: "25 × 48 = 1200. It's currently 31°C and cloudy in Mumbai."
```

### Limitations of Tool Calling Prompts

| Limitation | Explanation |
|------------|-------------|
| **LLM may call wrong tool** | Ambiguous queries can trigger incorrect tool selection |
| **No reasoning trace** | You see the decision but not *why* LLM chose that tool |
| **Single-step only** | Standard tool calling doesn't handle multi-step reasoning chains |
| **No "think before acting"** | LLM jumps to action without showing its reasoning process |

### What the Next Concept Fixes

Concept 7 fixes the **no reasoning trace** and **single-step only** problems with ReAct prompts.

---
---

## Concept 7 — ReAct Prompts (Reason + Act)
**Notebook: 7ReActPrompt.ipynb**

### Core Concept

ReAct = **Re**asoning + **Act**ing. The LLM explicitly writes out its thoughts before taking each action, creating a transparent reasoning chain.

```
Without ReAct (blind action):
  Human: "Who is the CEO of the company that makes ChatGPT?"
  LLM:   calls search("CEO ChatGPT company")
         ← why this query? we don't know

With ReAct (transparent reasoning):
  Human: "Who is the CEO of the company that makes ChatGPT?"

  Thought: ChatGPT is made by OpenAI. I need to find OpenAI's CEO.
  Action: search("OpenAI CEO 2024")
  Observation: "Sam Altman is the CEO of OpenAI."
  Thought: I have the answer.
  Final Answer: "Sam Altman is the CEO of OpenAI, the company behind ChatGPT."
```

### The ReAct Loop

```
Question
  ↓
Thought:       "What do I know? What do I need to find out?"
  ↓
Action:        tool_name(args)
  ↓
Observation:   tool result
  ↓
Thought:       "What does this tell me? Do I have enough?"
  ↓
  [loop if more info needed]
  ↓
Final Answer:  answer to original question
```

### What You Actually Build

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

# Pull the standard ReAct prompt from LangChain hub
react_prompt = hub.pull("hwchase17/react")

# The prompt instructs the LLM to output:
# Thought: ...
# Action: tool_name
# Action Input: ...
# Observation: (tool fills this in)
# ... (loop)
# Final Answer: ...

agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = executor.invoke({"input": "What is the capital of the country with the most RAG researchers?"})
```

### Thought Format Matters

```
Good Thought (specific, planful):
  "The user wants to know LangSmith's CEO. I'll search for OpenAI leadership 
   since LangSmith is a LangChain product. Let me first confirm who owns LangSmith."

Bad Thought (vague):
  "I need to find information about this."

The quality of Thought directly determines Action quality.
```

### Limitations of ReAct Prompts

| Limitation | Explanation |
|------------|-------------|
| **Verbose** | Thought/Action/Observation adds many tokens per step |
| **Can loop indefinitely** | Without a max_iterations limit, may get stuck |
| **Text parsing brittle** | If LLM deviates from Thought/Action format, parsing fails |
| **All tools treated equally** | No built-in way to prefer one tool over another |
| **Prompt sensitivity** | Slight prompt changes can break the format completely |

### What the Next Concept Fixes

Concept 8 fixes the **text parsing brittle** problem with Partial Prompts and dynamic template composition.

---
---

## Concept 8 — Partial Prompts & Dynamic Templates
**Notebook: 8PartialPrompts.ipynb**

### Core Concept

Partial prompts let you **pre-fill some variables** in a template while leaving others for later.  
Useful when some values are known at startup (API version, model config) and others at runtime (user input).

```
Full template:  "You are a {role} assistant. Today is {date}. Answer: {question}"

Partial fill (at startup):
  role = "Python expert"
  date = datetime.now().strftime("%Y-%m-%d")   ← computed once

Runtime fill (per request):
  question = user's actual question

→ Final prompt builds cleanly with values from two different sources/times
```

### What You Actually Build

```python
from langchain_core.prompts import PromptTemplate
from datetime import datetime

base_template = PromptTemplate(
    template="You are a {role}. Today is {date}. User asks: {question}",
    input_variables=["role", "date", "question"]
)

# Pre-fill known values — returns a new template with fewer variables
partial_template = base_template.partial(
    role="senior data scientist",
    date=lambda: datetime.now().strftime("%Y-%m-%d")  # function called at runtime
)

# Now only "question" needs to be provided
response = llm.invoke(partial_template.format(question="What is overfitting?"))
```

### Dynamic Template Composition

```python
# Build different system messages depending on context
def get_system_prompt(user_type: str) -> str:
    templates = {
        "beginner":  "Explain concepts simply. Use analogies. Avoid jargon.",
        "expert":    "Be technical. Use correct terminology. Skip basics.",
        "child":     "Use very simple words. Give fun examples."
    }
    return templates.get(user_type, "Be helpful and clear.")

# Compose final prompt dynamically
prompt = ChatPromptTemplate.from_messages([
    ("system", get_system_prompt(user_type)),
    ("human",  "{question}")
])
```

### Limitations of Partial Prompts

| Limitation | Explanation |
|------------|-------------|
| **Still static at composition time** | Logic for picking template must be written by you — LLM doesn't choose |
| **No example selection** | If you want dynamic examples, you need a separate selector |
| **Complex multi-persona logic is manual** | Routing to the right prompt template must be coded explicitly |

### What the Next Concept Fixes

Concept 9 fixes the **complex routing is manual** problem with Pipeline Prompts.

---
---

## Concept 9 — Pipeline Prompts & Prompt Composition
**Notebook: 9PipelinePrompts.ipynb**

### Core Concept

Pipeline Prompts let you **compose smaller prompt pieces into one final prompt**.  
Each piece can be templated independently, then assembled.

```
Piece 1 — Introduction:
  "You are a {role} at {company}."

Piece 2 — Instructions:
  "Your task: {task}. Always format output as: {format}."

Piece 3 — User Input:
  "Question: {question}"

→ Assemble:
  "You are a data scientist at Acme Corp.
   Your task: analyze data. Always format output as: bullet points.
   Question: What are the main trends in Q3 sales?"
```

### What You Actually Build

```python
from langchain_core.prompts import PipelinePromptTemplate, PromptTemplate

# Define sub-prompts
intro_prompt = PromptTemplate.from_template("You are a {role} at {company}.")
task_prompt  = PromptTemplate.from_template("Task: {task}. Format: {output_format}.")
input_prompt = PromptTemplate.from_template("Question: {question}")

# Assemble the full prompt
full_template = PromptTemplate.from_template(
    "{intro}\n{task}\n{input}"
)

pipeline_prompt = PipelinePromptTemplate(
    final_prompt=full_template,
    pipeline_prompts=[
        ("intro", intro_prompt),
        ("task",  task_prompt),
        ("input", input_prompt),
    ]
)

result = pipeline_prompt.format(
    role="data analyst",
    company="Acme Corp",
    task="summarize quarterly results",
    output_format="3 bullet points",
    question="What changed in Q3 vs Q2?"
)
```

### When Pipeline Prompts Help

```
Without composition (one giant template):
  Hard to maintain — change one part, risk breaking others
  Hard to reuse — can't use the "intro" piece in another prompt

With composition (modular pieces):
  Each piece is independently testable
  Mix and match pieces across different prompts
  Swap one piece without touching others
```

### Limitations of Pipeline Prompts

| Limitation | Explanation |
|------------|-------------|
| **Verbose setup** | More code for simple cases |
| **Still no example optimization** | Dynamic example selection still requires extra tooling |
| **Context window still finite** | Composed prompts can grow large and hit token limits |

---
---

## Concept 10 — Prompt Optimization & Best Practices
**Notebook: 10PromptOptimization.ipynb**

### Core Concept

Prompt Engineering is not guesswork — there are systematic patterns that consistently produce better outputs.

### The 5 Dimensions of a Strong Prompt

```
1. ROLE        → Who is the LLM?
2. CONTEXT     → What does the LLM need to know?
3. TASK        → What exactly should the LLM do?
4. FORMAT      → How should the output look?
5. CONSTRAINTS → What should the LLM avoid?
```

**Example — Weak vs Strong:**

```
Weak:
  "Write about machine learning."

Strong:
  ROLE:        "You are a technical writer with 10 years of ML experience."
  CONTEXT:     "The audience is junior developers who know Python but not statistics."
  TASK:        "Write a 200-word introduction to supervised learning."
  FORMAT:      "Use 2-3 short paragraphs. Include one concrete real-world example."
  CONSTRAINTS: "Avoid mathematical notation. Do not mention neural networks."
```

### Prompt Patterns Library

```
Pattern 1 — Role Assignment:
  "You are a [specific expert]. Your job is to..."
  → More focused, domain-accurate responses

Pattern 2 — Chain of Thought:
  "Think step by step before answering."
  → Better reasoning for math, logic, multi-step problems

Pattern 3 — Output Format Enforcement:
  "Respond ONLY as valid JSON with keys: name, age, city."
  → Parseable, consistent output

Pattern 4 — Negative Constraints:
  "Do NOT make up information. If unsure, say 'I don't know'."
  → Reduces hallucination

Pattern 5 — Persona + Tone:
  "Explain this as if teaching a 10-year-old. Be enthusiastic and use analogies."
  → Audience-appropriate communication

Pattern 6 — Grounding:
  "Answer ONLY based on the following context. If the answer is not in the context,
   say 'I don't know'. Context: {context}"
  → Prevents hallucination (standard RAG pattern)

Pattern 7 — One-Shot / Few-Shot:
  "Here is an example: Input: X → Output: Y. Now do the same for: Input: Z"
  → Pattern learning beats instruction alone

Pattern 8 — Self-Consistency:
  "Give me 3 different answers to this question, then pick the best one."
  → Reduces randomness in critical decisions
```

### Token Optimization

```
Verbose system message (expensive):
  "You are a very helpful assistant that is always willing to help the user with
   any question they might have, no matter how simple or complex it is."

Tight system message (same behavior, fewer tokens):
  "You are a helpful assistant."

Rule: Every token costs money. Be precise, not wordy.
```

### Temperature Guide

```
Temperature 0.0:  Deterministic. Best for: extraction, classification, structured output.
Temperature 0.3:  Slight creativity. Best for: summarization, Q&A, analysis.
Temperature 0.7:  Balanced. Best for: chatbots, explanation, general use.
Temperature 1.0:  Creative. Best for: brainstorming, stories, varied outputs.
Temperature 1.5+: Very random. Rarely useful.
```

---
---

## How Each Concept Builds on the Previous One

```
Basic PromptTemplate
  Problem: No role separation, no conversation support
      ↓ solved by
ChatPromptTemplate + Message Types
  Problem: No examples to show LLM what "good" looks like
      ↓ solved by
Few-Shot Prompting
  Problem: No memory — LLM forgets previous turns
      ↓ solved by
MessagesPlaceholder + Conversation History
  Problem: LLM returns unstructured text, hard to use in code
      ↓ solved by
Structured Output Prompts
  Problem: LLM can answer but can't take real-world actions
      ↓ solved by
Tool & Function Calling
  Problem: Actions taken without visible reasoning
      ↓ solved by
ReAct Prompts (Reason + Act)
  Problem: Prompt logic hardcoded, hard to reuse
      ↓ solved by
Partial Prompts + Dynamic Templates
  Problem: Large prompts unmanageable as single string
      ↓ solved by
Pipeline Prompts + Composition
  Problem: No systematic approach to quality
      ↓ solved by
Prompt Optimization + Best Practices
```

---

## Learning Order & Status

| # | Concept | Status | Key Skill Learned |
|---|---------|--------|-------------------|
| 1 | Basic PromptTemplate | To do | String templates, variable substitution |
| 2 | ChatPromptTemplate & Messages | To do | Roles, System/Human/AI message types |
| 3 | Few-Shot Prompting | To do | Example-driven learning, FewShotChatMessagePromptTemplate |
| 4 | MessagesPlaceholder & History | To do | Multi-turn conversation, memory strategies |
| 5 | Structured Output | To do | Pydantic schemas, JSON output, output parsers |
| 6 | Tool & Function Calling | To do | Tool binding, tool loops, ToolMessage |
| 7 | ReAct Prompts | To do | Thought/Action/Observation loop, agent reasoning |
| 8 | Partial Prompts | To do | Pre-filling variables, dynamic composition |
| 9 | Pipeline Prompts | To do | Modular prompt assembly, reusable pieces |
| 10 | Prompt Optimization | To do | Patterns, token efficiency, temperature |

---

## Quick Reference Card

| Concept | One-Line Summary |
|---------|-----------------|
| Basic PromptTemplate | Fill `{variables}` into a string template |
| ChatPromptTemplate | Structure prompt as [System, Human, AI] message list |
| Few-Shot | Show examples → LLM learns pattern from demonstrations |
| MessagesPlaceholder | Reserve a slot for injecting conversation history |
| Structured Output | Force LLM to return JSON / Pydantic model |
| Tool Calling | LLM decides which tool to call and with what args |
| ReAct | LLM reasons out loud before each action |
| Partial Prompts | Pre-fill some variables, leave rest for runtime |
| Pipeline Prompts | Compose modular prompt pieces into one final prompt |
| Prompt Optimization | Role + Context + Task + Format + Constraints = strong prompt |

---

## Message Types — Cheat Sheet

| Message Type | Role | When Used |
|--------------|------|-----------|
| `SystemMessage` | `system` | Persona, rules, constraints — set once per session |
| `HumanMessage` | `human` | User input — every turn |
| `AIMessage` | `ai` | LLM's past response — injected for memory |
| `ToolMessage` | `tool` | Tool/function output — returned after tool call |
| `ChatMessage` | custom | Any custom role (rarely needed) |
