# Agents

This folder contains agent implementations built on top of LLMs. Agents are programs where a language model can decide to call tools — functions you write — to answer questions. The model acts as the brain, your code acts as the hands.

## What is a Tool-Calling Agent?

A regular LLM just takes text in and gives text back. An agent extends this by giving the model access to tools — things like a calculator, a web search, a database query, or an API call. The model reads the tool descriptions and decides on its own when to use them, what inputs to pass, and how to present the result.

The flow looks like this:

```
You ask a question
  → Model thinks "I need a tool for this"
  → Model calls your tool with specific inputs
  → Your code runs the tool and returns a result
  → Model reads the result and answers you
```

This loop is the foundation of every agent ever built.

---

## Files

### `basic_agent.py`

A tool-calling agent built with the Groq API and Llama 4 Scout. It has three tools:

- **calculator** — evaluates math expressions using Python's `eval` and the `math` module
- **word_count** — counts words in a string
- **get_weather** — fetches live current weather for any city using the Open-Meteo API (no API key required)

The agent runs a `while True` loop: it sends the conversation to the model, checks if the model wants to call a tool, runs the tool, appends the result to the conversation history, and loops back. When the model is done calling tools it returns a final answer and the loop exits.

A good example of the agent's reasoning ability — when asked "Is it warmer in London or Paris today?", the model calls `get_weather` twice on its own (once per city), compares the results, and answers in plain English. You never told it to call the tool twice — it figured that out from the question alone.

---

## How the Code Works

**1. Tool definitions** — each tool is a dictionary that describes the tool to the model: its name, what it does (description), and what inputs it expects (parameters). The model reads these descriptions at inference time to decide when and how to use each tool.

**2. Tool execution** — the `run_tool` function maps tool names to actual Python code. The model never runs code itself — it just says "call this tool with these inputs" and your function does the real work.

**3. The agent loop** — inside `run_agent`, a `while True` loop sends the full conversation to the model on every iteration. If `finish_reason == "tool_calls"`, the model wants to call a tool. The result gets appended to the message history and the loop continues. If `finish_reason == "stop"`, the model is done and we return the final answer.

**4. Message history** — every turn (user message, model response, tool result) gets appended to the `messages` list. The model sees the full history on every request, which is how it knows what's already been tried and what the results were.

**5. System prompt** — a system message at the top of the conversation steers the model's behavior, telling it to always use tools for math rather than computing things itself.

**6. Weather tool** — uses two API calls: first to Open-Meteo's geocoding API to convert a city name into coordinates, then to Open-Meteo's forecast API to get current temperature, wind speed, and humidity. Completely free, no key needed.

---

## Problems Faced in Implementation

### 1. Groq tool-use model deprecations
The model `llama3-groq-70b-8192-tool-use-preview` — specifically fine-tuned for tool calling — was decommissioned mid-implementation. Switched to `llama-3.3-70b-versatile` but it produced malformed tool call syntax. Eventually settled on `meta-llama/llama-4-scout-17b-16e-instruct` which handles tool calling reliably.

### 2. Malformed tool call format
Groq/Llama models occasionally generate tool calls in an invalid format (e.g. `<function=calculator {...}` instead of proper JSON). This caused `400 Bad Request` errors from the API. Fixed by switching models and adding a system prompt that explicitly instructs the model to always use tools rather than answering directly.

### 3. `^` vs `**` for exponentiation
The model used `^` (XOR in Python) instead of `**` (exponentiation) when generating math expressions, causing a `TypeError`. Fixed by adding `.replace("^", "**")` in `run_tool` before passing the expression to `eval`.

### 4. API key exposure
A Groq API key was accidentally shared in plain text. The key was immediately revoked and regenerated. Fixed permanently by storing all secrets in a `.env` file, loading them with `python-dotenv`, and adding `.env` to `.gitignore` so keys are never committed to the repo.

---

## Setup

```bash
conda activate llm-lab
pip install groq python-dotenv requests
```

Create a `.env` file in the repo root:
```
GROQ_API_KEY="gsk_your_key_here"
```

Run the agent:
```bash
python basic_agent.py
```

---

## Key Concepts to Remember

| Concept | What it means |
|---|---|
| Tool definition | A JSON description the model reads to understand what a tool does |
| Tool execution | Your Python code that actually runs when the model calls a tool |
| Agent loop | `while True` — keep calling the model until it stops requesting tools |
| Message history | The full conversation sent to the model on every request |
| `finish_reason` | `"tool_calls"` = model wants a tool, `"stop"` = model is done |
| System prompt | A message that steers the model's overall behavior |
| Multi-tool reasoning | Model calls tools multiple times on its own when the question requires it |