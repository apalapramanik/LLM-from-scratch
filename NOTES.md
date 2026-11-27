# Notes

Short revision notes. One section per topic. Written so each `##` can become one slide.
Format: **What** → **Why** → **How** → **Trade-off** → **Interview line**.

---

## M1. Tool-calling agent

**What:** An LLM that can ask your code to run functions (tools), then use the results.

**Why:** LLMs can't do live things (weather, exact math). Tools give them hands.

**How (the loop):**
1. Send messages + tool descriptions to the model
2. Model replies either with an answer (`stop`) or a tool request (`tool_calls`)
3. Your code runs the tool, appends the result to messages
4. Repeat until the model answers

**Key points**
- The model never runs code. It only outputs the tool name + arguments as JSON.
- The tool description is the model's only guide. Bad description = wrong tool use.
- The full message history is sent every time. The model has no memory of its own.

**Interview line:** "The model is the brain, my code is the hands, and the loop is just
call → run tool → append result → repeat."

**Bug we fixed: duplicate assistant message**
- The model can request several tools in ONE message.
- Correct history: `assistant (once)` → `tool result 1` → `tool result 2`
- Our code appended the assistant message inside the `for` loop, so it was duplicated.
- Fix: append it once, above the loop.

**Interview line:** "History must stay well-formed: one assistant message, then one tool
result per call. Malformed history causes API errors or a confused model."

---

## M2. Making the agent reliable

### Safe calculator (no `eval`)

**What:** `eval` runs any text as Python code. Unsafe for model-written input.

**Why it matters:** the model's text can be steered by users or by documents it reads
(prompt injection). `eval("__import__('os').system(...)")` would run on your machine.

**How:** whitelist, not blacklist.
1. `ast.parse` → turns text into a tree, runs nothing
2. Walk the tree, allow only numbers, `+ - * / ** %`, a few math functions
3. Anything else → error

**Also:** cap exponents (`9**9**9` would hang). Return errors as text so the model
can retry, instead of crashing the agent.

**Trade-off:** whitelist is stricter (may reject valid stuff), but safe by default.

**Interview line:** "Never execute model output directly. Allow what you know is safe,
instead of trying to block everything dangerous."

### Tool safety net

**Rule:** a tool must never crash the agent. Return errors as text.

**Three protections:**
1. `timeout=5` on every network call: no infinite hangs
2. `try/except` around every tool run: error becomes `"Error running X: ..."`
3. Safe JSON parsing of model arguments: broken JSON becomes `{}`, not a crash

**Why:** the model can read an error message and retry or explain. A crash gives it nothing.

**Interview line:** "Treat tool output as part of the conversation. Failures are
information the model can use, so I return them instead of raising."

### Argument validation

**What:** before running a tool, check the model's arguments against the tool's own schema.

**Why:** models make typos (`cty` instead of `city`) or wrong types. A raw `KeyError` is
cryptic. A clear message ("Missing 'city'. Expected {city: string}") lets the model self-correct.

**How:** reuse the `tools = [...]` schema: check tool exists, required keys present, types match.
One source of truth, no duplicated rules.

**Interview line:** "Validate at the boundary, and make the error message tell the model
exactly how to fix its call."

---

## M3. Stopping agent loops

**What:** an agent loop only ends when the model says "done". It can get stuck.

**Ways it gets stuck:** same failing call retried; A→B→A→B cycle; never decides it has enough;
reply cut off (`finish_reason="length"`) matching no branch.

**Two guards:**
| Guard | Catches | Role |
|---|---|---|
| Repeat detection (`seen` set of name+args) | exact repeated calls | early warning |
| Max steps (`for step in range(N)`) | every loop type | hard limit |

**How:**
- Replace `while True` with `for step in range(MAX_STEPS)`
- On a repeat: don't run it, tell the model "you already made this call, try something else"
- On limit: return a clear "stopped" message
- Stop whenever the reply has no `tool_calls` (don't rely only on `finish_reason == "stop"`)

**Trade-off:** MAX_STEPS too low cuts off legit multi-step tasks; too high wastes money.
Repeat detection can't see near-duplicates (slightly different args).

**Other fixes to mention:** per-task token/cost budget, better tool descriptions,
a "give your best answer now" nudge on the last step, escalate to a human.

**Interview line:** "Repeat detection catches loops early, max steps is the guarantee.
I tell the model it repeated, rather than crashing."

---

## M4. Tracing

**What:** record every step of an agent run as structured data: a **trace** (one request)
made of **spans** (one unit of work each: name, input, output, duration, tokens, error).
Spans nest into a tree. In multi-agent systems each agent is a child span.

**Why:** when an answer is wrong you need to see *where*. Print statements can't tell you
which tool was called, what it returned, which step was slow, or what it cost.

**Debugging with a trace:** read top to bottom, find the **first** bad span.
The last span only shows the symptom. Causes: wrong tool chosen, bad tool output,
bad retrieval, repeated calls.

**How we built it (`agents/tracer.py`):**
- `with trace.span("name", input=...) as s:` times a block; set `s.output`, `s.meta`
- A stack tracks the open span, so new spans become its children
- `finally` records duration even when the code fails; `except` records the error and re-raises
- `trace.save()` writes `traces/<id>.json`

**What to record per span:** inputs, outputs, latency, tokens/cost, model name,
prompt version, errors, user/session id.

**Production tools:** LangSmith, Langfuse, OpenTelemetry. Same idea at scale.

**Interview line:** "I trace every model and tool call as a span with inputs, outputs,
latency and tokens, so I can find the first bad step instead of guessing."

---

## M5. Memory for long-running assistants

**Problem:** the model remembers nothing. We resend the whole `messages` list each turn.
Long history = higher cost, context-limit errors, and the model missing details in the middle.

**Three layers:**
| Layer | What | Good for | Weakness |
|---|---|---|---|
| Short-term | last N messages, exact | current thread | limited size |
| Summary | old turns compressed | gist of long chats | lossy (drops details) |
| Long-term | facts as embeddings, recalled by similarity (RAG for memory) | names, preferences, decisions | what to store? stale facts |

**Prompt shape:** `[system + summary] [last N messages] [recalled facts] [new message]`

**Trap:** never cut history mid tool-exchange. An assistant `tool_calls` message and its
`tool` results must stay together, so only cut at a **user** message.

**Long-term design questions:**
- What to store: durable facts only (not every message)
- When to recall: every turn, or only when needed
- Stale facts: timestamps, update or delete old ones
- Privacy: per-user filter, ability to delete (`forget_user`)

**Code:** `agents/memory.py` has `ConversationMemory` (window + summary) and
`LongTermMemory` (Chroma, per-user). Not yet wired into the agent loop.

**Interview line:** "Recent turns stay exact, old turns become a summary, and durable
facts go to a vector store I recall from. I only cut at user messages and keep memory per user."

---

(more topics added as we learn them)
