import json
import math
import os
import requests
import streamlit as st
import chromadb
from groq import Groq
from dotenv import load_dotenv

from agents.rag_agent import (
    retrieve as rag_retrieve,
    answer as rag_answer,
    ingest as rag_ingest,
    CHROMA_DIR,
    COLLECTION_NAME,
    DOCS_DIR,
)

load_dotenv()

st.set_page_config(
    page_title="LLM Agent Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 LLM Agent")
    st.markdown("Two agents built from scratch — no framework, just raw API calls.")
    st.divider()

    mode = st.radio(
        "Mode",
        ["🧰 Tool-Calling Agent", "📚 RAG Agent"],
        index=1,
        help="Switch between the two agent flavors.",
    )

    st.divider()

    if mode.startswith("🧰"):
        st.markdown("### 🛠 Available Tools")
        with st.expander("🧮 calculator", expanded=False):
            st.markdown("Evaluates any Python math expression.")
            st.code("sqrt(144)  →  12.0", language="python")
        with st.expander("📝 word_count", expanded=False):
            st.markdown("Counts words in a block of text.")
        with st.expander("🌤 get_weather", expanded=False):
            st.markdown("Live weather via Open-Meteo (no API key).")
    else:
        st.markdown("### 📚 Knowledge Base")
        st.markdown(f"`{DOCS_DIR}/`")
        for f in sorted(os.listdir(DOCS_DIR)) if os.path.isdir(DOCS_DIR) else []:
            if f.endswith(".md"):
                st.markdown(f"- `{f}`")
        st.markdown("### 🧠 Embedding")
        st.code("all-MiniLM-L6-v2", language="text")
        st.markdown("### 🗄 Vector Store")
        st.code("ChromaDB (persistent)", language="text")

    st.divider()
    st.markdown("### 🔬 LLM")
    st.code("meta-llama/llama-4-scout-17b-16e-instruct", language="text")
    st.markdown("### ⚡ Inference")
    st.code("Groq Cloud", language="text")

    st.divider()
    if not os.getenv("GROQ_API_KEY"):
        st.error("⚠️ GROQ_API_KEY not set.\nCreate a `.env` file in the repo root:\n```\nGROQ_API_KEY=\"gsk_...\"\n```")

    if st.button("🗑 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.tool_traces = {}
        st.session_state.rag_traces = {}
        st.rerun()

# ── Tool-calling agent setup ───────────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluates a math expression. Use this whenever the user asks a math question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "A valid Python math expression e.g. '2 + 2' or 'sqrt(144)'"}
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "word_count",
            "description": "Counts the number of words in a given text.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string", "description": "The text to count words in"}},
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Gets the current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name e.g. 'Tokyo'"}},
                "required": ["city"],
            },
        },
    },
]

def run_tool(name: str, inputs: dict) -> str:
    if name == "calculator":
        expression = inputs["expression"].replace("^", "**")
        return str(eval(expression, {"__builtins__": {}}, vars(math)))
    if name == "word_count":
        return str(len(inputs["text"].split()))
    if name == "get_weather":
        city = inputs["city"]
        geo = requests.get(f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1").json()
        if not geo.get("results"):
            return f"Could not find city: {city}"
        r = geo["results"][0]
        weather = requests.get(
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={r['latitude']}&longitude={r['longitude']}"
            f"&current=temperature_2m,wind_speed_10m,relative_humidity_2m"
            f"&temperature_unit=celsius"
        ).json()["current"]
        return (
            f"{city}, {r['country']}: {weather['temperature_2m']}°C, "
            f"wind {weather['wind_speed_10m']} km/h, "
            f"humidity {weather['relative_humidity_2m']}%"
        )

def run_tool_agent(user_message: str):
    if not os.getenv("GROQ_API_KEY"):
        return "⚠️ GROQ_API_KEY is not set. Add it to a `.env` file in the repo root.", []
    client = Groq()
    messages = [
        {"role": "system", "content": "You are a helpful assistant. When you need to calculate something, always use the calculator tool. Never compute things yourself."},
        {"role": "user", "content": user_message},
    ]
    tool_calls_made = []
    while True:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            max_tokens=1024,
            tools=TOOLS,
            tool_choice="auto",
            messages=messages,
        )
        msg = response.choices[0].message
        if response.choices[0].finish_reason == "stop":
            return msg.content, tool_calls_made
        for tc in msg.tool_calls:
            name = tc.function.name
            inputs = json.loads(tc.function.arguments)
            result = run_tool(name, inputs)
            tool_calls_made.append({"tool": name, "inputs": inputs, "result": result})
            messages.append(msg)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

# ── RAG agent setup ────────────────────────────────────────────────────────────
@st.cache_resource
def get_rag_collection():
    chroma = chromadb.PersistentClient(path=CHROMA_DIR)
    col = chroma.get_or_create_collection(COLLECTION_NAME)
    if col.count() == 0:
        rag_ingest(col)
    return col

@st.cache_resource
def get_groq_client():
    return Groq()

def run_rag_agent(user_message: str):
    if not os.getenv("GROQ_API_KEY"):
        return "⚠️ GROQ_API_KEY is not set. Add it to a `.env` file in the repo root.", []
    col = get_rag_collection()
    chunks, sources = rag_retrieve(col, user_message)
    reply = rag_answer(get_groq_client(), user_message, chunks, sources)
    traces = [{"source": m["source"], "chunk_idx": m["chunk"], "text": c} for c, m in zip(chunks, sources)]
    return reply, traces

# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "tool_traces" not in st.session_state:
    st.session_state.tool_traces = {}
if "rag_traces" not in st.session_state:
    st.session_state.rag_traces = {}

# ── Main UI ────────────────────────────────────────────────────────────────────
if mode.startswith("🧰"):
    st.markdown("# 🧠 Tool-Calling Agent")
    st.markdown("Ask anything. The agent decides which tools to use and chains them automatically.")
    suggestions = [
        "What's the weather in Tokyo?",
        "Is it warmer in London or Paris right now?",
        "What is sqrt(2) * 100 rounded to 2 decimal places?",
    ]
else:
    st.markdown("# 📚 RAG Agent")
    st.markdown("Ask about the docs in `data/docs/`. The agent retrieves the most relevant chunks, then answers grounded in them.")
    suggestions = [
        "What is QLoRA and why does it matter?",
        "When should I use fine-tuning instead of RAG?",
        "How does self-attention work?",
    ]

st.divider()

st.markdown("**Try asking:**")
cols = st.columns(3)
for i, (col, prompt) in enumerate(zip(cols, suggestions)):
    if col.button(prompt, key=f"sugg_{i}", use_container_width=True):
        st.session_state.pending_prompt = prompt

st.divider()

# Render chat history
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if i in st.session_state.tool_traces:
                traces = st.session_state.tool_traces[i]
                if traces:
                    with st.expander(f"🔧 {len(traces)} tool call(s)", expanded=False):
                        for t in traces:
                            st.markdown(f"**`{t['tool']}`**")
                            st.json(t["inputs"])
                            st.success(f"→ {t['result']}")
            if i in st.session_state.rag_traces:
                traces = st.session_state.rag_traces[i]
                if traces:
                    with st.expander(f"📚 Retrieved {len(traces)} chunks", expanded=False):
                        for t in traces:
                            st.markdown(f"**`{t['source']}` — chunk {t['chunk_idx']}**")
                            st.caption(t["text"])

pending = st.session_state.pop("pending_prompt", None)
user_input = st.chat_input("Ask the agent anything…") or pending

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        if mode.startswith("🧰"):
            with st.spinner("Thinking…"):
                reply, traces = run_tool_agent(user_input)
            st.markdown(reply)
            if traces:
                with st.expander(f"🔧 {len(traces)} tool call(s)", expanded=True):
                    for t in traces:
                        st.markdown(f"**`{t['tool']}`**")
                        st.json(t["inputs"])
                        st.success(f"→ {t['result']}")
            idx = len(st.session_state.messages)
            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.session_state.tool_traces[idx] = traces
        else:
            with st.spinner("Retrieving + generating…"):
                reply, traces = run_rag_agent(user_input)
            st.markdown(reply)
            if traces:
                with st.expander(f"📚 Retrieved {len(traces)} chunks", expanded=True):
                    for t in traces:
                        st.markdown(f"**`{t['source']}` — chunk {t['chunk_idx']}**")
                        st.caption(t["text"])
            idx = len(st.session_state.messages)
            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.session_state.rag_traces[idx] = traces
