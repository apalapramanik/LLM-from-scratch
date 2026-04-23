import json
import math
import os
import requests
import streamlit as st
from groq import Groq
from dotenv import load_dotenv

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
    st.markdown("Tool-calling agent powered by **Llama 4 Scout** via Groq.")
    st.divider()

    st.markdown("### 🛠 Available Tools")
    with st.expander("🧮 calculator", expanded=False):
        st.markdown("Evaluates any Python math expression.")
        st.code("sqrt(144)  →  12.0", language="python")
    with st.expander("📝 word_count", expanded=False):
        st.markdown("Counts words in a block of text.")
    with st.expander("🌤 get_weather", expanded=False):
        st.markdown("Fetches live weather for any city via Open-Meteo (no API key needed).")

    st.divider()
    st.markdown("### 🔬 Model")
    st.code("meta-llama/llama-4-scout-17b-16e-instruct", language="text")
    st.markdown("### ⚡ Inference")
    st.code("Groq Cloud (free tier)", language="text")

    st.divider()
    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key:
        st.error("⚠️ GROQ_API_KEY not set.\nCreate a `.env` file in the repo root:\n```\nGROQ_API_KEY=\"gsk_...\"\n```")

    if st.button("🗑 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.tool_traces = {}
        st.rerun()

# ── Tools ──────────────────────────────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluates a math expression. Use this whenever the user asks a math question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A valid Python math expression e.g. '2 + 2' or 'sqrt(144)'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "word_count",
            "description": "Counts the number of words in a given text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text to count words in"}
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Gets the current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name e.g. 'Tokyo'"}
                },
                "required": ["city"]
            }
        }
    }
]

def run_tool(name: str, inputs: dict) -> str:
    if name == "calculator":
        expression = inputs["expression"].replace("^", "**")
        result = eval(expression, {"__builtins__": {}}, vars(math))
        return str(result)
    elif name == "word_count":
        return str(len(inputs["text"].split()))
    elif name == "get_weather":
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

def run_agent(user_message: str):
    if not os.getenv("GROQ_API_KEY"):
        return "⚠️ GROQ_API_KEY is not set. Add it to a `.env` file in the repo root.", []
    client = Groq()
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. When you need to calculate something, always use the calculator tool. Never compute things yourself."
        },
        {"role": "user", "content": user_message}
    ]
    tool_calls_made = []

    while True:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            max_tokens=1024,
            tools=TOOLS,
            tool_choice="auto",
            messages=messages
        )
        msg = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        if finish_reason == "stop":
            return msg.content, tool_calls_made

        if finish_reason == "tool_calls":
            for tc in msg.tool_calls:
                name = tc.function.name
                inputs = json.loads(tc.function.arguments)
                result = run_tool(name, inputs)
                tool_calls_made.append({"tool": name, "inputs": inputs, "result": result})
                messages.append(msg)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "tool_traces" not in st.session_state:
    st.session_state.tool_traces = {}

# ── Main UI ────────────────────────────────────────────────────────────────────
st.markdown("# 🧠 LLM Agent — Live Demo")
st.markdown(
    "Ask anything. The agent decides which tools to use and chains them automatically."
)
st.divider()

# Suggested prompts
st.markdown("**Try asking:**")
cols = st.columns(3)
suggestions = [
    "What's the weather in Tokyo?",
    "Is it warmer in London or Paris right now?",
    "What is sqrt(2) * 100 rounded to 2 decimal places?",
]
for i, (col, prompt) in enumerate(zip(cols, suggestions)):
    if col.button(prompt, key=f"sugg_{i}", use_container_width=True):
        st.session_state.pending_prompt = prompt

st.divider()

# Render chat history
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and i in st.session_state.tool_traces:
            traces = st.session_state.tool_traces[i]
            if traces:
                with st.expander(f"🔧 {len(traces)} tool call(s) made", expanded=False):
                    for t in traces:
                        st.markdown(f"**`{t['tool']}`**")
                        st.json(t["inputs"])
                        st.success(f"→ {t['result']}")

# Handle suggested prompt click
pending = st.session_state.pop("pending_prompt", None)

# Chat input
user_input = st.chat_input("Ask the agent anything…") or pending

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            reply, traces = run_agent(user_input)
        st.markdown(reply)
        if traces:
            with st.expander(f"🔧 {len(traces)} tool call(s) made", expanded=True):
                for t in traces:
                    st.markdown(f"**`{t['tool']}`**")
                    st.json(t["inputs"])
                    st.success(f"→ {t['result']}")

    msg_index = len(st.session_state.messages)
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.session_state.tool_traces[msg_index] = traces
