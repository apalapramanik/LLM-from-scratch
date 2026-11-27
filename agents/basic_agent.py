import json
import math
import requests
from groq import Groq
from dotenv import load_dotenv
from safe_calc import safe_calc
from tool_utils import validate_args
from tracer import Trace   # run from agents/ folder: python basic_agent.py
load_dotenv()

client = Groq()

MAX_STEPS = 8   # hard limit on agent turns

# ── Tools ──────────────────────────────────────────────────────────────────────
tools = [
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

    # NEW TOOL ↓
    {
        "type": "function",
        "function": {
            "name": "word_count",
            "description": "Counts the number of words in a given text. Use this when the user asks how many words are in something.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to count words in"
                    }
                },
                "required": ["text"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Gets the current weather for a given city. Use this when the user asks about weather anywhere.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city e.g. 'Tokyo' or 'New York'"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

# ── Tool execution ─────────────────────────────────────────────────────────────
def run_tool(name: str, inputs: dict) -> str:
    """Safety net: a tool must never crash the agent. Errors become text the model can read."""
    problem = validate_args(tools, name, inputs)
    if problem:
        return f"Error: {problem}"          # clear, fixable message for the model
    try:
        result = _run_tool(name, inputs)
        return result if result is not None else f"Error: unknown tool '{name}'"
    except Exception as e:
        return f"Error running {name}: {type(e).__name__}: {e}"


def parse_args(raw: str) -> dict:
    """Model sometimes sends broken JSON. Return {} instead of crashing."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _run_tool(name: str, inputs: dict) -> str:
    if name == "calculator":
        # Whitelist-based calculator (no eval) — see safe_calc.py
        try:
            return str(safe_calc(inputs["expression"]))
        except Exception as e:
            return f"Calculator error: {e}"   # tell the model, don't crash
    
    elif name == "word_count":
        count = len(inputs["text"].split())
        return str(count)
    
    elif name == "get_weather":
        city = inputs["city"]
        
        # Step 1: convert city name to coordinates
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
        geo = requests.get(geo_url, timeout=5).json()
        
        if not geo.get("results"):
            return f"Could not find city: {city}"
        
        lat = geo["results"][0]["latitude"]
        lon = geo["results"][0]["longitude"]
        country = geo["results"][0]["country"]
        
        # Step 2: fetch weather using coordinates
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,wind_speed_10m,relative_humidity_2m"
            f"&temperature_unit=celsius"
        )
        weather = requests.get(weather_url, timeout=5).json()
        current = weather["current"]
        
        return (
            f"Weather in {city}, {country}: "
            f"{current['temperature_2m']}°C, "
            f"wind {current['wind_speed_10m']} km/h, "
            f"humidity {current['relative_humidity_2m']}%"
        )

# ── Agent loop ─────────────────────────────────────────────────────────────────
def finish(trace, answer):
    """Close the trace, print the tree, save JSON, return the answer."""
    trace.finish(answer)
    print(f"\n📈 Trace {trace.id} ({trace.total_tokens()} tokens):\n{trace.pretty()}")
    print(f"   saved → {trace.save()}")
    return answer


def run_agent(user_message: str):
    print(f"\n🧑 You: {user_message}")
    
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. When you need to calculate something, always use the calculator tool. Never try to compute things yourself."
        },
        {"role": "user", "content": user_message}
    ]
    
    seen = set()   # tool calls already made, for repeat detection
    trace = Trace(user_message)

    for step in range(MAX_STEPS):            # hard limit: replaces `while True`
        with trace.span("model_call", input={"messages": len(messages)}) as span:
            response = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                max_tokens=1024,
                tools=tools,
                tool_choice="auto",
                messages=messages
            )
            usage = getattr(response, "usage", None)
            if usage:
                span.meta["total_tokens"] = usage.total_tokens

        message = response.choices[0].message

        # No tool requested → the model is done (or was cut off). Either way, stop.
        if not message.tool_calls:
            print(f"🤖 Agent: {message.content}")
            return finish(trace, message.content)

        # Append the assistant's request ONCE, before its tool results.
        messages.append(message)
        for tool_call in message.tool_calls:
            name = tool_call.function.name
            inputs = parse_args(tool_call.function.arguments)

            key = (name, json.dumps(inputs, sort_keys=True))
            with trace.span(f"tool:{name}", input=inputs) as span:
                if key in seen:              # early guard: exact repeat
                    result = "You already made this exact call. Use a different approach or answer now."
                    span.meta["repeat_blocked"] = True
                else:
                    seen.add(key)
                    print(f"🔧 Tool: {name}({inputs})")
                    result = run_tool(name, inputs)
                span.output = result
            print(f"   → {result}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

    msg = f"Stopped after {MAX_STEPS} steps without a final answer."
    print(f"⚠️ {msg}")
    return finish(trace, msg)

# ── Run it ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

   
    run_agent("What is the square root of 144?")
    run_agent("What's the weather like in Tokyo right now?")
    run_agent("Is it warmer in London or Paris today?")