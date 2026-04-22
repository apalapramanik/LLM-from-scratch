import json
import math
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

client = Groq()

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
    }
]

# ── Tool execution ─────────────────────────────────────────────────────────────
def run_tool(name: str, inputs: dict) -> str:
    if name == "calculator":
        # Replace ^ with ** so the model can use either notation
        expression = inputs["expression"].replace("^", "**")
        result = eval(expression, {"__builtins__": {}}, vars(math))
        return str(result)
    
    elif name == "word_count":
        count = len(inputs["text"].split())
        return str(count)

# ── Agent loop ─────────────────────────────────────────────────────────────────
def run_agent(user_message: str):
    print(f"\n🧑 You: {user_message}")
    
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. When you need to calculate something, always use the calculator tool. Never try to compute things yourself."
        },
        {"role": "user", "content": user_message}
    ]
    
    while True:
        response = client.chat.completions.create(
            # Change the model in your run_agent function
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            max_tokens=1024,
            tools=tools,
            tool_choice="auto",
            messages=messages
        )
        
        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason
        
        if finish_reason == "stop":
            print(f"🤖 Agent: {message.content}")
            return message.content
        
        if finish_reason == "tool_calls":
            for tool_call in message.tool_calls:
                name = tool_call.function.name
                inputs = json.loads(tool_call.function.arguments)
                
                print(f"🔧 Tool: {name}({inputs})")
                result = run_tool(name, inputs)
                print(f"   → {result}")
                
                messages.append(message)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
# ── Run it ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    run_agent("What is the square root of 144?")
    run_agent("How many words are in 'The quick brown fox jumps over the lazy dog'?")
    run_agent("Calculate the area of a circle with radius 7. The formula is pi * r^2.")