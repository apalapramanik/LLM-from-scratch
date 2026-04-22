"""
Goal: load a HuggingFace model and run inference on Swan
Model: Qwen2.5-0.5B-Instruct (tiny, fast, good for testing)
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print()

# ── Load tokenizer ─────────────────────────────────────────────────────────────
# Tokenizer converts text → token IDs (numbers the model understands)
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# ── Load model ─────────────────────────────────────────────────────────────────
# torch_dtype=float16 — uses half precision, halves VRAM usage
# device_map="auto"   — automatically puts layers on GPU
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
    device_map="auto"
)

print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")
print()

# ── Run inference ──────────────────────────────────────────────────────────────
def ask(question: str):
    # Format as a chat message — instruct models expect this format
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": question}
    ]
    
    # apply_chat_template converts messages → model input format
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize — convert text to token IDs, move to GPU
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    
    # Generate — the model predicts next tokens one at a time
    with torch.no_grad():    # no_grad = don't track gradients, saves memory
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,    # max tokens to generate
            temperature=0.7,       # randomness (0=deterministic, 1=creative)
            do_sample=True,        # use sampling (vs greedy)
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode — convert token IDs back to text
    # Skip the input tokens, only decode the generated part
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)

# ── Test it ────────────────────────────────────────────────────────────────────
questions = [
    "What is machine learning in one sentence?",
    "Write a Python function that reverses a string.",
    "What is the capital of France?"
]

for q in questions:
    print(f"Q: {q}")
    print(f"A: {ask(q)}")
    print()