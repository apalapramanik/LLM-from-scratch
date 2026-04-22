#!/bin/bash
set -e  # stop on any error

echo "🚀 Setting up llm-lab env on Swan..."

module load anaconda

# Create env (skip if exists)
conda create -n llm-lab python=3.11 -y || echo "Env already exists, continuing..."
conda activate llm-lab

echo "📦 Installing PyTorch with CUDA 12.1..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "📦 Installing HuggingFace stack..."
pip install transformers datasets huggingface_hub
pip install peft trl accelerate
pip install bitsandbytes

echo "📦 Installing agent stack..."
pip install anthropic langgraph langchain chromadb

echo "📦 Installing utilities..."
pip install rich python-dotenv sentencepiece einops

echo "✅ Verifying install..."
python -c "
import torch
import transformers
import peft
import trl
import bitsandbytes
import anthropic
print('Swan env ✅')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available on login node: {torch.cuda.is_available()}')
print('(CUDA will be True on compute nodes — this is expected)')
"

echo "💾 Saving requirements..."
pip freeze > requirements_hpc.txt

echo "🎉 Done!"
