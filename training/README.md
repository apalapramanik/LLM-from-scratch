# Training

This folder contains scripts for running and training language models on an HPC cluster using GPU nodes.

---

## What is Inference?

Inference is running a model to get a response — no training involved:

```
Text → Tokenizer → Token IDs → GPU → Model → Token IDs → Tokenizer → Text
```

1. **Tokenizer** converts text into token IDs (numbers) the model understands
2. **Model** predicts the next token one at a time, feeding each output back as input
3. **Tokenizer** converts the generated token IDs back into human-readable text

Instruct models expect a specific chat format. The `apply_chat_template` function handles this automatically before passing input to the model.

---

## Files

### `inference.py`

Loads `Qwen2.5-0.5B-Instruct` (494M parameters) from HuggingFace and runs inference on a GPU.

What it demonstrates:
- Loading a tokenizer and model from HuggingFace Hub
- `device_map="auto"` — automatically places model layers on GPU
- `dtype=torch.float16` — half precision, halves VRAM usage
- `apply_chat_template` — formats messages correctly for instruct models
- `model.generate` — generates response tokens one at a time
- Decoding only the generated tokens, not the input prompt

### `slurm/`

SLURM job scripts for submitting jobs to the cluster scheduler.

| Script | Use for |
|---|---|
| `dev.sh` | Quick test runs, debugging |
| `train.sh` | Standard training runs |
| `big_train.sh` | Large model runs requiring more VRAM |

```bash
# Submit a job
sbatch training/slurm/dev.sh training/inference.py

# Check status
squeue -u $USER

# Interactive GPU session
srun --partition=gpu --gres=gpu:1 --mem=16G --time=02:00:00 --pty bash
```

---

## Environment

```bash
module load anaconda
conda activate llm-lab
```

Key packages:

```
torch          — PyTorch with CUDA support
transformers   — model loading, tokenizers, generation
datasets       — dataset loading and preprocessing
```