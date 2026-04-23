# LLM-from-scratch

A hands-on learning repo for building, fine-tuning, and deploying large language models and AI agents from the ground up. Everything here is built step by step — no black boxes.

---

## Live Demo

![Agent Demo](assets/demo.png)

> **Run it locally:**
> ```bash
> conda activate llm-lab
> streamlit run app.py
> ```
> Requires a `GROQ_API_KEY` in your `.env` file.

---

## Goal

The goal is to go from zero to being able to:
- Fine-tune open-source LLMs on custom datasets
- Build tool-calling agents that interact with real APIs and data
- Deploy models and agents in production

Both tracks are developed in parallel — agents on a local machine, model training on an HPC cluster with GPU nodes.

---

## What We Have So Far

### Agents — `agents/`

A tool-calling agent built from scratch using the Groq API and Llama 4 Scout. No frameworks — just raw API calls so you understand exactly what's happening under the hood.

The agent can:
- Do math using a calculator tool
- Count words in text
- Fetch live weather for any city in the world using the Open-Meteo API

The key insight: the model never runs any of this code itself. It reads tool descriptions, decides which tool to call and with what inputs, and your Python code does the actual work. The model is the brain. Your code is the hands.

### Training — `training/`

Inference pipeline running on a GPU cluster. Loads `Qwen2.5-0.5B-Instruct` (a 494M parameter model by Alibaba) from HuggingFace and runs it on a GPU — proving the full stack works end to end before moving to fine-tuning.

---

## Repo Structure

```
LLM-from-scratch/
├── agents/          # Tool-calling agents
├── training/        # Inference and fine-tuning scripts
│   └── slurm/       # HPC job submission scripts
├── data/            # Datasets and preprocessing
├── notebooks/       # Exploration and experiments
├── configs/         # Model and training configs
├── scripts/         # Utility and setup scripts
└── requirements.txt         # Mac/dev dependencies
    requirements_hpc.txt     # HPC dependencies
```

---

## Model Families

Open-source LLMs come from several major families. The ones used or planned in this repo:

| Family | Made by | Known for |
|---|---|---|
| **Llama** | Meta | Most popular open-source family, powers most of the OSS ecosystem |
| **Qwen** | Alibaba | Strong at coding and math, what we use for training |
| **Gemma** | Google | Lightweight, runs on smaller hardware |
| **Mistral** | Mistral AI | Efficient, punches above its weight |

The size of a model is measured in parameters — the numbers learned during training:

```
0.5B   →  tiny, fast, good for learning
7–8B   →  sweet spot for fine-tuning on a single GPU
70B    →  high quality, needs multiple GPUs
```

---

## Tech Stack

**Language & Environment**

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![Conda](https://img.shields.io/badge/conda-llm--lab-44A833?style=flat-square&logo=anaconda&logoColor=white)
![JupyterLab](https://img.shields.io/badge/JupyterLab-4.x-F37626?style=flat-square&logo=jupyter&logoColor=white)

**LLM APIs & SDKs**

![Groq](https://img.shields.io/badge/Groq-API-F55036?style=flat-square&logoColor=white)
![Anthropic](https://img.shields.io/badge/Anthropic-SDK-191919?style=flat-square&logo=anthropic&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Hub-FFD21E?style=flat-square&logo=huggingface&logoColor=black)

**Agent Frameworks**

![LangChain](https://img.shields.io/badge/LangChain-1.2-1C3C3C?style=flat-square&logo=langchain&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1.1-1C3C3C?style=flat-square&logo=langchain&logoColor=white)

**Vector Store / RAG**

![ChromaDB](https://img.shields.io/badge/ChromaDB-1.5-E85D04?style=flat-square&logo=databricks&logoColor=white)

**Model Training & Fine-tuning**

![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900?style=flat-square&logo=nvidia&logoColor=white)
![Transformers](https://img.shields.io/badge/🤗_Transformers-5.6-FFD21E?style=flat-square&logoColor=black)
![PEFT](https://img.shields.io/badge/PEFT-LoRA%20%2F%20QLoRA-FFD21E?style=flat-square&logoColor=black)
![TRL](https://img.shields.io/badge/TRL-SFT%20%2F%20RLHF-FFD21E?style=flat-square&logoColor=black)
![Accelerate](https://img.shields.io/badge/Accelerate-HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![BitsAndBytes](https://img.shields.io/badge/BitsAndBytes-4--bit%20Quant-6E40C9?style=flat-square&logoColor=white)

**Apple Silicon**

![MLX](https://img.shields.io/badge/Apple_MLX-0.31-000000?style=flat-square&logo=apple&logoColor=white)

**Infrastructure**

![SLURM](https://img.shields.io/badge/SLURM-HPC_Scheduler-0D6EFD?style=flat-square&logo=linux&logoColor=white)
![dotenv](https://img.shields.io/badge/.env-Secrets-ECD53F?style=flat-square&logo=dotenv&logoColor=black)

---

## Setup

**Mac (agent development):**
```bash
conda activate llm-lab
pip install -r requirements.txt
```

**HPC (model training):**
```bash
module load anaconda
conda activate llm-lab
```

Create a `.env` file in the repo root for API keys — never commit this file:
```
GROQ_API_KEY="gsk_..."
HF_TOKEN="hf_..."
```