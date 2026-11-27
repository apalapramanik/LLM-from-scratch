# Roadmap

Goal: be ready for an AI engineer interview in ~1 month. Each module builds on our code
and answers some of the 25 interview problems (P1–P25, listed at the bottom).

Legend: ✅ done · 🔨 in progress · ⬜ not started

No SLURM / HPC. Everything runs on the Mac (Groq API for LLMs, MLX for local models).

Rule for every module: **learn the concept → write/fix the code → add to NOTES.md → one spoken answer.**

---

## Already built (before this roadmap)

- ✅ `agents/basic_agent.py` — tool-calling loop (calculator, word_count, weather)
- ✅ `agents/rag_agent.py` — chunk → embed → Chroma → retrieve → grounded answer
- ✅ `app.py` — Streamlit UI for both
- ✅ `training/inference.py` — Qwen 0.5B inference

---

## Week 1 — Agents (P10, P11, P12, P23)

- ✅ **M1. Tool-calling agent, from the top** — walk through the code line by line
- ✅ **M2. Make it reliable** — dup-message bug, safe calc, timeouts, error handling, arg validation (P10)
- ✅ **M3. Stop loops** — max steps, repeated-call detection (P11)
- ✅ **M4. Tracing** — log every step: tool, input, output, latency, tokens (P23)
- ✅ **M5. Memory** (built in memory.py; not yet wired into the agent) — short-term window + summary + long-term vector memory (P12)

## Week 2 — RAG (P1, P2, P3, P4, P5)

- 🔨 **M6. RAG walkthrough** — what we built, why each step exists
- ⬜ **M7. Eval harness for retrieval** — 20 questions, hit@k, MRR (needed for everything after) (P2)
- ⬜ **M8. Better chunking** — recursive splitter, then compare vs fixed-size (P3)
- ⬜ **M9. BM25 + dense = hybrid search** (P4)
- ⬜ **M10. Reranker** — add it, measure if it helped (P5)
- ⬜ **M11. Scale thinking** — 10M docs: ANN indexes, sharding, filters (P1)

## Week 3 — Reliability, safety, evals (P13–P18, P24)

- ⬜ **M12. Structured output** — always-valid JSON (P13)
- ⬜ **M13. Hallucination** — abstain, groundedness check, citations (P14)
- ⬜ **M14. Eval suite** for a support bot (P15)
- ⬜ **M15. LLM-as-judge** — and how to trust it (P16)
- ⬜ **M16. Prompt injection + PII redaction** (P17, P18)
- ⬜ **M17. Prompt versioning + rollback** (P24)

## Week 4 — Cost, speed, models, system design (P6–P9, P19–P22, P25)

- ⬜ **M18. Streaming (SSE) + first-token latency** (P7, P22)
- ⬜ **M19. Semantic cache** with hit-rate tracking (P8)
- ⬜ **M20. Model router** cheap vs smart; cost cutting (P6, P9)
- ⬜ **M21. Fine-tune vs RAG vs prompting** (P19)
- ⬜ **M22. LoRA on a small model with MLX** (P20)
- ⬜ **M23. Serving concepts** — batching, KV cache, quantization (P21)
- ⬜ **M24. System design: ChatGPT for 1M users** (P25)

## Final days

- ⬜ Convert NOTES.md into slides
- ⬜ Say each of the 25 answers out loud, timed (3 min each)

---

## The 25 problems

1 RAG for 10M docs · 2 Debug wrong RAG answers · 3 Chunking for legal contracts · 4 BM25 vs dense vs hybrid ·
5 Reranker + measure · 6 Cut cost 70% · 7 First-token latency · 8 Semantic cache · 9 Cheap/smart routing ·
10 Reliable tool agent · 11 Agent stuck in loop · 12 Long-running memory · 13 Valid JSON · 14 Hallucinations ·
15 Eval suite · 16 LLM-as-judge · 17 Prompt injection · 18 PII redaction · 19 Fine-tune vs RAG vs prompt ·
20 LoRA on 5K examples · 21 Serve at scale · 22 SSE streaming · 23 Trace multi-agent · 24 Roll back prompt ·
25 ChatGPT-style for 1M users
