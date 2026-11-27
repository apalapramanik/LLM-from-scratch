# Roadmap

Goal: be ready for an AI engineer interview, and learn to build and fine-tune LLM systems.
Each module builds on our code and answers some of the 25 interview problems (P1–P25, listed at the bottom).

Legend: ✅ done · 🔨 in progress · ⬜ not started

No SLURM / HPC. Everything runs on the Mac (Groq API for hosted LLMs, MLX for local models).

Rule for every module: **learn the concept → write/fix the code → add to NOTES.md → one spoken answer.**

Plan length: about 5 weeks. If we need to hit 4, Week 5 is mostly concepts and can be compressed.

---

## Already built (before this roadmap)

- ✅ `agents/basic_agent.py` — tool-calling loop (calculator, word_count, weather)
- ✅ `agents/rag_agent.py` — chunk → embed → Chroma → retrieve → grounded answer
- ✅ `app.py` — Streamlit UI for both
- ✅ `training/inference.py` — Qwen 0.5B inference

---

## Week 1 — Agents ✅ (P10, P11, P12, P23)

- ✅ **M1. Tool-calling agent, from the top** — walk through the code line by line
- ✅ **M2. Make it reliable** — dup-message bug, safe calc, timeouts, error handling, arg validation (P10)
- ✅ **M3. Stop loops** — max steps, repeated-call detection (P11)
- ✅ **M4. Tracing** — spans with tool, input, output, latency, tokens (P23)
- ✅ **M5. Memory** — window + summary + long-term vector memory (P12); built in `memory.py`, not yet wired into the agent

## Week 2 — RAG (P1, P2, P3, P4, P5)

- 🔨 **M6. RAG walkthrough** — what we built, why each step exists
- ⬜ **M7. Eval harness for retrieval** — 20 questions, hit@k, MRR (everything after depends on it) (P2)
- ⬜ **M8. Better chunking** — recursive splitter, compare vs fixed-size (P3)
- ⬜ **M9. BM25 + dense = hybrid search** (P4)
- ⬜ **M10. Reranker** — add it, measure if it helped (P5)
- ⬜ **M11. Scale thinking** — 10M docs: ANN indexes, sharding, filters (P1)

## Week 3 — Fine-tuning and local models (P19, P20)  ← moved earlier

- ⬜ **M12. Local inference with MLX** — run Qwen on the Mac; tokens, sampling, chat templates
- ⬜ **M13. Fine-tune vs RAG vs prompting** — when to pick which, and how to defend it (P19)
- ⬜ **M14. Prepare a 5K-example dataset** — format, clean, train/validation split
- ⬜ **M15. LoRA fine-tune with MLX** — what LoRA is (rank, alpha, target modules), run it (P20)
- ⬜ **M16. Evaluate base vs tuned** — same prompts side by side, a small metric, spot overfitting

## Week 4 — Reliability, safety, evals (P13–P18, P24)

- ⬜ **M17. Structured output** — always-valid JSON (P13)
- ⬜ **M18. Hallucination** — abstain, groundedness check, citations (P14)
- ⬜ **M19. Eval suite** for a customer support bot (P15)
- ⬜ **M20. LLM-as-judge** — and how to trust it (P16)
- ⬜ **M21. Prompt injection + PII redaction** (P17, P18)
- ⬜ **M22. Prompt versioning + rollback** (P24)

## Week 5 — Cost, speed, serving, system design (P6–P9, P21, P22, P25)

- ⬜ **M23. Streaming (SSE) + first-token latency** (P7, P22)
- ⬜ **M24. Semantic cache** with hit-rate tracking (P8)
- ⬜ **M25. Model router** cheap vs smart; cutting cost (P6, P9)
- ⬜ **M26. Serving concepts** — batching, KV cache, quantization (P21)
- ⬜ **M27. System design: ChatGPT-style chat for 1M users** (P25)

## Final days

- ⬜ Convert NOTES.md into slides
- ⬜ Say each of the 25 answers out loud, timed (3 min each)

---

## Extras (after the main plan — things we dropped or deferred)

Pick these up if time allows. None are needed for the interview list.

- ⬜ Wire `memory.py` into the agent loop, and add the tracer to `app.py`
- ⬜ `web_search` tool for the agent (DuckDuckGo or Tavily free tier)
- ⬜ LangGraph version of the agent — same behavior as a state machine, for framework comparison
- ⬜ A tiny GPT from scratch in PyTorch (attention, blocks, training loop) — the "LLM from scratch" part of this repo's name
- ⬜ QLoRA on a 7B model, if GPU access comes back (needs a GPU box, not the Mac)
- ⬜ Deploy the Streamlit app (public URL) with a rate limit and cost cap

---

## The 25 problems

1 RAG for 10M docs · 2 Debug wrong RAG answers · 3 Chunking for legal contracts · 4 BM25 vs dense vs hybrid ·
5 Reranker + measure · 6 Cut cost 70% · 7 First-token latency · 8 Semantic cache · 9 Cheap/smart routing ·
10 Reliable tool agent · 11 Agent stuck in loop · 12 Long-running memory · 13 Valid JSON · 14 Hallucinations ·
15 Eval suite · 16 LLM-as-judge · 17 Prompt injection · 18 PII redaction · 19 Fine-tune vs RAG vs prompt ·
20 LoRA on 5K examples · 21 Serve at scale · 22 SSE streaming · 23 Trace multi-agent · 24 Roll back prompt ·
25 ChatGPT-style for 1M users
