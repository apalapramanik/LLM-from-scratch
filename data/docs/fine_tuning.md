# Fine-tuning vs RAG

These are two different ways to adapt a language model to your data, and they solve different problems.

**Fine-tuning** changes the model's weights by training it on examples. The new knowledge or behavior gets baked into the parameters. Use it for:
- Teaching the model a new style or tone
- Specializing on a domain's vocabulary (medical, legal, code)
- Making the model follow a specific output format reliably

**RAG** keeps the model frozen and retrieves relevant context at query time. Use it for:
- Answering questions about documents the model hasn't seen
- Working with frequently-changing data
- Producing answers with traceable citations

**Common mistake:** people fine-tune when they should RAG. If your goal is "the model should know about our company's internal docs," fine-tuning is usually the wrong tool — you'll burn GPU time, and the model will hallucinate when asked about details that weren't in the training set. RAG handles this case far better.

**The combined approach:** Many production systems use both. Fine-tune for behavior (tone, format, refusals), then RAG for knowledge (current data, citations). The fine-tune teaches the model *how* to respond; RAG provides *what* to respond about.

**Cost:** Fine-tuning has a one-time training cost (GPU hours) plus the cost of serving the new model. RAG has ongoing per-query costs (embedding + retrieval + larger prompts) but no training cost. For most teams, RAG is cheaper to start with.
