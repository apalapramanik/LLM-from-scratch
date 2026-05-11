"""
RAG agent — retrieve from a vector DB, then generate.

Flow:
  docs → chunks → embeddings → ChromaDB        (ingest, once)
  query → top-k chunks → LLM with context      (every query)

No frameworks — just chromadb + raw Groq API so you can see what each step does.
"""
import os
import glob
import chromadb
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────
DOCS_DIR        = "data/docs"
CHROMA_DIR      = "chroma_db"
COLLECTION_NAME = "knowledge_base"
TOP_K           = 3
CHUNK_SIZE      = 400   # characters per chunk
CHUNK_OVERLAP   = 50    # overlap so context isn't cut mid-thought
MODEL           = "meta-llama/llama-4-scout-17b-16e-instruct"

# ── Chunking ──────────────────────────────────────────────────────────────────
# Real systems use semantic / sentence-aware chunkers (e.g. langchain's
# RecursiveCharacterTextSplitter). For learning purposes a fixed-size sliding
# window with overlap is the simplest thing that actually works.
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks

# ── Ingest ────────────────────────────────────────────────────────────────────
# Loads every .md file under DOCS_DIR, chunks each one, and adds the chunks to
# Chroma. Chroma uses its default embedding function (sentence-transformers
# all-MiniLM-L6-v2) — downloads the model on first run, then caches it.
def ingest(collection):
    docs, metas, ids = [], [], []
    paths = sorted(glob.glob(f"{DOCS_DIR}/*.md"))

    for path in paths:
        with open(path) as f:
            text = f.read()
        source = os.path.basename(path)
        for i, chunk in enumerate(chunk_text(text)):
            docs.append(chunk)
            metas.append({"source": source, "chunk": i})
            ids.append(f"{source}::{i}")

    collection.add(documents=docs, metadatas=metas, ids=ids)
    sources = sorted({m["source"] for m in metas})
    print(f"📥 Ingested {len(docs)} chunks from {len(sources)} files: {sources}")

# ── Retrieve ──────────────────────────────────────────────────────────────────
# Chroma embeds the query with the same model used at ingest, then returns the
# top-k chunks by cosine similarity.
def retrieve(collection, query: str, top_k: int = TOP_K):
    results = collection.query(query_texts=[query], n_results=top_k)
    return results["documents"][0], results["metadatas"][0]

# ── Generate ──────────────────────────────────────────────────────────────────
# Pack retrieved chunks into a single context string and pass it to the LLM.
# The system prompt is what makes this "grounded" — the model is told to use
# ONLY the context, not its own knowledge.
def answer(client: Groq, query: str, chunks: list, sources: list) -> str:
    context = "\n\n---\n\n".join(
        f"[Source: {m['source']}, chunk {m['chunk']}]\n{c}"
        for c, m in zip(chunks, sources)
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You answer questions using ONLY the context below. "
                "If the answer isn't in the context, say 'I don't have that information.' "
                "Cite the source filename in parentheses after each claim."
            )
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}"
        }
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=512,
        temperature=0.2,    # low temp — we want grounded answers, not creative ones
    )
    return response.choices[0].message.content

# ── Run it ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        raise SystemExit("⚠️  GROQ_API_KEY not set. Add it to .env in the repo root.")

    # PersistentClient writes the DB to disk so we don't re-embed every run
    chroma = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma.get_or_create_collection(COLLECTION_NAME)

    if collection.count() == 0:
        ingest(collection)
    else:
        print(f"📚 Using existing collection with {collection.count()} chunks")

    client = Groq()

    queries = [
        "What is QLoRA and why does it matter?",
        "When should I use fine-tuning instead of RAG?",
        "What's the difference between self-attention and a feed-forward layer?",
        "What's the best pizza topping?",   # not in the docs — model should say so
    ]

    for q in queries:
        print(f"\n{'─' * 70}")
        print(f"🧑  {q}")

        chunks, sources = retrieve(collection, q)
        print(f"📚  Retrieved {len(chunks)} chunks:")
        for s in sources:
            print(f"    • {s['source']} (chunk {s['chunk']})")

        print(f"🤖  {answer(client, q, chunks, sources)}")
