# Retrieval-Augmented Generation (RAG)

RAG is a way to give a language model access to information it wasn't trained on, without fine-tuning. The model stays frozen — you change what it sees, not what it knows.

**The flow:**
1. **Ingest** — split your documents into chunks (a few hundred characters each)
2. **Embed** — convert each chunk into a vector using an embedding model
3. **Store** — put the vectors into a vector database (Chroma, Pinecone, Weaviate, etc.)
4. **Retrieve** — at query time, embed the user's question, find the top-k most similar chunks
5. **Generate** — pass the retrieved chunks to the LLM as context and ask it to answer

**Why this beats fine-tuning for many use cases:**
- **Fresh data** — you can add new documents instantly without retraining
- **Citations** — you know which document the answer came from
- **Smaller models** — a small model with good retrieved context often beats a large model relying on memorized knowledge

**Embeddings 101:** An embedding model maps text to a fixed-size vector (typically 384, 768, or 1536 dimensions). Texts with similar meaning end up close together in vector space. The default in ChromaDB is `all-MiniLM-L6-v2`, a 384-dimensional sentence-transformer model that's tiny and fast.

**Retrieval quality matters more than the LLM.** If retrieval pulls in the wrong chunks, even GPT-4 will give a bad answer. Good chunking (semantic, not just fixed-size) and hybrid search (vector + keyword) are usually the biggest levers.
