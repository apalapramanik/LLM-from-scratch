"""
Conversation memory: short-term window + rolling summary.

  [system] [summary of old turns] [last N turns, exact]

When history grows past `max_messages`, the oldest turns are summarized and
dropped. We only ever cut at a USER message, so an assistant tool-call is never
separated from its tool results (that would make the API reject the request).

`summarize` is a function you pass in (text -> short text). In the real agent it
calls the LLM; in tests it can be a fake.
"""


class ConversationMemory:
    def __init__(self, summarize, max_messages=12, keep_last=6):
        self.summarize = summarize
        self.max_messages = max_messages   # trim when history exceeds this
        self.keep_last = keep_last         # roughly how many recent messages to keep
        self.summary = ""
        self.messages = []                 # user / assistant / tool messages (no system)

    def add(self, message):
        self.messages.append(message)
        if len(self.messages) > self.max_messages:
            self._compress()

    def _cut_point(self):
        """Index of the first USER message at/after len-keep_last. Safe boundary."""
        start = max(len(self.messages) - self.keep_last, 0)
        for i in range(start, len(self.messages)):
            if _role(self.messages[i]) == "user":
                return i
        return 0   # no safe boundary → don't cut

    def _compress(self):
        cut = self._cut_point()
        if cut == 0:
            return
        old, self.messages = self.messages[:cut], self.messages[cut:]
        text = "\n".join(f"{_role(m)}: {_content(m)}" for m in old if _content(m))
        self.summary = self.summarize(
            f"Existing summary:\n{self.summary}\n\nNew conversation to merge in:\n{text}\n\n"
            "Write a short summary. Keep names, numbers, preferences and decisions."
        )

    def build(self, system_prompt):
        """The list of messages to send to the model."""
        sys_text = system_prompt
        if self.summary:
            sys_text += f"\n\nSummary of earlier conversation:\n{self.summary}"
        return [{"role": "system", "content": sys_text}] + self.messages


class LongTermMemory:
    """Durable facts stored as embeddings (same idea as RAG). Per-user, timestamped, deletable."""

    def __init__(self, client, name="long_term_memory"):
        self.col = client.get_or_create_collection(name)

    def remember(self, user_id, fact):
        import time, uuid
        self.col.add(
            documents=[fact],
            metadatas=[{"user_id": user_id, "ts": time.time()}],
            ids=[uuid.uuid4().hex],
        )

    def recall(self, user_id, query, k=3):
        if self.col.count() == 0:
            return []
        res = self.col.query(query_texts=[query], n_results=k, where={"user_id": user_id})
        return res["documents"][0]

    def forget_user(self, user_id):
        self.col.delete(where={"user_id": user_id})   # privacy: delete on request


def _role(m):
    return m["role"] if isinstance(m, dict) else m.role

def _content(m):
    return (m.get("content") if isinstance(m, dict) else m.content) or ""


if __name__ == "__main__":
    fake_summarize = lambda prompt: f"[summary of {prompt.count(chr(10))} lines]"
    mem = ConversationMemory(fake_summarize, max_messages=6, keep_last=3)
    for i in range(5):
        mem.add({"role": "user", "content": f"question {i}"})
        mem.add({"role": "assistant", "content": f"answer {i}"})
    print("summary:", mem.summary)
    for m in mem.build("You are helpful."):
        print(f"{m['role']:10} {m['content'][:70]!r}")
