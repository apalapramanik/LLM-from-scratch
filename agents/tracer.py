"""
Tiny tracer — records every step of an agent run as a tree of spans.

A span = one unit of work: name, input, output, duration, extra data (tokens, errors).
A trace = all spans for one user request. Spans can nest (children).

Usage:
    trace = Trace("user question")
    with trace.span("tool:get_weather", input={"city": "Paris"}) as s:
        s.output = run_tool(...)
    trace.save()          # writes traces/<id>.json
    print(trace.pretty())
"""
import json
import os
import time
import uuid
from contextlib import contextmanager

TRACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "traces")


class Span:
    def __init__(self, name, input=None):
        self.name = name
        self.input = input
        self.output = None
        self.error = None
        self.meta = {}              # tokens, model name, etc.
        self.start = time.time()
        self.duration = None
        self.children = []

    def to_dict(self):
        return {
            "name": self.name, "input": self.input, "output": self.output,
            "error": self.error, "meta": self.meta,
            "duration_s": round(self.duration or 0, 3),
            "children": [c.to_dict() for c in self.children],
        }


class Trace:
    def __init__(self, user_input):
        self.id = uuid.uuid4().hex[:8]
        self.user_input = user_input
        self.root = Span("request", input=user_input)
        self._stack = [self.root]   # current open span is the top

    @contextmanager
    def span(self, name, input=None):
        s = Span(name, input)
        self._stack[-1].children.append(s)   # attach to current parent
        self._stack.append(s)
        try:
            yield s
        except Exception as e:               # record the error, then re-raise
            s.error = f"{type(e).__name__}: {e}"
            raise
        finally:
            s.duration = time.time() - s.start
            self._stack.pop()

    def finish(self, output=None):
        self.root.output = output
        self.root.duration = time.time() - self.root.start

    def total_tokens(self):
        def walk(s):
            return s.meta.get("total_tokens", 0) + sum(walk(c) for c in s.children)
        return walk(self.root)

    def save(self):
        os.makedirs(TRACE_DIR, exist_ok=True)
        path = os.path.join(TRACE_DIR, f"{self.id}.json")
        with open(path, "w") as f:
            json.dump({"id": self.id, **self.root.to_dict()}, f, indent=2, default=str)
        return path

    def pretty(self):
        lines = []
        def walk(s, depth):
            flag = " ❌ " + s.error if s.error else ""
            tok = f" {s.meta['total_tokens']}tok" if "total_tokens" in s.meta else ""
            lines.append(f"{'  ' * depth}├─ {s.name} ({(s.duration or 0):.2f}s{tok}){flag}")
            for c in s.children:
                walk(c, depth + 1)
        walk(self.root, 0)
        return "\n".join(lines)


if __name__ == "__main__":
    t = Trace("Is it warmer in London or Paris?")
    with t.span("model_call") as s:
        time.sleep(0.05); s.meta["total_tokens"] = 340
    with t.span("tool:get_weather", input={"city": "London"}) as s:
        time.sleep(0.02); s.output = "12C"
    try:
        with t.span("tool:get_weather", input={"city": "???"}) as s:
            raise ValueError("city not found")
    except ValueError:
        pass
    t.finish("Paris is warmer")
    print(t.pretty())
    print("total tokens:", t.total_tokens())
