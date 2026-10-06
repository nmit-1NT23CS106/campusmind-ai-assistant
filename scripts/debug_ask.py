"""Show exactly what the model says at every step of one question.

Usage:  python -m scripts.debug_ask "What is the minimum attendance to sit the exams?" --student 20260001
        python -m scripts.debug_ask "Which students are below 75% in CS301?" --admin
Needs no running server. Prints no secrets.
"""
import argparse
import json

from app.llm import get_llm
from app.services import build_state
from app.sql_tool import Scope


class Tap:
    """Wraps the real LLM and prints each exchange."""

    def __init__(self, inner):
        self.inner, self.calls = inner, 0

    def complete(self, system, messages, max_tokens=None):
        self.calls += 1
        if self.calls > 1:
            print(f"\n>>> observation sent to the model:\n{messages[-1]['content'][:500]}")
        reply = self.inner.complete(system, messages) if max_tokens is None else self.inner.complete(system, messages, max_tokens)
        print(f"\n=== step {self.calls}: model replied ({reply.output_tokens} tokens, {reply.latency_ms / 1000:.1f}s) ===")
        print(reply.text if reply.text.strip() else "(EMPTY REPLY)")
        return reply


def main():
    p = argparse.ArgumentParser()
    p.add_argument("question")
    who = p.add_mutually_exclusive_group()
    who.add_argument("--admin", action="store_true")
    who.add_argument("--student", type=int, default=20260001)
    a = p.parse_args()

    state = build_state()
    state._llm = Tap(get_llm())
    scope = Scope("admin") if a.admin else Scope("student", a.student)
    result = state.agent().ask(a.question, scope)

    print("\n================ RESULT ================")
    print("answer :", result.answer)
    print("refused:", result.refused, "| tools:", result.tools_used, "| steps:", result.steps)
    print("sources:", result.sources)
    print("trace  :", json.dumps(result.trace, indent=1)[:1500])


if __name__ == "__main__":
    main()
