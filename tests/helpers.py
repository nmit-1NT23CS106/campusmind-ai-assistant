import json

from app.llm import LLMResponse


class ScriptedLLM:
    """Plays back fixed replies so agent behaviour can be tested without a real model."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, system, messages, max_tokens=1024):
        self.calls.append({"system": system, "messages": [dict(m) for m in messages]})
        if not self.replies:
            raise AssertionError("The agent asked the LLM more times than the script allows.")
        reply = self.replies.pop(0)
        return LLMResponse(text=reply if isinstance(reply, str) else json.dumps(reply),
                           input_tokens=10, output_tokens=5, latency_ms=1.0, model="scripted")


def act(action, **fields):
    return json.dumps({"action": action, **fields})
