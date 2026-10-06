"""Find out why a model returns empty replies, and which configuration works.

Usage:  python -m scripts.probe_llm
Sends the exact first request the agent would send (same prompt, same question) to several
Groq model settings and reports where each reply ended up. Prints no secrets.
"""
import json
import os
import sys
import time

import requests
from dotenv import load_dotenv

from app.agent import parse_action
from app.llm import GroqClient
from app.prompts import build_system_prompt
from app.services import Settings
from app.sql_tool import Scope, SQLTool

QUESTION = "What is the minimum attendance to sit the exams?"
VARIANTS = [
    ("A  app default (gpt-oss-120b, effort low)", "openai/gpt-oss-120b", {"reasoning_effort": "low"}),
    ("B  gpt-oss-120b, no effort setting", "openai/gpt-oss-120b", {}),
    ("C  gpt-oss-120b, effort medium", "openai/gpt-oss-120b", {"reasoning_effort": "medium"}),
    ("D  gpt-oss-20b, effort low", "openai/gpt-oss-20b", {"reasoning_effort": "low"}),
    ("E  qwen3.8-27b, no extras", "qwen/qwen3.8-27b", {}),
]


def build_messages():
    settings = Settings.from_env()
    schema = SQLTool(settings.db_path).schema_text()
    system = build_system_prompt(schema, Scope("student", 20260001))
    return [{"role": "system", "content": system}, {"role": "user", "content": QUESTION}]


def describe(name, response):
    if response.status_code != 200:
        return [f"{name}\n   HTTP {response.status_code}: {response.text[:250]}"]
    data = response.json()
    choice = data["choices"][0]
    message = choice["message"]
    content = message.get("content") or ""
    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
    tool_calls = message.get("tool_calls")
    try:
        parse_action(content)
        parses = "YES"
    except Exception:
        parses = "no"
    shown = content or reasoning or json.dumps(tool_calls)
    return [
        f"{name}",
        f"   finish_reason={choice.get('finish_reason')}  completion_tokens={data.get('usage', {}).get('completion_tokens')}",
        f"   content_chars={len(content)}  reasoning_chars={len(reasoning)}  tool_calls={'yes' if tool_calls else 'no'}"
        f"  usable_as_agent_reply={parses}",
        f"   message fields: {sorted(message.keys())}",
        f"   preview: {shown[:260]!r}",
    ]


def main(post=requests.post, sleep=time.sleep):
    load_dotenv()
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        sys.exit("GROQ_API_KEY is not set in .env")
    messages = build_messages()
    for name, model, extra in VARIANTS:
        body = {"model": model, "max_completion_tokens": 4096, "temperature": 0, "messages": messages, **extra}
        try:
            response = post(GroqClient.url, headers={"Authorization": f"Bearer {key}", "content-type": "application/json"},
                            json=body, timeout=120)
            lines = describe(name, response)
        except requests.RequestException as exc:
            lines = [f"{name}\n   network error: {type(exc).__name__}"]
        print("\n".join(lines) + "\n")
        sleep(2)


if __name__ == "__main__":
    main()
