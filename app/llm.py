"""One small interface over several LLM providers, so the agent works with whichever key you have.

complete(system, messages) -> LLMResponse. Messages alternate user/assistant and start with user.
Set LLM_PROVIDER (anthropic | openai | groq | gemini), the matching *_API_KEY, and optionally LLM_MODEL.
Default model names change over time: if one is rejected, set LLM_MODEL to a current one.
"""
import os
import time
from dataclasses import dataclass

import requests

DEFAULT_MODELS = {
    "anthropic": "claude-haiku-4-5-20251001",
    "openai": "gpt-4o-mini",
    "groq": "llama-3.3-70b-versatile",
    "gemini": "gemini-2.0-flash",
}
KEY_VARS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY",
            "groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY"}
RETRY_STATUSES = {429, 500, 502, 503, 529}


class LLMError(RuntimeError):
    """Provider failure. Never contains the API key."""


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    model: str = ""


class _BaseClient:
    provider = ""

    def __init__(self, api_key, model=None, timeout=60, max_retries=2, sleep=time.sleep):
        if not api_key:
            raise LLMError(f"{KEY_VARS[self.provider]} is not set.")
        self.api_key = api_key
        self.model = model or DEFAULT_MODELS[self.provider]
        self.timeout, self.max_retries, self._sleep = timeout, max_retries, sleep

    def _post(self, url, headers, body):
        last = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(url, headers=headers, json=body, timeout=self.timeout)
            except requests.RequestException as exc:
                last = f"network error: {type(exc).__name__}"
            else:
                if resp.status_code == 200:
                    return resp.json()
                last = f"HTTP {resp.status_code}: {resp.text[:200]}"
                if resp.status_code not in RETRY_STATUSES:
                    break
            if attempt < self.max_retries:
                self._sleep(1.5 * (attempt + 1))
        raise LLMError(f"{self.provider} request failed ({last.replace(self.api_key, '***')})")

    def complete(self, system, messages, max_tokens=1024):
        started = time.perf_counter()
        text, tin, tout = self._call(system, messages, max_tokens)
        return LLMResponse(text, tin, tout, round((time.perf_counter() - started) * 1000, 1), self.model)


class AnthropicClient(_BaseClient):
    provider = "anthropic"

    def _call(self, system, messages, max_tokens):
        data = self._post("https://api.anthropic.com/v1/messages",
                          {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                          {"model": self.model, "max_tokens": max_tokens, "temperature": 0, "system": system, "messages": messages})
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        usage = data.get("usage", {})
        return text, usage.get("input_tokens", 0), usage.get("output_tokens", 0)


class OpenAICompatClient(_BaseClient):
    """OpenAI and Groq share the same chat-completions format."""
    provider = "openai"
    url = "https://api.openai.com/v1/chat/completions"

    def _call(self, system, messages, max_tokens):
        data = self._post(self.url, {"Authorization": f"Bearer {self.api_key}", "content-type": "application/json"},
                          {"model": self.model, "max_tokens": max_tokens, "temperature": 0,
                           "messages": [{"role": "system", "content": system}] + messages})
        usage = data.get("usage", {})
        return data["choices"][0]["message"]["content"] or "", usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


class GroqClient(OpenAICompatClient):
    provider = "groq"
    url = "https://api.groq.com/openai/v1/chat/completions"


class GeminiClient(_BaseClient):
    provider = "gemini"

    def _call(self, system, messages, max_tokens):
        contents = [{"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]} for m in messages]
        data = self._post(f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
                          {"x-goog-api-key": self.api_key, "content-type": "application/json"},  # key in a header, not the URL
                          {"systemInstruction": {"parts": [{"text": system}]}, "contents": contents,
                           "generationConfig": {"temperature": 0, "maxOutputTokens": max_tokens}})
        try:
            text = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
        except (KeyError, IndexError):
            raise LLMError("gemini returned no text (the response may have been blocked)")
        usage = data.get("usageMetadata", {})
        return text, usage.get("promptTokenCount", 0), usage.get("candidatesTokenCount", 0)


CLIENTS = {"anthropic": AnthropicClient, "openai": OpenAICompatClient, "groq": GroqClient, "gemini": GeminiClient}


def llm_configured(env=None):
    env = os.environ if env is None else env
    provider = env.get("LLM_PROVIDER", "").lower()
    if provider:
        return provider in CLIENTS and bool(env.get(KEY_VARS[provider]))
    return any(env.get(v) for v in KEY_VARS.values())


def get_llm(env=None):
    env = os.environ if env is None else env
    provider = env.get("LLM_PROVIDER", "").lower()
    if not provider:
        found = [p for p, var in KEY_VARS.items() if env.get(var)]
        if len(found) != 1:
            raise LLMError("Set LLM_PROVIDER (anthropic, openai, groq or gemini) and the matching API key in .env.")
        provider = found[0]
    if provider not in CLIENTS:
        raise LLMError(f"Unknown LLM_PROVIDER '{provider}'. Choose one of: {', '.join(CLIENTS)}.")
    return CLIENTS[provider](env.get(KEY_VARS[provider]), env.get("LLM_MODEL") or None)
