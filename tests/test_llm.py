import pytest

from app import llm
from app.llm import AnthropicClient, GeminiClient, GroqClient, LLMError, OpenAICompatClient, get_llm, llm_configured


class FakeResp:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code, self._payload, self.text = status, payload or {}, text

    def json(self):
        return self._payload


@pytest.fixture()
def capture(monkeypatch):
    calls, queue = [], []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return queue.pop(0)

    monkeypatch.setattr(llm.requests, "post", fake_post)
    return calls, queue


MSGS = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "{}"}, {"role": "user", "content": "more"}]


def test_anthropic_request_and_parse(capture):
    calls, queue = capture
    queue.append(FakeResp(payload={"content": [{"type": "text", "text": "ok"}], "usage": {"input_tokens": 11, "output_tokens": 3}}))
    r = AnthropicClient("sk-test").complete("SYS", MSGS)
    c = calls[0]
    assert c["url"].endswith("/v1/messages") and c["headers"]["x-api-key"] == "sk-test"
    assert c["json"]["system"] == "SYS" and c["json"]["messages"] == MSGS and c["json"]["temperature"] == 0
    assert (r.text, r.input_tokens, r.output_tokens) == ("ok", 11, 3) and r.latency_ms >= 0


def test_openai_puts_system_first_and_parses(capture):
    calls, queue = capture
    queue.append(FakeResp(payload={"choices": [{"message": {"content": "yo"}}], "usage": {"prompt_tokens": 5, "completion_tokens": 2}}))
    r = OpenAICompatClient("k").complete("SYS", MSGS)
    assert calls[0]["json"]["messages"][0] == {"role": "system", "content": "SYS"}
    assert calls[0]["headers"]["Authorization"] == "Bearer k"
    assert (r.text, r.input_tokens, r.output_tokens) == ("yo", 5, 2)


def test_groq_uses_its_own_endpoint(capture):
    calls, queue = capture
    queue.append(FakeResp(payload={"choices": [{"message": {"content": "x"}}]}))
    GroqClient("k").complete("S", MSGS)
    assert "groq.com" in calls[0]["url"]


def test_gemini_maps_roles_and_keeps_key_out_of_url(capture):
    calls, queue = capture
    queue.append(FakeResp(payload={"candidates": [{"content": {"parts": [{"text": "hey"}]}}],
                                   "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 4}}))
    r = GeminiClient("SECRETKEY", model="gemini-test").complete("SYS", MSGS)
    c = calls[0]
    assert "SECRETKEY" not in c["url"] and c["headers"]["x-goog-api-key"] == "SECRETKEY"
    assert [x["role"] for x in c["json"]["contents"]] == ["user", "model", "user"]
    assert "gemini-test" in c["url"] and (r.text, r.input_tokens, r.output_tokens) == ("hey", 7, 4)


def test_gemini_blocked_response_is_a_clear_error(capture):
    _, queue = capture
    queue.append(FakeResp(payload={"candidates": []}))
    with pytest.raises(LLMError, match="no text"):
        GeminiClient("k").complete("S", MSGS)


def test_retries_on_429_then_succeeds(capture):
    calls, queue = capture
    sleeps = []
    queue += [FakeResp(429, text="slow down"), FakeResp(payload={"content": [{"type": "text", "text": "ok"}]})]
    r = AnthropicClient("k", sleep=sleeps.append).complete("S", MSGS)
    assert r.text == "ok" and len(calls) == 2 and sleeps == [1.5]


def test_gives_up_after_retries_and_never_leaks_the_key(capture):
    _, queue = capture
    queue += [FakeResp(503, text="boom sk-LEAK boom")] * 3
    with pytest.raises(LLMError) as exc:
        AnthropicClient("sk-LEAK", sleep=lambda s: None).complete("S", MSGS)
    assert "sk-LEAK" not in str(exc.value) and "503" in str(exc.value)


def test_non_retryable_error_fails_fast(capture):
    calls, queue = capture
    queue.append(FakeResp(401, text="bad key"))
    with pytest.raises(LLMError, match="401"):
        AnthropicClient("k", sleep=lambda s: None).complete("S", MSGS)
    assert len(calls) == 1


def test_missing_key_is_a_clear_error():
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        AnthropicClient("")


def test_get_llm_picks_the_only_configured_provider():
    assert isinstance(get_llm({"GROQ_API_KEY": "g"}), GroqClient)


def test_get_llm_respects_explicit_provider_and_model():
    c = get_llm({"LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "a", "LLM_MODEL": "my-model", "OPENAI_API_KEY": "o"})
    assert isinstance(c, AnthropicClient) and c.model == "my-model"


def test_get_llm_errors():
    with pytest.raises(LLMError, match="Set LLM_PROVIDER"):
        get_llm({})
    with pytest.raises(LLMError, match="Set LLM_PROVIDER"):
        get_llm({"GROQ_API_KEY": "g", "OPENAI_API_KEY": "o"})  # ambiguous
    with pytest.raises(LLMError, match="Unknown"):
        get_llm({"LLM_PROVIDER": "nope"})


def test_llm_configured():
    assert not llm_configured({})
    assert llm_configured({"OPENAI_API_KEY": "x"})
    assert not llm_configured({"LLM_PROVIDER": "openai"})
    assert llm_configured({"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "x"})
