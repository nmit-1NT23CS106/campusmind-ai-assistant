import json
import pytest

from app import llm
from app.llm import AnthropicClient, GeminiClient, GroqClient, LLMError, OpenAICompatClient, get_llm, llm_configured


class FakeResp:
    def __init__(self, status=200, payload=None, text="", headers=None):
        self.status_code, self._payload, self.text, self.headers = status, payload or {}, text, headers or {}

    def json(self):
        return self._payload


@pytest.fixture()
def capture(monkeypatch):
    calls, queue = [], []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

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


# ---------- reasoning models (Groq gpt-oss / qwen) ----------
def ok_payload(content="x", finish="stop"):
    return {"choices": [{"message": {"content": content}, "finish_reason": finish}], "usage": {}}


def test_groq_default_model_is_qwen():
    # chosen by running scripts.probe_llm: gpt-oss models return their JSON in the reasoning field
    assert GroqClient("k").model == "qwen/qwen3.8-27b"


def test_uses_max_completion_tokens_with_a_generous_default(capture, monkeypatch):
    calls, queue = capture
    monkeypatch.delenv("LLM_MAX_TOKENS", raising=False)
    queue.append(FakeResp(payload=ok_payload()))
    GroqClient("k").complete("S", MSGS)
    assert calls[0]["json"]["max_completion_tokens"] == 4096 and "max_tokens" not in calls[0]["json"]


def test_max_tokens_can_be_overridden_by_env_and_ignores_junk(capture, monkeypatch):
    calls, queue = capture
    queue.append(FakeResp(payload=ok_payload()))
    monkeypatch.setenv("LLM_MAX_TOKENS", "777")
    GroqClient("k").complete("S", MSGS)
    assert calls[0]["json"]["max_completion_tokens"] == 777
    queue.append(FakeResp(payload=ok_payload()))
    monkeypatch.setenv("LLM_MAX_TOKENS", "# a stray comment")
    GroqClient("k").complete("S", MSGS)
    assert calls[1]["json"]["max_completion_tokens"] == 4096


def test_gpt_oss_on_groq_defaults_to_low_reasoning_effort(capture, monkeypatch):
    calls, queue = capture
    monkeypatch.delenv("LLM_REASONING_EFFORT", raising=False)
    queue.append(FakeResp(payload=ok_payload()))
    GroqClient("k", model="openai/gpt-oss-20b").complete("S", MSGS)
    assert calls[0]["json"]["reasoning_effort"] == "low"


def test_reasoning_effort_env_overrides_and_other_models_get_none(capture, monkeypatch):
    calls, queue = capture
    monkeypatch.setenv("LLM_REASONING_EFFORT", "high")
    queue.append(FakeResp(payload=ok_payload()))
    GroqClient("k").complete("S", MSGS)
    assert calls[0]["json"]["reasoning_effort"] == "high"
    monkeypatch.delenv("LLM_REASONING_EFFORT")
    queue.append(FakeResp(payload=ok_payload()))
    GroqClient("k", model="qwen/qwen3.8-27b").complete("S", MSGS)
    assert "reasoning_effort" not in calls[1]["json"]
    queue.append(FakeResp(payload=ok_payload()))
    OpenAICompatClient("k").complete("S", MSGS)
    assert "reasoning_effort" not in calls[2]["json"]


def test_empty_reply_from_running_out_of_tokens_is_a_clear_error(capture):
    _, queue = capture
    queue.append(FakeResp(payload=ok_payload(content="", finish="length")))
    with pytest.raises(LLMError, match="LLM_MAX_TOKENS"):
        GroqClient("k").complete("S", MSGS)


def test_unknown_model_error_tells_you_how_to_fix_it_without_leaking_the_key(capture):
    _, queue = capture
    queue.append(FakeResp(404, text='{"error":{"message":"The model `old-model` does not exist","code":"model_not_found"}}'))
    with pytest.raises(LLMError) as exc:
        GroqClient("gsk-SECRET", model="old-model").complete("S", MSGS)
    assert "LLM_MODEL" in str(exc.value) and "gsk-SECRET" not in str(exc.value)


# ---------- gpt-oss quirks seen in a real run ----------
def test_json_is_recovered_from_the_reasoning_field_when_content_is_empty(capture):
    _, queue = capture
    reasoning = 'Need policy. Use search_docs.{"action":"search_docs","query":"minimum attendance"}'
    queue.append(FakeResp(payload={"choices": [{"message": {"content": "", "reasoning": reasoning}, "finish_reason": "stop"}], "usage": {}}))
    text = GroqClient("k", model="openai/gpt-oss-120b").complete("S", MSGS).text
    assert json.loads(text) == {"action": "search_docs", "query": "minimum attendance"}


def test_content_wins_over_reasoning(capture):
    _, queue = capture
    queue.append(FakeResp(payload={"choices": [{"message": {"content": '{"action": "refuse", "reason": "x"}',
                                                            "reasoning": '{"action": "search_docs"}'}, "finish_reason": "stop"}], "usage": {}}))
    assert "refuse" in GroqClient("k").complete("S", MSGS).text


def test_reasoning_without_json_stays_empty(capture):
    _, queue = capture
    queue.append(FakeResp(payload={"choices": [{"message": {"content": "", "reasoning": "just thinking, no answer"}, "finish_reason": "stop"}], "usage": {}}))
    assert GroqClient("k").complete("S", MSGS).text == ""


def test_tool_use_failed_error_suggests_a_working_model(capture):
    _, queue = capture
    queue.append(FakeResp(400, text='{"error":{"message":"Tool choice is none, but model called a tool","code":"tool_use_failed"}}'))
    with pytest.raises(LLMError, match="qwen/qwen3.8-27b"):
        GroqClient("k", model="openai/gpt-oss-20b").complete("S", MSGS)


# ---------------- rate limits and flaky networks (seen in a real evaluation run) ----------------
OK = {"content": [{"type": "text", "text": "ok"}]}
KEY = "sk-test-key-123"  # realistic key: a one-letter key would be redacted out of ordinary words


def test_429_waits_for_the_retry_after_header(capture):
    _, queue = capture
    sleeps = []
    queue += [FakeResp(429, headers={"retry-after": "3"}), FakeResp(payload=OK)]
    assert AnthropicClient(KEY, sleep=sleeps.append).complete("S", MSGS).text == "ok"
    assert sleeps == [3.5]


@pytest.mark.parametrize("text,expected", [("Please try again in 2.5s.", 3.0), ("try again in 250ms", 0.75),
                                           ("try again in 1m5s", 65.5)])
def test_429_waits_as_long_as_the_message_says(capture, text, expected):
    calls, queue = capture
    sleeps = []
    queue += [FakeResp(429, text=text), FakeResp(payload=OK)]
    try:
        AnthropicClient(KEY, sleep=sleeps.append).complete("S", MSGS)
    except LLMError:
        assert expected > 60  # a long wait is refused instead of slept through
        assert len(calls) == 1 and sleeps == []
    else:
        assert sleeps == [pytest.approx(expected)]


def test_a_long_rate_limit_stops_immediately_and_says_how_long(capture):
    calls, queue = capture
    queue.append(FakeResp(429, text="Rate limit reached. Please try again in 7m23.3s."))
    with pytest.raises(LLMError, match="wait about 443s"):
        AnthropicClient(KEY, sleep=lambda s: None).complete("S", MSGS)
    assert len(calls) == 1


def test_429_is_retried_up_to_five_times_then_gives_up(capture):
    calls, queue = capture
    sleeps = []
    queue += [FakeResp(429, text="slow down")] * 6
    with pytest.raises(LLMError, match="429"):
        AnthropicClient(KEY, sleep=sleeps.append).complete("S", MSGS)
    assert len(calls) == 6 and len(sleeps) == 5


def test_network_errors_are_retried_with_growing_pauses(capture):
    calls, queue = capture
    sleeps = []
    queue += [llm.requests.ConnectionError(), llm.requests.ConnectionError(), FakeResp(payload=OK)]
    assert AnthropicClient(KEY, sleep=sleeps.append).complete("S", MSGS).text == "ok"
    assert sleeps == [1.5, 3.0] and len(calls) == 3


def test_persistent_network_failure_becomes_a_clear_error(capture):
    _, queue = capture
    queue += [llm.requests.ConnectionError()] * 5
    with pytest.raises(LLMError, match="network error"):
        AnthropicClient(KEY, sleep=lambda s: None).complete("S", MSGS)
