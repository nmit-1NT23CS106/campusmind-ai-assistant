import json

from app.services import Settings
from data.seed import build
from scripts import probe_llm


class Resp:
    def __init__(self, status, payload=None, text=""):
        self.status_code, self._p, self.text = status, payload or {}, text

    def json(self):
        return self._p


def reply(content="", reasoning="", tool_calls=None, finish="stop"):
    message = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning"] = reasoning
    if tool_calls:
        message["tool_calls"] = tool_calls
    return Resp(200, {"choices": [{"message": message, "finish_reason": finish}], "usage": {"completion_tokens": 33}})


def test_describe_flags_usable_json_replies():
    out = "\n".join(probe_llm.describe("X", reply(content='{"action": "search_docs", "query": "q"}')))
    assert "usable_as_agent_reply=YES" in out and "content_chars=" in out


def test_describe_shows_where_an_empty_reply_went():
    reasoning_only = "\n".join(probe_llm.describe("X", reply(reasoning="thinking about it")))
    assert "usable_as_agent_reply=no" in reasoning_only and "reasoning_chars=17" in reasoning_only
    tool = "\n".join(probe_llm.describe("X", reply(tool_calls=[{"function": {"name": "search_docs"}}])))
    assert "tool_calls=yes" in tool and "search_docs" in tool


def test_describe_reports_http_errors():
    assert "HTTP 400" in probe_llm.describe("X", Resp(400, text="bad request"))[0]


def test_main_runs_every_variant_and_never_prints_the_key(monkeypatch, capsys, tmp_path):
    db = build(tmp_path / "probe.db", n_students=5)  # the test brings its own database
    monkeypatch.setattr(Settings, "from_env", classmethod(lambda cls: cls(db_path=db)))
    monkeypatch.setenv("GROQ_API_KEY", "gsk-SECRET")
    monkeypatch.setattr(probe_llm, "load_dotenv", lambda: None)
    sent = []

    def fake_post(url, headers=None, json=None, timeout=None):
        sent.append(json)
        return reply(content='{"action": "refuse", "reason": "x"}')

    probe_llm.main(post=fake_post, sleep=lambda s: None)
    out = capsys.readouterr().out
    assert len(sent) == len(probe_llm.VARIANTS) and "gsk-SECRET" not in out
    assert sent[0]["reasoning_effort"] == "low" and "reasoning_effort" not in sent[1]
    assert sent[0]["messages"][1]["content"] == probe_llm.QUESTION and "CREATE" in sent[0]["messages"][0]["content"]
