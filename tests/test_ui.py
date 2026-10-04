from unittest.mock import MagicMock, patch

import pytest
from streamlit.testing.v1 import AppTest

from tests.conftest import ROOT
from ui.chat import api_headers, call_api

ANSWER = {"answer": "You need at least 75% in each course.", "sources": ["attendance_policy > Minimum attendance requirement"],
          "refused": False, "tools_used": ["search_docs"], "steps": 2, "input_tokens": 100, "output_tokens": 20,
          "latency_ms": 1500.0, "retriever": "hybrid", "role": "student",
          "sql_runs": [{"sql": "SELECT 1", "columns": ["x"], "rows": [[1]], "row_count": 1, "truncated": False, "error": None}],
          "trace": [{"step": 1, "action": "search_docs", "detail": "attendance", "retrieved": ["attendance_policy > Minimum attendance requirement"]}]}


def resp(status=200, payload=None):
    m = MagicMock(status_code=status, text=str(payload))
    m.json.return_value = payload
    return m


def fake_get(url, **kw):
    if url.endswith("/health"):
        return resp(200, {"status": "ok", "retriever": "bm25", "llm_configured": True, "warning": None})
    return resp(200, [{"student_id": 20260001, "name": "Zoya Iyer", "lowest_attendance": 70.0}])


def test_api_headers():
    assert api_headers("a", "b") == {"X-API-Token": "a", "X-Admin-Token": "b"}
    assert api_headers("", "") == {}


def test_call_api_success_and_errors():
    with patch("ui.chat.requests.post", return_value=resp(200, ANSWER)):
        assert call_api({"question": "q"}) == (True, ANSWER)
    with patch("ui.chat.requests.post", return_value=resp(401, {"detail": "Missing or invalid X-API-Token."})):
        ok, msg = call_api({"question": "q"})
        assert not ok and msg.startswith("401") and "X-API-Token" in msg
    import requests
    with patch("ui.chat.requests.post", side_effect=requests.ConnectionError()):
        ok, msg = call_api({"question": "q"}, base_url="http://x:1")
        assert not ok and "Can't reach" in msg


@pytest.fixture()
def app():
    with patch("requests.get", side_effect=fake_get):
        yield AppTest.from_file(str(ROOT / "ui" / "chat.py"), default_timeout=20)


def test_page_renders_without_errors(app):
    app.run()
    assert not app.exception
    assert app.title[0].value.endswith("CampusMind")
    assert any("API online" in s.value for s in app.sidebar.success)


def test_asking_a_question_shows_answer_sources_and_details(app):
    with patch("requests.post", return_value=resp(200, ANSWER)) as post:
        app.run()
        app.chat_input[0].set_value("What is the minimum attendance?").run()
    assert not app.exception
    body = post.call_args.kwargs["json"]
    assert body == {"question": "What is the minimum attendance?", "role": "student", "student_id": 20260001}
    texts = [m.markdown[0].value for m in app.chat_message]
    assert texts == ["What is the minimum attendance?", ANSWER["answer"]]
    assert any("attendance_policy" in c.value for c in app.caption)


def test_api_failure_is_shown_politely(app):
    with patch("requests.post", return_value=resp(503, {"detail": "The language model is not configured."})):
        app.run()
        app.chat_input[0].set_value("hello there").run()
    assert not app.exception
    assert "not configured" in app.chat_message[1].markdown[0].value
