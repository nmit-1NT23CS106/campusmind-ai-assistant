import json

import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.chunking import load_chunks
from app.llm import LLMError
from app.retrieval import BM25Retriever
from app.services import Settings, build_state
from data.seed import build
from tests.conftest import DOCS
from tests.helpers import ScriptedLLM, act


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    return build(tmp_path_factory.mktemp("api") / "a.db", seed=9, n_students=20)


def client_for(db, tmp_path, replies=(), **settings):
    st = Settings(db_path=db, docs_dir=DOCS, log_path=tmp_path / "log.jsonl", retriever="bm25", **settings)
    llm = ScriptedLLM(replies) if replies is not None else None
    state = build_state(st, llm=llm, retriever=BM25Retriever(load_chunks(DOCS)))
    return TestClient(create_app(state)), state


DOC_FLOW = [act("search_docs", query="minimum attendance"), act("answer", answer="75%", sources=["[1]"])]


def test_health(db, tmp_path):
    c, _ = client_for(db, tmp_path)
    body = c.get("/health").json()
    assert body["status"] == "ok" and body["retriever"] == "bm25" and body["llm_configured"] is True


def test_ask_student_end_to_end_and_logging(db, tmp_path):
    c, state = client_for(db, tmp_path, DOC_FLOW)
    r = c.post("/ask", json={"question": "What is the minimum attendance?", "role": "student", "student_id": 20260001})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "75%" and body["sources"] and body["tools_used"] == ["search_docs"]
    line = json.loads(state.settings.log_path.read_text().splitlines()[0])
    assert line["role"] == "student" and line["input_tokens"] == 20 and "rows" not in json.dumps(line)


def test_admin_ask(db, tmp_path):
    c, _ = client_for(db, tmp_path, [act("query_sql", sql="SELECT COUNT(*) FROM students"), act("answer", answer="20", sources=[])])
    r = c.post("/ask", json={"question": "How many students?", "role": "admin"})
    assert r.status_code == 200 and r.json()["sql_runs"][0]["rows"] == [[20]]


def test_student_role_requires_a_real_student_id(db, tmp_path):
    c, _ = client_for(db, tmp_path)
    assert c.post("/ask", json={"question": "hello there", "role": "student"}).status_code == 422
    assert c.post("/ask", json={"question": "hello there", "role": "student", "student_id": 1}).status_code == 404


@pytest.mark.parametrize("body", [{"question": "hi"}, {"question": "x" * 501}, {"question": "hello", "role": "teacher"}, {}])
def test_input_validation(db, tmp_path, body):
    c, _ = client_for(db, tmp_path)
    assert c.post("/ask", json=body).status_code == 422


def test_api_token_is_enforced_when_configured(db, tmp_path):
    c, _ = client_for(db, tmp_path, DOC_FLOW, api_token="secret")
    body = {"question": "What is the minimum attendance?", "role": "student", "student_id": 20260001}
    assert c.post("/ask", json=body).status_code == 401
    assert c.post("/ask", json=body, headers={"X-API-Token": "wrong"}).status_code == 401
    assert c.post("/ask", json=body, headers={"X-API-Token": "secret"}).status_code == 200


def test_admin_needs_the_admin_token(db, tmp_path):
    c, _ = client_for(db, tmp_path, [act("refuse", reason="x")], admin_token="boss")
    body = {"question": "How many students?", "role": "admin"}
    assert c.post("/ask", json=body).status_code == 403
    assert c.post("/ask", json=body, headers={"X-Admin-Token": "nope"}).status_code == 403
    assert c.post("/ask", json=body, headers={"X-Admin-Token": "boss"}).status_code == 200


def test_student_does_not_need_the_admin_token(db, tmp_path):
    c, _ = client_for(db, tmp_path, [act("refuse", reason="x")], admin_token="boss")
    assert c.post("/ask", json={"question": "hello there", "role": "student", "student_id": 20260001}).status_code == 200


def test_rate_limit(db, tmp_path):
    c, _ = client_for(db, tmp_path, [act("refuse", reason="x")] * 5, rate_limit_per_min=3)
    codes = [c.post("/ask", json={"question": "hello there", "role": "admin"}).status_code for _ in range(5)]
    assert codes == [200, 200, 200, 429, 429]


def test_llm_not_configured_gives_503(db, tmp_path, monkeypatch):
    for var in ("LLM_PROVIDER", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    c, _ = client_for(db, tmp_path, replies=None)
    r = c.post("/ask", json={"question": "hello there", "role": "admin"})
    assert r.status_code == 503 and "LLM_PROVIDER" in r.json()["detail"]


def test_llm_failure_is_a_clean_502_and_is_logged(db, tmp_path):
    class Broken:
        def complete(self, *a, **k):
            raise LLMError("anthropic request failed (HTTP 500)")

    st = Settings(db_path=db, docs_dir=DOCS, log_path=tmp_path / "log.jsonl", retriever="bm25")
    c = TestClient(create_app(build_state(st, llm=Broken(), retriever=BM25Retriever(load_chunks(DOCS)))))
    r = c.post("/ask", json={"question": "hello there", "role": "admin"})
    assert r.status_code == 502 and "HTTP 500" not in r.text
    assert "llm:" in json.loads(st.log_path.read_text().splitlines()[0])["error"]


def test_demo_students_lists_at_risk_first(db, tmp_path):
    c, _ = client_for(db, tmp_path)
    rows = c.get("/demo/students?limit=3").json()
    assert len(rows) == 3 and all({"student_id", "name"} <= set(r) for r in rows)
    assert rows[0]["lowest_attendance"] < 75


def test_retriever_falls_back_to_bm25_if_the_model_cannot_load(db, tmp_path, monkeypatch):
    import app.services as services

    real = services.build_retriever

    def flaky(name, chunks, *a, **k):
        if name != "bm25":
            raise OSError("no network")
        return real(name, chunks)

    monkeypatch.setattr(services, "build_retriever", flaky)
    st = Settings(db_path=db, docs_dir=DOCS, log_path=tmp_path / "l.jsonl", retriever="hybrid")
    state = build_state(st)
    assert state.retriever_name == "bm25" and "bm25" in state.warning
