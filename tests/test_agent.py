import pytest

from app.agent import Agent, parse_action
from app.chunking import load_chunks
from app.retrieval import BM25Retriever
from app.sql_tool import Scope, SQLTool
from data.seed import build
from tests.conftest import DOCS
from tests.helpers import ScriptedLLM, act

ADMIN = Scope("admin")
ME = 20260002
STUDENT = Scope("student", ME)


@pytest.fixture(scope="module")
def parts(tmp_path_factory):
    db = build(tmp_path_factory.mktemp("agent") / "a.db", seed=5, n_students=20)
    return BM25Retriever(load_chunks(DOCS)), SQLTool(db)


def make(parts, replies, **kw):
    retriever, sql = parts
    llm = ScriptedLLM(replies)
    return Agent(llm, retriever, sql, **kw), llm


def last_observation(llm, call=-1):
    return llm.calls[call]["messages"][-1]["content"]


# ---------- parsing ----------
@pytest.mark.parametrize("text", [
    '{"action": "answer", "answer": "x"}',
    '```json\n{"action": "answer", "answer": "x"}\n```',
    'Sure! {"action": "answer", "answer": "x"} Hope that helps.',
])
def test_parse_action_tolerates_fences_and_prose(text):
    assert parse_action(text)["action"] == "answer"


@pytest.mark.parametrize("text", ["no json here", "{not json}", '{"no_action": 1}', "[1,2]"])
def test_parse_action_rejects_garbage(text):
    with pytest.raises(ValueError):
        parse_action(text)


# ---------- document questions ----------
def test_document_answer_with_verified_citation(parts):
    agent, llm = make(parts, [act("search_docs", query="minimum attendance for exam"),
                              act("answer", answer="You need 75% in each course.", sources=["[1]"])])
    r = agent.ask("What attendance do I need?", ADMIN)
    assert r.answer == "You need 75% in each course." and not r.refused
    assert r.tools_used == ["search_docs"] and r.sources[0].startswith("attendance_policy")
    assert "[1] attendance_policy" in last_observation(llm)
    assert (r.input_tokens, r.output_tokens, r.steps) == (20, 10, 2) and r.latency_ms >= 0


def test_hallucinated_sources_are_dropped(parts):
    agent, _ = make(parts, [act("search_docs", query="attendance"),
                            act("answer", answer="75%", sources=["[9]", "made_up_doc > Nothing", "[1]", "[1]"])])
    r = agent.ask("q", ADMIN)
    assert len(r.sources) == 1 and r.sources[0].startswith("attendance_policy")


def test_sources_can_be_given_as_full_citation_strings(parts):
    retriever, _ = parts
    top = retriever.search("revaluation fee", k=4)[0][0].citation
    agent, _ = make(parts, [act("search_docs", query="revaluation fee"), act("answer", answer="Rs. 800", sources=[top])])
    assert agent.ask("q", ADMIN).sources == [top]


def test_repeated_search_reuses_passage_numbers(parts):
    agent, llm = make(parts, [act("search_docs", query="attendance"), act("search_docs", query="attendance"),
                              act("answer", answer="ok", sources=["[1]"])])
    agent.ask("q", ADMIN)
    first, second = llm.calls[1]["messages"][-1]["content"], llm.calls[2]["messages"][-1]["content"]
    assert first.split("\n")[2].startswith("[1]") and second.split("\n")[2].startswith("[1]")


# ---------- guardrails on the model's behaviour ----------
def test_answer_before_any_tool_is_rejected_then_accepted(parts):
    agent, llm = make(parts, [act("answer", answer="from memory", sources=[]),
                              act("search_docs", query="attendance"),
                              act("answer", answer="75%", sources=["[1]"])])
    r = agent.ask("q", ADMIN)
    assert "before answering" in last_observation(llm, 1) and r.answer == "75%"
    assert [t["action"] for t in r.trace][0] == "answer_rejected"


def test_empty_answer_is_rejected(parts):
    agent, llm = make(parts, [act("search_docs", query="fees"), act("answer", answer="  ", sources=[]),
                              act("answer", answer="Rs. 100 per day", sources=["[1]"])])
    assert agent.ask("q", ADMIN).answer == "Rs. 100 per day"


def test_invalid_json_gets_a_second_chance(parts):
    agent, llm = make(parts, ["I think the answer is 75", act("search_docs", query="attendance"),
                              act("answer", answer="75%", sources=["[1]"])])
    r = agent.ask("q", ADMIN)
    assert r.answer == "75%" and r.trace[0]["action"] == "invalid_reply"
    assert "not valid" in last_observation(llm, 1)


def test_unknown_action_is_reported_back(parts):
    agent, llm = make(parts, [act("delete_everything"), act("search_docs", query="fees"),
                              act("answer", answer="ok", sources=["[1]"])])
    agent.ask("q", ADMIN)
    assert "unknown action" in last_observation(llm, 1)


def test_empty_search_query_is_an_error_observation(parts):
    agent, llm = make(parts, [act("search_docs", query=""), act("search_docs", query="fees"),
                              act("answer", answer="ok", sources=["[1]"])])
    agent.ask("q", ADMIN)
    assert "empty" in last_observation(llm, 1)


def test_refusal(parts):
    agent, _ = make(parts, [act("refuse", reason="That's outside what I can answer.")])
    r = agent.ask("What's on the cafeteria menu?", ADMIN)
    assert r.refused and r.answer == "That's outside what I can answer." and r.tools_used == []


def test_step_limit_ends_cleanly(parts):
    agent, _ = make(parts, [act("search_docs", query="attendance")] * 3, max_steps=3)
    r = agent.ask("q", ADMIN)
    assert r.refused and r.steps == 3 and r.trace[-1]["action"] == "step_limit"


# ---------- SQL ----------
def test_sql_error_is_fed_back_and_the_model_recovers(parts):
    agent, llm = make(parts, [act("query_sql", sql="SELECT * FROM users"),
                              act("query_sql", sql="SELECT COUNT(*) AS n FROM students"),
                              act("answer", answer="20 students", sources=[])])
    r = agent.ask("How many students?", ADMIN)
    assert "not available" in r.sql_runs[0]["error"] and r.sql_runs[1]["rows"] == [[20]]
    assert "ERROR" in last_observation(llm, 1) and "20" in last_observation(llm, 2)
    assert r.tools_used == ["query_sql", "query_sql"] and r.sources == []


def test_dangerous_sql_never_executes(parts):
    agent, _ = make(parts, [act("query_sql", sql="DROP TABLE students"),
                            act("query_sql", sql="SELECT COUNT(*) FROM students"),
                            act("answer", answer="still 20", sources=[])])
    r = agent.ask("q", ADMIN)
    assert "Only SELECT" in r.sql_runs[0]["error"] and r.sql_runs[1]["rows"] == [[20]]


def test_student_scope_is_enforced_through_the_agent(parts):
    sql = act("query_sql", sql="SELECT COUNT(*) AS n FROM students")
    for scope, expected in ((STUDENT, 1), (ADMIN, 20)):
        agent, _ = make(parts, [sql, act("answer", answer="x", sources=[])])
        assert agent.ask("How many students are there?", scope).sql_runs[0]["rows"] == [[expected]]


def test_large_results_are_truncated_for_the_model_but_flagged(parts):
    agent, llm = make(parts, [act("query_sql", sql="SELECT student_id FROM students"),
                              act("answer", answer="listed", sources=[])])
    agent.ask("List ids", ADMIN)
    obs = last_observation(llm, 1)
    assert "20 row(s)" in obs and "showing the first 20" not in obs  # 20 rows fit
    agent, llm = make(parts, [act("query_sql", sql="SELECT enrollment_id FROM enrollments"),
                              act("answer", answer="listed", sources=[])])
    agent.ask("List ids", ADMIN)
    assert "showing the first 20" in last_observation(llm, 1)


def test_mixed_question_uses_both_tools(parts):
    agent, _ = make(parts, [act("query_sql", sql="SELECT course_code, attendance_pct FROM enrollment_summary"),
                            act("search_docs", query="minimum attendance for exam eligibility"),
                            act("answer", answer="Compare with 75%.", sources=["[1]"])])
    r = agent.ask("Am I eligible for the exam?", STUDENT)
    assert r.tools_used == ["query_sql", "search_docs"] and r.sources and r.sql_runs[0]["row_count"] == 5


# ---------- prompt ----------
def test_system_prompt_carries_scope_and_schema(parts):
    agent, llm = make(parts, [act("refuse", reason="no")])
    agent.ask("hi", STUDENT)
    system = llm.calls[0]["system"]
    assert str(ME) in system and "enrollment_summary" in system and "CREATE" in system
    agent, llm = make(parts, [act("refuse", reason="no")])
    agent.ask("hi", ADMIN)
    assert "administrator" in llm.calls[0]["system"] and str(ME) not in llm.calls[0]["system"]


def test_conversation_alternates_roles(parts):
    agent, llm = make(parts, [act("search_docs", query="a"), act("search_docs", query="b"),
                              act("answer", answer="ok", sources=["[1]"])])
    agent.ask("q", ADMIN)
    roles = [m["role"] for m in llm.calls[-1]["messages"]]
    assert roles == ["user", "assistant", "user", "assistant", "user"]
