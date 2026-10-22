import json

import pytest

from app.agent import Agent
from app.chunking import load_chunks
from app.retrieval import BM25Retriever
from app.sql_guard import validate_select
from app.sql_tool import Scope, SQLTool
from data.seed import build
from eval.agent_eval import (AGENT_QUESTIONS, load_cases, markdown_table, pick_demo_student, results_match,
                             run_eval, score_case, summarize)
from tests.conftest import DOCS
from tests.helpers import ScriptedLLM, act


@pytest.fixture(scope="module")
def tool(tmp_path_factory):
    return SQLTool(build(tmp_path_factory.mktemp("ev") / "full.db"))  # the real 200-student dataset


@pytest.fixture(scope="module")
def retriever():
    return BM25Retriever(load_chunks(DOCS))


# ---------- the question set itself ----------
def test_case_mix_and_unique_ids():
    cases = load_cases()
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids))
    cats = {c["category"] for c in cases}
    assert cats == {"docs", "refuse", "sql", "mixed"} and len(cases) >= 50


def test_every_gold_sql_passes_the_guard_and_returns_a_comparable_result(tool):
    student = Scope("student", pick_demo_student(tool))
    for case in json.loads(AGENT_QUESTIONS.read_text()):
        if case["category"] != "sql":
            continue
        scope = student if case["role"] == "student" else Scope("admin")
        for gold in case["gold_sql"]:
            validate_select(gold)
            result = tool.run(gold, scope)
            assert 0 < result.row_count <= 50, (case["id"], gold, result.row_count)


def test_top_n_questions_have_no_ties_at_the_boundary(tool):
    # "top 2 by CGPA" is only well-defined if rank 2 and rank 3 differ
    cgpa = [r[0] for r in tool.run("SELECT cgpa FROM students ORDER BY cgpa DESC LIMIT 3", Scope("admin")).rows]
    assert cgpa[1] != cgpa[2]
    avg = [r[1] for r in tool.run("SELECT course_code, AVG(attendance_pct) FROM enrollment_summary GROUP BY course_code ORDER BY 2 LIMIT 2", Scope("admin")).rows]
    assert avg[0] != avg[1]


def test_demo_student_has_a_condonation_band_course(tool):
    sid = pick_demo_student(tool)
    rows = tool.run("SELECT attendance_pct FROM enrollment_summary", Scope("student", sid)).rows
    assert any(65 <= r[0] < 75 for r in rows)


# ---------- comparison logic ----------
def test_results_match_ignores_order_names_and_extra_columns():
    assert results_match(["n"], [[1], [2]], ["student", "x"], [[2, "a"], [1, "b"]])
    assert results_match(["c"], [[5.0]], ["count"], [[5]])
    assert not results_match(["n"], [[1], [2]], ["n"], [[1], [3]])
    assert not results_match(["n"], [[1], [2]], ["n"], [[1], [2], [3]])


def test_results_match_needs_every_gold_column():
    assert results_match(["a", "b"], [[1, "x"]], ["b", "a", "c"], [["x", 1, 9]])
    assert not results_match(["a", "b"], [[1, "x"]], ["a"], [[1]])


# ---------- scoring with a scripted agent ----------
def agent_for(retriever, tool, replies):
    return Agent(ScriptedLLM(replies), retriever, tool)


def ask_and_score(case, retriever, tool, replies, scope=None):
    scope = scope or Scope("admin")
    result = agent_for(retriever, tool, replies).ask(case["question"], scope)
    return score_case(case, result, tool, scope)


SQL_CASE = {"id": "t", "category": "sql", "question": "How many students?", "gold_sql": ["SELECT COUNT(*) FROM students"]}


def test_sql_case_passes_when_results_match(retriever, tool):
    ok, why = ask_and_score(SQL_CASE, retriever, tool,
                            [act("query_sql", sql="SELECT COUNT(*) AS total FROM students"), act("answer", answer="200", sources=[])])
    assert ok, why


def test_sql_case_fails_on_wrong_data(retriever, tool):
    ok, why = ask_and_score(SQL_CASE, retriever, tool,
                            [act("query_sql", sql="SELECT COUNT(*) FROM courses"), act("answer", answer="5", sources=[])])
    assert not ok and "differs" in why


def test_sql_case_uses_last_successful_query_after_an_error(retriever, tool):
    ok, _ = ask_and_score(SQL_CASE, retriever, tool,
                          [act("query_sql", sql="SELECT * FROM users"), act("query_sql", sql="SELECT COUNT(*) FROM students"),
                           act("answer", answer="200", sources=[])])
    assert ok


def test_refuse_case(retriever, tool):
    case = {"id": "r", "category": "refuse", "question": "Cafeteria menu?"}
    assert ask_and_score(case, retriever, tool, [act("refuse", reason="not in my documents")])[0]
    ok, why = ask_and_score(case, retriever, tool, [act("search_docs", query="menu"), act("answer", answer="pizza", sources=["[1]"])])
    assert not ok and "cannot answer" in why


def test_docs_case_requires_search_and_the_right_citation(retriever, tool):
    case = {"id": "d", "category": "docs", "question": "min attendance?", "gold_doc": "attendance_policy"}
    assert ask_and_score(case, retriever, tool, [act("search_docs", query="minimum attendance exam"), act("answer", answer="75%", sources=["[1]"])])[0]
    ok, why = ask_and_score(case, retriever, tool, [act("search_docs", query="library fine"), act("answer", answer="x", sources=["[1]"])])
    assert not ok and "did not cite" in why
    ok, why = ask_and_score(case, retriever, tool, [act("refuse", reason="no")])
    assert not ok and "refused" in why


def test_mixed_case_needs_both_tools(retriever, tool):
    case = {"id": "m", "category": "mixed", "question": "eligible?", "gold_doc": "attendance_policy"}
    me = Scope("student", pick_demo_student(tool))
    both = [act("query_sql", sql="SELECT attendance_pct FROM enrollment_summary"), act("search_docs", query="minimum attendance exam"),
            act("answer", answer="compare", sources=["[1]"])]
    assert ask_and_score(case, retriever, tool, both, me)[0]
    only_docs = [act("search_docs", query="minimum attendance exam"), act("answer", answer="75%", sources=["[1]"])]
    ok, why = ask_and_score(case, retriever, tool, only_docs, me)
    assert not ok and "used only" in why


# ---------- the whole loop ----------
def test_run_eval_summary_and_table(retriever, tool):
    cases = [SQL_CASE, {"id": "r", "category": "refuse", "question": "Cafeteria menu?", "role": "admin"}]
    cases[0] = {**SQL_CASE, "role": "admin"}
    llm = ScriptedLLM([act("query_sql", sql="SELECT COUNT(*) FROM students"), act("answer", answer="200", sources=[]),
                       act("answer", answer="pizza", sources=[]), act("search_docs", query="menu"), act("answer", answer="pizza", sources=["[1]"])])
    summary = run_eval(Agent(llm, retriever, tool), tool, cases, student_id=20260001, progress=lambda s: None)
    assert summary["by_category"] == {"sql": {"n": 1, "pass_rate": 1.0}, "refuse": {"n": 1, "pass_rate": 0.0}}
    assert summary["overall_pass_rate"] == 0.5 and summary["avg_tokens"] > 0
    table = markdown_table(summary)
    assert "Text-to-SQL" in table and "**50%**" in table


def test_a_provider_error_is_reported_not_a_crash(retriever, tool):
    class Boom:
        def complete(self, *a, **k):
            raise RuntimeError("provider down")

    summary = run_eval(Agent(Boom(), retriever, tool), tool, [{**SQL_CASE, "role": "admin"}], 20260001, progress=lambda s: None)
    assert summary["errors"] == 1 and summary["n_scored"] == 0 and "provider down" in summary["details"][0]["reason"]


def test_empty_summary_does_not_divide_by_zero():
    assert summarize([])["overall_pass_rate"] == 0


# ---------------- provider errors must not distort the scores ----------------
class FlakyOnce:
    def __init__(self, inner):
        self.inner, self.failed = inner, False

    def complete(self, system, messages, max_tokens=None):
        if not self.failed:
            self.failed = True
            raise RuntimeError("HTTP 429 rate limit")
        return self.inner.complete(system, messages)


def test_provider_errors_are_retried_then_scored(retriever, tool):
    llm = FlakyOnce(ScriptedLLM([act("query_sql", sql="SELECT COUNT(*) FROM students"), act("answer", answer="200", sources=[])]))
    summary = run_eval(Agent(llm, retriever, tool), tool, [{**SQL_CASE, "role": "admin"}], 20260001,
                       progress=lambda s: None, retry_errors=1, retry_wait=0)
    assert summary["errors"] == 0 and summary["overall_pass_rate"] == 1.0


def test_errors_are_excluded_from_pass_rates_and_reported(retriever, tool):
    llm = FlakyOnce(ScriptedLLM([act("query_sql", sql="SELECT COUNT(*) FROM students"), act("answer", answer="200", sources=[])]))
    cases = [{**SQL_CASE, "role": "admin"}, {**SQL_CASE, "id": "t2", "role": "admin"}]
    summary = run_eval(Agent(llm, retriever, tool), tool, cases, 20260001, progress=lambda s: None)
    assert (summary["n_questions"], summary["n_scored"], summary["errors"]) == (2, 1, 1)
    assert summary["overall_pass_rate"] == 1.0  # the one scored question passed; the error is not a failure
    assert "NOT counted" in markdown_table(summary)


def test_sql_errors_the_agent_recovered_from_are_recorded(retriever, tool):
    llm = ScriptedLLM([act("query_sql", sql="SELECT * FROM users"), act("query_sql", sql="SELECT COUNT(*) FROM students"),
                       act("answer", answer="200", sources=[])])
    lines = []
    summary = run_eval(Agent(llm, retriever, tool), tool, [{**SQL_CASE, "role": "admin"}], 20260001, progress=lines.append)
    detail = summary["details"][0]
    assert detail["passed"] and detail["sql_errors"] == 1 and "not available" in detail["sql_error_samples"][0]
    assert any("SQL error the agent hit" in line for line in lines)
