"""End-to-end agent evaluation (needs a real LLM key).

Categories and how each is scored (all deterministic, no LLM judge):
  docs     the agent searched the documents, did not refuse, and cited the gold document
  refuse   questions the documents cannot answer: the agent must refuse instead of guessing
  sql      the agent's last successful query returns the same data as a hand-written gold query
           ("execution accuracy": same row count, and every gold column appears with identical values)
  mixed    questions needing data AND policy: both tools used, gold document cited, no refusal

Answer wording is not scored automatically; spot-check a few answers by hand.

Usage:
  python -m eval.agent_eval                    # everything (~55 questions)
  python -m eval.agent_eval --limit 10         # quick smoke test
  python -m eval.agent_eval --sleep 4          # pause between questions for free-tier rate limits
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from app.services import build_state
from app.sql_tool import Scope, SQLToolError

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "eval" / "results"
DOC_QUESTIONS = ROOT / "eval" / "questions.json"
AGENT_QUESTIONS = ROOT / "eval" / "agent_questions.json"
PAYLOAD_ROW_CAP = 50


def load_cases():
    cases = []
    for q in json.loads(DOC_QUESTIONS.read_text()):
        cases.append({"id": q["id"], "category": "docs" if q["answerable"] else "refuse", "role": "admin",
                      "question": q["question"], "gold_doc": q["gold_doc"]})
    cases.extend(json.loads(AGENT_QUESTIONS.read_text()))
    return cases


def pick_demo_student(sql_tool):
    """A student with at least one course in the 65-75% condonation band, so mixed questions are meaningful."""
    conn = sql_tool._connect()
    try:
        row = conn.execute("SELECT student_id FROM enrollment_summary WHERE attendance_pct >= 65 AND attendance_pct < 75 "
                           "GROUP BY student_id ORDER BY student_id LIMIT 1").fetchone()
    finally:
        conn.close()
    return row[0] if row else sql_tool.demo_students(1)[0]["student_id"]


def _norm(value):
    return round(float(value), 2) if isinstance(value, (int, float)) and not isinstance(value, bool) else value


def _column(rows, j):
    return sorted((_norm(r[j]) for r in rows), key=repr)


def results_match(gold_columns, gold_rows, got_columns, got_rows):
    """Same number of rows, and every gold column appears (any order, any name) in the agent's result."""
    if len(gold_rows) != len(got_rows):
        return False
    for j in range(len(gold_columns)):
        expected = _column(gold_rows, j)
        if not any(_column(got_rows, k) == expected for k in range(len(got_columns))):
            return False
    return True


def gold_results(sql_tool, gold_sql, scope):
    return [sql_tool.run(sql, scope) for sql in gold_sql]


def score_case(case, result, sql_tool, scope):
    """Return (passed, reason)."""
    category = case["category"]
    if category == "refuse":
        return (True, "refused") if result.refused else (False, "answered a question the documents cannot answer")
    if result.refused:
        return False, f"refused: {result.answer[:80]}"
    if category == "docs":
        if "search_docs" not in result.tools_used:
            return False, "did not search the documents"
        gold = case["gold_doc"]
        return (True, "ok") if any(s.startswith(gold) for s in result.sources) else (False, f"did not cite {gold}")
    if category == "mixed":
        if not {"search_docs", "query_sql"} <= set(result.tools_used):
            return False, f"used only {sorted(set(result.tools_used))}"
        gold = case["gold_doc"]
        return (True, "ok") if any(s.startswith(gold) for s in result.sources) else (False, f"did not cite {gold}")
    if category == "sql":
        ok_runs = [r for r in result.sql_runs if not r["error"]]
        if not ok_runs:
            return False, "no successful SQL query"
        got = ok_runs[-1]
        if got["row_count"] > PAYLOAD_ROW_CAP:
            return False, "result too large to compare"
        for gold in gold_results(sql_tool, case["gold_sql"], scope):
            if results_match(gold.columns, gold.rows, got["columns"], got["rows"]):
                return True, "ok"
        return False, f"query result differs from gold: {got['sql'][:100]}"
    raise ValueError(f"unknown category {category}")


def evaluate_case(agent, sql_tool, case, student_id):
    scope = Scope("student", student_id) if case["role"] == "student" else Scope("admin")
    base = {"id": case["id"], "category": case["category"], "question": case["question"]}
    try:
        result = agent.ask(case["question"], scope)
        passed, reason = score_case(case, result, sql_tool, scope)
    except Exception as exc:  # provider hiccups (rate limit, network) are errors, not wrong answers
        return {**base, "passed": False, "error": True, "reason": f"{type(exc).__name__}: {str(exc)[:160]}"}
    return {**base, "passed": passed, "reason": reason,
            "tokens": result.input_tokens + result.output_tokens, "latency_ms": result.latency_ms,
            "steps": result.steps, "sql_runs": len(result.sql_runs),
            "sql_errors": sum(1 for r in result.sql_runs if r["error"]),
            "sql_error_samples": [r["error"][:120] for r in result.sql_runs if r["error"]]}


def run_eval(agent, sql_tool, cases, student_id, sleep=0.0, progress=print, retry_errors=0, retry_wait=30.0):
    details = []

    def show(i, d):
        label = "ERROR" if d.get("error") else ("PASS" if d["passed"] else "FAIL")
        line = f"[{i}/{len(cases)}] {label} {d['id']} {d['question'][:60]}"
        if label != "PASS":
            line += f"  -> {d['reason']}"
        for message in d.get("sql_error_samples", []):
            line += f"\n      (SQL error the agent hit: {message})"
        progress(line)

    for i, case in enumerate(cases, 1):
        details.append(evaluate_case(agent, sql_tool, case, student_id))
        show(i, details[-1])
        if sleep and i < len(cases):
            time.sleep(sleep)

    for _ in range(retry_errors):
        bad = [i for i, d in enumerate(details) if d.get("error")]
        if not bad:
            break
        progress(f"\nRetrying {len(bad)} question(s) that hit provider errors (waiting {retry_wait:.0f}s first)...")
        if retry_wait:
            time.sleep(retry_wait)
        for i in bad:
            details[i] = evaluate_case(agent, sql_tool, cases[i], student_id)
            show(i + 1, details[i])
            if sleep:
                time.sleep(sleep)
    return summarize(details)


def summarize(details):
    """Provider errors are excluded from pass rates and reported separately: they say nothing about the agent."""
    errors = [d for d in details if d.get("error")]
    scored = [d for d in details if not d.get("error")]
    by_cat = defaultdict(list)
    for d in scored:
        by_cat[d["category"]].append(d)
    n = max(len(scored), 1)
    sql_runs = sum(d["sql_runs"] for d in scored)
    answerable = [d for d in scored if d["category"] != "refuse"]
    return {
        "n_questions": len(details), "n_scored": len(scored), "errors": len(errors),
        "overall_pass_rate": round(sum(d["passed"] for d in scored) / len(scored), 3) if scored else 0,
        "by_category": {c: {"n": len(v), "pass_rate": round(sum(x["passed"] for x in v) / len(v), 3)} for c, v in by_cat.items()},
        "false_refusal_rate": round(sum(1 for d in answerable if d["reason"].startswith("refused")) / max(len(answerable), 1), 3),
        "sql_error_rate": round(sum(d["sql_errors"] for d in scored) / max(sql_runs, 1), 3),
        "avg_steps": round(sum(d["steps"] for d in scored) / n, 2),
        "avg_tokens": round(sum(d["tokens"] for d in scored) / n),
        "avg_latency_s": round(sum(d["latency_ms"] for d in scored) / n / 1000, 2),
        "details": details,
    }


def markdown_table(summary):
    names = {"docs": "Document questions (cited the right document)", "refuse": "Unanswerable questions (refused)",
             "sql": "Text-to-SQL (execution accuracy)", "mixed": "Data + policy (both tools, right citation)"}
    lines = ["| Category | Questions | Pass rate |", "|---|---|---|"]
    for cat in ("docs", "refuse", "sql", "mixed"):
        if cat in summary["by_category"]:
            c = summary["by_category"][cat]
            lines.append(f"| {names[cat]} | {c['n']} | {c['pass_rate']:.0%} |")
    lines.append(f"| **Overall** | {summary['n_scored']} | **{summary['overall_pass_rate']:.0%}** |")
    lines += ["", f"False refusal rate {summary['false_refusal_rate']:.0%} | SQL error rate {summary['sql_error_rate']:.0%} | "
              f"avg {summary['avg_steps']} steps, {summary['avg_tokens']} tokens, {summary['avg_latency_s']}s per question"]
    if summary["errors"]:
        lines += ["", f"{summary['errors']} of {summary['n_questions']} question(s) failed on provider errors (rate limits or "
                      "network) and are NOT counted above. Run again to retry them."]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None, help="evaluate only every Nth-spaced sample of this many questions")
    p.add_argument("--categories", nargs="*", default=None, choices=["docs", "refuse", "sql", "mixed"])
    p.add_argument("--sleep", type=float, default=0.0, help="seconds to wait between questions")
    p.add_argument("--retry-errors", type=int, default=1, help="extra passes over questions that hit provider errors")
    p.add_argument("--retry-wait", type=float, default=30.0, help="seconds to wait before a retry pass")
    a = p.parse_args()

    state = build_state()
    if not state.llm_ready:
        raise SystemExit("No LLM configured. Set LLM_PROVIDER and the matching API key in .env first.")
    cases = [c for c in load_cases() if not a.categories or c["category"] in a.categories]
    if a.limit and a.limit < len(cases):
        step = len(cases) / a.limit
        cases = [cases[int(i * step)] for i in range(a.limit)]
    student_id = pick_demo_student(state.sql_tool)
    print(f"Retrieval: {state.retriever_name} | demo student: {student_id} | {len(cases)} questions\n")

    summary = run_eval(state.agent(), state.sql_tool, cases, student_id, sleep=a.sleep,
                       retry_errors=a.retry_errors, retry_wait=a.retry_wait)
    table = markdown_table(summary)
    print("\n" + table)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "agent_eval.json").write_text(json.dumps(summary, indent=2))
    (RESULTS / "agent_eval.md").write_text(table + "\n")


if __name__ == "__main__":
    main()
