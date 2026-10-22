"""Draft resume bullets from your MEASURED evaluation results. Prints only numbers that exist.

Usage:  python -m scripts.resume_bullets --tests 231
Run `python -m eval.run_eval --all` and `python -m eval.agent_eval --sleep 8` first.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "eval" / "results"
MIN_AGENT_QUESTIONS = 40


def pct(x):
    return f"{x:.0%}"


def _default_config(r):
    return (r.get("strategy") == "heading" and not r.get("chunk_kwargs")
            and all(v == 1.0 for v in r.get("retriever_options", {}).values()))


def load_retrieval(results_dir=RESULTS):
    """Newest result file per retriever, using default chunking and equal fusion weights."""
    found = {}
    for path in sorted(Path(results_dir).glob("*.json"), key=lambda p: p.stat().st_mtime):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if isinstance(data, dict) and "retriever" in data and "hit@3" in data and _default_config(data):
            found[data["retriever"]] = data
    return found


def load_agent(results_dir=RESULTS):
    path = Path(results_dir) / "agent_eval.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def retrieval_bullet(found):
    """Returns (bullet_or_None, warnings)."""
    bm25, hybrid, dense = found.get("bm25"), found.get("hybrid"), found.get("dense")
    if not bm25 or not hybrid:
        return None, ["Retrieval: run `python -m eval.run_eval --all` first (needs bm25 and hybrid results)."]
    n = hybrid["n_questions"]
    base = ("Built a hybrid retrieval pipeline over university policy documents (BM25 + bge-small embeddings merged with "
            "Reciprocal Rank Fusion) with cited answers")
    if hybrid["hit@3"] > bm25["hit@3"]:
        return (f"{base}; on a {n}-question evaluation set, hit@3 rose from {pct(bm25['hit@3'])} (keyword search) to "
                f"{pct(hybrid['hit@3'])} (hit@1 {pct(bm25['hit@1'])} to {pct(hybrid['hit@1'])})."), []
    parts = f"BM25 {pct(bm25['hit@3'])}" + (f", dense {pct(dense['hit@3'])}" if dense else "") + f", hybrid {pct(hybrid['hit@3'])}"
    return (f"{base}; compared BM25, dense and hybrid retrieval on a {n}-question evaluation set (hit@3: {parts})."), [
        "Retrieval: hybrid did not beat BM25 on hit@3. Do NOT write 'improved'. Report the comparison honestly, or tune and re-measure."]


def agent_bullet(agent):
    if not agent:
        return None, ["Agent: run `python -m eval.agent_eval --sleep 8` first."]
    warnings = []
    if agent["errors"]:
        warnings.append(f"Agent: {agent['errors']} question(s) hit provider errors and are not counted. Re-run before quoting numbers.")
    if agent["n_scored"] < MIN_AGENT_QUESTIONS:
        warnings.append(f"Agent: only {agent['n_scored']} questions were scored (full set is ~55). Run the full evaluation before quoting.")
    cats = agent["by_category"]

    def frac(c):  # counts next to percentages, so a 5-question category cannot look stronger than it is
        return f"{round(c['pass_rate'] * c['n'])} of {c['n']} ({pct(c['pass_rate'])})"

    claims = []
    if "docs" in cats:
        claims.append(f"cited the correct source on {frac(cats['docs'])} document questions")
    if "sql" in cats:
        claims.append(f"matched gold SQL results on {frac(cats['sql'])} data questions")
    if "refuse" in cats:
        claims.append(f"refused {frac(cats['refuse'])} unanswerable questions")
    if "mixed" in cats:
        claims.append(f"used both tools and cited the right policy on {frac(cats['mixed'])} mixed questions")
    if not claims:
        return None, warnings + ["Agent: results file has no categories."]
    return (f"Developed an LLM agent (Qwen via Groq) that routes each question to document search, guarded text-to-SQL, both, "
            f"or refusal; on a {agent['n_scored']}-question end-to-end evaluation it " + ", ".join(claims[:-1])
            + (", and " if len(claims) > 1 else "") + claims[-1] + "."), warnings


def safety_bullet(tests):
    count = f"{tests}+ automated tests" if tests else "automated tests"
    return ("Secured model-written SQL with five layers (parser-based validator, read-only connection, authorizer, per-student "
            "row-level scoping, row limit and timeout), verified by " + count + " including attempts to read other students' data.")


def build(results_dir=RESULTS, tests=None):
    bullets, warnings = [], []
    for text, warn in (retrieval_bullet(load_retrieval(results_dir)), agent_bullet(load_agent(results_dir))):
        warnings += warn
        if text:
            bullets.append(text)
    bullets.append(safety_bullet(tests))
    return bullets, warnings


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tests", type=int, default=None, help="how many tests pytest reports as passing")
    a = p.parse_args()
    bullets, warnings = build(tests=a.tests)
    print("CampusMind: AI Academic Assistant | Python, FastAPI, Streamlit, SQLite, BM25, embeddings (fastembed), Groq LLM API, pytest\n")
    for b in bullets:
        print(f"  - {b}\n")
    if warnings:
        print("CHECK BEFORE USING:")
        for w in warnings:
            print(f"  ! {w}")
    print("\nAll numbers come from a small synthetic corpus and a self-written question set. Say 'internal evaluation set' if asked.")


if __name__ == "__main__":
    main()
