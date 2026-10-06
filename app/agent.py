"""The agent: an LLM that chooses between document search, guarded SQL, both, or refusing.

It runs a short loop (ReAct style): the model replies with one JSON action, the agent runs the tool,
feeds the observation back, and repeats until the model answers or refuses. Everything the model
claims is checked: citations must point at passages that were really retrieved, and an answer
before any tool call is rejected.
"""
import json
import re
import time
from dataclasses import asdict, dataclass, field

from .prompts import build_system_prompt
from .sql_tool import SQLToolError

LAST_STEP_NOTE = "\n\nThis is your last step. Reply now with action 'answer' (using what you have) or 'refuse'."
REPEAT_SEARCH_NOTE = "\n\nYou already searched for this. Use the passages above to answer, or refuse if they do not contain the answer."
OBSERVATION_ROWS = 20  # rows shown to the model per query (the UI still gets more)
PAYLOAD_ROWS = 50


@dataclass
class AgentResult:
    answer: str
    sources: list = field(default_factory=list)
    refused: bool = False
    tools_used: list = field(default_factory=list)
    sql_runs: list = field(default_factory=list)
    trace: list = field(default_factory=list)
    steps: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0

    def to_dict(self):
        return asdict(self)


def parse_action(text):
    """Pull the first JSON object out of a model reply (tolerates code fences and stray prose)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)  # some models print their reasoning
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found")
    data = json.loads(cleaned[start:end + 1], strict=False)  # strict=False tolerates raw newlines inside strings
    if not isinstance(data, dict) or "action" not in data:
        raise ValueError("JSON object has no 'action'")
    return data


def format_rows(result):
    shown = result.rows[:OBSERVATION_ROWS]
    lines = [" | ".join(result.columns)] + [" | ".join("NULL" if v is None else str(v) for v in row) for row in shown]
    note = f"{result.row_count} row(s)"
    if result.truncated:
        note += f" (query hit the {result.row_count}-row limit; more exist)"
    if result.row_count > len(shown):
        note += f", showing the first {len(shown)}"
    return f"SQL result: {note}\n" + "\n".join(lines)


class Agent:
    def __init__(self, llm, retriever, sql_tool, max_steps=6, doc_k=4):
        self.llm, self.retriever, self.sql_tool = llm, retriever, sql_tool
        self.max_steps, self.doc_k = max_steps, doc_k
        self._schema = None

    def ask(self, question, scope, today=None):
        if self._schema is None:
            self._schema = self.sql_tool.schema_text()
        system = build_system_prompt(self._schema, scope, today)
        messages = [{"role": "user", "content": question}]
        result = AgentResult(answer="")
        passages = []  # chunks retrieved so far; position + 1 is the [n] shown to the model
        started = time.perf_counter()

        for step in range(1, self.max_steps + 1):
            reply = self.llm.complete(system, messages)
            result.steps = step
            result.input_tokens += reply.input_tokens
            result.output_tokens += reply.output_tokens
            messages.append({"role": "assistant", "content": reply.text})

            try:
                action = parse_action(reply.text)
            except (ValueError, json.JSONDecodeError) as exc:
                result.trace.append({"step": step, "action": "invalid_reply", "detail": str(exc),
                                     "reply": reply.text[:300]})
                note = f"Your reply was not valid ({str(exc)[:80]}). Reply with exactly one JSON object and nothing else."
                if step == self.max_steps - 1:
                    note += LAST_STEP_NOTE
                messages.append({"role": "user", "content": note})
                continue

            kind = action.get("action")
            if kind == "search_docs":
                observation = self._search(action.get("query"), passages, result, step)
            elif kind == "query_sql":
                observation = self._sql(action.get("sql"), scope, result, step)
            elif kind == "answer":
                text = action.get("answer")
                if not isinstance(text, str) or not text.strip():
                    observation = "ERROR: the answer was empty."
                    result.trace.append({"step": step, "action": "answer_rejected", "detail": "empty answer"})
                elif not result.tools_used:
                    observation = "ERROR: use search_docs or query_sql before answering. Do not answer from memory."
                    result.trace.append({"step": step, "action": "answer_rejected", "detail": "no tool used"})
                else:
                    result.answer = text.strip()
                    result.sources = self._resolve_sources(action.get("sources"), passages)
                    result.trace.append({"step": step, "action": "answer", "detail": f"{len(result.sources)} verified source(s)"})
                    return self._finish(result, started)
            elif kind == "refuse":
                result.refused = True
                result.answer = str(action.get("reason") or "I can't help with that.").strip()
                result.trace.append({"step": step, "action": "refuse", "detail": result.answer})
                return self._finish(result, started)
            else:
                observation = f"ERROR: unknown action '{kind}'. Use search_docs, query_sql, answer or refuse."
                result.trace.append({"step": step, "action": "unknown_action", "detail": str(kind)})

            if step == self.max_steps - 1:
                observation += LAST_STEP_NOTE
            messages.append({"role": "user", "content": observation})

        result.refused = True
        result.answer = "I couldn't complete that request. Please try rephrasing your question."
        result.trace.append({"step": result.steps, "action": "step_limit", "detail": f"stopped after {self.max_steps} steps"})
        return self._finish(result, started)

    # ----- tools -----
    def _search(self, query, passages, result, step):
        result.tools_used.append("search_docs")
        if not isinstance(query, str) or not query.strip():
            result.trace.append({"step": step, "action": "search_docs", "detail": "empty query"})
            return "ERROR: the search query was empty."
        repeated = any(t["action"] == "search_docs" and t.get("detail", "").lower() == query.strip().lower()
                       for t in result.trace)
        hits = [chunk for chunk, _ in self.retriever.search(query.strip(), k=self.doc_k)]
        blocks = []
        for chunk in hits:
            if chunk.citation not in [p.citation for p in passages]:
                passages.append(chunk)
            number = [p.citation for p in passages].index(chunk.citation) + 1
            blocks.append(f"[{number}] {chunk.citation}\n{chunk.text}")
        result.trace.append({"step": step, "action": "search_docs", "detail": query.strip(),
                             "retrieved": [c.citation for c in hits]})
        return "Passages found:\n\n" + "\n\n".join(blocks) + (REPEAT_SEARCH_NOTE if repeated else "")

    def _sql(self, sql, scope, result, step):
        result.tools_used.append("query_sql")
        run = {"sql": sql if isinstance(sql, str) else "", "columns": [], "rows": [], "row_count": 0,
               "truncated": False, "error": None}
        try:
            res = self.sql_tool.run(sql, scope)
        except SQLToolError as exc:
            run["error"] = str(exc)
            result.sql_runs.append(run)
            result.trace.append({"step": step, "action": "query_sql", "detail": run["sql"], "error": run["error"]})
            return f"ERROR: {exc}"
        run.update(sql=res.sql, columns=res.columns, rows=res.rows[:PAYLOAD_ROWS], row_count=res.row_count,
                   truncated=res.truncated)
        result.sql_runs.append(run)
        result.trace.append({"step": step, "action": "query_sql", "detail": res.sql, "rows": res.row_count})
        return format_rows(res)

    @staticmethod
    def _resolve_sources(requested, passages):
        """Keep only sources that match passages that were actually retrieved."""
        by_citation = {p.citation: p.citation for p in passages}
        resolved = []
        for item in requested if isinstance(requested, list) else []:
            text = str(item).strip()
            match = re.fullmatch(r"\[?(\d+)\]?", text)
            if match and 1 <= int(match.group(1)) <= len(passages):
                citation = passages[int(match.group(1)) - 1].citation
            else:
                citation = by_citation.get(text)
            if citation and citation not in resolved:
                resolved.append(citation)
        return resolved

    @staticmethod
    def _finish(result, started):
        result.latency_ms = round((time.perf_counter() - started) * 1000, 1)
        return result
