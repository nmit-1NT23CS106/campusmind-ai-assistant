"""Append one JSON line per question: latency, tokens, tools. No row data is stored."""
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

_lock = threading.Lock()


def log_interaction(path, *, question, role, result=None, error=None, student_id=None):
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "role": role, "student_id": student_id, "question": question, "error": error,
    }
    if result is not None:
        record.update(
            refused=result.refused, tools=result.tools_used, steps=result.steps,
            input_tokens=result.input_tokens, output_tokens=result.output_tokens,
            latency_ms=result.latency_ms, sources=result.sources,
            sql=[r["sql"] for r in result.sql_runs], sql_errors=sum(1 for r in result.sql_runs if r["error"]),
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock, path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def summarize(path):
    path = Path(path)
    if not path.exists():
        return {"questions": 0}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    done = [r for r in rows if r.get("latency_ms") is not None]
    n = max(len(done), 1)
    return {
        "questions": len(rows), "errors": sum(1 for r in rows if r.get("error")),
        "avg_latency_ms": round(sum(r["latency_ms"] for r in done) / n, 1),
        "avg_input_tokens": round(sum(r["input_tokens"] for r in done) / n),
        "avg_output_tokens": round(sum(r["output_tokens"] for r in done) / n),
    }
