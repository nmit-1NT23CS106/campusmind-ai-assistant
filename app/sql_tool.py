"""Run validated, read-only SQL with row-level scoping.

Defence in depth (each layer is tested separately):
  1. sql_guard: one SELECT, allowlisted tables only
  2. read-only SQLite connection (mode=ro) plus PRAGMA query_only
  3. authorizer: only SELECT / READ / allowed functions may execute
  4. row-level scope: for students, TEMP views shadow the real tables so every query,
     join, subquery or UNION can only see that student's own rows
  5. row limit and wall-clock timeout
"""
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

from .sql_guard import ALLOWED_TABLES, BLOCKED_FUNCTIONS, GuardError, validate_select


class SQLToolError(Exception):
    """Any failure running a query. The message is safe to show to the model."""


@dataclass(frozen=True)
class Scope:
    role: str
    student_id: int | None = None

    def __post_init__(self):
        if self.role not in ("admin", "student"):
            raise ValueError(f"Unknown role '{self.role}'")
        if self.role == "student" and not isinstance(self.student_id, int):
            raise ValueError("A student scope needs an integer student_id")


@dataclass
class SQLResult:
    sql: str
    columns: list
    rows: list
    truncated: bool = False
    row_count: int = 0
    seconds: float = 0.0


def _scoped_views(student_id):
    sid = int(student_id)  # integer literal only, never a string from outside
    own = f"e.student_id = {sid}"
    return [
        f"CREATE TEMP VIEW students AS SELECT * FROM main.students WHERE student_id = {sid}",
        f"CREATE TEMP VIEW enrollments AS SELECT * FROM main.enrollments WHERE student_id = {sid}",
        "CREATE TEMP VIEW attendance_records AS SELECT r.* FROM main.attendance_records r "
        f"JOIN main.enrollments e ON e.enrollment_id = r.enrollment_id WHERE {own}",
        "CREATE TEMP VIEW assessments AS SELECT a.* FROM main.assessments a "
        f"JOIN main.enrollments e ON e.enrollment_id = a.enrollment_id WHERE {own}",
        f"CREATE TEMP VIEW enrollment_summary AS SELECT * FROM main.enrollment_summary WHERE student_id = {sid}",
    ]


def _authorizer(action, arg1, arg2, dbname, source):
    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_RECURSIVE}
    if action in allowed:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() not in BLOCKED_FUNCTIONS:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


class SQLTool:
    def __init__(self, db_path, max_rows=100, timeout_s=2.0):
        self.db_path = Path(db_path).resolve()
        self.max_rows = max_rows
        self.timeout_s = timeout_s

    def _connect(self):
        return sqlite3.connect(f"{self.db_path.as_uri()}?mode=ro", uri=True)

    def schema_text(self):
        """DDL of the allowed tables, shown to the model so it writes correct SQL."""
        conn = self._connect()
        try:
            rows = conn.execute("SELECT name, sql FROM sqlite_master WHERE type IN ('table','view') AND sql IS NOT NULL").fetchall()
        finally:
            conn.close()
        return "\n\n".join(sql for name, sql in sorted(rows) if name in ALLOWED_TABLES)

    def student_exists(self, student_id):
        conn = self._connect()
        try:
            return conn.execute("SELECT 1 FROM students WHERE student_id = ?", (int(student_id),)).fetchone() is not None
        finally:
            conn.close()

    def demo_students(self, limit=5):
        """A few fake students for the demo UI. Prefers ones with at least one course under 75%."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT s.student_id, s.name, MIN(m.attendance_pct) AS low FROM students s "
                "JOIN enrollment_summary m ON m.student_id = s.student_id "
                "GROUP BY s.student_id ORDER BY (low < 75) DESC, s.student_id LIMIT ?", (int(limit),)).fetchall()
        finally:
            conn.close()
        return [{"student_id": r[0], "name": r[1], "lowest_attendance": r[2]} for r in rows]

    def run(self, sql, scope):
        """Validate then execute. Raises SQLToolError with a model-readable message on any failure."""
        try:
            checked = validate_select(sql)
        except GuardError as exc:
            raise SQLToolError(str(exc)) from exc
        return self._execute(checked, scope)

    def _execute(self, checked_sql, scope):
        conn = self._connect()
        started = time.perf_counter()
        try:
            if scope.role == "student":
                for statement in _scoped_views(scope.student_id):
                    conn.execute(statement)
            conn.execute("PRAGMA query_only = ON")
            conn.set_authorizer(_authorizer)
            conn.set_progress_handler(lambda: 1 if time.perf_counter() - started > self.timeout_s else 0, 10_000)

            cursor = conn.execute(f"SELECT * FROM ({checked_sql}) LIMIT ?", (self.max_rows + 1,))
            columns = [d[0] for d in cursor.description]
            rows = cursor.fetchall()
        except sqlite3.OperationalError as exc:
            message = str(exc)
            if "interrupted" in message:
                message = f"The query took longer than {self.timeout_s} seconds. Simplify it."
            raise SQLToolError(f"SQL error: {message}") from exc
        except sqlite3.DatabaseError as exc:
            raise SQLToolError(f"SQL error: {exc}") from exc
        finally:
            conn.close()

        truncated = len(rows) > self.max_rows
        rows = [list(r) for r in rows[: self.max_rows]]
        return SQLResult(sql=checked_sql, columns=columns, rows=rows, truncated=truncated,
                         row_count=len(rows), seconds=round(time.perf_counter() - started, 4))
