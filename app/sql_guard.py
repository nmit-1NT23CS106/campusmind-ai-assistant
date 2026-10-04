"""Validate model-written SQL BEFORE it touches the database.

Rules: exactly one statement; it must be a SELECT (or UNION of SELECTs); only allowlisted tables;
no schema-qualified names (main.x, temp.x); no table-valued functions; no dangerous functions.
The validated query is regenerated from the parse tree, so what runs is what was checked.
This is layer 1 of defence in depth; sql_tool adds a read-only connection, an authorizer,
row-level scoping, a row limit and a timeout.
"""
import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

ALLOWED_TABLES = frozenset({"students", "courses", "enrollments", "attendance_records",
                            "assessments", "enrollment_summary"})
BLOCKED_FUNCTIONS = frozenset({"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer",
                               "zeroblob", "randomblob", "printf_unbounded"})
MAX_SQL_CHARS = 2000
_SET_OPERATION = getattr(exp, "SetOperation", exp.Union)


class GuardError(ValueError):
    """The SQL was rejected. The message is safe to show to the model so it can correct itself."""


def validate_select(sql, allowed_tables=ALLOWED_TABLES):
    if not isinstance(sql, str) or not sql.strip():
        raise GuardError("The SQL query is empty.")
    if len(sql) > MAX_SQL_CHARS:
        raise GuardError(f"The SQL query is too long (limit {MAX_SQL_CHARS} characters).")

    try:
        statements = [s for s in sqlglot.parse(sql, read="sqlite") if s is not None]
    except SqlglotError as exc:
        raise GuardError(f"The SQL could not be parsed: {str(exc).splitlines()[0][:150]}") from exc

    if len(statements) != 1:
        raise GuardError("Send exactly one SQL statement.")
    root = statements[0]
    if not isinstance(root, (exp.Select, _SET_OPERATION)):
        raise GuardError("Only SELECT queries are allowed.")

    cte_names = {cte.alias_or_name.lower() for cte in root.find_all(exp.CTE)}
    for table in root.find_all(exp.Table):
        name = table.name.lower()
        if table.db or table.catalog:
            raise GuardError("Schema-qualified table names (like main.students) are not allowed.")
        if not name:
            raise GuardError("Table-valued functions are not allowed.")
        if name in cte_names:
            continue
        if name not in allowed_tables:
            raise GuardError(f"Table '{table.name}' is not available. Allowed tables: {', '.join(sorted(allowed_tables))}.")

    for func in root.find_all(exp.Func):
        fname = (func.name if isinstance(func, exp.Anonymous) else func.sql_name()).lower()
        if fname in BLOCKED_FUNCTIONS:
            raise GuardError(f"Function '{fname}' is not allowed.")

    return root.sql(dialect="sqlite", comments=False)
