import sqlite3

import pytest

from app.sql_tool import Scope, SQLTool, SQLToolError
from data.seed import CLASSES_PER_COURSE, COURSES, build


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    return build(tmp_path_factory.mktemp("db") / "t.db", seed=11, n_students=25)


@pytest.fixture()
def tool(db):
    return SQLTool(db, max_rows=100, timeout_s=1.0)


ADMIN = Scope("admin")
ME = 20260003
STUDENT = Scope("student", ME)


def scalar(tool, sql, scope):
    return tool.run(sql, scope).rows[0][0]


def test_admin_sees_everything(tool):
    assert scalar(tool, "SELECT COUNT(*) FROM students", ADMIN) == 25
    assert scalar(tool, "SELECT COUNT(*) FROM enrollments", ADMIN) == 25 * len(COURSES)


def test_result_shape(tool):
    r = tool.run("SELECT name, cgpa FROM students ORDER BY cgpa DESC LIMIT 3", ADMIN)
    assert r.columns == ["name", "cgpa"] and r.row_count == 3 and not r.truncated


def test_row_limit_sets_truncated_flag(db):
    small = SQLTool(db, max_rows=10)
    r = small.run("SELECT * FROM attendance_records", ADMIN)
    assert r.row_count == 10 and r.truncated is True


# ---------- row-level scoping: a student can only ever see their own rows ----------
def test_student_sees_only_own_student_row(tool):
    assert scalar(tool, "SELECT COUNT(*) FROM students", STUDENT) == 1
    assert scalar(tool, "SELECT student_id FROM students", STUDENT) == ME


def test_student_scope_applies_to_every_table(tool):
    assert scalar(tool, "SELECT COUNT(*) FROM enrollments", STUDENT) == len(COURSES)
    assert scalar(tool, "SELECT COUNT(*) FROM attendance_records", STUDENT) == len(COURSES) * CLASSES_PER_COURSE
    assert scalar(tool, "SELECT COUNT(*) FROM assessments", STUDENT) == len(COURSES) * 3
    assert scalar(tool, "SELECT COUNT(*) FROM enrollment_summary", STUDENT) == len(COURSES)


# (query, expected row count). Everything a student can reach is limited to their own rows.
@pytest.mark.parametrize("sql,expected", [
    (f"SELECT COUNT(*) FROM students WHERE student_id <> {ME}", 0),
    ("SELECT COUNT(*) FROM students s JOIN enrollments e ON e.student_id <> s.student_id", 0),
    (f"WITH everyone AS (SELECT * FROM students) SELECT COUNT(*) FROM everyone WHERE student_id <> {ME}", 0),
    (f"SELECT COUNT(*) FROM enrollment_summary WHERE student_id <> {ME}", 0),
    (f"SELECT COUNT(*) FROM students WHERE student_id IN (SELECT student_id FROM enrollments WHERE student_id <> {ME})", 0),
    ("SELECT COUNT(*) FROM (SELECT * FROM students UNION SELECT * FROM students)", 1),  # only their own row exists
    ("SELECT COUNT(DISTINCT student_id) FROM enrollments", 1),
    ("SELECT COUNT(DISTINCT e.student_id) FROM enrollments e CROSS JOIN students s", 1),
])
def test_student_cannot_reach_other_students_by_any_trick(tool, sql, expected):
    assert scalar(tool, sql, STUDENT) == expected


def test_student_cannot_bypass_scope_with_qualified_names(tool):
    with pytest.raises(SQLToolError, match="Schema-qualified"):
        tool.run("SELECT * FROM main.students", STUDENT)


def test_courses_are_public_reference_data(tool):
    assert scalar(tool, "SELECT COUNT(*) FROM courses", STUDENT) == len(COURSES)


def test_student_scope_matches_admin_filter(tool):
    mine = tool.run("SELECT course_code, attendance_pct FROM enrollment_summary ORDER BY course_code", STUDENT).rows
    admin = tool.run(f"SELECT course_code, attendance_pct FROM enrollment_summary WHERE student_id = {ME} ORDER BY course_code", ADMIN).rows
    assert mine == admin


# ---------- layers behind the guard (guard deliberately bypassed) ----------
def test_connection_is_read_only_even_if_guard_is_bypassed(tool):
    with pytest.raises(SQLToolError):
        tool._execute("DELETE FROM students", ADMIN)


def test_authorizer_blocks_pragma_even_if_guard_is_bypassed(tool):
    with pytest.raises(SQLToolError):
        tool._execute("PRAGMA table_info(students)", ADMIN)


def test_raw_connection_is_read_only(tool):
    conn = tool._connect()
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("INSERT INTO students VALUES (1,'x','x@x','x',1,1.0)")
    conn.close()


def test_runaway_query_is_stopped_by_timeout(tool):
    with pytest.raises(SQLToolError, match="longer than"):
        tool.run("WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) SELECT COUNT(*) FROM c", ADMIN)


def test_sql_errors_become_model_readable_messages(tool):
    with pytest.raises(SQLToolError, match="SQL error"):
        tool.run("SELECT no_such_column FROM students", ADMIN)


def test_guard_errors_surface_as_tool_errors(tool):
    with pytest.raises(SQLToolError, match="Only SELECT"):
        tool.run("DROP TABLE students", ADMIN)


def test_scope_validation():
    with pytest.raises(ValueError):
        Scope("teacher")
    with pytest.raises(ValueError):
        Scope("student")
    with pytest.raises(ValueError):
        Scope("student", "20260003")  # must be an int, never a raw string


def test_schema_text_lists_only_allowed_objects(tool):
    text = tool.schema_text()
    for name in ("students", "enrollment_summary", "attendance_records"):
        assert name in text
    assert "sqlite_" not in text
