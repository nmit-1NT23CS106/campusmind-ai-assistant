import pytest

from app.sql_guard import GuardError, validate_select

GOOD = [
    "SELECT * FROM students",
    "SELECT name FROM students WHERE cgpa > 9 ORDER BY cgpa DESC LIMIT 5;",
    "SELECT course_code, AVG(attendance_pct) FROM enrollment_summary GROUP BY course_code",
    "SELECT s.name, c.name FROM students s JOIN enrollments e ON e.student_id = s.student_id JOIN courses c ON c.course_id = e.course_id",
    "WITH low AS (SELECT * FROM enrollment_summary WHERE attendance_pct < 75) SELECT student_name FROM low",
    "SELECT name FROM students UNION SELECT name FROM courses",
    "SELECT * FROM students WHERE student_id IN (SELECT student_id FROM enrollments)",
    "SELECT COUNT(*) FROM students -- a harmless comment",
    "select round(avg(cgpa), 2) from students",
]

BAD = {
    "DROP TABLE students": "Only SELECT",
    "DELETE FROM students": "Only SELECT",
    "INSERT INTO students VALUES (1,'a','b','c',1,1)": "Only SELECT",
    "UPDATE students SET cgpa = 10": "Only SELECT",
    "CREATE TABLE x (a INT)": "Only SELECT",
    "PRAGMA table_info(students)": "Only SELECT",
    "ATTACH DATABASE 'x.db' AS x": "Only SELECT",
    "SELECT 1; DROP TABLE students": "exactly one",
    "SELECT * FROM students; SELECT * FROM courses": "exactly one",
    "SELECT * FROM users": "not available",
    "SELECT * FROM sqlite_master": "not available",
    "SELECT * FROM main.students": "Schema-qualified",
    "SELECT * FROM temp.students": "Schema-qualified",
    "SELECT * FROM json_each('[1,2]')": "Table-valued",
    "SELECT load_extension('evil')": "not allowed",
    "SELECT zeroblob(1000000000)": "not allowed",
    "SELECT * FROM students WHERE student_id IN (SELECT id FROM secrets)": "not available",
    "": "empty",
    "   ": "empty",
    "this is not sql at all": None,
}


@pytest.mark.parametrize("sql", GOOD)
def test_valid_selects_pass_and_are_normalized(sql):
    out = validate_select(sql)
    assert out.upper().startswith(("SELECT", "WITH")) and not out.rstrip().endswith(";")


@pytest.mark.parametrize("sql,reason", list(BAD.items()), ids=[s[:40] or "empty" for s in BAD])
def test_dangerous_or_invalid_sql_is_rejected(sql, reason):
    with pytest.raises(GuardError) as exc:
        validate_select(sql)
    if reason:
        assert reason.lower() in str(exc.value).lower()


def test_overlong_query_rejected():
    with pytest.raises(GuardError, match="too long"):
        validate_select("SELECT " + "1," * 2000 + "1")


def test_non_string_rejected():
    with pytest.raises(GuardError):
        validate_select(None)


def test_what_runs_is_what_was_parsed():
    """The returned SQL is regenerated from the parse tree, so comments and tricks are dropped."""
    out = validate_select("SELECT /* sneaky */ name FROM students -- trailing")
    assert "sneaky" not in out and "trailing" not in out
