import sqlite3

from data.seed import CLASSES_PER_COURSE, COURSES, build


def make(tmp_path, n=30):
    path = build(tmp_path / "t.db", seed=7, n_students=n)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def test_row_counts(tmp_path):
    c = make(tmp_path)
    assert c.execute("SELECT COUNT(*) FROM students").fetchone()[0] == 30
    assert c.execute("SELECT COUNT(*) FROM enrollments").fetchone()[0] == 30 * len(COURSES)
    assert c.execute("SELECT COUNT(*) FROM attendance_records").fetchone()[0] == 30 * len(COURSES) * CLASSES_PER_COURSE
    assert c.execute("SELECT COUNT(*) FROM assessments").fetchone()[0] == 30 * len(COURSES) * 3


def test_foreign_keys_are_consistent(tmp_path):
    assert make(tmp_path).execute("PRAGMA foreign_key_check").fetchall() == []


def test_summary_view_values_are_sane(tmp_path):
    rows = make(tmp_path).execute(
        "SELECT attendance_pct, classes_held, internal_marks, internal_max FROM enrollment_summary").fetchall()
    assert rows
    for pct, held, marks, mx in rows:
        assert 0 <= pct <= 100 and held == CLASSES_PER_COURSE and 0 <= marks <= mx == 40


def test_data_has_both_safe_and_at_risk_students(tmp_path):
    c = make(tmp_path, n=80)
    below = c.execute("SELECT COUNT(*) FROM enrollment_summary WHERE attendance_pct < 75").fetchone()[0]
    total = c.execute("SELECT COUNT(*) FROM enrollment_summary").fetchone()[0]
    assert 0 < below < total * 0.4


def test_same_seed_gives_identical_data(tmp_path):
    a = sqlite3.connect(build(tmp_path / "a.db", seed=3, n_students=10)).execute("SELECT * FROM students").fetchall()
    b = sqlite3.connect(build(tmp_path / "b.db", seed=3, n_students=10)).execute("SELECT * FROM students").fetchall()
    assert a == b
