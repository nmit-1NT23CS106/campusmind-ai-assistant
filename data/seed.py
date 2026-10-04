"""Create a reproducible fake university database (SQLite) for CampusMind.

Usage: python -m data.seed            (writes data/campus.db)
All data is synthetic. The same seed always produces the same database.
"""
import random
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "campus.db"
N_STUDENTS = 200
FIRST = ["Aarav", "Diya", "Kabir", "Meera", "Rohan", "Sana", "Vikram", "Isha", "Arjun", "Neha", "Tara", "Yash",
         "Anaya", "Dev", "Kavya", "Nikhil", "Pooja", "Rahul", "Sneha", "Varun", "Zoya", "Aditya", "Bhavna", "Chetan"]
LAST = ["Rao", "Nair", "Shah", "Iyer", "Das", "Khan", "Joshi", "Menon", "Patel", "Reddy", "Bose", "Gupta",
        "Sharma", "Kulkarni", "Hegde", "Bhat", "Singh", "Verma", "Pillai", "Mishra"]
DEPARTMENTS = ["Computer Science", "Information Science", "Electronics"]
COURSES = [  # code, name, credits, department
    ("CS301", "Database Management Systems", 4, "Computer Science"),
    ("CS302", "Operating Systems", 4, "Computer Science"),
    ("CS303", "Machine Learning", 3, "Computer Science"),
    ("CS304", "Computer Networks", 3, "Computer Science"),
    ("CS305", "Software Engineering", 3, "Computer Science"),
]
SEMESTER_START = date(2026, 8, 3)
CLASSES_PER_COURSE = 36

SCHEMA = """
CREATE TABLE students (
  student_id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT NOT NULL UNIQUE,
  department TEXT NOT NULL, semester INTEGER NOT NULL, cgpa REAL NOT NULL);
CREATE TABLE courses (
  course_id INTEGER PRIMARY KEY, code TEXT NOT NULL UNIQUE, name TEXT NOT NULL,
  credits INTEGER NOT NULL, department TEXT NOT NULL);
CREATE TABLE enrollments (
  enrollment_id INTEGER PRIMARY KEY,
  student_id INTEGER NOT NULL REFERENCES students(student_id),
  course_id INTEGER NOT NULL REFERENCES courses(course_id),
  UNIQUE (student_id, course_id));
CREATE TABLE attendance_records (
  record_id INTEGER PRIMARY KEY,
  enrollment_id INTEGER NOT NULL REFERENCES enrollments(enrollment_id),
  class_date TEXT NOT NULL, present INTEGER NOT NULL CHECK (present IN (0, 1)));
CREATE TABLE assessments (
  assessment_id INTEGER PRIMARY KEY,
  enrollment_id INTEGER NOT NULL REFERENCES enrollments(enrollment_id),
  kind TEXT NOT NULL CHECK (kind IN ('IA1', 'IA2', 'Assignment')),
  marks REAL NOT NULL, max_marks REAL NOT NULL);
CREATE INDEX idx_att_enrollment ON attendance_records(enrollment_id);
CREATE INDEX idx_assess_enrollment ON assessments(enrollment_id);
CREATE INDEX idx_enroll_student ON enrollments(student_id);

CREATE VIEW enrollment_summary AS
SELECT e.enrollment_id, s.student_id, s.name AS student_name, s.department, s.cgpa,
       c.code AS course_code, c.name AS course_name,
       COUNT(a.record_id) AS classes_held,
       SUM(a.present) AS classes_attended,
       ROUND(100.0 * SUM(a.present) / COUNT(a.record_id), 1) AS attendance_pct,
       (SELECT SUM(marks) FROM assessments x WHERE x.enrollment_id = e.enrollment_id) AS internal_marks,
       (SELECT SUM(max_marks) FROM assessments x WHERE x.enrollment_id = e.enrollment_id) AS internal_max
FROM enrollments e
JOIN students s ON s.student_id = e.student_id
JOIN courses c ON c.course_id = e.course_id
LEFT JOIN attendance_records a ON a.enrollment_id = e.enrollment_id
GROUP BY e.enrollment_id;
"""


def class_dates():
    """Two classes per week, Monday and Thursday, skipping nothing (synthetic)."""
    out, d = [], SEMESTER_START
    while len(out) < CLASSES_PER_COURSE:
        if d.weekday() in (0, 3):
            out.append(d)
        d += timedelta(days=1)
    return out


def build(path=DB_PATH, seed=42, n_students=N_STUDENTS):
    rng = random.Random(seed)
    path = Path(path)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.executemany("INSERT INTO courses (code, name, credits, department) VALUES (?,?,?,?)", COURSES)

    dates = class_dates()
    used_emails = set()
    for i in range(n_students):
        sid = 20260001 + i
        first, last = rng.choice(FIRST), rng.choice(LAST)
        email = f"{first}.{last}.{sid}@northfield.edu".lower()
        used_emails.add(email)
        diligence = min(0.99, max(0.45, rng.betavariate(11, 1.3)))  # per-student habit, ~15% of enrollments end up under 75%
        ability = min(0.98, max(0.35, rng.betavariate(5, 2)))
        cgpa = round(3.8 + 6.2 * (0.6 * ability + 0.4 * diligence), 2)
        conn.execute("INSERT INTO students VALUES (?,?,?,?,?,?)",
                     (sid, f"{first} {last}", email, rng.choice(DEPARTMENTS), 5, min(cgpa, 9.95)))

        for course_id in range(1, len(COURSES) + 1):
            cur = conn.execute("INSERT INTO enrollments (student_id, course_id) VALUES (?,?)", (sid, course_id))
            eid = cur.lastrowid
            p_present = min(0.995, max(0.3, diligence + rng.uniform(-0.06, 0.06)))
            conn.executemany(
                "INSERT INTO attendance_records (enrollment_id, class_date, present) VALUES (?,?,?)",
                [(eid, d.isoformat(), 1 if rng.random() < p_present else 0) for d in dates])
            for kind, mx in (("IA1", 15), ("IA2", 15), ("Assignment", 10)):
                score = round(min(mx, max(0, mx * (0.25 + 0.75 * ability) + rng.gauss(0, mx * 0.08))) * 2) / 2
                conn.execute("INSERT INTO assessments (enrollment_id, kind, marks, max_marks) VALUES (?,?,?,?)",
                             (eid, kind, score, mx))
    conn.commit()
    conn.close()
    return path


if __name__ == "__main__":
    out = build()
    conn = sqlite3.connect(out)
    n = conn.execute("SELECT COUNT(*) FROM enrollment_summary WHERE attendance_pct < 75").fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM enrollment_summary").fetchone()[0]
    print(f"Created {out}: {N_STUDENTS} students, {total} enrollments, {n} below 75% attendance")
    sys.exit(0)
