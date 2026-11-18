"""SQLite access layer for the attendance system."""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "attendance.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            roll_no TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            embedding BLOB NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            roll_no TEXT NOT NULL,
            name TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            FOREIGN KEY (student_id) REFERENCES students(id),
            UNIQUE(student_id, date)
        )
    """)
    conn.commit()
    conn.close()


def add_student(roll_no, name, embedding_bytes):
    conn = get_connection()
    cur = conn.execute(
        "INSERT INTO students (roll_no, name, embedding) VALUES (?, ?, ?)",
        (roll_no, name, embedding_bytes),
    )
    conn.commit()
    student_id = cur.lastrowid
    conn.close()
    return student_id


def get_student_by_id(student_id):
    conn = get_connection()
    row = conn.execute(
        "SELECT id, roll_no, name FROM students WHERE id = ?", (student_id,)
    ).fetchone()
    conn.close()
    return row


def get_all_students():
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, roll_no, name FROM students ORDER BY id"
    ).fetchall()
    conn.close()
    return rows


def get_all_students_with_embeddings():
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, roll_no, name, embedding FROM students ORDER BY id"
    ).fetchall()
    conn.close()
    return rows


def mark_attendance(student_id, roll_no, name, date_str, time_str):
    """Inserts one attendance row per student per day. Returns False if already marked today."""
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO attendance (student_id, roll_no, name, date, time) VALUES (?, ?, ?, ?, ?)",
            (student_id, roll_no, name, date_str, time_str),
        )
        conn.commit()
        marked = True
    except sqlite3.IntegrityError:
        marked = False
    conn.close()
    return marked


def get_attendance(date_str=None):
    conn = get_connection()
    if date_str:
        rows = conn.execute(
            "SELECT roll_no, name, date, time FROM attendance WHERE date = ? ORDER BY time",
            (date_str,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT roll_no, name, date, time FROM attendance ORDER BY date DESC, time DESC"
        ).fetchall()
    conn.close()
    return rows
