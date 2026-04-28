import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "attendance.db")

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            date TEXT,
            time TEXT,
            status TEXT,
            bus_no TEXT
        )
    """)
    conn.commit()
    conn.close()

def mark_attendance(name, bus_no):
    conn = get_connection()
    cursor = conn.cursor()

    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    cursor.execute("SELECT * FROM attendance WHERE name=? AND date=?", (name, date))
    result = cursor.fetchone()

    if result is None:
        cursor.execute(
            "INSERT INTO attendance (name, date, time, status, bus_no) VALUES (?, ?, ?, ?, ?)",
            (name, date, time, "Present", bus_no)
        )
        conn.commit()
        msg = f"{name} marked present"
    else:
        msg = f"{name} already marked today"

    conn.close()
    return msg

init_db()
