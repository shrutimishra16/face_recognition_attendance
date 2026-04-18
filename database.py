import pyodbc
from datetime import datetime

def get_connection():
    conn = pyodbc.connect(
        "DRIVER={SQL Server};"
        "SERVER=SHRUTIMISHRA\SQLEXPRESS;"   
        "DATABASE=FaceRecognitionDB;"
        "Trusted_Connection=yes;"
    )
    return conn


def mark_attendance(name, bus_no):
    conn = get_connection()
    cursor = conn.cursor()

    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    cursor.execute("""
        SELECT * FROM attendance WHERE name=? AND date=?
    """, (name, date))

    result = cursor.fetchone()

    if result is None:
        cursor.execute("""
            INSERT INTO attendance (name, date, time, status, bus_no)
            VALUES (?, ?, ?, ?, ?)
        """, (name, date, time, "Present", bus_no))
        conn.commit()
        return f"{name} marked present"
    else:
        return f"{name} already marked today"

    conn.close()