import sqlite3

DB_NAME = "smartlabeler.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_path TEXT,
        label TEXT,
        labeled INTEGER
    )
    """)

    conn.commit()
    conn.close()