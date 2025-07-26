from contextlib import contextmanager
import psycopg2
import os
from dotenv import load_dotenv


load_dotenv()
DATABASE_URL = os.environ.get("DATABASE_URL")

def get_db_connection():
    return psycopg2.connect(DATABASE_URL)

@contextmanager
def db_cursor(commit=False):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        yield cursor
        if commit:
            conn.commit()
    finally:
        cursor.close()
        conn.close()

