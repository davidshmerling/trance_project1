from db import db_cursor  # מניח שזה הקובץ שלך


def reset_tracks_table():
    create_sql = """
    DROP TABLE IF EXISTS tracks;

    CREATE TABLE tracks (
    id SERIAL PRIMARY KEY,
    link TEXT NOT NULL UNIQUE,
    title TEXT,  -- שם השיר
    goa FLOAT DEFAULT 0,
    retro_goa FLOAT DEFAULT 0,
    full_on FLOAT DEFAULT 0,
    hitech FLOAT DEFAULT 0,
    psy FLOAT DEFAULT 0,
    darkpsy FLOAT DEFAULT 0,
    voters_count INT DEFAULT 1
    );

    """

    with db_cursor(commit=True) as cur:
        cur.execute(create_sql)
        print("✅ טבלת 'tracks' אופסה ונוצרה מחדש בהצלחה!")


if __name__ == "__main__":
    reset_tracks_table()

