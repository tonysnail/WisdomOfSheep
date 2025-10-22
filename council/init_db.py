# init_db.py
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "wisdom_of_sheep.sql"
SCHEMA_PATH = ROOT / "council_schema.sql"

ROOT.mkdir(parents=True, exist_ok=True)

with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
    schema = f.read()

con = sqlite3.connect(str(DB_PATH))
con.executescript(schema)
con.commit()
con.close()

print(f"Database created at {DB_PATH}")
