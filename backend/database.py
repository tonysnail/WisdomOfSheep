from __future__ import annotations

import json
import shutil
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .config import DB_PATH, DB_TEMPLATE_PATH, SCHEMA_PATH

LOGGER_NAME = "backend.database"

_DB_READY_LOCK = threading.Lock()
_DB_READY_PATH: Path | None = None


def _handle_wal_files(base_path: Path, dest_base: Path | None) -> None:
    for suffix in ("-wal", "-shm"):
        candidate = base_path.with_name(f"{base_path.name}{suffix}")
        if not candidate.exists():
            continue
        try:
            if dest_base is not None:
                dest = dest_base.with_name(f"{dest_base.name}{suffix}")
                shutil.move(str(candidate), str(dest))
            else:
                candidate.unlink()
        except OSError:
            continue


def _read_schema() -> str:
    if not SCHEMA_PATH.exists():
        raise RuntimeError(f"Missing council schema at {SCHEMA_PATH}")
    return SCHEMA_PATH.read_text(encoding="utf-8")


def _initialize_db_file(target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    _handle_wal_files(target, None)
    if target.exists():
        target.unlink()

    if target == DB_PATH and DB_TEMPLATE_PATH.exists():
        shutil.copyfile(DB_TEMPLATE_PATH, target)
    else:
        schema_sql = _read_schema()
        with sqlite3.connect(str(target)) as conn:
            conn.executescript(schema_sql)
            conn.commit()


def _is_db_healthy(path: Path) -> bool:
    try:
        with sqlite3.connect(str(path)) as conn:
            row = conn.execute("PRAGMA quick_check").fetchone()
    except sqlite3.DatabaseError:
        return False
    if not row:
        return False
    return str(row[0]).strip().lower() == "ok"


def _backup_corrupt_db(path: Path) -> Path | None:
    if not path.exists():
        return None
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    backup = path.with_name(f"{path.stem}.corrupt-{timestamp}{path.suffix}")
    shutil.move(str(path), str(backup))
    _handle_wal_files(path, backup)
    return backup


def ensure_database_ready(logger=None) -> None:
    global _DB_READY_PATH
    current = Path(DB_PATH)
    if _DB_READY_PATH is not None and _DB_READY_PATH == current:
        return

    with _DB_READY_LOCK:
        if _DB_READY_PATH is not None and _DB_READY_PATH == current:
            return

        current.parent.mkdir(parents=True, exist_ok=True)
        if not current.exists():
            if logger:
                logger.warning("council database missing at %s; creating new file", current)
            _initialize_db_file(current)
        elif not _is_db_healthy(current):
            backup = _backup_corrupt_db(current)
            if logger:
                if backup is not None:
                    logger.error(
                        "council database at %s was corrupt; backed up to %s and reinitialised",
                        current,
                        backup,
                    )
                else:
                    logger.error("council database at %s was corrupt; reinitialising", current)
            _initialize_db_file(current)

        _DB_READY_PATH = current


def connect() -> sqlite3.Connection:
    ensure_database_ready()
    if not DB_PATH.exists():
        raise RuntimeError(f"DB not found: {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def query_all(conn: sqlite3.Connection, sql: str, params: Sequence[Any]) -> List[sqlite3.Row]:
    cur = conn.execute(sql, tuple(params))
    return cur.fetchall()


def query_one(conn: sqlite3.Connection, sql: str, params: Sequence[Any]) -> Optional[sqlite3.Row]:
    cur = conn.execute(sql, tuple(params))
    return cur.fetchone()


def execute(conn: sqlite3.Connection, sql: str, params: Sequence[Any]) -> None:
    conn.execute(sql, tuple(params))
    conn.commit()


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
    CREATE TABLE IF NOT EXISTS posts (
      post_id    TEXT PRIMARY KEY,
      platform   TEXT,
      source     TEXT,
      url        TEXT,
      title      TEXT,
      author     TEXT,
      scraped_at TEXT,
      posted_at  TEXT,
      score      INTEGER,
      text       TEXT
    )"""
    )
    conn.execute(
        """
    CREATE TABLE IF NOT EXISTS stages (
      post_id    TEXT NOT NULL,
      stage      TEXT NOT NULL,
      created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
      payload    TEXT NOT NULL,
      PRIMARY KEY (post_id, stage, created_at)
    )"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_stages_post_stage_created ON stages(post_id, stage, created_at)"
    )
    conn.execute(
        """
    CREATE TABLE IF NOT EXISTS post_extras (
      post_id      TEXT PRIMARY KEY,
      payload_json TEXT NOT NULL
    )"""
    )
    conn.execute(
        """
    CREATE TABLE IF NOT EXISTS council_analysis_errors (
      id           INTEGER PRIMARY KEY AUTOINCREMENT,
      post_id      TEXT NOT NULL,
      stage        TEXT,
      stage_label  TEXT,
      message      TEXT,
      log_excerpt  TEXT,
      job_id       TEXT,
      occurred_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
    )"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_council_analysis_errors_post ON council_analysis_errors(post_id)"
    )
    conn.execute(
        """
    CREATE TABLE IF NOT EXISTS council_stage_interest (
      post_id             TEXT PRIMARY KEY,
      ticker              TEXT,
      status              TEXT NOT NULL DEFAULT 'ok',
      interest_score      REAL,
      interest_label      TEXT,
      interest_why        TEXT,
      council_recommended INTEGER,
      council_priority    TEXT,
      error_code          TEXT,
      error_message       TEXT,
      created_at          TEXT,
      debug_json          TEXT
    )"""
    )
    try:
        cols = conn.execute("PRAGMA table_info(council_stage_interest)").fetchall()
        existing_cols: Set[str] = set()
        for row in cols:
            try:
                existing_cols.add(row["name"])
            except Exception:  # noqa: BLE001
                if isinstance(row, (tuple, list)) and len(row) > 1:
                    existing_cols.add(str(row[1]))
    except Exception:  # noqa: BLE001
        existing_cols = set()
    alters = [
        (
            "status",
            "ALTER TABLE council_stage_interest ADD COLUMN status TEXT DEFAULT 'ok'",
        ),
        (
            "interest_score",
            "ALTER TABLE council_stage_interest ADD COLUMN interest_score REAL",
        ),
        (
            "interest_label",
            "ALTER TABLE council_stage_interest ADD COLUMN interest_label TEXT",
        ),
        (
            "interest_why",
            "ALTER TABLE council_stage_interest ADD COLUMN interest_why TEXT",
        ),
        (
            "council_recommended",
            "ALTER TABLE council_stage_interest ADD COLUMN council_recommended INTEGER",
        ),
        (
            "council_priority",
            "ALTER TABLE council_stage_interest ADD COLUMN council_priority TEXT",
        ),
        (
            "error_code",
            "ALTER TABLE council_stage_interest ADD COLUMN error_code TEXT",
        ),
        (
            "error_message",
            "ALTER TABLE council_stage_interest ADD COLUMN error_message TEXT",
        ),
        (
            "created_at",
            "ALTER TABLE council_stage_interest ADD COLUMN created_at TEXT",
        ),
        (
            "debug_json",
            "ALTER TABLE council_stage_interest ADD COLUMN debug_json TEXT",
        ),
    ]
    for col_name, sql in alters:
        if col_name in existing_cols:
            continue
        try:
            conn.execute(sql)
        except sqlite3.OperationalError:
            continue
    conn.commit()


def upsert_post_row(conn: sqlite3.Connection, post_vals: Dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO posts (post_id, platform, source, url, title, author, scraped_at, posted_at, score, text)
        VALUES (:post_id, :platform, :source, :url, :title, :author, :scraped_at, :posted_at, :score, :text)
        ON CONFLICT(post_id) DO UPDATE SET
            platform=excluded.platform,
            source=excluded.source,
            url=excluded.url,
            title=excluded.title,
            author=excluded.author,
            scraped_at=excluded.scraped_at,
            posted_at=excluded.posted_at,
            score=excluded.score,
            text=excluded.text
    """,
        post_vals,
    )
    conn.commit()


def upsert_extras(conn: sqlite3.Connection, post_id: str, extras_vals: Dict[str, Any]) -> None:
    payload_json = json.dumps(extras_vals, ensure_ascii=False)
    conn.execute(
        """
        INSERT INTO post_extras (post_id, payload_json)
        VALUES (?, ?)
        ON CONFLICT(post_id) DO UPDATE SET payload_json=excluded.payload_json
    """,
        (post_id, payload_json),
    )
    conn.commit()


def load_extras_dict(conn: sqlite3.Connection, post_id: str) -> Dict[str, Any]:
    row = query_one(conn, "SELECT payload_json FROM post_extras WHERE post_id = ?", (post_id,))
    if not row or not row["payload_json"]:
        return {}
    try:
        data = json.loads(row["payload_json"])
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def save_extras_dict(conn: sqlite3.Connection, post_id: str, extras: Dict[str, Any]) -> None:
    upsert_extras(conn, post_id, extras)


def insert_stage_payload(
    conn: sqlite3.Connection,
    post_id: str,
    stage: str,
    payload: Dict[str, Any],
    *,
    created_at: Optional[str] = None,
) -> str:
    created = created_at or datetime.utcnow().replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
    conn.execute(
        "INSERT INTO stages (post_id, stage, created_at, payload) VALUES (?, ?, ?, ?)",
        (post_id, stage, created, json.dumps(payload, ensure_ascii=False)),
    )
    conn.commit()
    return created


def latest_stage_payload(conn: sqlite3.Connection, post_id: str, stage: str) -> Optional[Dict[str, Any]]:
    row = query_one(
        conn,
        """
        SELECT payload FROM stages
        WHERE post_id = ? AND stage = ?
        ORDER BY datetime(created_at) DESC
        LIMIT 1
    """,
        (post_id, stage),
    )
    if not row or not row["payload"]:
        return None
    try:
        return json.loads(row["payload"])
    except Exception:
        return None


def strip_research_from_extras(conn: sqlite3.Connection, post_id: Optional[str] = None) -> int:
    if post_id is not None:
        targets = [post_id]
    else:
        rows = conn.execute("SELECT post_id FROM post_extras").fetchall()
        targets = [row["post_id"] for row in rows if row and row["post_id"]]

    updated = 0
    for pid in targets:
        extras = load_extras_dict(conn, pid)
        if "research" not in extras:
            continue
        extras.pop("research", None)
        save_extras_dict(conn, pid, extras)
        updated += 1
    return updated
