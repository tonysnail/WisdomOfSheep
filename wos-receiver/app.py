"""

Oracle Server News Corpus Handler

Use to write scraped news articles from Pi, or read online from WOS Frontend.


# Save the file on the Oracle Server at /opt/wos-receiver/app.py, then restart:

sudo systemctl daemon-reload
sudo systemctl restart wos-receiver
sudo systemctl status wos-receiver --no-pager


"""

from __future__ import annotations

import os
import json
import gzip
import time
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import PlainTextResponse
import asyncio

# -------------------------
# Config & paths
# -------------------------
APP_USER = os.getenv("WOS_BASIC_USER", "")
APP_PASS = os.getenv("WOS_BASIC_PASS", "")

DATA = Path("/var/wos")
DATA.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA / "posts.sqlite"
LIVE_NDJSON = DATA / "raw_posts_log.ndjson"
MAX_BYTES = 100 * 1024 * 1024  # rotate NDJSON at ~100MB

# Single-flight writer: prevents concurrent writers
WRITE_LOCK = asyncio.Lock()

app = FastAPI(title="Wisdom Of Sheep Receiver", version="1.1")

# -------------------------
# Legacy CSV schema
# -------------------------
# This is the canonical column order / field set that the rest of WOS expects.
CSV_FIELDS = [
    "scraped_at",
    "platform",
    "source",
    "post_id",
    "url",
    "title",
    "text",
    "final_url",
    "fetch_status",
    "domain",
]

# -------------------------
# Helpers
# -------------------------
def _auth(request: Request) -> bool:
    """Basic auth if creds configured; open if not."""
    if not (APP_USER and APP_PASS):
        return True
    auth = request.headers.get("authorization") or ""
    if not auth.lower().startswith("basic "):
        return False
    import base64
    try:
        user, pwd = base64.b64decode(auth.split(" ", 1)[1]).decode().split(":", 1)
        return (user == APP_USER and pwd == APP_PASS)
    except Exception:
        return False


def _open_db_raw() -> sqlite3.Connection:
    """
    Low-level: open SQLite connection (no schema). Use this ONLY if caller
    is going to handle exceptions. This is useful for readiness checks,
    because we don't want a failing DB open to crash the whole request.
    """
    conn = sqlite3.connect(
        DB_PATH,
        timeout=30,
        isolation_level=None,  # autocommit
    )
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def _db() -> sqlite3.Connection:
    """
    Normal DB accessor. Ensures schema exists.
    Raises if SQLite truly can't open.
    """
    conn = _open_db_raw()

    # Ensure schema
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS seen(
            platform TEXT NOT NULL,
            post_id  TEXT NOT NULL,
            PRIMARY KEY(platform, post_id)
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS posts(
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            scraped_at TEXT,
            platform   TEXT,
            post_id    TEXT,
            json       TEXT,
            UNIQUE(platform, post_id)
        );
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS ix_posts_scrapedat ON posts(scraped_at);")
    return conn


def _rotate_if_needed() -> None:
    """Rotate NDJSON audit file when it exceeds MAX_BYTES."""
    if LIVE_NDJSON.exists() and LIVE_NDJSON.stat().st_size >= MAX_BYTES:
        ts = int(time.time())
        gz = DATA / f"raw_posts_log_{ts}.ndjson.gz"
        with open(LIVE_NDJSON, "rb") as fi, gzip.open(gz, "wb") as fo:
            fo.write(fi.read())
        LIVE_NDJSON.unlink(missing_ok=True)


def _append_rows_ndjson(rows: List[Dict[str, Any]]) -> None:
    """Append rows to NDJSON with fsync to ensure durability."""
    if not rows:
        return
    _rotate_if_needed()
    with open(LIVE_NDJSON, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _row_from_json(js: Dict[str, Any]) -> Dict[str, str]:
    """
    Coerce incoming row values to strings (or "") for storage.

    Critically: enforce the canonical CSV schema used by WOS ingestion.
    Anything extra in js is ignored (we don't need to persist it for compat).
    """
    out: Dict[str, str] = {}
    for field in CSV_FIELDS:
        v = js.get(field, "")
        out[field] = "" if v is None else str(v)
    return out


def _ensure_csv_shape(row: Dict[str, Any]) -> Dict[str, str]:
    """
    Take a dict we pulled back out of SQLite and guarantee it matches
    the legacy CSV schema/columns exactly, with "" defaults.

    This is what we send to the client in /wos/next-after and friends.
    """
    shaped: Dict[str, str] = {}
    for field in CSV_FIELDS:
        v = row.get(field, "")
        shaped[field] = "" if v is None else str(v)
    return shaped


def _resolve_after_ts(
    conn: sqlite3.Connection,
    platform: Optional[str],
    post_id: Optional[str],
    after_ts: Optional[str],
) -> Optional[str]:
    """Resolve 'after' timestamp from (platform, post_id) if provided."""
    if after_ts:
        return after_ts
    if platform and post_id:
        cur = conn.execute(
            "SELECT scraped_at FROM posts WHERE platform=? AND post_id=? LIMIT 1;",
            (platform, post_id),
        )
        row = cur.fetchone()
        if row and row[0]:
            return row[0]
    return None


def _try_db_ready() -> Dict[str, Any]:
    """
    Lightweight readiness probe.
    We DO NOT create tables here — we only verify:
    - DB file can be opened
    - we can run a trivial query
    - WRITE_LOCK isn't currently held by another writer
    """
    info: Dict[str, Any] = {
        "db_ok": False,
        "db_count_posts": None,
        "write_locked": WRITE_LOCK.locked(),
    }

    try:
        conn = _open_db_raw()
    except Exception as exc:
        # can't even open sqlite
        info["db_error"] = f"open-failed:{exc}"
        return info

    try:
        # Does posts table exist?
        cur = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='posts';"
        )
        exists_posts = cur.fetchone()[0] > 0

        if exists_posts:
            cur2 = conn.execute("SELECT COUNT(*) FROM posts;")
            info["db_count_posts"] = cur2.fetchone()[0]
        else:
            info["db_count_posts"] = 0

        info["db_ok"] = True
    except Exception as exc:
        info["db_error"] = f"query-failed:{exc}"
    finally:
        conn.close()

    return info


# -------------------------
# Endpoints
# -------------------------
@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    """
    Super-lightweight liveness check for systemd etc.
    Returns current NDJSON size.
    """
    sz = LIVE_NDJSON.stat().st_size if LIVE_NDJSON.exists() else 0
    return f"OK {sz} bytes\n"


@app.get("/wos/ready")
def ready(request: Request):
    """
    Readiness check for *writers* (Pi harvester) and *readers* (frontend).

    Returns:
    - ready: bool (true only if DB is reachable AND no active writer)
    - write_locked: whether WRITE_LOCK is engaged
    - db_ok / db_error
    - db_count_posts (if known)
    """
    if not _auth(request):
        raise HTTPException(401, "unauthorized")

    info = _try_db_ready()

    # We consider "ready" == DB is OK AND not currently in WRITE_LOCK.
    is_ready = (info.get("db_ok") is True) and (info.get("write_locked") is False)

    return {
        "ready": is_ready,
        "write_locked": info.get("write_locked"),
        "db_ok": info.get("db_ok"),
        "db_error": info.get("db_error"),
        "db_count_posts": info.get("db_count_posts"),
    }


@app.get("/wos/stats")
def stats(request: Request):
    """
    Returns global stats for monitoring / debugging.
    Currently:
    - total_posts
    - total_seen_keys
    - ndjson_size_bytes
    - write_locked
    """
    if not _auth(request):
        raise HTTPException(401, "unauthorized")

    write_locked = WRITE_LOCK.locked()
    ndjson_size = LIVE_NDJSON.stat().st_size if LIVE_NDJSON.exists() else 0

    try:
        conn = _db()
        cur_posts = conn.execute("SELECT COUNT(*) FROM posts;")
        total_posts = cur_posts.fetchone()[0]

        cur_seen = conn.execute("SELECT COUNT(*) FROM seen;")
        total_seen = cur_seen.fetchone()[0]
        conn.close()
    except Exception as exc:
        raise HTTPException(
            500,
            detail={"error": "db-failed", "exc": str(exc), "write_locked": write_locked},
        )

    return {
        "total_posts": total_posts,
        "total_seen_keys": total_seen,
        "ndjson_size_bytes": ndjson_size,
        "write_locked": write_locked,
    }


@app.get("/wos/exists")
def exists(platform: str, post_id: str, request: Request):
    """
    Check if a post_id for a platform has already been ingested.
    """
    if not _auth(request):
        raise HTTPException(401, "unauthorized")

    try:
        conn = _db()
    except Exception as exc:
        raise HTTPException(503, f"db-unavailable:{exc}")

    cur = conn.execute(
        "SELECT 1 FROM seen WHERE platform=? AND post_id=? LIMIT 1;",
        (platform.strip(), post_id.strip()),
    )
    ex = cur.fetchone() is not None
    conn.close()
    return {"exists": ex}


@app.post("/wos/raw-posts")
async def receive_posts(req: Request, resp: Response):
    """
    Pi harvester calls this with {"rows": [ {scraped_at,...} ] }
    Exactly one row per request.

    Behaviour:
    - refuse concurrent writers (return 409 with Retry-After)
    - insert into seen/posts
    - append to rolling NDJSON (with rotation)
    """
    if not _auth(req):
        raise HTTPException(401, "unauthorized")

    # If another writer is active, tell caller to back off.
    if WRITE_LOCK.locked():
        resp.status_code = 409
        resp.headers["Retry-After"] = "1"
        return {"status": "busy", "ready": False}

    try:
        payload = await req.json()
    except Exception:
        raise HTTPException(400, "invalid json")

    rows = payload.get("rows") or []
    if not isinstance(rows, list):
        raise HTTPException(400, "rows must be a list")

    # Enforce exactly one row per write request. This flattens memory usage
    # and makes backpressure + retry logic predictable.
    if len(rows) != 1:
        raise HTTPException(400, "exactly one row per request is required")

    async with WRITE_LOCK:
        # during this 'with', /wos/ready will report write_locked=True
        try:
            conn = _db()
        except Exception as exc:
            # if we can't even open/init DB, that's a server failure
            raise HTTPException(503, f"db-unavailable:{exc}")

        # keep WAL from growing forever during heavy ingest
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        cur = conn.cursor()

        r0 = _row_from_json(rows[0])
        plat = r0.get("platform", "").strip()
        pid = r0.get("post_id", "").strip()
        ts = r0.get("scraped_at", "")

        dups = 0
        accepted = 0

        if plat and pid:
            cur.execute(
                "INSERT OR IGNORE INTO seen(platform, post_id) VALUES(?,?)",
                (plat, pid),
            )
            if cur.rowcount == 0:
                dups = 1

        if dups == 0:
            js = json.dumps(r0, ensure_ascii=False)
            cur.execute(
                """
                INSERT OR IGNORE INTO posts(scraped_at, platform, post_id, json)
                VALUES(?,?,?,?)
                """,
                (ts, plat, pid, js),
            )
            if cur.rowcount == 1:
                accepted = 1
                _append_rows_ndjson([r0])
            else:
                dups = 1

        conn.commit()
        conn.close()

    # lock is released here: safe to accept next write
    return {
        "status": "ok",
        "accepted": accepted,
        "duplicates": dups,
        "ready": True,
    }


@app.get("/wos/next-after")
def next_after(
    request: Request,
    platform: Optional[str] = None,
    post_id: Optional[str] = None,
    after_ts: Optional[str] = None,
    limit: int = 1,
):
    """
    Fetch the next (or next N) rows sorted by scraped_at after the given reference.

    Reference can be (platform, post_id) OR an explicit after_ts string.

    Returns JSON:
    {
        "items": [ {scraped_at:..., platform:..., ..., domain:...}, ... ],
        "count": N,
        "after_ts": "..."   # the timestamp we used as our lower bound
    }

    Every item matches the legacy CSV schema exactly, so downstream
    code can treat each dict like a CSV line.
    """
    if not _auth(request):
        raise HTTPException(401, "unauthorized")

    limit = max(1, min(int(limit), 500))

    try:
        conn = _db()
    except Exception as exc:
        raise HTTPException(503, f"db-unavailable:{exc}")

    ref_ts = _resolve_after_ts(conn, platform, post_id, after_ts)

    if ref_ts:
        cur = conn.execute(
            """
            SELECT json
            FROM posts
            WHERE scraped_at > ?
            ORDER BY scraped_at ASC, id ASC
            LIMIT ?;
            """,
            (ref_ts, limit),
        )
    else:
        # If no reference, return the earliest rows in the DB
        cur = conn.execute(
            """
            SELECT json
            FROM posts
            ORDER BY scraped_at ASC, id ASC
            LIMIT ?;
            """,
            (limit,),
        )

    items: List[Dict[str, str]] = []
    for (js,) in cur.fetchall():
        try:
            raw_row = json.loads(js)
        except Exception:
            # corrupted row: skip
            continue

        # force CSV-compatible schema/keys/defaults
        shaped = _ensure_csv_shape(raw_row)
        items.append(shaped)

    conn.close()
    return {
        "items": items,
        "count": len(items),
        "after_ts": ref_ts,
    }
