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
from fastapi.responses import PlainTextResponse, JSONResponse
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
MAX_BYTES = 100 * 1024 * 1024  # rotate NDJSON at ~100MB (adjust if you like)

# Single-flight writer
WRITE_LOCK = asyncio.Lock()

app = FastAPI(title="Wisdom Of Sheep Receiver", version="1.0")


# -------------------------
# Helpers
# -------------------------
def _auth(request: Request) -> bool:
    """Basic Auth if creds configured; open if not."""
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


def _db() -> sqlite3.Connection:
    """Open SQLite and ensure schema exists."""
    conn = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)  # autocommit mode
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA foreign_keys=ON;")

    # Schema
    conn.execute("""
    CREATE TABLE IF NOT EXISTS seen(
        platform TEXT NOT NULL,
        post_id  TEXT NOT NULL,
        PRIMARY KEY(platform, post_id)
    );
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS posts(
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        scraped_at TEXT,
        platform   TEXT,
        post_id    TEXT,
        json       TEXT,
        UNIQUE(platform, post_id)
    );
    """)
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
    """Coerce incoming row values to strings (or empty) for storage."""
    out: Dict[str, str] = {}
    for k, v in js.items():
        out[k] = "" if v is None else str(v)
    for key in ("scraped_at", "platform", "source", "post_id", "url", "title", "text"):
        out.setdefault(key, "")
    return out


def _resolve_after_ts(conn: sqlite3.Connection,
                      platform: Optional[str],
                      post_id: Optional[str],
                      after_ts: Optional[str]) -> Optional[str]:
    """Resolve 'after' timestamp from (platform, post_id) if provided."""
    if after_ts:
        return after_ts
    if platform and post_id:
        cur = conn.execute(
            "SELECT scraped_at FROM posts WHERE platform=? AND post_id=? LIMIT 1;",
            (platform, post_id)
        )
        row = cur.fetchone()
        if row and row[0]:
            return row[0]
    return None


# -------------------------
# Endpoints
# -------------------------
@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    sz = LIVE_NDJSON.stat().st_size if LIVE_NDJSON.exists() else 0
    return f"OK {sz} bytes\n"


@app.get("/wos/ready")
def ready(request: Request):
    if not _auth(request):
        raise HTTPException(401, "unauthorized")
    return {"ready": (not WRITE_LOCK.locked())}


@app.get("/wos/exists")
def exists(platform: str, post_id: str, request: Request):
    if not _auth(request):
        raise HTTPException(401, "unauthorized")
    conn = _db()
    cur = conn.execute(
        "SELECT 1 FROM seen WHERE platform=? AND post_id=? LIMIT 1;",
        (platform.strip(), post_id.strip())
    )
    ex = cur.fetchone() is not None
    conn.close()
    return {"exists": ex}


@app.post("/wos/raw-posts")
async def receive_posts(req: Request, resp: Response):
    if not _auth(req):
        raise HTTPException(401, "unauthorized")

    # If another writer is active, signal busy so client waits/retries.
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

    # Strict: exactly 1 row per request (keeps memory/IO flat even for very large 'text')
    if len(rows) != 1:
        raise HTTPException(400, "exactly one row per request is required")

    async with WRITE_LOCK:
        conn = _db()
        # keep WAL tidy during sustained writes
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        cur = conn.cursor()

        r0 = _row_from_json(rows[0])
        plat = r0.get("platform", "").strip()
        pid  = r0.get("post_id", "").strip()
        ts   = r0.get("scraped_at", "")

        dups = 0
        accepted = 0

        if plat and pid:
            cur.execute("INSERT OR IGNORE INTO seen(platform, post_id) VALUES(?,?)", (plat, pid))
            if cur.rowcount == 0:
                dups = 1

        if dups == 0:
            js = json.dumps(r0, ensure_ascii=False)
            cur.execute(
                "INSERT OR IGNORE INTO posts(scraped_at, platform, post_id, json) VALUES(?,?,?,?)",
                (ts, plat, pid, js)
            )
            if cur.rowcount == 1:
                accepted = 1
                _append_rows_ndjson([r0])
            else:
                dups = 1

        conn.commit()
        conn.close()

    # leaving the lock means the write is truly finished; OK to accept next post
    return {"status": "ok", "accepted": accepted, "duplicates": dups, "ready": True}


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
    Reference can be (platform, post_id) or an explicit after_ts (ISO8601).
    """
    if not _auth(request):
        raise HTTPException(401, "unauthorized")

    limit = max(1, min(int(limit), 500))
    conn = _db()

    ref_ts = _resolve_after_ts(conn, platform, post_id, after_ts)
    if ref_ts:
        cur = conn.execute(
            "SELECT json FROM posts WHERE scraped_at > ? ORDER BY scraped_at ASC, id ASC LIMIT ?;",
            (ref_ts, limit)
        )
    else:
        # If no reference, just return earliest rows
        cur = conn.execute(
            "SELECT json FROM posts ORDER BY scraped_at ASC, id ASC LIMIT ?;",
            (limit,)
        )

    items = []
    for (js,) in cur.fetchall():
        try:
            items.append(json.loads(js))
        except Exception:
            pass

    conn.close()
    return {"items": items, "count": len(items), "after_ts": ref_ts}
