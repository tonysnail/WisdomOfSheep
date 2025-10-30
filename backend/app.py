# backend/app.py

#  python -m uvicorn app:app --reload --port 8000

from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import sys
import sqlite3
import csv
import math
import re
import random
import shutil
import threading
import time
import uuid

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Callable, Literal

from urllib.parse import urljoin


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

COUNCIL_FAILURES_PATH = ROOT / "council_failures.json"
COUNCIL_FAILURES_LOCK = threading.Lock()

from fastapi import Body, FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.encoders import jsonable_encoder

import requests

from backend.council_time import CouncilTimeModel, approximate_token_count


# round_table.py pulls in heavy optional dependencies (pandas, etc.). During
# unit tests we don't need the real implementation, so we tolerate import
# failures and allow tests to monkeypatch a fake runner instead.
try:  # pragma: no cover - exercised indirectly in tests
    from round_table import run_stages_for_post  # type: ignore[attr-defined]
    ROUND_TABLE_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001
    run_stages_for_post = None  # type: ignore[assignment]
    ROUND_TABLE_IMPORT_ERROR = exc

try:  # pragma: no cover - exercised indirectly in tests
    from stock_window import get_stock_window
    STOCK_WINDOW_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001
    get_stock_window = None  # type: ignore[assignment]
    STOCK_WINDOW_IMPORT_ERROR = exc

try:  # pragma: no cover - optional dependency for conversation hub ingestion
    from ticker_conversation_hub import ConversationHub, SQLiteStore, compute_ticker_signal
    CONVO_HUB_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001
    ConversationHub = None  # type: ignore[assignment]
    SQLiteStore = None  # type: ignore[assignment]
    CONVO_HUB_IMPORT_ERROR = exc

try:  # pragma: no cover - optional interest score computation
    from council.interest_score import compute_interest_for_post
    INTEREST_SCORE_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001
    compute_interest_for_post = None  # type: ignore[assignment]
    INTEREST_SCORE_IMPORT_ERROR = exc


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float | None) -> Optional[float]:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return value


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() not in {"", "0", "false", "no"}


# =========================
# Config (env-overridable)
# =========================
LOGGER = logging.getLogger(__name__)
DB_PATH = ROOT / "council" / "wisdom_of_sheep.sql"
DB_TEMPLATE_PATH = ROOT / "council" / "wisdom_of_sheep-empty.sql"
SCHEMA_PATH = ROOT / "council" / "council_schema.sql"
TIME_MODEL_PATH = ROOT / "council" / "council_time_model.json"
CSV_PATH = ROOT / "raw_posts_log.csv"
TICKERS_DIR = ROOT / "tickers"  # <- your new folder


DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

ORACLE_CURSOR_PATH = DATA_DIR / "oracle_cursor.json"
ORACLE_RETRY_STATE_PATH = DATA_DIR / "oracle_retry_state.json"
ORACLE_SKIPPED_PATH = DATA_DIR / "skipped_articles.json"
ORACLE_UNSUMMARISED_PATH = DATA_DIR / "oracle_unsummarised.json"
ORACLE_DEFAULT_BASE_URL = (os.getenv("WOS_ORACLE_BASE_URL", "") or "").strip()
ORACLE_USER = os.getenv("WOS_ORACLE_USER", "")
ORACLE_PASS = os.getenv("WOS_ORACLE_PASS", "")
ORACLE_BATCH_SIZE = max(1, _env_int("WOS_ORACLE_BATCH_SIZE", 32))
_ORACLE_BATCH_SLEEP_RAW = _env_float("WOS_ORACLE_BATCH_SLEEP", 0.2)
ORACLE_BATCH_SLEEP = 0.0 if _ORACLE_BATCH_SLEEP_RAW is None else max(0.0, _ORACLE_BATCH_SLEEP_RAW)
_ORACLE_POLL_BASE_RAW = _env_float("WOS_ORACLE_POLL_BASE_SECONDS", 10.0)
ORACLE_POLL_BASE = 10.0 if _ORACLE_POLL_BASE_RAW is None else max(1.0, _ORACLE_POLL_BASE_RAW)
_ORACLE_POLL_MAX_RAW = _env_float("WOS_ORACLE_POLL_MAX_SECONDS", 60.0)
ORACLE_POLL_MAX = 60.0 if _ORACLE_POLL_MAX_RAW is None else max(ORACLE_POLL_BASE, _ORACLE_POLL_MAX_RAW)
_ORACLE_BACKOFF_MIN_RAW = _env_float("WOS_ORACLE_BACKOFF_MIN_SECONDS", 2.0)
ORACLE_BACKOFF_MIN = 2.0 if _ORACLE_BACKOFF_MIN_RAW is None else max(0.5, _ORACLE_BACKOFF_MIN_RAW)
_ORACLE_BACKOFF_MAX_RAW = _env_float("WOS_ORACLE_BACKOFF_MAX_SECONDS", 30.0)
ORACLE_BACKOFF_MAX = 30.0 if _ORACLE_BACKOFF_MAX_RAW is None else max(ORACLE_BACKOFF_MIN, _ORACLE_BACKOFF_MAX_RAW)
ORACLE_REQUEST_TIMEOUT = _env_float("WOS_ORACLE_TIMEOUT_SECONDS", 30.0)
ORACLE_MAX_RETRIES = max(0, _env_int("WOS_ORACLE_MAX_RETRIES", 3))


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


_DB_READY_LOCK = threading.Lock()
_DB_READY_PATH: Path | None = None


def _ensure_database_ready() -> None:
    global _DB_READY_PATH
    current = Path(DB_PATH)
    if _DB_READY_PATH is not None and _DB_READY_PATH == current:
        return

    with _DB_READY_LOCK:
        if _DB_READY_PATH is not None and _DB_READY_PATH == current:
            return

        current.parent.mkdir(parents=True, exist_ok=True)
        if not current.exists():
            LOGGER.warning("council database missing at %s; creating new file", current)
            _initialize_db_file(current)
        elif not _is_db_healthy(current):
            backup = _backup_corrupt_db(current)
            if backup is not None:
                LOGGER.error("council database at %s was corrupt; backed up to %s and reinitialised", current, backup)
            else:
                LOGGER.error("council database at %s was corrupt; reinitialising", current)
            _initialize_db_file(current)

        _DB_READY_PATH = current

PAGE_SIZE_DEFAULT = _env_int("WOS_PAGE_SIZE", 50)
BATCH_LIMIT = _env_int("WOS_BATCH_LIMIT", 200)
STDIO_TRIM = _env_int("WOS_STDIO_TRIM", 100_000)  # 100KB

ROUND_TABLE_MODEL = os.getenv("WOS_MODEL_NAME", "mistral")
ROUND_TABLE_HOST = os.getenv("WOS_MODEL_HOST", "http://localhost:11434")
ROUND_TABLE_TIMEOUT_RAW = _env_float("WOS_MODEL_TIMEOUT_SECS", None)
ROUND_TABLE_TIMEOUT = (
    None if ROUND_TABLE_TIMEOUT_RAW is None or ROUND_TABLE_TIMEOUT_RAW <= 0 else ROUND_TABLE_TIMEOUT_RAW
)
ROUND_TABLE_VERBOSE = _env_flag("WOS_RUNNER_VERBOSE", "1")
ROUND_TABLE_AUTOFILL = _env_flag("WOS_RUNNER_AUTOFILL", "1")
ROUND_TABLE_PRETTY = _env_flag("WOS_RUNNER_PRETTY", "0")
ROUND_TABLE_DUMP_DIR = os.getenv("WOS_RUNNER_DUMP_DIR") or None
ROUND_TABLE_EVIDENCE_LOOKBACK = _env_int("WOS_EVIDENCE_LOOKBACK_DAYS", 120)
ROUND_TABLE_MAX_EVIDENCE = _env_int("WOS_MAX_EVIDENCE_PER_CLAIM", 3)

CONVO_STORE_PATH = ROOT / "convos" / "conversations.sqlite"
CONVO_MODEL = os.getenv("WOS_CONVO_MODEL", ROUND_TABLE_MODEL)

CONVO_HUB_LOCK = threading.Lock()
CONVO_HUB_INSTANCE: Optional["ConversationHub"] = None
_TICKER_UNIVERSE: Optional[Set[str]] = None

COUNCIL_TIME_MODEL = CouncilTimeModel(TIME_MODEL_PATH)


# ───────────────────────── Technical / Sentiment tool palettes ─────────────────────────
# We’ll use these to (a) strip sentiment-only steps from the technical plan and
# (b) avoid throwing “unknown_tool” inside the technical executor.
TECH_TOOLS: Set[str] = {
    "price_window",
    "compute_indicators",
    "trend_strength",
    "volatility_state",
    "support_resistance_check",
    "bollinger_breakout_scan",
    "obv_trend",
    "mfi_flow",
}
SENTIMENT_TOOLS: Set[str] = {"news_hub_score", "news_hub_ask_as_of"}

# ================
# Job persistence
# ================
JOBS_DIR = ROOT / ".jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

ACTIVE_JOB_ID: Optional[str] = None
JOB_INDEX_LOCK = threading.Lock()

ACTIVE_COUNCIL_JOB_ID: Optional[str] = None
COUNCIL_JOB_LOCK = threading.Lock()

COUNCIL_STAGE_LABELS: Dict[str, str] = {
    "entity": "Entity Council",
    "claims": "Claims Council",
    "context": "Context Council",
    "verifier": "Verifier Council",
    "for": "Bull Council",
    "against": "Bear Council",
    "direction": "Direction Council",
    "research": "Researcher",
    "researcher": "Researcher",
    "chairman": "Chairman Verdict",
}

COUNCIL_ANALYSIS_SEQUENCE: Tuple[str, ...] = (
    "entity",
    "research",
    "claims",
    "context",
    "verifier",
    "for",
    "against",
    "direction",
    "chairman",
)

COUNCIL_PRESERVE_STAGES: Tuple[str, ...] = ("summariser", "conversation_hub")

JOB_LOG_KEEP = int(os.getenv("WOS_JOB_LOG_KEEP", "200"))


def _council_stage_label(stage: str) -> str:
    label = COUNCIL_STAGE_LABELS.get(stage)
    if label:
        return label
    safe = stage.replace("_", " ").strip()
    return safe.title() if safe else stage

# =========================
# App + CORS
# =========================
app = FastAPI(title="Wisdom Of Sheep Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("WOS_CORS_ORIGINS", "http://localhost:5173,http://localhost:5174").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup_init() -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_database_ready()
    with _connect() as conn:
        _ensure_schema(conn)

# =========================
# DB helpers
# =========================
def _connect() -> sqlite3.Connection:
    _ensure_database_ready()
    if not DB_PATH.exists():
        raise RuntimeError(f"DB not found: {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _q_all(conn: sqlite3.Connection, sql: str, params: Tuple[Any, ...]) -> List[sqlite3.Row]:
    cur = conn.execute(sql, params)
    return cur.fetchall()

def _q_one(conn: sqlite3.Connection, sql: str, params: Tuple[Any, ...]) -> Optional[sqlite3.Row]:
    cur = conn.execute(sql, params)
    return cur.fetchone()

def _exec(conn: sqlite3.Connection, sql: str, params: Tuple[Any, ...]) -> None:
    conn.execute(sql, params)
    conn.commit()


def _norm_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        text = str(value).strip()
    except Exception:  # noqa: BLE001
        return None
    return text or None

def _upsert_post_row(conn: sqlite3.Connection, post_vals: Dict[str, Any]) -> None:
    conn.execute("""
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
    """, post_vals)
    conn.commit()

def _upsert_extras(conn: sqlite3.Connection, post_id: str, extras_vals: Dict[str, Any]) -> None:
    payload_json = json.dumps(extras_vals, ensure_ascii=False)
    conn.execute("""
        INSERT INTO post_extras (post_id, payload_json)
        VALUES (?, ?)
        ON CONFLICT(post_id) DO UPDATE SET payload_json=excluded.payload_json
    """, (post_id, payload_json))
    conn.commit()

def _ensure_schema(conn: sqlite3.Connection):
    # posts: canonical article fields
    conn.execute("""
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
    )""")
    # stages: immutable history of stage outputs
    conn.execute("""
    CREATE TABLE IF NOT EXISTS stages (
      post_id    TEXT NOT NULL,
      stage      TEXT NOT NULL,
      created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
      payload    TEXT NOT NULL,
      PRIMARY KEY (post_id, stage, created_at)
    )""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_stages_post_stage_created ON stages(post_id, stage, created_at)")
    # post_extras: aux JSON per post
    conn.execute("""
    CREATE TABLE IF NOT EXISTS post_extras (
      post_id      TEXT PRIMARY KEY,
      payload_json TEXT NOT NULL
    )""")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS council_analysis_errors (
      id           INTEGER PRIMARY KEY AUTOINCREMENT,
      post_id      TEXT NOT NULL,
      stage        TEXT,
      stage_label  TEXT,
      message      TEXT,
      log_excerpt  TEXT,
      job_id       TEXT,
      occurred_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
    )""")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_council_analysis_errors_post ON council_analysis_errors(post_id)"
    )
    conn.execute("""
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
    )""")
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
    
# =========================
# Schemas
# =========================


class InterestRecord(BaseModel):
    status: str = "pending"
    ticker: Optional[str] = None
    interest_score: Optional[float] = None
    interest_label: Optional[str] = None
    interest_why: Optional[str] = None
    council_recommended: Optional[bool] = None
    council_priority: Optional[str] = None
    calculated_at: Optional[str] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)


class PostListItem(BaseModel):
    post_id: str
    title: str
    platform: str
    source: str
    url: str
    scraped_at: Optional[str] = None
    posted_at: Optional[str] = None
    preview: str
    markets: List[str] = Field(default_factory=list)
    signal: Dict[str, Any] = Field(default_factory=dict)
    has_summary: bool = False
    has_analysis: bool = False
    summary_bullets: List[str] = Field(default_factory=list)
    assets_mentioned: List[Dict[str, Optional[str]]] = Field(default_factory=list)
    spam_likelihood_pct: int = 0
    spam_why: str = ""
    interest: Optional["InterestRecord"] = None
    chairman_plain_english: Optional[str] = None
    chairman_direction: Optional[str] = None


class CalendarDay(BaseModel):
    date: str
    count: int
    analysed_count: int = 0


class PostsCalendarResponse(BaseModel):
    days: List[CalendarDay] = Field(default_factory=list)


class PostRow(BaseModel):
    post_id: str
    platform: Optional[str] = None
    source: Optional[str] = None
    url: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None
    scraped_at: Optional[str] = None
    posted_at: Optional[str] = None
    score: Optional[int] = None
    text: Optional[str] = None

class PostDetail(BaseModel):
    post: PostRow
    extras: Dict[str, Any] = Field(default_factory=dict)
    stages: Dict[str, Any] = Field(default_factory=dict)
    interest: Optional[InterestRecord] = None

class RunStageRequest(BaseModel):
    post_id: str
    stages: List[str]
    overwrite: bool
    refresh_from_csv: bool = False
    echo_post: bool = False

class RunStageResponse(BaseModel):
    ok: bool
    post_id: str
    stages_run: List[str]
    stdout: str = Field(default="")
    stderr: str = Field(default="")


def _parse_interest_debug(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="ignore")
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            data = json.loads(text)
        except Exception:  # noqa: BLE001
            return {}
        return data if isinstance(data, dict) else {}
    if isinstance(raw, dict):
        return raw
    return {}


def _build_interest_record(
    *,
    status: Any,
    ticker: Any,
    score: Any,
    label: Any,
    why: Any,
    recommended: Any,
    priority: Any,
    calculated_at: Any,
    error_code: Any,
    error_message: Any,
    debug_json: Any,
) -> Optional[InterestRecord]:
    status_raw = _norm_optional_str(status)
    score_val: Optional[float]
    try:
        score_val = float(score) if score is not None else None
    except Exception:  # noqa: BLE001
        score_val = None

    ticker_str = _norm_optional_str(ticker)
    label_str = _norm_optional_str(label)
    why_str = _norm_optional_str(why)
    priority_str = _norm_optional_str(priority)
    created_str = _norm_optional_str(calculated_at)
    error_code_str = _norm_optional_str(error_code)
    error_message_str = _norm_optional_str(error_message)

    recommended_bool: Optional[bool]
    if recommended is None:
        recommended_bool = None
    else:
        raw = str(recommended).strip()
        if raw == "" or raw.lower() == "none":
            recommended_bool = None
        else:
            try:
                recommended_bool = bool(int(float(raw)))
            except Exception:  # noqa: BLE001
                recommended_bool = bool(recommended)

    if (
        status_raw is None
        and score_val is None
        and error_code_str is None
        and error_message_str is None
    ):
        return None

    debug_dict = _parse_interest_debug(debug_json)
    metrics: Dict[str, Any] = {}
    if isinstance(debug_dict, dict):
        metrics_payload = debug_dict.get("metrics")
        if isinstance(metrics_payload, dict):
            metrics = dict(metrics_payload)
        else:
            metrics = {}
        for key in ("spam_pct", "platform", "source", "posted_at"):
            if key in debug_dict and key not in metrics:
                metrics[key] = debug_dict[key]

    status_clean = status_raw or ("ok" if score_val is not None else "pending")

    return InterestRecord(
        status=status_clean,
        ticker=ticker_str,
        interest_score=score_val,
        interest_label=label_str,
        interest_why=why_str,
        council_recommended=recommended_bool,
        council_priority=priority_str,
        calculated_at=created_str,
        error_code=error_code_str,
        error_message=error_message_str,
        metrics=metrics,
    )


class ResearchTickerPayload(BaseModel):
    ticker: str
    article_time: str
    hypotheses: List[Dict[str, Any]] = Field(default_factory=list)
    rationale: str = ""
    plan: Dict[str, Any] = Field(default_factory=dict)
    technical: Dict[str, Any] = Field(default_factory=dict)
    sentiment: Dict[str, Any] = Field(default_factory=dict)
    summary_text: str = ""
    updated_at: str
    session_id: Optional[str] = None
    log: str = ""


class ResearchPayload(BaseModel):
    article_time: str
    updated_at: str
    ordered_tickers: List[str] = Field(default_factory=list)
    tickers: Dict[str, ResearchTickerPayload] = Field(default_factory=dict)


class ResearchResponse(BaseModel):
    ok: bool
    research: ResearchPayload


@app.get("/api/stocks/window")
def api_get_stock_window(
    ticker: str = Query(..., description="Ticker symbol to fetch, e.g., AAPL"),
    center: Optional[str] = Query(None, description="Center timestamp (ISO8601)"),
    before: Optional[str] = Query(None, description="Span before the center, e.g., 3d, 6h"),
    after: Optional[str] = Query(None, description="Span after the center, e.g., 3d, 6h"),
    interval: Optional[str] = Query(None, description="Explicit Yahoo Finance interval, e.g., 1m, 1d"),
):
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker is required")

    if not callable(get_stock_window):
        hint = f": {STOCK_WINDOW_IMPORT_ERROR}" if STOCK_WINDOW_IMPORT_ERROR else ""
        raise HTTPException(status_code=503, detail=f"stock-window-unavailable{hint}")

    try:
        window = get_stock_window(
            ticker=ticker,
            center=center,
            before=before,
            after=after,
            interval=interval,
            include_data=True,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - pass through unexpected errors
        raise HTTPException(status_code=502, detail=f"Failed to fetch stock window: {exc}") from exc

    return jsonable_encoder(asdict(window))

class BatchRunFilter(BaseModel):
    query: Optional[str] = None
    platform: Optional[str] = None
    source: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None

class BatchRunRequest(BaseModel):
    stages: List[str]
    overwrite: bool
    post_ids: Optional[List[str]] = None
    filter: Optional[BatchRunFilter] = None
    refresh_from_csv: bool = False

class BatchRunResponse(BaseModel):
    ok: bool
    submitted: int
    results: List[RunStageResponse]

# =========================
# Utilities
# =========================
def _build_markets_from_entity(payload: Any) -> List[str]:
    if not payload:
        return []
    try:
        assets = payload.get("assets") or []
        vals = []
        for a in assets:
            t = (a.get("ticker") or "").strip()
            if t:
                vals.append(t)
        # de-dup preserving order
        seen = set()
        uniq = []
        for x in vals:
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        return uniq[:8]  # short list for badges
    except Exception:
        return []

def _build_signal_from_dir_or_mod(payload: Any) -> Dict[str, Any]:
    if not payload:
        return {}
    out = {}
    # common fields we’ve used historically
    for k in ("bias", "direction", "stance", "verdict", "confidence"):
        if k in payload:
            out[k] = payload[k]
    # normalize a tiny view
    if "bias" not in out and "direction" in out:
        out["bias"] = out["direction"]
    return out


_SPAM_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_CASHTAG_RE = re.compile(r"\$[A-Z]{1,5}\b")


def _parse_spam_likelihood(value: Any) -> int:
    """Return a clamped 0-100 integer spam likelihood from arbitrary inputs."""

    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return 0
        pct = int(float(value))
    elif isinstance(value, str):
        match = _SPAM_NUMBER_RE.search(value)
        if not match:
            return 0
        try:
            pct = int(float(match.group(0)))
        except (TypeError, ValueError):
            return 0
    else:
        return 0

    return max(0, min(100, pct))


def _parse_spam_reason(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                parts.append(text)
        return "; ".join(parts)
    return ""


def _get_conversation_hub() -> Optional["ConversationHub"]:
    if not callable(ConversationHub) or SQLiteStore is None:
        return None
    global CONVO_HUB_INSTANCE
    with CONVO_HUB_LOCK:
        if CONVO_HUB_INSTANCE is None:
            try:
                store = SQLiteStore(str(CONVO_STORE_PATH))
                CONVO_HUB_INSTANCE = ConversationHub(store=store, model=CONVO_MODEL)
            except Exception as exc:  # noqa: BLE001
                logging.exception("conversation hub init failed: %s", exc)
                CONVO_HUB_INSTANCE = None
        return CONVO_HUB_INSTANCE


def _load_ticker_universe() -> Set[str]:
    global _TICKER_UNIVERSE
    if _TICKER_UNIVERSE is not None:
        return _TICKER_UNIVERSE

    tickers_path = TICKERS_DIR / "tickers_enriched.csv"
    universe: Set[str] = set()
    if tickers_path.exists():
        try:
            with tickers_path.open("r", encoding="utf-8", newline="") as fh:
                reader = csv.reader(fh)
                header = next(reader, [])
                ticker_idx = 0
                for idx, name in enumerate(header):
                    label = (name or "").strip().lower()
                    if label in {"ticker", "symbol", "ticker_symbol"}:
                        ticker_idx = idx
                        break
                for row in reader:
                    if not row:
                        continue
                    try:
                        raw_symbol = row[ticker_idx]
                    except IndexError:
                        continue
                    symbol = (raw_symbol or "").strip().upper()
                    if symbol:
                        universe.add(symbol)
        except Exception as exc:  # noqa: BLE001
            logging.warning("Failed to load ticker universe from %s: %s", tickers_path, exc)

    _TICKER_UNIVERSE = universe
    return universe


def _filter_valid_tickers(candidates: Sequence[str]) -> List[str]:
    universe = _load_ticker_universe()
    seen: Set[str] = set()
    filtered: List[str] = []
    for raw in candidates:
        symbol = (raw or "").strip().upper()
        if not symbol or symbol in seen:
            continue
        if universe and symbol not in universe:
            continue
        seen.add(symbol)
        filtered.append(symbol)
    return filtered


def _fetch_recent_deltas(store: Any, ticker: str, as_of_iso: str, limit: int = 5) -> List[Dict[str, Any]]:
    try:
        records = store.records_before(ticker, as_of_iso, limit=4000)
    except Exception:  # noqa: BLE001
        return []

    norm = (ticker or "").strip().upper()
    deltas: List[Dict[str, Any]] = []
    for rec in records:
        if not isinstance(rec, dict) or rec.get("type") != "delta":
            continue
        data = rec.get("data") or {}
        who = data.get("who") or []
        if norm not in {str(w).strip().upper() for w in who if isinstance(w, str)}:
            continue
        entry = {
            "t": data.get("t"),
            "sum": data.get("sum"),
            "dir": data.get("dir"),
            "impact": data.get("impact"),
            "why": data.get("why"),
            "chan": data.get("chan"),
            "cat": data.get("cat"),
            "src": data.get("src"),
            "url": data.get("url"),
        }
        deltas.append(entry)

    deltas.sort(key=lambda item: (item.get("t") or ""))
    return deltas[-limit:]


def _extract_convo_tickers(post_row: sqlite3.Row, summariser: Dict[str, Any]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []

    def _add(raw: Any) -> None:
        if not raw:
            return
        value = str(raw).strip().upper()
        if not value:
            return
        if value not in seen:
            seen.add(value)
            ordered.append(value)

    primary = summariser.get("primary_ticker") or summariser.get("primaryTicker")
    if isinstance(primary, str):
        _add(primary)

    assets = summariser.get("assets_mentioned") or summariser.get("assets") or []
    if isinstance(assets, list):
        for asset in assets:
            if isinstance(asset, dict):
                _add(asset.get("ticker"))

    if isinstance(post_row, dict):
        title = str(post_row.get("title") or "")
        body = str(post_row.get("text") or "")
    else:
        title = str(post_row["title"] or "")
        body = str(post_row["text"] or "")

    text_blobs = [title, body]
    for blob in text_blobs:
        if not blob:
            continue
        for match in _CASHTAG_RE.findall(blob.upper()):
            _add(match[1:])

    return ordered[:6]


def _ingest_conversation_hub(conn: sqlite3.Connection, post_id: str) -> Optional[Dict[str, Any]]:
    hub = _get_conversation_hub()
    if not hub:
        return None

    post_row = _q_one(
        conn,
        "SELECT post_id, platform, source, url, title, text, scraped_at, posted_at FROM posts WHERE post_id = ?",
        (post_id,),
    )
    if not post_row:
        return None

    summariser = _latest_stage_payload(conn, post_id, "summariser")
    if not summariser:
        return None

    tickers = _extract_convo_tickers(post_row, summariser)
    if not tickers:
        return None

    bullets = _clean_strings(summariser.get("summary_bullets"))
    if not bullets:
        fallback = summariser.get("summary") or summariser.get("summary_text")
        if isinstance(fallback, str) and fallback.strip():
            bullets = [fallback.strip()]

    extras = _load_extras_dict(conn, post_id)
    ts = (
        post_row["posted_at"]
        or post_row["scraped_at"]
        or extras.get("posted_at")
        or extras.get("scraped_at")
        or _now_iso()
    )
    url = extras.get("final_url") or post_row["url"] or ""
    source = post_row["source"] or post_row["platform"] or extras.get("source") or "news"

    try:
        result = hub.ingest_article(
            tickers=tickers,
            title=post_row["title"] or "",
            bullets=bullets,
            url=url,
            ts=ts,
            source=source,
            verbose=False,
            post_id=post_id,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"conversation-hub-ingest-failed: {exc}") from exc

    delta = result.get("delta") or {}
    payload = {
        "post_id": post_id,
        "tickers": result.get("tickers") or tickers,
        "appended": result.get("appended", 0),
        "skipped_old": result.get("skipped_old", 0),
        "reason": result.get("reason"),
        "delta": delta,
        "ts": delta.get("t") or ts,
        "source": source,
        "url": url,
        "ingested_at": _now_iso(),
    }

    _insert_stage_payload(conn, post_id, "conversation_hub", payload)
    extras["conversation_hub"] = payload
    _save_extras_dict(conn, post_id, extras)
    return payload


def _latest_stage_payload(conn: sqlite3.Connection, post_id: str, stage: str) -> Optional[Dict[str, Any]]:
    row = _q_one(conn, """
        SELECT payload FROM stages
        WHERE post_id = ? AND stage = ?
        ORDER BY datetime(created_at) DESC
        LIMIT 1
    """, (post_id, stage))
    if not row or not row["payload"]:
        return None
    try:
        return json.loads(row["payload"])
    except Exception:
        return None


def _load_extras_dict(conn: sqlite3.Connection, post_id: str) -> Dict[str, Any]:
    row = _q_one(conn, "SELECT payload_json FROM post_extras WHERE post_id = ?", (post_id,))
    if not row or not row["payload_json"]:
        return {}
    try:
        data = json.loads(row["payload_json"])
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _save_extras_dict(conn: sqlite3.Connection, post_id: str, extras: Dict[str, Any]) -> None:
    _upsert_extras(conn, post_id, extras)


def _strip_research_from_extras(conn: sqlite3.Connection, post_id: Optional[str] = None) -> int:
    """Remove cached research payloads from ``post_extras``.

    Returns the number of posts whose extras were updated.
    """

    if post_id is not None:
        targets = [post_id]
    else:
        rows = conn.execute("SELECT post_id FROM post_extras").fetchall()
        targets = [row["post_id"] for row in rows if row and row["post_id"]]

    updated = 0
    for pid in targets:
        extras = _load_extras_dict(conn, pid)
        if "research" not in extras:
            continue
        extras.pop("research", None)
        _save_extras_dict(conn, pid, extras)
        updated += 1
    return updated


def _insert_stage_payload(
    conn: sqlite3.Connection,
    post_id: str,
    stage: str,
    payload: Dict[str, Any],
) -> str:
    created_at = _now_iso()
    conn.execute(
        "INSERT INTO stages (post_id, stage, created_at, payload) VALUES (?, ?, ?, ?)",
        (post_id, stage, created_at, json.dumps(payload, ensure_ascii=False)),
    )
    conn.commit()
    return created_at


def _extract_summary_text(payload: Optional[Dict[str, Any]]) -> str:
    if not isinstance(payload, dict):
        return ""
    parts: List[str] = []
    summary_text = payload.get("summary_text")
    if isinstance(summary_text, str):
        text = summary_text.strip()
        if text:
            parts.append(text)
    bullets = payload.get("summary_bullets")
    if isinstance(bullets, list):
        for bullet in bullets:
            text_val: Optional[str] = None
            if isinstance(bullet, str):
                text_val = bullet
            elif isinstance(bullet, dict):
                raw = bullet.get("text") or bullet.get("summary")
                if isinstance(raw, str):
                    text_val = raw
            if not text_val:
                continue
            text = text_val.strip()
            if text:
                parts.append(text)
    if not parts:
        return ""
    return "\n".join(parts)


def _article_summary_tokens(post_id: str) -> Tuple[int, int]:
    with _connect() as conn:
        _ensure_schema(conn)
        row = _q_one(conn, "SELECT text FROM posts WHERE post_id = ?", (post_id,))
        article_text = ""
        if row and row.get("text"):
            article_text = str(row["text"] or "")
        summary_payload = _latest_stage_payload(conn, post_id, "summariser") or {}
    summary_text = _extract_summary_text(summary_payload)
    return approximate_token_count(article_text), approximate_token_count(summary_text)


def _record_council_error(
    post_id: str,
    stage: str,
    stage_label: str,
    message: str,
    job_id: str,
    *,
    log_excerpt: Optional[str] = None,
) -> None:
    trimmed_message = (message or "").strip()
    trimmed_excerpt = (log_excerpt or "").strip()
    if trimmed_excerpt and len(trimmed_excerpt) > 8000:
        trimmed_excerpt = trimmed_excerpt[:8000]
    try:
        with _connect() as conn:
            _ensure_schema(conn)
            conn.execute(
                """
                INSERT INTO council_analysis_errors
                  (post_id, stage, stage_label, message, log_excerpt, job_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (post_id, stage, stage_label, trimmed_message or None, trimmed_excerpt or None, job_id),
            )
            conn.commit()
    except Exception:  # noqa: BLE001
        logging.exception("failed to record council error for %s stage %s", post_id, stage)


def _format_council_error_detail(detail: Any) -> str:
    if isinstance(detail, str):
        return detail
    if isinstance(detail, dict):
        error_text = str(detail.get("error") or detail.get("message") or "").strip()
        extras: List[str] = []
        for key, value in detail.items():
            if key in {"error", "message"}:
                continue
            if value is None:
                continue
            if isinstance(value, (list, tuple, set)):
                values = [str(item).strip() for item in value if str(item or "").strip()]
                if not values:
                    continue
                extras.append(f"{key}=[{', '.join(values)}]")
            else:
                text = str(value).strip()
                if not text:
                    continue
                extras.append(f"{key}={text}")
        if extras:
            extras_text = "; ".join(extras)
            if error_text:
                return f"{error_text} ({extras_text})"
            return extras_text
        if error_text:
            return error_text
        try:
            return json.dumps(detail, ensure_ascii=False)
        except Exception:  # noqa: BLE001
            return str(detail)
    return str(detail)


def _now_iso() -> str:
    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_article_time(raw: Optional[str]) -> str:
    if not raw:
        return _now_iso()
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")
    except Exception:
        return raw


def _clean_strings(value: Any) -> List[str]:
    items: List[str] = []
    if isinstance(value, list):
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                items.append(text)
    elif isinstance(value, str):
        for part in value.splitlines():
            text = part.strip()
            if text:
                items.append(text)
    return items


def _extract_claim_texts(payload: Dict[str, Any]) -> List[str]:
    claims = payload.get("claims")
    texts: List[str] = []
    if isinstance(claims, list):
        for claim in claims:
            if isinstance(claim, dict):
                text = claim.get("text")
                if text is None:
                    continue
                clean = str(text).strip()
                if clean:
                    texts.append(clean)
    return texts


def _direction_estimate(direction_payload: Optional[Dict[str, Any]], summariser_payload: Optional[Dict[str, Any]]) -> str:
    if isinstance(direction_payload, dict):
        raw = direction_payload.get("implied_direction")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    if isinstance(summariser_payload, dict):
        stance = summariser_payload.get("author_stance")
        if isinstance(stance, str) and stance.strip():
            s = stance.strip().lower()
            mapping = {
                "bullish": "up",
                "bearish": "down",
                "neutral": "neutral",
                "uncertain": "uncertain",
                "up": "up",
                "down": "down",
            }
            return mapping.get(s, stance.strip())
    return "uncertain"


def _primary_ticker(
    summariser_payload: Optional[Dict[str, Any]],
    entity_payload: Optional[Dict[str, Any]],
) -> Optional[str]:
    candidates: List[str] = []
    if isinstance(summariser_payload, dict):
        assets = summariser_payload.get("assets_mentioned") or []
        if isinstance(assets, list):
            for asset in assets:
                if isinstance(asset, dict):
                    ticker = asset.get("ticker")
                    if isinstance(ticker, str) and ticker.strip():
                        candidates.append(ticker.strip().upper())
    if not candidates and isinstance(entity_payload, dict):
        assets = entity_payload.get("assets") or []
        if isinstance(assets, list):
            for asset in assets:
                if isinstance(asset, dict):
                    ticker = asset.get("ticker")
                    if isinstance(ticker, str) and ticker.strip():
                        candidates.append(ticker.strip().upper())
    return candidates[0] if candidates else None


def _try_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_tool_label(tool: str) -> str:
    return tool.replace("_", " ").replace("-", " ").title()


def _format_tool_result(tool: str, result: Dict[str, Any]) -> str:
    if not isinstance(result, dict):
        return f"{_format_tool_label(tool)}: no data"

    if tool == "price_window":
        rows = result.get("data")
        if isinstance(rows, list) and rows:
            first = rows[0]
            last = rows[-1]
            first_close = _try_float(first.get("Close"))
            last_close = _try_float(last.get("Close"))
            if first_close is not None and last_close is not None:
                change = last_close - first_close
                pct = (change / first_close * 100.0) if first_close else None
                pct_str = f" ({pct:+.1f}%)" if pct is not None else ""
                return f"Price window close {last_close:.2f}{pct_str}".strip()
        note = result.get("note") or "no data"
        return f"Price window: {note}"

    if tool == "compute_indicators":
        rsi = _try_float(result.get("rsi14"))
        macd = result.get("macd") if isinstance(result.get("macd"), dict) else {}
        hist = _try_float(macd.get("hist")) if isinstance(macd, dict) else None
        close = _try_float(result.get("close"))
        parts = []
        if rsi is not None:
            parts.append(f"RSI14 {rsi:.1f}")
        if hist is not None:
            parts.append(f"MACD hist {hist:+.2f}")
        if close is not None:
            parts.append(f"Close {close:.2f}")
        if not parts:
            note = result.get("note") or "no indicators"
            parts.append(str(note))
        return "; ".join(parts)

    if tool == "trend_strength":
        direction = result.get("direction")
        strength = result.get("strength")
        slope = _try_float(result.get("slope_pct_per_day"))
        r2 = _try_float(result.get("r2"))
        parts = []
        if direction:
            parts.append(f"Trend {direction}")
        if strength is not None:
            parts.append(f"Strength {strength}")
        if slope is not None:
            parts.append(f"Slope {slope:+.2f}%/day")
        if r2 is not None:
            parts.append(f"R² {r2:.2f}")
        return "; ".join(parts) or "Trend strength: no data"

    if tool == "volatility_state":
        state = result.get("state")
        ratio = _try_float(result.get("ratio"))
        rv = _try_float(result.get("realized_vol_annual_pct"))
        if state or ratio is not None or rv is not None:
            parts = []
            if state:
                parts.append(f"Vol {state}")
            if ratio is not None:
                parts.append(f"Ratio {ratio:.2f}")
            if rv is not None:
                parts.append(f"RV {rv:.2f}%")
            return "; ".join(parts)
        note = result.get("note") or "volatility unavailable"
        return str(note)

    if tool == "support_resistance_check":
        sup = _try_float(result.get("nearest_support"))
        res = _try_float(result.get("nearest_resistance"))
        pct_sup = _try_float(result.get("distance_to_support_pct"))
        pct_res = _try_float(result.get("distance_to_resistance_pct"))
        parts = []
        if sup is not None:
            if pct_sup is not None:
                parts.append(f"Support {sup:.2f} ({pct_sup:+.1f}%)")
            else:
                parts.append(f"Support {sup:.2f}")
        if res is not None:
            if pct_res is not None:
                parts.append(f"Resistance {res:.2f} ({pct_res:+.1f}%)")
            else:
                parts.append(f"Resistance {res:.2f}")
        if parts:
            return "; ".join(parts)
        note = result.get("note") or "no levels"
        return str(note)

    if tool == "bollinger_breakout_scan":
        event = result.get("last_event")
        date = result.get("last_event_date")
        pct_b = _try_float(result.get("%b"))
        bw = _try_float(result.get("bandwidth"))
        parts = []
        if event:
            parts.append(f"Last {event.replace('_', ' ')}")
        if date:
            parts.append(f"on {date}")
        if pct_b is not None:
            parts.append(f"%B {pct_b:.2f}")
        if bw is not None:
            parts.append(f"Bandwidth {bw:.3f}")
        return " ".join(parts).strip() or "Bollinger scan: no signal"

    if tool == "obv_trend":
        trend = result.get("trend")
        slope = _try_float(result.get("slope"))
        r2 = _try_float(result.get("r2"))
        parts = []
        if trend:
            parts.append(f"OBV {trend}")
        if slope is not None:
            parts.append(f"Slope {slope:+.0f}")
        if r2 is not None:
            parts.append(f"R² {r2:.2f}")
        return "; ".join(parts) or "OBV trend: no data"

    if tool == "mfi_flow":
        mfi = _try_float(result.get("mfi"))
        state = result.get("state")
        parts = []
        if mfi is not None:
            parts.append(f"MFI {mfi:.0f}")
        if state:
            parts.append(state)
        return " ".join(parts).strip() or "MFI flow: no data"

    note = result.get("note")
    return f"{_format_tool_label(tool)}: {note or 'no data'}"


def _summarize_technical_results(results: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    insights: List[Dict[str, Any]] = []
    lines: List[str] = []
    for record in results:
        tool = record.get("tool") or "unknown"
        status = record.get("status") or ("ok" if record.get("result") is not None else "error")
        if status != "ok":
            text = f"{_format_tool_label(tool)}: {record.get('error', 'failed')}"
        else:
            text = _format_tool_result(tool, record.get("result") or {})
        insights.append({"tool": tool, "text": text, "status": status})
        lines.append(text)
    return insights, lines


def _summarize_sentiment(data: Dict[str, Any]) -> str:
    if "error" in data:
        return f"Sentiment unavailable: {data['error']}"
    ticker = data.get("ticker")
    lookback = data.get("lookback_days")
    ticker_series = data.get("series_ticker")
    sector_series = data.get("series_sector")
    parts: List[str] = []
    if isinstance(ticker_series, list) and ticker_series:
        last = ticker_series[-1]
        avg = _try_float(last.get("avg_combined"))
        posts = last.get("posts")
        if avg is not None:
            part = f"Ticker avg {avg:+.2f}"
            if posts is not None:
                part += f" across {posts} post(s)"
            parts.append(part)
    if isinstance(sector_series, list) and sector_series:
        last = sector_series[-1]
        avg = _try_float(last.get("avg_combined"))
        posts = last.get("posts")
        if avg is not None:
            part = f"Sector avg {avg:+.2f}"
            if posts is not None:
                part += f" across {posts} post(s)"
            parts.append(part)
    counts = data.get("counts")
    if isinstance(counts, dict):
        considered = counts.get("considered")
        if considered:
            parts.append(f"Considered {considered} post(s)")
    prefix = f"{ticker} sentiment ({lookback}d)" if ticker else "Sentiment"
    return f"{prefix}: " + ("; ".join(parts) if parts else "no signal")


def _build_research_summary_text(
    ticker: str,
    article_time: str,
    hypotheses: List[Dict[str, Any]],
    rationale: str,
    technical_lines: List[str],
    sentiment_summary: Optional[str],
) -> str:
    lines: List[str] = [f"Research focus: {ticker}", f"Article time: {article_time}"]
    if hypotheses:
        lines.append("")
        lines.append("Hypotheses:")
        for idx, hyp in enumerate(hypotheses, start=1):
            if not isinstance(hyp, dict):
                continue
            text = str(hyp.get("text") or "").strip()
            hyp_type = str(hyp.get("type") or "").strip()
            if text:
                if hyp_type:
                    lines.append(f"{idx}. ({hyp_type}) {text}")
                else:
                    lines.append(f"{idx}. {text}")
    if rationale:
        lines.append("")
        lines.append(f"Rationale: {rationale}")
    if technical_lines:
        lines.append("")
        lines.append("Technical checks:")
        for entry in technical_lines:
            lines.append(f"- {entry}")
    if sentiment_summary:
        lines.append("")
        lines.append(sentiment_summary)
    return "\n".join(lines).strip()


def _execute_technical_plan(plan: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    from technical_analyser import (
        tool_bollinger_breakout_scan,
        tool_compute_indicators,
        tool_mfi_flow,
        tool_obv_trend,
        tool_price_window,
        tool_support_resistance_check,
        tool_trend_strength,
        tool_volatility_state,
    )

    steps = plan.get("steps") if isinstance(plan, dict) else None
    if not isinstance(steps, list):
        steps = []

    results: List[Dict[str, Any]] = []
    for idx, raw_step in enumerate(steps):
        record: Dict[str, Any] = {
            "index": idx,
            "tool": None,
            "status": "skipped",
        }
        if not isinstance(raw_step, dict):
            record["error"] = "invalid_step"
            results.append(record)
            continue

        tool = raw_step.get("tool")
        record.update(
            {
                "tool": tool,
                "covers": raw_step.get("covers") or [],
                "tests": raw_step.get("tests") or [],
                "pass_if": raw_step.get("pass_if"),
                "fail_if": raw_step.get("fail_if"),
                "why": raw_step.get("why"),
            }
        )

        args = raw_step.get("args")
        if not isinstance(args, dict):
            record["status"] = "error"
            record["error"] = "invalid_args"
            results.append(record)
            continue

        try:
            if tool == "price_window":
                res = tool_price_window(
                    str(args["ticker"]).strip(),
                    str(args["from"]).strip(),
                    str(args["to"]).strip(),
                    str(args.get("interval", "1d")).strip() or "1d",
                )
            elif tool == "compute_indicators":
                res = tool_compute_indicators(
                    str(args["ticker"]).strip(),
                    int(args.get("window_days", 60)),
                )
            elif tool == "trend_strength":
                res = tool_trend_strength(
                    str(args["ticker"]).strip(),
                    int(args.get("lookback_days", 30)),
                )
            elif tool == "volatility_state":
                res = tool_volatility_state(
                    str(args["ticker"]).strip(),
                    int(args.get("days", 20)),
                    int(args.get("baseline_days", 10)),
                )
            elif tool == "support_resistance_check":
                res = tool_support_resistance_check(
                    str(args["ticker"]).strip(),
                    int(args.get("days", 30)),
                )
            elif tool == "bollinger_breakout_scan":
                res = tool_bollinger_breakout_scan(
                    str(args["ticker"]).strip(),
                    int(args.get("days", 20)),
                )
            elif tool == "obv_trend":
                res = tool_obv_trend(
                    str(args["ticker"]).strip(),
                    int(args.get("lookback_days", 30)),
                )
            elif tool == "mfi_flow":
                res = tool_mfi_flow(
                    str(args["ticker"]).strip(),
                    int(args.get("period", 14)),
                )
            else:
                # If a non-technical tool sneaks in, don’t fail the whole block — skip it.
                record["status"] = "skipped"
                record["error"] = "non-technical-tool"
                results.append(record)
                continue

            record["status"] = "ok"
            record["result"] = res
        except Exception as exc:  # noqa: BLE001
            record["status"] = "error"
            record["error"] = str(exc)
        results.append(record)

    insights, summary_lines = _summarize_technical_results(results)
    status = "ok"
    if not results:
        status = "empty"
    elif any(r.get("status") == "error" for r in results):
        status = "partial"

    payload = {
        "steps": steps,
        "results": results,
        "insights": insights,
        "summary_lines": summary_lines,
        "status": status,
    }
    return payload, summary_lines


def _run_sentiment_block(
    ticker: str,
    article_time: str,
    *,
    channel: str = "social",
    lookback_days: int = 7,
    burst_hours: int = 6,
    conversation_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    hub = _get_conversation_hub()
    if not hub:
        raise RuntimeError("conversation-hub-unavailable")

    store = getattr(hub, "store", None)
    if store is None:
        raise RuntimeError("conversation-store-unavailable")

    norm_ticker = (ticker or "").strip().upper()
    if not norm_ticker:
        raise ValueError("ticker-required")

    try:
        score = compute_ticker_signal(
            store,
            norm_ticker,
            article_time,
            lookback_days=lookback_days,
            peers=None,
            channel_filter=channel,
            burst_hours=burst_hours,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "error": f"hub-score-failed: {exc}",
            "ticker": norm_ticker,
            "as_of": article_time,
            "channel": channel,
            "window_days": lookback_days,
        }

    signal = score.get("signal") if isinstance(score, dict) else None
    if not isinstance(signal, dict):
        signal = {}

    des_raw = signal.get("des_raw") if isinstance(signal.get("des_raw"), (int, float)) else None
    des_sector = signal.get("des_sector") if isinstance(signal.get("des_sector"), (int, float)) else None
    des_idio = signal.get("des_idio") if isinstance(signal.get("des_idio"), (int, float)) else None
    confidence = signal.get("confidence") if isinstance(signal.get("confidence"), (int, float)) else None
    n_deltas_raw = signal.get("n_deltas")
    n_deltas = int(n_deltas_raw) if isinstance(n_deltas_raw, (int, float)) else None

    recent_deltas = _fetch_recent_deltas(store, norm_ticker, article_time, limit=6)

    payload: Dict[str, Any] = {
        "ticker": norm_ticker,
        "as_of": article_time,
        "channel": score.get("channel") if isinstance(score.get("channel"), str) else channel,
        "window_days": score.get("window_days") if isinstance(score.get("window_days"), int) else lookback_days,
        "burst_hours": score.get("burst_hours") if isinstance(score.get("burst_hours"), int) else burst_hours,
        "des_raw": des_raw,
        "des_sector": des_sector,
        "des_idio": des_idio,
        "confidence": confidence,
        "n_deltas": n_deltas,
        "DES_adj": (des_idio or 0.0) * (confidence or 0.0) if (des_idio is not None and confidence is not None) else None,
        "raw": score,
    }

    if recent_deltas:
        payload["deltas"] = recent_deltas
        payload["latest_delta"] = recent_deltas[-1]

    if isinstance(conversation_payload, dict):
        delta = conversation_payload.get("delta")
        if isinstance(delta, dict):
            who = delta.get("who") or []
            matches = norm_ticker in {str(w).strip().upper() for w in who if isinstance(w, str)}
            if matches:
                payload["article_delta"] = {
                    "t": delta.get("t"),
                    "sum": delta.get("sum"),
                    "dir": delta.get("dir"),
                    "impact": delta.get("impact"),
                    "why": delta.get("why"),
                    "chan": delta.get("chan"),
                    "cat": delta.get("cat"),
                    "src": delta.get("src"),
                    "url": delta.get("url"),
                }

    try:
        narrative = hub.ask_as_of(
            norm_ticker,
            "Summarize tone and catalysts.",
            article_time,
        )
    except Exception as exc:  # noqa: BLE001
        payload["narrative_error"] = f"hub-narrative-failed: {exc}"
    else:
        if narrative:
            payload["narrative"] = narrative

    return payload

def _trim(s: str, n: int) -> str:
    if s and len(s) > n:
        return s[:n] + "\n…[truncated]…"
    return s or ""

def _safe_bool_env(flag: bool) -> str:
    return "1" if flag else "0"


class _CallbackWriter(io.TextIOBase):
    def __init__(self, callback: Optional[Callable[[str], None]]):
        super().__init__()
        self._callback = callback
        self._buffer = ""

    def write(self, s: str) -> int:  # type: ignore[override]
        if not self._callback or not s:
            return len(s or "")
        text = str(s)
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._callback(line.rstrip("\r"))
        return len(text)

    def flush(self) -> None:  # type: ignore[override]
        if self._callback and self._buffer:
            self._callback(self._buffer.rstrip("\r"))
        self._buffer = ""


def _load_council_failure_entries() -> List[Dict[str, Any]]:
    if not COUNCIL_FAILURES_PATH.exists():
        return []
    try:
        with COUNCIL_FAILURES_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return []
    if isinstance(data, dict):
        data = data.get("failures")
    if not isinstance(data, list):
        return []
    entries: List[Dict[str, Any]] = []
    for item in data:
        if isinstance(item, dict):
            entries.append(item)
    return entries


def _save_council_failure_entries(entries: List[Dict[str, Any]]) -> None:
    COUNCIL_FAILURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = COUNCIL_FAILURES_PATH.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(entries, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, COUNCIL_FAILURES_PATH)


def _load_failed_post_ids() -> Set[str]:
    failed_ids: Set[str] = set()
    for entry in _load_council_failure_entries():
        pid = str(entry.get("post_id") or "").strip()
        if pid:
            failed_ids.add(pid)
    return failed_ids


def _record_council_failure(
    stage: str,
    post_id: str,
    title: str,
    details: str,
    *,
    context: Optional[str] = None,
) -> None:
    safe_details = (details or "").replace("\n", " ").strip()
    entry = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stage": stage,
        "post_id": post_id,
        "title": title,
        "details": safe_details,
    }
    if context:
        entry["context"] = context
    with COUNCIL_FAILURES_LOCK:
        entries = _load_council_failure_entries()
        entries.append(entry)
        _save_council_failure_entries(entries)


# =========================
# Job helpers
# =========================
def _job_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def _load_job(job_id: str) -> Optional[Dict[str, Any]]:
    path = _job_path(job_id)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _job_save(job: Dict[str, Any]) -> None:
    path = _job_path(job["id"])
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(job, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def _job_mutate(job_id: str, mutator: Callable[[Dict[str, Any]], None]) -> Optional[Dict[str, Any]]:
    job = _load_job(job_id)
    if not job:
        return None
    mutator(job)
    job["updated_at"] = time.time()
    _job_save(job)
    return job


def _job_update_fields(job_id: str, **fields: Any) -> Optional[Dict[str, Any]]:
    return _job_mutate(job_id, lambda job: job.update(fields))


def _job_append_log(job_id: str, line: str, keep: int = JOB_LOG_KEEP) -> Optional[Dict[str, Any]]:
    def mutate(job: Dict[str, Any]) -> None:
        tail = list(job.get("log_tail") or [])
        tail.append(line)
        if len(tail) > keep:
            tail = tail[-keep:]
        job["log_tail"] = tail

    return _job_mutate(job_id, mutate)


def _job_increment(job_id: str, field: str, amount: int = 1) -> Optional[Dict[str, Any]]:
    def mutate(job: Dict[str, Any]) -> None:
        current = int(job.get(field, 0) or 0)
        job[field] = current + amount

    return _job_mutate(job_id, mutate)


def _default_oracle_cursor() -> Dict[str, str]:
    return {"platform": "", "post_id": "", "scraped_at": ""}


def _load_oracle_cursor() -> Dict[str, str]:
    if not ORACLE_CURSOR_PATH.exists():
        return _default_oracle_cursor()
    try:
        raw = json.loads(ORACLE_CURSOR_PATH.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return _default_oracle_cursor()
        cursor = {
            "platform": str(raw.get("platform", "") or ""),
            "post_id": str(raw.get("post_id", "") or ""),
            "scraped_at": str(raw.get("scraped_at", "") or ""),
        }
        return cursor
    except Exception:
        return _default_oracle_cursor()


def _save_oracle_cursor(cursor: Dict[str, Any]) -> None:
    payload = {
        "platform": str(cursor.get("platform", "") or ""),
        "post_id": str(cursor.get("post_id", "") or ""),
        "scraped_at": str(cursor.get("scraped_at", "") or ""),
    }
    tmp_path = ORACLE_CURSOR_PATH.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, ORACLE_CURSOR_PATH)


def _default_oracle_retry_state() -> Dict[str, Any]:
    return {
        "post_id": "",
        "attempts": 0,
        "scraped_at": "",
        "platform": "",
        "source": "",
        "title": "",
        "last_error": "",
    }


def _load_oracle_retry_state() -> Dict[str, Any]:
    if not ORACLE_RETRY_STATE_PATH.exists():
        return _default_oracle_retry_state()
    try:
        raw = json.loads(ORACLE_RETRY_STATE_PATH.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return _default_oracle_retry_state()
        state = _default_oracle_retry_state()
        for key in state:
            if key in raw:
                state[key] = str(raw[key] or "") if isinstance(state[key], str) else raw[key]
        attempts = raw.get("attempts")
        try:
            state["attempts"] = int(attempts) if attempts is not None else 0
        except (TypeError, ValueError):
            state["attempts"] = 0
        return state
    except Exception:
        return _default_oracle_retry_state()


def _save_oracle_retry_state(state: Dict[str, Any]) -> None:
    payload = {
        "post_id": str(state.get("post_id", "") or ""),
        "attempts": int(state.get("attempts", 0) or 0),
        "scraped_at": str(state.get("scraped_at", "") or ""),
        "platform": str(state.get("platform", "") or ""),
        "source": str(state.get("source", "") or ""),
        "title": str(state.get("title", "") or ""),
        "last_error": str(state.get("last_error", "") or ""),
    }
    tmp_path = ORACLE_RETRY_STATE_PATH.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, ORACLE_RETRY_STATE_PATH)


def _save_oracle_unsummarised(entries: List[Dict[str, Any]], *, cutoff_iso: str) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cutoff": cutoff_iso,
        "count": len(entries),
        "articles": entries,
    }
    tmp_path = ORACLE_UNSUMMARISED_PATH.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, ORACLE_UNSUMMARISED_PATH)


def _append_skipped_article_entry(entry: Dict[str, Any], keep: int = 200) -> None:
    existing: List[Dict[str, Any]] = []
    if ORACLE_SKIPPED_PATH.exists():
        try:
            raw = json.loads(ORACLE_SKIPPED_PATH.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                existing = [item for item in raw if isinstance(item, dict)]
        except Exception:
            existing = []
    existing.append(entry)
    if keep > 0 and len(existing) > keep:
        existing = existing[-keep:]
    tmp_path = ORACLE_SKIPPED_PATH.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(existing, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, ORACLE_SKIPPED_PATH)


def _oracle_auth_tuple() -> Optional[Tuple[str, str]]:
    user = (ORACLE_USER or "").strip()
    pwd = (ORACLE_PASS or "").strip()
    if not (user and pwd):
        return None
    return user, pwd


def _oracle_join(base_url: str, path: str) -> str:
    base = (base_url or "").strip()
    if not base:
        return path
    if not base.endswith("/"):
        base = base + "/"
    return urljoin(base, path.lstrip("/"))


# =========================
# Endpoints
# =========================
def _worker_refresh_summaries(job_id: str):
    global ACTIVE_JOB_ID
    job = _load_job(job_id)
    if not job:
        return

    snap_raw = job.get("snapshot")
    snap_path = Path(snap_raw) if snap_raw else None
    backlog: List[Dict[str, Any]] = list(job.get("backlog") or [])
    only_new = bool(job.get("only_new"))
    collect_new_posts = bool(job.get("collect_new_posts"))
    oracle_online = bool(job.get("oracle_online"))
    oracle_base_url = str(job.get("oracle_base_url") or "").strip()
    backlog_total = len(backlog)
    csv_total = int(job.get("csv_total") or 0)
    existing_posts: set[str] = set()
    _job_update_fields(
        job_id,
        status="running",
        started_at=job.get("started_at") or time.time(),
        oracle_status=(job.get("oracle_status") if not oracle_online else "connecting"),
    )

    def _ensure_not_cancelled() -> bool:
        current = _load_job(job_id)
        if not current:
            return False
        if current.get("cancelled"):
            _job_append_log(job_id, "⚠ refresh cancelled by user")
            _job_update_fields(
                job_id,
                status="cancelled",
                phase="",
                current="",
                ended_at=time.time(),
                cancelled=True,
            )
            return False
        return True

    interest_notice_done = False
    interest_total = 0
    interest_done = 0

    def _interest_schedule(count: int) -> None:
        nonlocal interest_total
        if count <= 0:
            return
        interest_total += count

    def _interest_before_run(label: str, pid: str, title: str) -> None:
        if not label.strip():
            label = "Interest"
        total = interest_total if interest_total > 0 else interest_done + 1
        index = interest_done + 1
        remaining = max(total - index, 0)
        safe_pid = pid or "(unknown post)"
        safe_title = (title or "")[:80]
        _job_update_fields(
            job_id,
            phase=f"{label} {index}/{total}",
            current=f"{label}: {safe_pid} — {safe_title} ({remaining} remaining)",
        )

    def _interest_after_run() -> None:
        nonlocal interest_done
        interest_done += 1

    def _interest_supported() -> bool:
        nonlocal interest_notice_done
        if not callable(compute_interest_for_post):
            if not interest_notice_done:
                if INTEREST_SCORE_IMPORT_ERROR:
                    _job_append_log(job_id, f"⚠ interest scoring unavailable: {INTEREST_SCORE_IMPORT_ERROR}")
                else:
                    _job_append_log(job_id, "⚠ interest scoring unavailable")
                interest_notice_done = True
            return False
        return True

    def _should_run_interest(conn: sqlite3.Connection, pid: str) -> bool:
        if not _interest_supported():
            return False
        try:
            row = conn.execute(
                "SELECT status FROM council_stage_interest WHERE post_id = ? LIMIT 1",
                (pid,),
            ).fetchone()
        except Exception:  # noqa: BLE001
            return True
        if not row:
            return True
        status = _norm_optional_str(row["status"]) or ""
        return status.lower() != "ok"

    def _run_interest_for_post(pid: str, title: str = "") -> Optional[Dict[str, Any]]:
        if not _interest_supported():
            return None
        if not _ensure_not_cancelled():
            return None
        try:
            result = compute_interest_for_post(
                pid,
                db_path=str(DB_PATH),
                conv_db_path=str(CONVO_STORE_PATH),
                persist=True,
            )
        except Exception as exc:  # noqa: BLE001
            logging.exception("interest scoring failed for %s", pid)
            _job_append_log(job_id, f"✗ interest error {pid}: {exc}")
            return None

        status = _norm_optional_str(result.get("status")) or ""
        ticker = _norm_optional_str(result.get("ticker")) or ""
        if status.lower() == "ok":
            score_val = result.get("interest_score")
            label = _norm_optional_str(result.get("interest_label"))
            if isinstance(score_val, (int, float)):
                score_txt = f"{int(round(score_val))}%"
            else:
                score_txt = "ok"
            if label:
                score_txt = f"{score_txt} {label}"
            suffix = f" [{ticker}]" if ticker else ""
            _job_append_log(job_id, f"✓ interest {pid}{suffix} {score_txt}")
        else:
            reason = _norm_optional_str(result.get("error_code")) or _norm_optional_str(result.get("error_message")) or "failed"
            suffix = f" [{ticker}]" if ticker else ""
            _job_append_log(job_id, f"⚠ interest failed {pid}{suffix}: {reason}")
        return result

    oracle_interest_threshold = 0.0
    try:
        oracle_interest_threshold = float(job.get("oracle_interest_threshold") or 0.0)
    except (TypeError, ValueError):
        oracle_interest_threshold = 0.0
    oracle_interest_threshold = max(0.0, min(100.0, oracle_interest_threshold))
    oracle_handles_council = bool(job.get("oracle_handles_council"))
    warmup_cutoff_iso = str(job.get("oracle_warmup_cutoff") or datetime.now(timezone.utc).isoformat())


    def _run_oracle_loop(conn: sqlite3.Connection, summarised: Set[str]) -> None:
        if not oracle_base_url:
            _job_append_log(job_id, "✗ Oracle base URL missing; cannot synchronise")
            _job_update_fields(
                job_id,
                status="error",
                error="oracle-base-url-missing",
                phase="",
                current="",
                oracle_status="error",
                ended_at=time.time(),
            )
            return

        session = requests.Session()
        auth_tuple = _oracle_auth_tuple()
        if auth_tuple:
            session.auth = auth_tuple
        session.headers.update({"User-Agent": "WisdomOfSheep-OracleClient/1.0"})

        timeout: Optional[float]
        if ORACLE_REQUEST_TIMEOUT is None or ORACLE_REQUEST_TIMEOUT <= 0:
            timeout = None
        else:
            timeout = ORACLE_REQUEST_TIMEOUT

        last_request_at = 0.0

        def _handle_unauthorized() -> None:
            _job_append_log(job_id, "✗ Oracle auth invalid. Set WOS_ORACLE_USER and WOS_ORACLE_PASS.")
            _job_update_fields(
                job_id,
                status="error",
                error="oracle-unauthorized",
                oracle_status="unauthorized",
                phase="",
                current="",
                oracle_poll_seconds=None,
                oracle_idle_since=None,
                ended_at=time.time(),
            )

        def _request(
            path: str,
            *,
            params: Optional[Dict[str, Any]] = None,
            min_interval: float = 0.0,
        ) -> Optional[requests.Response]:
            nonlocal last_request_at
            backoff = ORACLE_BACKOFF_MIN
            while _ensure_not_cancelled():
                if min_interval > 0:
                    elapsed = time.time() - last_request_at
                    if elapsed < min_interval:
                        time.sleep(min_interval - elapsed)
                try:
                    resp = session.get(
                        _oracle_join(oracle_base_url, path),
                        params=params,
                        timeout=timeout,
                    )
                    last_request_at = time.time()
                except requests.RequestException as exc:
                    last_request_at = time.time()
                    _job_append_log(job_id, f"⚠ Oracle request failed: {exc}")
                    _job_update_fields(
                        job_id,
                        oracle_status="connecting",
                        phase="Oracle request retry",
                        current=str(exc),
                    )
                    time.sleep(min(backoff, ORACLE_BACKOFF_MAX))
                    backoff = min(backoff * 2, ORACLE_BACKOFF_MAX)
                    continue

                if resp.status_code == 401:
                    _handle_unauthorized()
                    return None
                if resp.status_code == 409:
                    time.sleep(1.0)
                    continue
                if resp.status_code >= 500:
                    _job_append_log(
                        job_id,
                        f"⚠ Oracle server error {resp.status_code}: {resp.text.strip()[:200]}",
                    )
                    time.sleep(min(backoff, ORACLE_BACKOFF_MAX))
                    backoff = min(backoff * 2, ORACLE_BACKOFF_MAX)
                    continue
                if resp.status_code >= 400:
                    _job_append_log(
                        job_id,
                        f"⚠ Oracle request {path} returned {resp.status_code}: {resp.text.strip()[:200]}",
                    )
                    time.sleep(min(backoff, ORACLE_BACKOFF_MAX))
                    backoff = min(backoff * 2, ORACLE_BACKOFF_MAX)
                    continue

                return resp
            return None

        def _fetch_batch(
            ref_platform: Optional[str],
            ref_post_id: Optional[str],
            limit: int,
            *,
            min_interval: float = 0.0,
        ) -> Optional[List[Dict[str, Any]]]:
            params: Dict[str, Any] = {
                "limit": max(1, limit),
                "platform": ref_platform or "",
                "post_id": ref_post_id or "",
            }
            resp = _request("/wos/next-after", params=params, min_interval=min_interval)
            if resp is None:
                return None
            try:
                payload = resp.json()
            except Exception as exc:  # noqa: BLE001
                _job_append_log(job_id, f"⚠ Oracle next-after parse error: {exc}")
                return []
            items = payload.get("items")
            if not isinstance(items, list):
                return []
            return [item for item in items if isinstance(item, dict)]

        def _parse_scraped(value: str) -> Optional[datetime]:
            text = (value or "").strip()
            if not text:
                return None
            try:
                if text.endswith("Z"):
                    text = text[:-1] + "+00:00"
                return datetime.fromisoformat(text)
            except ValueError:
                return None

        def _format_scraped(value: str) -> str:
            dt = _parse_scraped(value)
            if not dt:
                return (value or "").strip() or "unknown time"
            return dt.strftime("%d %b %Y %H:%M")

        def _set_progress(total: int, index: int, stage: str, message: str) -> None:
            _job_update_fields(
                job_id,
                oracle_progress_total=max(int(total or 0), 0),
                oracle_progress_index=max(int(index or 0), 0),
                oracle_progress_stage=stage,
                oracle_progress_message=message,
            )

        def _run_oracle_council(post_id: str, title: str, interest_score: Optional[float]) -> None:
            nonlocal failed_post_ids
            if not oracle_handles_council:
                return
            if interest_score is None or interest_score < oracle_interest_threshold:
                return
            score_txt = f"{int(round(interest_score))}%"
            _job_append_log(job_id, f"→ council {post_id} {score_txt} {title[:80]}")
            try:
                clear_post_analysis(post_id)
            except HTTPException as exc:
                detail = _format_council_error_detail(exc.detail)
                _record_council_failure(post_id, "clear_analysis", "Clear Council Analysis", detail, context="oracle-auto")
                _job_append_log(job_id, f"✗ council clear failed {post_id}: {detail}")
                return
            except Exception as exc:  # noqa: BLE001
                message = str(exc) or repr(exc)
                _record_council_failure(post_id, "clear_analysis", "Clear Council Analysis", message, context="oracle-auto")
                _job_append_log(job_id, f"✗ council clear failed {post_id}: {message}")
                return

            stage_ok = True
            for stage in COUNCIL_ANALYSIS_SEQUENCE:
                label = _council_stage_label(stage)
                _job_append_log(job_id, f"→ {label} for {post_id}")
                err_output = ""
                try:
                    if stage == "research":
                        _run_research_pipeline(post_id)
                    else:
                        code, _out, err = _run_round_table(post_id, [stage], False, False)
                        if code != 0:
                            err_output = err or ""
                            raise RuntimeError(err or f"{stage} failed")
                except HTTPException as exc:
                    detail = _format_council_error_detail(exc.detail)
                    _record_council_failure(post_id, stage, label, detail, context="oracle-auto")
                    _job_append_log(job_id, f"✗ {label} failed {post_id}: {detail}")
                    stage_ok = False
                    break
                except Exception as exc:  # noqa: BLE001
                    message = str(exc) or repr(exc)
                    _record_council_failure(post_id, stage, label, message, context="oracle-auto")
                    _job_append_log(job_id, f"✗ {label} failed {post_id}: {message}")
                    stage_ok = False
                    break
            if stage_ok:
                _job_append_log(job_id, f"✓ council analysis completed {post_id}")

        _job_update_fields(
            job_id,
            oracle_status="connecting",
            phase="Oracle health check",
            current="Connecting to Oracle…",
            oracle_poll_seconds=None,
            oracle_idle_since=None,
        )

        health_resp = _request("/healthz")
        if health_resp is None:
            return
        if health_resp.status_code >= 400:
            _job_append_log(job_id, f"✗ Oracle /healthz returned {health_resp.status_code}")
            _job_update_fields(
                job_id,
                status="error",
                error=f"oracle-healthz-{health_resp.status_code}",
                oracle_status="error",
                phase="",
                current="",
                oracle_poll_seconds=None,
                oracle_idle_since=None,
                ended_at=time.time(),
            )
            return

        _job_update_fields(
            job_id,
            oracle_status="connecting",
            phase="Oracle ready check",
            current="Waiting for Oracle receiver…",
            oracle_poll_seconds=None,
            oracle_idle_since=None,
        )

        ready_backoff = max(2.0, ORACLE_BACKOFF_MIN)
        while _ensure_not_cancelled():
            ready_resp = _request("/wos/ready")
            if ready_resp is None:
                return
            if ready_resp.status_code >= 400:
                _job_append_log(job_id, f"✗ Oracle /wos/ready {ready_resp.status_code}")
                _job_update_fields(
                    job_id,
                    status="error",
                    error=f"oracle-ready-{ready_resp.status_code}",
                    oracle_status="error",
                    phase="",
                    current="",
                    oracle_poll_seconds=None,
                    oracle_idle_since=None,
                    ended_at=time.time(),
                )
                return
            try:
                ready_payload = ready_resp.json()
            except Exception as exc:  # noqa: BLE001
                _job_append_log(job_id, f"⚠ Oracle ready payload error: {exc}")
                time.sleep(ready_backoff)
                continue
            if ready_payload.get("ready"):
                break
            _job_append_log(job_id, "Oracle busy, retrying ready check…")
            time.sleep(ready_backoff)

        _job_update_fields(
            job_id,
            oracle_status="warmup",
            phase="Oracle warm-up scan",
            current="Scanning Oracle backlog…",
            oracle_poll_seconds=None,
            oracle_idle_since=None,
        )

        pending_article: Optional[Dict[str, Any]] = None
        ref_platform: Optional[str] = None
        ref_post_id: Optional[str] = None
        retry_state = _load_oracle_retry_state()
        failed_post_ids = _load_failed_post_ids()
        warmup_queue: List[Dict[str, Any]] = []
        warmup_records: List[Dict[str, Any]] = []
        warmup_cutoff_dt = _parse_scraped(warmup_cutoff_iso)
        warmup_total_estimate = 0
        scanned_count = 0
        last_seen_article: Optional[Dict[str, Any]] = None
        warmup_finished = False

        stats_resp = _request("/wos/stats", min_interval=0.2)
        if stats_resp is not None and stats_resp.ok:
            try:
                stats_payload = stats_resp.json()
                warmup_total_estimate = int(stats_payload.get("total_posts") or 0)
            except Exception as exc:  # noqa: BLE001
                _job_append_log(job_id, f"⚠ Oracle stats parse error: {exc}")
                warmup_total_estimate = 0

        warmup_initial_msg = (
            f"Warm-up scan 0/{warmup_total_estimate}: preparing…"
            if warmup_total_estimate
            else "Warm-up scan starting…"
        )
        _set_progress(warmup_total_estimate, 0, "warmup", warmup_initial_msg)
        _job_update_fields(job_id, phase="Oracle warm-up scan", current=warmup_initial_msg)

        def _set_retry_state(state: Dict[str, Any]) -> None:
            nonlocal retry_state
            retry_state = state
            _save_oracle_retry_state(state)

        def _clear_retry_state() -> None:
            _set_retry_state(_default_oracle_retry_state())

        while _ensure_not_cancelled() and not warmup_finished:
            batch = _fetch_batch(ref_platform, ref_post_id, ORACLE_BATCH_SIZE, min_interval=0.2)
            if batch is None:
                return
            if not batch:
                break
            for article in batch:
                last_seen_article = article
                scanned_count += 1
                scraped_txt = str(article.get("scraped_at") or "")
                platform_txt = str(article.get("platform") or "")
                pid = str(article.get("post_id") or "").strip()
                total_display = warmup_total_estimate or scanned_count
                human_time = _format_scraped(scraped_txt)
                progress_msg = f"Warm-up scan {scanned_count}/{total_display}: {human_time}"
                _set_progress(total_display, scanned_count, "warmup", progress_msg)
                current_msg = (
                    f"Checking {scanned_count}/{total_display} — {human_time} {platform_txt}:{pid}"
                ).strip()
                _job_update_fields(job_id, phase="Oracle warm-up scan", current=current_msg)
                _job_append_log(
                    job_id,
                    f"checking article {scanned_count}/{total_display}: {human_time} {platform_txt}:{pid}".strip(),
                )

                scraped_dt = _parse_scraped(scraped_txt)
                if warmup_cutoff_dt and scraped_dt and scraped_dt > warmup_cutoff_dt:
                    warmup_finished = True
                    break

                if not pid or pid in summarised or pid in failed_post_ids:
                    continue

                warmup_queue.append(article)
                warmup_records.append({"post_id": pid, "scraped_at": scraped_txt})

            tail = batch[-1]
            ref_platform = str(tail.get("platform") or "") or None
            ref_post_id = str(tail.get("post_id") or "") or None

            _save_oracle_unsummarised(warmup_records, cutoff_iso=warmup_cutoff_iso)

        warmup_total = len(warmup_queue)
        warmup_summary_msg = (
            f"Warm-up queued {warmup_total} unsummarised article(s)"
            if warmup_total
            else "Warm-up scan complete"
        )
        _set_progress(warmup_total_estimate or scanned_count, scanned_count, "warmup", warmup_summary_msg)
        _job_update_fields(job_id, phase="Oracle warm-up scan", current=warmup_summary_msg)
        if warmup_total:
            _job_append_log(job_id, f"→ warm-up queued {warmup_total} unsummarised Oracle article(s)")

        global_article_index = scanned_count
        backlog_processed = 0

        cursor_state = _default_oracle_cursor()
        if last_seen_article:
            cursor_state = {
                "platform": str(last_seen_article.get("platform") or ""),
                "post_id": str(last_seen_article.get("post_id") or ""),
                "scraped_at": str(last_seen_article.get("scraped_at") or ""),
            }
        _save_oracle_cursor(cursor_state)
        _job_update_fields(job_id, oracle_cursor=cursor_state)

        if pending_article is None:
            ref_platform = cursor_state.get("platform") or None
            ref_post_id = cursor_state.get("post_id") or None

        pending_stage = "warmup"
        pending_progress_total = 0
        pending_progress_index = 0
        poll_interval = ORACLE_POLL_BASE
        idle_started: Optional[float] = None

        while _ensure_not_cancelled():
            if pending_article is None:
                if warmup_queue:
                    pending_article = warmup_queue.pop(0)
                    backlog_processed += 1
                    pending_stage = "backlog"
                    pending_progress_total = max(warmup_total, 1)
                    pending_progress_index = backlog_processed
                    idle_started = None
                else:
                    stats_total = 0
                    stats_resp = _request("/wos/stats", min_interval=1.0)
                    if stats_resp is None:
                        return
                    if stats_resp.ok:
                        try:
                            stats_payload = stats_resp.json()
                            stats_total = int(stats_payload.get("total_posts") or 0)
                        except Exception as exc:  # noqa: BLE001
                            _job_append_log(job_id, f"⚠ Oracle stats parse error: {exc}")
                    total_articles = stats_total or max(global_article_index + 1, 1)
                    next_index = global_article_index + 1
                    prep_msg = f"Preparing article {next_index}/{total_articles}…"
                    _set_progress(total_articles, next_index, "live", prep_msg)
                    _job_update_fields(
                        job_id,
                        phase="Oracle fetch",
                        current=prep_msg,
                        oracle_status="processing",
                        oracle_poll_seconds=None,
                        oracle_idle_since=None,
                    )
                    batch = _fetch_batch(ref_platform, ref_post_id, 1, min_interval=1.0)
                    if batch is None:
                        return
                    if not batch:
                        now = time.time()
                        if idle_started is None:
                            idle_started = now
                        wait = min(poll_interval, ORACLE_POLL_MAX)
                        jitter = wait * random.uniform(-0.2, 0.2)
                        wait = max(1.0, wait + jitter)
                        elapsed = now - idle_started
                        last_desc = "last: (none)"
                        if any(cursor_state.values()):
                            last_desc = (
                                f"last: {cursor_state.get('scraped_at') or ''} "
                                f"{cursor_state.get('platform') or ''}:{cursor_state.get('post_id') or ''}"
                            ).strip()
                        idle_msg = f"Idle for {int(elapsed)}s — {last_desc}"
                        _set_progress(total_articles, global_article_index, "idle", idle_msg)
                        _job_update_fields(
                            job_id,
                            oracle_status="idle",
                            phase=f"Idle (polling every {wait:.1f}s)",
                            current=idle_msg,
                            oracle_poll_seconds=wait,
                            oracle_idle_since=idle_started,
                        )
                        _job_append_log(
                            job_id,
                            f"No new articles on Oracle ({last_desc}). Polling again in {wait:.1f}s…",
                        )
                        time.sleep(wait)
                        poll_interval = ORACLE_POLL_BASE
                        continue
                    pending_article = batch[0]
                    pending_stage = "live"
                    pending_progress_total = total_articles
                    pending_progress_index = next_index
                    idle_started = None
                    poll_interval = ORACLE_POLL_BASE

            details = _normalize_article(pending_article)
            pid = details["post_id"]
            title = details["title"]
            scraped_at = details["scraped_at"]
            platform = details["platform"]
            source = details["source"]
            display_title = title[:80]

            if retry_state.get("post_id") != pid:
                _set_retry_state(
                    {
                        "post_id": pid,
                        "attempts": 0,
                        "scraped_at": scraped_at,
                        "platform": platform,
                        "source": source,
                        "title": title,
                        "last_error": "",
                    }
                )

            if pending_stage == "backlog":
                stage_label = f"Backlog summarising {pending_progress_index}/{pending_progress_total}"
                stage_message = (
                    f"Backlog {pending_progress_index}/{pending_progress_total}: "
                    f"{scraped_at} {platform}:{pid} — {display_title}"
                ).strip()
                log_prefix = "backlog"
            else:
                stage_label = f"Oracle summarising {pending_progress_index}/{pending_progress_total}"
                stage_message = (
                    f"Summarising {pending_progress_index}/{pending_progress_total}: "
                    f"{scraped_at} {platform}:{pid} — {display_title}"
                ).strip()
                log_prefix = "oracle"

            _set_progress(max(pending_progress_total, 1), pending_progress_index, pending_stage, stage_message)
            _job_update_fields(
                job_id,
                oracle_status="processing",
                phase=stage_label,
                current=stage_message,
                oracle_poll_seconds=None,
                oracle_idle_since=None,
            )
            _job_append_log(
                job_id,
                f"→ {log_prefix} summarising {scraped_at} {platform}:{pid} — {display_title}",
            )

            if pid in failed_post_ids:
                skip_message = (
                    f"↷ {log_prefix} skipping known failure {scraped_at} {platform}:{pid} — {display_title}"
                )
                _job_append_log(job_id, skip_message)
                cursor_state = {
                    "platform": platform,
                    "post_id": pid,
                    "scraped_at": scraped_at,
                }
                _save_oracle_cursor(cursor_state)
                _job_update_fields(
                    job_id,
                    oracle_status="processing",
                    phase="Oracle skipping known failure",
                    current=f"{scraped_at} {platform}:{pid} — skipped",
                    oracle_cursor=cursor_state,
                    oracle_poll_seconds=None,
                    oracle_idle_since=None,
                )
                _job_increment(job_id, "done", 1)
                _clear_retry_state()
                ref_platform = platform or None
                ref_post_id = pid or None
                pending_article = None
                if pending_stage == "live":
                    global_article_index = max(global_article_index, pending_progress_index)
                continue

            post_vals = {
                "post_id": pid,
                "platform": platform or None,
                "source": source or None,
                "url": pending_article.get("url"),
                "title": title,
                "author": pending_article.get("author"),
                "scraped_at": scraped_at,
                "posted_at": pending_article.get("posted_at") or scraped_at,
                "score": (
                    int(pending_article["score"])
                    if str(pending_article.get("score") or "").isdigit()
                    else None
                ),
                "text": pending_article.get("text"),
            }
            extras_vals = {
                "final_url": pending_article.get("final_url"),
                "fetch_status": pending_article.get("fetch_status"),
                "domain": pending_article.get("domain"),
            }

            _upsert_post_row(conn, post_vals)
            _upsert_extras(conn, pid, extras_vals)

            if only_new:
                existing_posts.add(pid)

            success, error_text, log_lines = _stream_summariser(pid, title)
            if not success:
                reason = error_text or (log_lines[-1] if log_lines else "Unknown summariser failure")
                log_tail = "; ".join(log_lines[-5:]) if log_lines else ""
                attempts = int(retry_state.get("attempts", 0) or 0) + 1
                _set_retry_state(
                    {
                        "post_id": pid,
                        "attempts": attempts,
                        "scraped_at": scraped_at,
                        "platform": platform,
                        "source": source,
                        "title": title,
                        "last_error": reason,
                    }
                )

                if attempts < ORACLE_MAX_RETRIES:
                    retry_message = (
                        f"⚠ summariser retry {scraped_at} {platform}:{pid} "
                        f"attempt {attempts}/{ORACLE_MAX_RETRIES}: {reason}"
                    )
                    if log_tail:
                        retry_message += f" | log_tail={log_tail}"
                    _job_append_log(job_id, retry_message)
                    time.sleep(min(ORACLE_BACKOFF_MIN, 5.0))
                    continue

                skip_message = (
                    f"✗ summariser skipped {scraped_at} {platform}:{pid} after {attempts} attempts"
                )
                if reason:
                    skip_message += f": {reason}"
                _job_append_log(job_id, skip_message)
                _record_council_failure(
                    "summariser",
                    pid,
                    title,
                    f"{reason}" if not log_tail else f"{reason} | log_tail={log_tail}",
                    context=(
                        f"scraped_at={scraped_at}, platform={platform}, source={source}, "
                        f"display_title={display_title}, attempts={attempts}"
                    ),
                )
                failed_post_ids.add(pid)
                _append_skipped_article_entry(
                    {
                        "post_id": pid,
                        "scraped_at": scraped_at,
                        "platform": platform,
                        "source": source,
                        "title": title,
                        "attempts": attempts,
                        "reason": reason,
                        "skipped_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    }
                )
                _clear_retry_state()
                _job_increment(job_id, "done", 1)
                cursor_state = {
                    "platform": platform,
                    "post_id": pid,
                    "scraped_at": scraped_at,
                }
                _save_oracle_cursor(cursor_state)
                _job_update_fields(
                    job_id,
                    oracle_cursor=cursor_state,
                    oracle_poll_seconds=None,
                    oracle_idle_since=None,
                )
                ref_platform = platform or None
                ref_post_id = pid or None
                pending_article = None
                if pending_stage == "live":
                    global_article_index = max(global_article_index, pending_progress_index)
                continue

            _clear_retry_state()
            summarised.add(pid)
            _job_append_log(
                job_id,
                f"✓ summarised {scraped_at} {platform}:{pid} — {display_title}",
            )

            try:
                convo_payload = _ingest_conversation_hub(conn, pid)
                if convo_payload:
                    tickers_txt = ",".join(convo_payload.get("tickers") or [])
                    if tickers_txt:
                        _job_append_log(job_id, f"✓ convo hub {pid} [{tickers_txt}]")
            except Exception as exc:  # noqa: BLE001
                logging.exception("conversation hub ingest failed for oracle %s", pid)
                _job_append_log(job_id, f"✗ convo hub failed {pid}: {exc}")

            interest_result = None
            try:
                if _should_run_interest(conn, pid):
                    _interest_schedule(1)
                    label = "Interest backlog" if pending_stage == "backlog" else "Interest oracle"
                    _interest_before_run(label, pid, title)
                    interest_result = _run_interest_for_post(pid, title)
                    _interest_after_run()
            except Exception as exc:  # noqa: BLE001
                logging.exception("interest scoring check failed for oracle %s", pid)
                _job_append_log(job_id, f"⚠ interest check failed {pid}: {exc}")

            interest_score_val: Optional[float] = None
            if interest_result is not None:
                raw_score = interest_result.get("interest_score")
                try:
                    interest_score_val = float(raw_score) if raw_score is not None else None
                except (TypeError, ValueError):
                    interest_score_val = None
            _record_new_post(pid, title, interest_score_val)

            if pending_stage == "live":
                global_article_index = max(global_article_index, pending_progress_index)

            if oracle_handles_council:
                _run_oracle_council(pid, title, interest_score_val)

            _job_increment(job_id, "done", 1)

            cursor_state = {
                "platform": platform,
                "post_id": pid,
                "scraped_at": scraped_at,
            }
            _save_oracle_cursor(cursor_state)
            _job_update_fields(
                job_id,
                oracle_cursor=cursor_state,
                oracle_poll_seconds=None,
                oracle_idle_since=None,
            )

            ref_platform = platform or None
            ref_post_id = pid or None
            pending_article = None


        def _record_new_post(pid: str, title: str, interest_score: Optional[float]) -> None:
            if not collect_new_posts:
                return

            safe_pid = (pid or "").strip()
            if not safe_pid:
                return

            def mutate(data: Dict[str, Any]) -> None:
                entries = list(data.get("new_posts") or [])
                entry: Dict[str, Any] = {"post_id": safe_pid}
                safe_title = (title or "").strip()
                if safe_title:
                    entry["title"] = safe_title
                if interest_score is not None:
                    entry["interest_score"] = interest_score
                entries.append(entry)
                data["new_posts"] = entries

            _job_mutate(job_id, mutate)

        def _backfill_interest_scores(conn: sqlite3.Connection) -> None:
            if not _interest_supported():
                return
            rows = _q_all(
                conn,
                """
                SELECT DISTINCT s.post_id, COALESCE(p.title, '') AS title
                FROM stages s
                JOIN posts p ON p.post_id = s.post_id
                LEFT JOIN council_stage_interest ci ON ci.post_id = s.post_id
                WHERE s.stage = 'summariser'
                  AND (
                        ci.post_id IS NULL
                     OR LOWER(COALESCE(ci.status, '')) != 'ok'
                     OR LOWER(COALESCE(ci.error_code, '')) = 'price_data_unavailable'
                     OR INSTR(LOWER(COALESCE(ci.interest_why, '')), 'price data being unavailable for ticker') > 0
                  )
                ORDER BY datetime(COALESCE(p.scraped_at, p.posted_at)) DESC
                """,
                tuple(),
            )
            if not rows:
                return
            total = len(rows)
            _interest_schedule(total)
            _job_append_log(job_id, f"→ backfilling interest scores for {total} post(s)")
            for row in rows:
                if not _ensure_not_cancelled():
                    return
                pid = row["post_id"]
                title = row["title"] or ""
                _interest_before_run("Interest backfill", pid, title)
                _run_interest_for_post(pid, title)
                _interest_after_run()

    def _stream_summariser(post_id: str, title: str) -> Tuple[bool, Optional[str], List[str]]:
        log_lines: List[str] = []

        def emit(line: str) -> None:
            clean_line = line.rstrip("\r")
            log_lines.append(clean_line)
            _job_append_log(job_id, clean_line)

        writer = _CallbackWriter(emit)
        try:
            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                run_stages_for_post(
                    **_round_table_kwargs(
                        post_id=post_id,
                        stages=["summariser"],
                        refresh_from_csv=False,
                        echo_post=False,
                        title=title,
                        text="",
                        log_callback=emit,
                    )
                )
        except Exception as exc:
            writer.flush()
            logging.exception("summariser stage failed for %s", post_id)
            message = str(exc)
            emit(f"✗ summariser failed {post_id}: {message}")
            return False, message, log_lines

        writer.flush()
        return True, None, log_lines

    try:
        try:
            with _connect() as conn:
                _ensure_schema(conn)

                rows = _q_all(
                    conn,
                    "SELECT DISTINCT post_id FROM stages WHERE stage = 'summariser'",
                    tuple(),
                )
                summarised: set[str] = {row["post_id"] for row in rows}

                if only_new:
                    post_rows = _q_all(conn, "SELECT post_id FROM posts", tuple())
                    existing_posts = {row["post_id"] for row in post_rows if row["post_id"]}
                else:
                    _backfill_interest_scores(conn)
        except Exception as ex:
            _job_update_fields(job_id, status="error", error=str(ex), ended_at=time.time())
            with JOB_INDEX_LOCK:
                if ACTIVE_JOB_ID == job_id:
                    ACTIVE_JOB_ID = None
            return

        backlog_total = len(backlog)
        if backlog_total:
            _job_append_log(job_id, f"→ fixing {backlog_total} unsummarised backlog post(s)")
        for index, item in enumerate(backlog, start=1):
            if not _ensure_not_cancelled():
                return

            pid = (item.get("post_id") or "").strip()
            title = (item.get("title") or "").strip()
            remaining_backlog = max(backlog_total - index, 0)
            display_pid = pid or "(missing post id)"
            display_title = title[:80]
            _job_update_fields(
                job_id,
                phase=f"Fixing backlog {index}/{backlog_total}",
                current=f"Backlog: {display_pid} — {display_title} ({remaining_backlog} remaining)",
            )
            if not pid:
                _job_increment(job_id, "done", 1)
                continue

            if pid in summarised:
                _job_append_log(job_id, f"↷ backlog {pid} already summarised; skipping")
                _job_increment(job_id, "done", 1)
                continue

            _job_append_log(job_id, f"→ backlog summarising {pid}")

            success, error_text, log_lines = _stream_summariser(pid, title)
            if success:
                summarised.add(pid)
                _job_append_log(job_id, f"✓ summarised backlog {pid}")
                try:
                    with _connect() as convo_conn:
                        convo_payload = _ingest_conversation_hub(convo_conn, pid)
                    if convo_payload:
                        tickers_txt = ",".join(convo_payload.get("tickers") or [])
                        if tickers_txt:
                            _job_append_log(job_id, f"✓ convo hub backlog {pid} [{tickers_txt}]")
                except Exception as exc:  # noqa: BLE001
                    logging.exception("conversation hub ingest failed for backlog %s", pid)
                    _job_append_log(job_id, f"✗ convo hub backlog failed {pid}: {exc}")
                try:
                    with _connect() as interest_conn:
                        if _should_run_interest(interest_conn, pid):
                            _interest_schedule(1)
                            _interest_before_run("Interest backlog", pid, title)
                            _run_interest_for_post(pid, title)
                            _interest_after_run()
                except Exception as exc:  # noqa: BLE001
                    logging.exception("interest scoring check failed for backlog %s", pid)
                    _job_append_log(job_id, f"⚠ interest backlog check failed {pid}: {exc}")
            else:
                reason = error_text or (log_lines[-1] if log_lines else "Unknown summariser failure")
                log_tail = "; ".join(log_lines[-5:]) if log_lines else ""
                _job_append_log(job_id, f"✗ summariser failed backlog {pid}; skipping")
                _record_council_failure(
                    "summariser",
                    pid,
                    title,
                    f"{reason}" if not log_tail else f"{reason} | log_tail={log_tail}",
                    context=(
                        f"backlog_index={index}/{backlog_total}, remaining={remaining_backlog}, "
                        f"display_title={display_title}"
                    ),
                )

            _job_increment(job_id, "done", 1)

            if not _ensure_not_cancelled():
                return

        with _connect() as conn:
            _ensure_schema(conn)

            rows = _q_all(conn, "SELECT DISTINCT post_id FROM stages WHERE stage = 'summariser'", tuple())
            summarised.update(row["post_id"] for row in rows)

            if oracle_online:
                _job_update_fields(job_id, oracle_status="warmup")
                _run_oracle_loop(conn, summarised)
                return

            seen_csv: set[str] = set()

            if backlog:
                _job_update_fields(job_id, phase="processing snapshot", current="loading CSV snapshot")

            csv_processed = 0
            with snap_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if not _ensure_not_cancelled():
                        return

                    pid = (row.get("post_id") or "").strip()
                    if not pid or pid in seen_csv:
                        continue
                    seen_csv.add(pid)

                    if pid in summarised:
                        continue

                    if only_new and pid in existing_posts:
                        continue

                    title = row.get("title") or ""
                    csv_processed += 1
                    remaining_csv = max(csv_total - csv_processed, 0)
                    display_title = title[:80]
                    _job_update_fields(
                        job_id,
                        phase=f"Processing snapshot {csv_processed}/{csv_total}",
                        current=f"Processing: {pid} — {display_title} ({remaining_csv} remaining)",
                    )
                    _job_update_fields(
                        job_id,
                        phase=f"Summarising snapshot {csv_processed}/{csv_total}",
                        current=f"Summarising: {pid} — {display_title} ({remaining_csv} remaining)",
                    )

                    post_vals = {
                        "post_id": pid,
                        "platform": row.get("platform"),
                        "source": row.get("source"),
                        "url": row.get("url"),
                        "title": title,
                        "author": row.get("author"),
                        "scraped_at": row.get("scraped_at"),
                        "posted_at": row.get("posted_at") or row.get("scraped_at"),
                        "score": (int(row["score"]) if (row.get("score") or "").isdigit() else None),
                        "text": row.get("text"),
                    }
                    extras_vals = {
                        "final_url": row.get("final_url"),
                        "fetch_status": row.get("fetch_status"),
                        "domain": row.get("domain"),
                    }

                    _upsert_post_row(conn, post_vals)
                    _upsert_extras(conn, pid, extras_vals)

                    if only_new:
                        existing_posts.add(pid)

                    _job_update_fields(
                        job_id,
                        phase=f"Summarising snapshot {csv_processed}/{csv_total}",
                        current=f"Summarising: {pid} — {display_title} ({remaining_csv} remaining)",
                    )
                    _job_append_log(job_id, f"→ summarising {pid}")

                    success, error_text, log_lines = _stream_summariser(pid, title)
                    if success:
                        summarised.add(pid)
                        _job_append_log(job_id, f"✓ summarised {pid}")
                        try:
                            convo_payload = _ingest_conversation_hub(conn, pid)
                            if convo_payload:
                                tickers_txt = ",".join(convo_payload.get("tickers") or [])
                                if tickers_txt:
                                    _job_append_log(job_id, f"✓ convo hub {pid} [{tickers_txt}]")
                        except Exception as exc:  # noqa: BLE001
                            logging.exception("conversation hub ingest failed for %s", pid)
                            _job_append_log(job_id, f"✗ convo hub failed {pid}: {exc}")
                        interest_result: Optional[Dict[str, Any]] = None
                        try:
                            if _should_run_interest(conn, pid):
                                _interest_schedule(1)
                                _interest_before_run("Interest snapshot", pid, title)
                                interest_result = _run_interest_for_post(pid, title)
                                _interest_after_run()
                        except Exception as exc:  # noqa: BLE001
                            logging.exception("interest scoring check failed for %s", pid)
                            _job_append_log(job_id, f"⚠ interest check failed {pid}: {exc}")
                        interest_score_val: Optional[float] = None
                        if interest_result is not None:
                            raw_score = interest_result.get("interest_score")
                            try:
                                interest_score_val = float(raw_score) if raw_score is not None else None
                            except (TypeError, ValueError):
                                interest_score_val = None
                        _record_new_post(pid, title, interest_score_val)
                    else:
                        reason = error_text or (log_lines[-1] if log_lines else "Unknown summariser failure")
                        log_tail = "; ".join(log_lines[-5:]) if log_lines else ""
                        _job_append_log(job_id, f"✗ summariser failed {pid}; skipping")
                        _record_council_failure(
                            "summariser",
                            pid,
                            title,
                            f"{reason}" if not log_tail else f"{reason} | log_tail={log_tail}",
                            context=(
                                f"csv_index={csv_processed}/{csv_total}, remaining={remaining_csv}, "
                                f"display_title={display_title}"
                            ),
                        )

                    _job_increment(job_id, "done", 1)

                    if not _ensure_not_cancelled():
                        return

        _job_update_fields(job_id, status="done", ended_at=time.time(), phase="", current="")
    except Exception as ex:
        _job_update_fields(job_id, status="error", error=str(ex), ended_at=time.time())
    finally:
        if snap_path is not None:
            try:
                snap_path.unlink(missing_ok=True)
            except Exception:
                pass
        with JOB_INDEX_LOCK:
            if ACTIVE_JOB_ID == job_id:
                ACTIVE_JOB_ID = None


def _worker_council_analysis(job_id: str) -> None:
    global ACTIVE_COUNCIL_JOB_ID
    job = _load_job(job_id)
    if not job:
        return

    queue = list(job.get("queue") or [])
    total = int(job.get("total") or len(queue))
    _job_update_fields(
        job_id,
        status="running",
        started_at=job.get("started_at") or time.time(),
        remaining=len(queue),
    )

    def _ensure_not_cancelled() -> bool:
        current = _load_job(job_id)
        if not current:
            return False
        if current.get("cancelled"):
            _job_append_log(job_id, "⚠ council analysis cancelled by user")
            _job_update_fields(
                job_id,
                status="cancelled",
                current="",
                ended_at=time.time(),
                remaining=max(int(current.get("remaining") or 0), 0),
                current_mode=None,
                current_eta_seconds=None,
                current_started_at=None,
                current_article_tokens=None,
                current_summary_tokens=None,
            )
            return False
        return True

    try:
        for index, entry in enumerate(queue):
            if not _ensure_not_cancelled():
                return

            post_id = str(entry.get("post_id") or "")
            title = str(entry.get("title") or "")
            score = entry.get("interest_score")
            mode = str(entry.get("mode") or "interest")
            score_txt = ""
            if isinstance(score, (int, float)):
                score_txt = f" {int(round(float(score)))}%"

            display_title = title[:120]
            remaining = max(total - index, 0)

            article_tokens = 0
            summary_tokens = 0
            try:
                article_tokens, summary_tokens = _article_summary_tokens(post_id)
            except Exception:  # noqa: BLE001
                logging.exception("failed to compute token counts for %s", post_id)

            eta_seconds: Optional[float] = None
            try:
                eta_seconds = COUNCIL_TIME_MODEL.predict(article_tokens, summary_tokens)
            except Exception:  # noqa: BLE001
                eta_seconds = None

            article_started_at = time.time()
            _job_update_fields(
                job_id,
                current=f"{post_id} — {display_title}",
                remaining=remaining,
                current_mode=mode,
                current_started_at=article_started_at,
                current_eta_seconds=eta_seconds,
                current_article_tokens=article_tokens,
                current_summary_tokens=summary_tokens,
            )

            prefix = "→ analysing"
            if mode == "repair":
                prefix = "→ repairing"
            _job_append_log(job_id, f"{prefix} {post_id}{score_txt} {display_title}".rstrip())

            try:
                clear_post_analysis(post_id)
            except HTTPException as exc:
                detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
                _record_council_error(post_id, "clear_analysis", "Clear Council Analysis", detail, job_id)
                _job_append_log(job_id, f"✗ clear failed {post_id}: {detail}")
                _job_update_fields(
                    job_id,
                    current_eta_seconds=None,
                    current_started_at=None,
                    current_article_tokens=None,
                    current_summary_tokens=None,
                    current_mode=None,
                )
                _job_increment(job_id, "done", 1)
                continue
            except Exception as exc:  # noqa: BLE001
                message = str(exc)
                _record_council_error(post_id, "clear_analysis", "Clear Council Analysis", message, job_id)
                _job_append_log(job_id, f"✗ clear failed {post_id}: {message}")
                _job_update_fields(
                    job_id,
                    current_eta_seconds=None,
                    current_started_at=None,
                    current_article_tokens=None,
                    current_summary_tokens=None,
                    current_mode=None,
                )
                _job_increment(job_id, "done", 1)
                continue

            stage_ok = True
            for stage in COUNCIL_ANALYSIS_SEQUENCE:
                if not _ensure_not_cancelled():
                    return
                label = _council_stage_label(stage)
                _job_append_log(job_id, f"→ {label} for {post_id}")
                err_output = ""
                try:
                    if stage == "research":
                        _run_research_pipeline(post_id)
                    else:
                        code, _out, err = _run_round_table(post_id, [stage], False, False)
                        if code != 0:
                            err_output = err or ""
                            raise RuntimeError(err or f"{stage} failed")
                except HTTPException as exc:
                    detail = _format_council_error_detail(exc.detail)
                    _record_council_error(post_id, stage, label, detail, job_id, log_excerpt=err_output)
                    _job_append_log(job_id, f"✗ {label} failed {post_id}: {detail}")
                    stage_ok = False
                    break
                except Exception as exc:  # noqa: BLE001
                    message = str(exc) or repr(exc)
                    _record_council_error(post_id, stage, label, message, job_id, log_excerpt=err_output)
                    _job_append_log(job_id, f"✗ {label} failed {post_id}: {message}")
                    stage_ok = False
                    break

            if stage_ok:
                _job_append_log(job_id, f"✓ council analysis completed {post_id}")
                try:
                    duration = time.time() - article_started_at
                    COUNCIL_TIME_MODEL.observe(article_tokens, summary_tokens, duration)
                except Exception:  # noqa: BLE001
                    logging.exception("failed to record duration for %s", post_id)

            _job_increment(job_id, "done", 1)
            if mode == "repair":
                _job_increment(job_id, "repairs_done", 1)
            remaining_after = max(total - (index + 1), 0)
            _job_update_fields(
                job_id,
                remaining=remaining_after,
                current_eta_seconds=None,
                current_started_at=None,
                current_article_tokens=None,
                current_summary_tokens=None,
            )

        _job_update_fields(
            job_id,
            status="done",
            ended_at=time.time(),
            current="",
            remaining=0,
            current_mode=None,
            current_eta_seconds=None,
            current_started_at=None,
            current_article_tokens=None,
            current_summary_tokens=None,
        )
    except Exception as exc:  # noqa: BLE001
        _job_update_fields(
            job_id,
            status="error",
            error=str(exc),
            ended_at=time.time(),
            current_mode=None,
            current_eta_seconds=None,
            current_started_at=None,
            current_article_tokens=None,
            current_summary_tokens=None,
        )
    finally:
        _job_update_fields(job_id, queue=[])
        with COUNCIL_JOB_LOCK:
            if ACTIVE_COUNCIL_JOB_ID == job_id:
                ACTIVE_COUNCIL_JOB_ID = None


class CouncilAnalysisStartRequest(BaseModel):
    interest_min: float = Field(0.0, ge=0.0, le=100.0)
    repair_missing: bool = False
    post_ids: Optional[List[str]] = None


@app.post("/api/council-analysis/start")
def start_council_analysis(payload: CouncilAnalysisStartRequest):
    global ACTIVE_COUNCIL_JOB_ID
    threshold = float(payload.interest_min)
    target_post_order: Dict[str, int] = {}
    target_post_ids: Optional[Set[str]] = None
    if payload.post_ids is not None:
        cleaned: List[str] = []
        for raw in payload.post_ids:
            pid = str(raw or "").strip()
            if not pid:
                continue
            if pid in target_post_order:
                continue
            target_post_order[pid] = len(cleaned)
            cleaned.append(pid)
        target_post_ids = set(cleaned)

    with COUNCIL_JOB_LOCK:
        if ACTIVE_COUNCIL_JOB_ID:
            active_job = _load_job(ACTIVE_COUNCIL_JOB_ID)
            if active_job and active_job.get("status") in {"queued", "running", "cancelling"}:
                raise HTTPException(
                    status_code=409,
                    detail={"job_id": ACTIVE_COUNCIL_JOB_ID, "status": active_job.get("status")},
                )
            ACTIVE_COUNCIL_JOB_ID = None

        job_id = str(uuid.uuid4())
        ACTIVE_COUNCIL_JOB_ID = job_id

    repair_queue: List[Dict[str, Any]] = []
    repair_ids: Set[str] = set()
    skipped_below_threshold = 0

    with _connect() as conn:
        _ensure_schema(conn)
        if payload.repair_missing:
            repair_rows = _q_all(
                conn,
                """
                SELECT
                  p.post_id,
                  COALESCE(p.title, '') AS title,
                  COALESCE(ci.interest_score, 0) AS interest_score,
                  COALESCE(p.posted_at, p.scraped_at, '') AS article_time
                FROM posts p
                LEFT JOIN (
                    SELECT DISTINCT post_id
                    FROM stages
                    WHERE stage = 'chairman'
                ) AS sc ON sc.post_id = p.post_id
                LEFT JOIN council_stage_interest ci ON ci.post_id = p.post_id
                WHERE sc.post_id IS NULL
                ORDER BY datetime(COALESCE(p.posted_at, p.scraped_at)) ASC,
                         p.post_id ASC
                """,
                tuple(),
            )
            for row in repair_rows:
                pid = str(row["post_id"] or "").strip()
                if not pid:
                    continue
                if target_post_ids is not None and pid not in target_post_ids:
                    continue
                raw_score = row["interest_score"]
                try:
                    score_val = float(raw_score if raw_score is not None else 0.0)
                except (TypeError, ValueError):
                    score_val = 0.0
                if score_val < threshold:
                    skipped_below_threshold += 1
                    continue
                entry: Dict[str, Any] = {
                    "post_id": pid,
                    "title": row["title"],
                    "interest_score": score_val,
                    "article_time": row["article_time"] or "",
                    "mode": "repair",
                }
                repair_queue.append(entry)
                repair_ids.add(pid)

        rows = _q_all(
            conn,
            """
            SELECT
              p.post_id,
              COALESCE(p.title, '') AS title,
              COALESCE(ci.interest_score, 0) AS interest_score,
              COALESCE(p.posted_at, p.scraped_at, '') AS article_time,
              CASE WHEN sc.post_id IS NOT NULL THEN 1 ELSE 0 END AS has_chairman
            FROM posts p
            JOIN council_stage_interest ci ON ci.post_id = p.post_id
            LEFT JOIN (
                SELECT DISTINCT post_id
                FROM stages
                WHERE stage = 'chairman'
            ) AS sc ON sc.post_id = p.post_id
            WHERE ci.interest_score IS NOT NULL
              AND LOWER(COALESCE(ci.status, '')) = 'ok'
              AND ci.interest_score >= ?
            ORDER BY datetime(COALESCE(p.posted_at, p.scraped_at)) ASC,
                     ci.interest_score DESC,
                     p.post_id ASC
            """,
            (threshold,),
        )

    interest_queue: List[Dict[str, Any]] = []
    skipped_with_chairman = 0
    for row in rows:
        pid = str(row["post_id"] or "").strip()
        if not pid or pid in repair_ids:
            continue
        if target_post_ids is not None and pid not in target_post_ids:
            continue
        if int(row["has_chairman"] or 0):
            skipped_with_chairman += 1
            continue
        raw_score = row["interest_score"]
        try:
            score_val = float(raw_score if raw_score is not None else 0.0)
        except (TypeError, ValueError):
            score_val = 0.0
        if score_val < threshold:
            skipped_below_threshold += 1
            continue
        interest_queue.append(
            {
                "post_id": pid,
                "title": row["title"],
                "interest_score": score_val,
                "article_time": row["article_time"] or "",
                "mode": "interest",
            }
        )

    repair_queue.sort(key=lambda entry: (entry.get("article_time") or "", entry.get("post_id")))
    interest_queue.sort(key=lambda entry: (entry.get("article_time") or "", entry.get("post_id")))
    queue: List[Dict[str, Any]] = repair_queue + interest_queue

    if target_post_ids is not None:
        queue = [entry for entry in queue if entry.get("post_id") in target_post_ids]
        if target_post_order:
            queue.sort(key=lambda entry: target_post_order.get(entry.get("post_id"), len(target_post_order)))

    total = len(queue)
    now = time.time()
    job_data: Dict[str, Any] = {
        "id": job_id,
        "type": "council_analysis",
        "status": "queued" if total > 0 else "done",
        "total": total,
        "done": 0,
        "queue": queue,
        "interest_min": threshold,
        "current": "",
        "remaining": total,
        "skipped_with_chairman": skipped_with_chairman,
        "skipped_below_threshold": skipped_below_threshold,
        "repairs_total": len(repair_queue),
        "repairs_done": 0,
        "repair_missing": bool(payload.repair_missing),
        "log_tail": [],
        "error": "",
        "started_at": None,
        "ended_at": None,
        "cancelled": False,
        "created_at": now,
        "updated_at": now,
        "current_mode": None,
        "current_eta_seconds": None,
        "current_started_at": None,
        "current_article_tokens": None,
        "current_summary_tokens": None,
    }

    _job_save(job_data)

    if total > 0:
        if target_post_ids is not None:
            summary_bits = [
                f"→ queued {total} selected post(s) for council analysis at ≥{int(round(threshold))}%"
            ]
        else:
            summary_bits = [
                f"→ queued {total} post(s) for council analysis at ≥{int(round(threshold))}%"
            ]
    else:
        summary_bits = [
            "→ no selected posts meet the council threshold"
            if target_post_ids is not None
            else "→ no posts meet the council threshold"
        ]
    if repair_queue:
        plural_repair = "s" if len(repair_queue) != 1 else ""
        summary_bits.append(f"(repairing {len(repair_queue)} missing verdict{plural_repair})")
    if skipped_with_chairman:
        plural = "s" if skipped_with_chairman != 1 else ""
        summary_bits.append(f"(skipped {skipped_with_chairman} post{plural} with chairman verdict)")
    if skipped_below_threshold:
        plural_threshold = "s" if skipped_below_threshold != 1 else ""
        summary_bits.append(
            f"(skipped {skipped_below_threshold} post{plural_threshold} below {int(round(threshold))}% interest)"
        )
    summary_line = " ".join(summary_bits)
    _job_append_log(job_id, summary_line)

    if total == 0:
        _job_update_fields(
            job_id,
            ended_at=now,
            remaining=0,
            current_mode=None,
            current_eta_seconds=None,
            current_started_at=None,
            current_article_tokens=None,
            current_summary_tokens=None,
        )
        with COUNCIL_JOB_LOCK:
            if ACTIVE_COUNCIL_JOB_ID == job_id:
                ACTIVE_COUNCIL_JOB_ID = None
        return {
            "ok": True,
            "job_id": job_id,
            "total": total,
            "interest_min": threshold,
            "skipped_with_chairman": skipped_with_chairman,
            "skipped_below_threshold": skipped_below_threshold,
            "repairs_total": len(repair_queue),
        }

    thread = threading.Thread(target=_worker_council_analysis, args=(job_id,), daemon=True)
    thread.start()

    return {
        "ok": True,
        "job_id": job_id,
        "total": total,
        "interest_min": threshold,
        "skipped_with_chairman": skipped_with_chairman,
        "skipped_below_threshold": skipped_below_threshold,
        "repairs_total": len(repair_queue),
    }


@app.get("/api/council-analysis/{job_id}")
def get_council_analysis(job_id: str):
    job = _load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job-not-found")

    remaining_val = job.get("remaining")
    msg_parts: List[str] = []
    if isinstance(remaining_val, (int, float)):
        remaining_int = max(int(remaining_val), 0)
        msg_parts.append(f"{remaining_int} remaining")
    current = str(job.get("current") or "").strip()
    if current:
        msg_parts.append(current)

    data = dict(job)
    data["message"] = " — ".join(part for part in msg_parts if part)
    return data


@app.get("/api/council-analysis/active")
def get_active_council_analysis():
    with COUNCIL_JOB_LOCK:
        job_id = ACTIVE_COUNCIL_JOB_ID

    if not job_id:
        return Response(status_code=204)

    job = _load_job(job_id)
    if not job:
        with COUNCIL_JOB_LOCK:
            if ACTIVE_COUNCIL_JOB_ID == job_id:
                ACTIVE_COUNCIL_JOB_ID = None
        raise HTTPException(status_code=404, detail="job-not-found")

    return {
        "job_id": job_id,
        "status": job.get("status"),
        "total": job.get("total"),
        "done": job.get("done"),
        "remaining": job.get("remaining"),
        "interest_min": job.get("interest_min"),
    }


@app.post("/api/council-analysis/{job_id}/stop")
def stop_council_analysis(job_id: str):
    job = _load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job-not-found")

    if job.get("status") in {"done", "error", "cancelled"}:
        return {"ok": True, "status": job.get("status")}

    def mutate(data: Dict[str, Any]) -> None:
        data["cancelled"] = True
        if data.get("status") in {"running", "queued"}:
            data["status"] = "cancelling"

    _job_mutate(job_id, mutate)
    _job_append_log(job_id, "⚠ cancellation requested")
    return {"ok": True, "status": "cancelling"}


@app.post("/api/council-analysis/erase-all")
def erase_all_council_analysis():
    with COUNCIL_JOB_LOCK:
        active_job_id = ACTIVE_COUNCIL_JOB_ID

    if active_job_id:
        job = _load_job(active_job_id)
        if job and job.get("status") in {"queued", "running", "cancelling"}:
            raise HTTPException(
                status_code=409,
                detail={"job_id": active_job_id, "status": job.get("status")},
            )

    with _connect() as conn:
        _ensure_schema(conn)
        placeholders = ",".join("?" for _ in COUNCIL_PRESERVE_STAGES)
        cur = conn.execute(
            f"""
            DELETE FROM stages
            WHERE stage NOT IN ({placeholders})
            """,
            COUNCIL_PRESERVE_STAGES,
        )
        deleted = int(cur.rowcount or 0)
        conn.commit()

        cleared = _strip_research_from_extras(conn)

    return {
        "ok": True,
        "deleted_stages": deleted,
        "cleared_research_posts": cleared,
    }


@app.get("/api/refresh-summaries/{job_id}")
def get_refresh_summaries(job_id: str):
    job = _load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job-not-found")
    # return a copy with computed message
    msg = ""
    if job.get("total", 0) > 0:
        msg = f'{job.get("phase", "")} {job.get("done", 0)}/{job.get("total", 0)} — {job.get("current", "")}'
    data = dict(job)
    data["message"] = msg.strip()
    return data


@app.get("/api/refresh-summaries/active")
def get_active_refresh_summaries():
    global ACTIVE_JOB_ID
    with JOB_INDEX_LOCK:
        job_id = ACTIVE_JOB_ID
    if not job_id:
        return Response(status_code=204)

    job = _load_job(job_id)
    if not job:
        with JOB_INDEX_LOCK:
            if ACTIVE_JOB_ID == job_id:
                ACTIVE_JOB_ID = None
        raise HTTPException(status_code=404, detail="job-not-found")

    return {"job_id": job_id, "status": job.get("status"), "total": job.get("total"), "done": job.get("done")}


@app.post("/api/refresh-summaries/{job_id}/stop")
def stop_refresh_summaries(job_id: str):
    job = _load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job-not-found")

    if job.get("status") in {"done", "error", "cancelled"}:
        return {"ok": True, "status": job.get("status")}

    def mutate(data: Dict[str, Any]) -> None:
        data["cancelled"] = True
        if data.get("status") in {"running", "queued"}:
            data["status"] = "cancelling"

    _job_mutate(job_id, mutate)
    _job_append_log(job_id, "⚠ cancellation requested")

    return {"ok": True, "status": "cancelling"}


@app.get("/api/posts/calendar")
def posts_calendar(
    year: int = Query(..., ge=1970, le=3000),
    month: int = Query(..., ge=1, le=12),
) -> PostsCalendarResponse:
    start = datetime(year, month, 1, tzinfo=timezone.utc)
    if month == 12:
        end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        end = datetime(year, month + 1, 1, tzinfo=timezone.utc)

    start_iso = start.isoformat().replace("+00:00", "Z")
    end_iso = end.isoformat().replace("+00:00", "Z")

    sql = """
        SELECT
          DATE(datetime(COALESCE(p.posted_at, p.scraped_at))) AS day,
          COUNT(*) AS count,
          SUM(
            CASE
              WHEN EXISTS (
                SELECT 1 FROM stages s
                WHERE s.post_id = p.post_id AND s.stage = 'chairman'
              ) THEN 1
              ELSE 0
            END
          ) AS analysed_count
        FROM posts p
        WHERE datetime(COALESCE(p.posted_at, p.scraped_at)) >= ?
          AND datetime(COALESCE(p.posted_at, p.scraped_at)) < ?
        GROUP BY day
        ORDER BY day
    """

    days: List[CalendarDay] = []
    with _connect() as conn:
        rows = _q_all(conn, sql, (start_iso, end_iso))
        for row in rows:
            day = row["day"]
            if not day:
                continue
            count = int(row["count"] or 0)
            analysed = int(row["analysed_count"] or 0)
            if count <= 0 and analysed <= 0:
                continue
            days.append(
                CalendarDay(
                    date=str(day),
                    count=count,
                    analysed_count=max(analysed, 0),
                )
            )

    return PostsCalendarResponse(days=days)


def _list_posts_from_db(
    *,
    q: Optional[str],
    platform: Optional[str],
    source: Optional[str],
    date_from: Optional[str],
    date_to: Optional[str],
    interest_min: Optional[float],
    page: int,
    page_size: int,
) -> Dict[str, Any]:
    offset = (page - 1) * page_size

    where = ["1=1"]
    params: List[Any] = []

    if q:
        # minimal LIKE search; replace with FTS later if desired
        where.append("(p.title LIKE ? OR p.text LIKE ?)")
        like = f"%{q}%"
        params.extend([like, like])

    if platform:
        where.append("p.platform = ?")
        params.append(platform)

    if source:
        where.append("p.source = ?")
        params.append(source)

    if date_from:
        where.append("(p.posted_at >= ? OR p.scraped_at >= ?)")
        params.extend([date_from, date_from])

    if date_to:
        where.append("(p.posted_at < ? OR p.scraped_at < ?)")
        params.extend([date_to, date_to])

    if isinstance(interest_min, (int, float)):
        where.append(
            "EXISTS (SELECT 1 FROM council_stage_interest ci WHERE ci.post_id = p.post_id AND ci.interest_score IS NOT NULL"
            " AND LOWER(COALESCE(ci.status, '')) = 'ok' AND ci.interest_score >= ?)"
        )
        params.append(float(interest_min))

    where_sql = " AND ".join(where)

    sql_items = f"""
        SELECT
          p.post_id, p.title, p.platform, p.source, p.url,
          p.scraped_at, p.posted_at,
          substr(replace(replace(p.text, char(10), ' '), char(13), ' '), 1, 400) AS preview,
          EXISTS(
            SELECT 1 FROM stages s
            WHERE s.post_id = p.post_id AND s.stage = 'summariser'
          ) AS has_summary,
          EXISTS(
            SELECT 1 FROM stages s
            WHERE s.post_id = p.post_id AND s.stage IN (
              'direction','moderator','researcher','technical_research','sentiment_research','conversation_hub','chairman'
            )
          ) AS has_analysis
          , ci.status AS interest_status
          , ci.ticker AS interest_ticker
          , ci.interest_score AS interest_score
          , ci.interest_label AS interest_label
          , ci.interest_why AS interest_why
          , ci.council_recommended AS interest_recommended
          , ci.council_priority AS interest_priority
          , ci.error_code AS interest_error_code
          , ci.error_message AS interest_error_message
          , ci.created_at AS interest_created_at
          , ci.debug_json AS interest_debug_json
        FROM posts p
        LEFT JOIN council_stage_interest ci ON ci.post_id = p.post_id
        WHERE {where_sql}
        ORDER BY datetime(COALESCE(p.scraped_at, p.posted_at)) DESC
        LIMIT ? OFFSET ?
    """
    sql_count = f"SELECT COUNT(*) AS n FROM posts p WHERE {where_sql}"

    with _connect() as conn:
        total_row = _q_one(conn, sql_count, tuple(params))
        total = int(total_row["n"] if total_row and total_row["n"] is not None else 0)
        rows = _q_all(conn, sql_items, tuple(params + [page_size, offset]))

        items: List[Dict[str, Any]] = []
        for r in rows:
            post_id = r["post_id"]

            # markets from latest 'entity'
            entity = _latest_stage_payload(conn, post_id, "entity")
            markets = _build_markets_from_entity(entity)

            # signal from 'direction' or fallback 'moderator'
            direction = _latest_stage_payload(conn, post_id, "direction") or {}
            if not direction:
                direction = _latest_stage_payload(conn, post_id, "moderator") or {}
            signal = _build_signal_from_dir_or_mod(direction)

            summariser_payload = _latest_stage_payload(conn, post_id, "summariser") or {}
            chairman_payload = _latest_stage_payload(conn, post_id, "chairman") or {}
            summary_bullets: List[str] = []
            assets_mentioned: List[Dict[str, Optional[str]]] = []
            spam_likelihood_pct = 0
            spam_why = ""
            chairman_plain = ""
            chairman_direction = ""
            if isinstance(summariser_payload, dict):
                raw_bullets = summariser_payload.get("summary_bullets") or []
                for bullet in raw_bullets:
                    if bullet is None:
                        continue
                    text = str(bullet).strip()
                    if text:
                        summary_bullets.append(text)

                raw_assets = summariser_payload.get("assets_mentioned") or []
                for asset in raw_assets:
                    if not isinstance(asset, dict):
                        continue
                    clean_asset: Dict[str, Optional[str]] = {}
                    for key in ("ticker", "name_or_description", "exchange_or_market"):
                        value = asset.get(key)
                        clean_asset[key] = None if value is None else str(value)
                    assets_mentioned.append(clean_asset)

                raw_spam_pct = summariser_payload.get("spam_likelihood_pct")
                spam_likelihood_pct = _parse_spam_likelihood(raw_spam_pct)

                raw_spam_why = summariser_payload.get("spam_why")
                if not (isinstance(raw_spam_why, str) and raw_spam_why.strip()):
                    raw_spam_why = summariser_payload.get("spam_reasons")
            spam_why = _parse_spam_reason(raw_spam_why)

            if isinstance(chairman_payload, dict):
                raw_plain = chairman_payload.get("plain_english_result")
                if isinstance(raw_plain, str):
                    chairman_plain = raw_plain.strip()
                final_metrics = chairman_payload.get("final_metrics")
                if isinstance(final_metrics, dict):
                    raw_direction = final_metrics.get("implied_direction")
                    if isinstance(raw_direction, str):
                        chairman_direction = raw_direction.strip()

            has_summary = bool(r["has_summary"])
            has_analysis = bool(r["has_analysis"])

            interest_record = _build_interest_record(
                status=r["interest_status"],
                ticker=r["interest_ticker"],
                score=r["interest_score"],
                label=r["interest_label"],
                why=r["interest_why"],
                recommended=r["interest_recommended"],
                priority=r["interest_priority"],
                calculated_at=r["interest_created_at"],
                error_code=r["interest_error_code"],
                error_message=r["interest_error_message"],
                debug_json=r["interest_debug_json"],
            )

            item = PostListItem(
                post_id=post_id,
                title=r["title"] or "",
                platform=r["platform"] or "",
                source=r["source"] or "",
                url=r["url"] or "",
                scraped_at=r["scraped_at"],
                posted_at=r["posted_at"],
                preview=r["preview"] or "",
                markets=markets,
                signal=signal,
                has_summary=has_summary,     # ← now coming straight from SQL
                has_analysis=has_analysis,   # ← now coming straight from SQL
                summary_bullets=summary_bullets,
                assets_mentioned=assets_mentioned,
                spam_likelihood_pct=spam_likelihood_pct,
                spam_why=spam_why,
                interest=interest_record,
                chairman_plain_english=chairman_plain or None,
                chairman_direction=chairman_direction or None,
            )
            items.append(item.model_dump())

    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@app.get("/api/posts")
def list_posts(
    q: Optional[str] = Query(None, description="Search in title/text"),
    platform: Optional[str] = None,
    source: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    interest_min: Optional[float] = Query(None, ge=0.0, le=100.0),
    page: int = Query(1, ge=1),
    page_size: int = Query(PAGE_SIZE_DEFAULT, ge=1, le=500),
):
    if not isinstance(q, str):
        q = None
    if not isinstance(platform, str):
        platform = None
    if not isinstance(source, str):
        source = None
    if not isinstance(date_from, str):
        date_from = None
    if not isinstance(date_to, str):
        date_to = None

    try:
        return _list_posts_from_db(
            q=q,
            platform=platform,
            source=source,
            date_from=date_from,
            date_to=date_to,
            interest_min=float(interest_min) if isinstance(interest_min, (int, float)) else None,
            page=page,
            page_size=page_size,
        )
    except sqlite3.DatabaseError as exc:
        LOGGER.error("list-posts-db-error: %s", exc)
        raise HTTPException(status_code=500, detail="database-unavailable") from exc

def _get_post_from_db(post_id: str) -> Dict[str, Any]:
    with _connect() as conn:
        p = _q_one(conn, "SELECT * FROM posts WHERE post_id = ?", (post_id,))
        if not p:
            raise HTTPException(status_code=404, detail="post-not-found")

        # extras
        e = _q_one(conn, "SELECT payload_json FROM post_extras WHERE post_id = ?", (post_id,))
        extras = {}
        if e and e["payload_json"]:
            try:
                extras = json.loads(e["payload_json"])
            except Exception:
                extras = {}

        # stages
        stg_rows = _q_all(
            conn,
            """
            SELECT stage, payload, created_at
            FROM stages
            WHERE post_id = ?
            ORDER BY datetime(created_at) DESC
            """,
            (post_id,),
        )
        stages: Dict[str, Any] = {}
        for row in stg_rows:
            stage_name = row["stage"]
            if stage_name in stages:
                continue
            payload = None
            if row["payload"]:
                try:
                    payload = json.loads(row["payload"])
                except Exception:
                    payload = {"_raw": row["payload"], "_error": "json-parse-failed"}
            stages[stage_name] = payload

        post = PostRow(**{k: p[k] for k in p.keys()})

        interest_row = _q_one(
            conn,
            """
            SELECT status, ticker, interest_score, interest_label, interest_why,
                   council_recommended, council_priority, error_code, error_message,
                   created_at, debug_json
            FROM council_stage_interest
            WHERE post_id = ?
            LIMIT 1
            """,
            (post_id,),
        )
        interest_record = None
        if interest_row:
            interest_record = _build_interest_record(
                status=interest_row["status"],
                ticker=interest_row["ticker"],
                score=interest_row["interest_score"],
                label=interest_row["interest_label"],
                why=interest_row["interest_why"],
                recommended=interest_row["council_recommended"],
                priority=interest_row["council_priority"],
                calculated_at=interest_row["created_at"],
                error_code=interest_row["error_code"],
                error_message=interest_row["error_message"],
                debug_json=interest_row["debug_json"],
            )

        return PostDetail(post=post, extras=extras, stages=stages, interest=interest_record).model_dump()


@app.get("/api/posts/{post_id:path}")
def get_post(post_id: str):
    try:
        return _get_post_from_db(post_id)
    except sqlite3.DatabaseError as exc:
        LOGGER.error("get-post-db-error: %s", exc)
        raise HTTPException(status_code=500, detail="database-unavailable") from exc

# ---------- Refresh from CSV ----------

def _csv_find_row(post_id: str) -> Optional[Dict[str, Any]]:
    # raw_posts_log.csv columns (typical):
    # scraped_at,platform,source,post_id,url,title,text,final_url,fetch_status,domain,posted_at?,author?,score?
    if not CSV_PATH.exists():
        return None
    import csv
    with CSV_PATH.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("post_id") or "").strip() == post_id:
                return row
    return None

@app.post("/api/posts/{post_id:path}/refresh-from-csv")
def refresh_post_from_csv(post_id: str):
    row = _csv_find_row(post_id)
    if not row:
        raise HTTPException(status_code=404, detail="not-found-in-csv")

    # Map into posts + post_extras upserts
    post_vals = {
        "post_id": row.get("post_id") or post_id,
        "platform": row.get("platform"),
        "source": row.get("source"),
        "url": row.get("url"),
        "title": row.get("title"),
        "author": row.get("author"),
        "scraped_at": row.get("scraped_at"),
        "posted_at": row.get("posted_at") or row.get("scraped_at"),
        "score": (int(row["score"]) if (row.get("score") or "").isdigit() else None),
        "text": row.get("text"),
    }
    extras_vals = {
        "final_url": row.get("final_url"),
        "fetch_status": row.get("fetch_status"),
        "domain": row.get("domain"),
    }
    with _connect() as conn:
        # upsert posts
        conn.execute("""
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
        """, post_vals)
        conn.commit()

        # upsert extras into post_extras(payload_json)
        payload_json = json.dumps(extras_vals, ensure_ascii=False)
        conn.execute("""
            INSERT INTO post_extras (post_id, payload_json)
            VALUES (?, ?)
            ON CONFLICT(post_id) DO UPDATE SET payload_json=excluded.payload_json
        """, (post_id, payload_json))
        conn.commit()

    return {"ok": True, "updated": True}


@app.post("/api/posts/{post_id:path}/clear-analysis")
def clear_post_analysis(post_id: str):
    with _connect() as conn:
        _ensure_schema(conn)

        exists = _q_one(conn, "SELECT 1 FROM posts WHERE post_id = ?", (post_id,))
        if not exists:
            raise HTTPException(status_code=404, detail="post-not-found")

        placeholders = ",".join("?" for _ in COUNCIL_PRESERVE_STAGES)
        cur = conn.execute(
            f"""
            DELETE FROM stages
            WHERE post_id = ?
              AND stage NOT IN ({placeholders})
            """,
            (post_id, *COUNCIL_PRESERVE_STAGES),
        )
        deleted = int(cur.rowcount or 0)
        conn.commit()

        removed_research = _strip_research_from_extras(conn, post_id) > 0

    return {"ok": True, "deleted_stages": deleted, "removed_research": removed_research}

class RefreshSummariesStartRequest(BaseModel):
    mode: Literal["full", "new_only"] = "full"
    collect_new_posts: bool = False
    oracle_online: bool = False
    oracle_base_url: Optional[str] = None
    interest_threshold: Optional[float] = None


@app.post("/api/refresh-summaries/start")
def start_refresh_summaries(
    payload: Optional[RefreshSummariesStartRequest] = Body(default=None),
):
    """
    Snapshot CSV, ingest any posts not in SQL, run summariser for each.
    Returns a job_id; poll /api/refresh-summaries/{job_id} for progress.
    """
    global ACTIVE_JOB_ID

    requested_mode = payload.mode if payload else "full"
    only_new = requested_mode == "new_only"
    collect_new_posts = bool(payload.collect_new_posts) if payload else False
    oracle_online = bool(payload.oracle_online) if payload else False
    oracle_base_url = (payload.oracle_base_url or ORACLE_DEFAULT_BASE_URL).strip() if payload else ORACLE_DEFAULT_BASE_URL
    interest_threshold = 0.0
    if payload and payload.interest_threshold is not None:
        try:
            interest_threshold = float(payload.interest_threshold)
        except (TypeError, ValueError):
            interest_threshold = 0.0
    interest_threshold = max(0.0, min(100.0, interest_threshold))
    if oracle_online and not oracle_base_url:
        raise HTTPException(status_code=400, detail="oracle-base-url-required")

    if not oracle_online and not CSV_PATH.exists():
        raise HTTPException(status_code=404, detail="csv-not-found")

    job_id = str(uuid.uuid4())

    with JOB_INDEX_LOCK:
        if ACTIVE_JOB_ID:
            active_job = _load_job(ACTIVE_JOB_ID)
            if active_job and active_job.get("status") in {"queued", "running"}:
                raise HTTPException(status_code=409, detail={"job_id": ACTIVE_JOB_ID, "status": active_job.get("status")})
            ACTIVE_JOB_ID = None
        ACTIVE_JOB_ID = job_id

    snapshot_path: Optional[Path] = None
    if not oracle_online:
        snapshot_path = CSV_PATH.with_suffix(f".snapshot-{int(time.time())}.csv")
        try:
            shutil.copyfile(CSV_PATH, snapshot_path)
        except Exception as ex:
            with JOB_INDEX_LOCK:
                if ACTIVE_JOB_ID == job_id:
                    ACTIVE_JOB_ID = None
            raise HTTPException(status_code=500, detail=f"snapshot-failed: {ex}")
    else:
        try:
            if not ORACLE_CURSOR_PATH.exists():
                _save_oracle_cursor(_default_oracle_cursor())
        except Exception:
            pass

    backlog: List[Dict[str, Any]] = []
    summarised_ids: set[str] = set()
    existing_post_ids: set[str] = set()
    try:
        with _connect() as conn:
            _ensure_schema(conn)
            if not only_new:
                backlog_rows = _q_all(
                    conn,
                    """
                    SELECT p.post_id, COALESCE(p.title, '') AS title
                    FROM posts p
                    WHERE NOT EXISTS (
                        SELECT 1
                        FROM stages s
                        WHERE s.post_id = p.post_id
                          AND s.stage = 'summariser'
                    )
                    ORDER BY datetime(COALESCE(p.scraped_at, p.posted_at)) ASC
                    """,
                    tuple(),
                )
                backlog = [
                    {"post_id": row["post_id"], "title": row["title"]}
                    for row in backlog_rows
                    if row["post_id"]
                ]
            summarised_rows = _q_all(
                conn,
                "SELECT DISTINCT post_id FROM stages WHERE stage = 'summariser'",
                tuple(),
            )
            summarised_ids = {row["post_id"] for row in summarised_rows}
            if only_new:
                post_rows = _q_all(
                    conn,
                    "SELECT post_id FROM posts",
                    tuple(),
                )
                existing_post_ids = {row["post_id"] for row in post_rows if row["post_id"]}
    except Exception as ex:
        with JOB_INDEX_LOCK:
            if ACTIVE_JOB_ID == job_id:
                ACTIVE_JOB_ID = None
        if snapshot_path is not None:
            try:
                snapshot_path.unlink(missing_ok=True)
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=f"db-prepare-failed: {ex}")

    backlog_ids = {item["post_id"] for item in backlog}
    total_backlog = len(backlog_ids)
    csv_total = 0
    if snapshot_path is not None:
        csv_seen: set[str] = set()
        try:
            with snapshot_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    pid = (row.get("post_id") or "").strip()
                    if not pid or pid in csv_seen:
                        continue
                    csv_seen.add(pid)
                    if pid in summarised_ids or pid in backlog_ids:
                        continue
                    if only_new and pid in existing_post_ids:
                        continue
                    csv_total += 1
        except Exception as ex:
            with JOB_INDEX_LOCK:
                if ACTIVE_JOB_ID == job_id:
                    ACTIVE_JOB_ID = None
            try:
                snapshot_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise HTTPException(status_code=500, detail=f"snapshot-read-failed: {ex}")

    total = total_backlog + csv_total
    now = time.time()
    warmup_cutoff_iso = datetime.now(timezone.utc).isoformat()
    job: Dict[str, Any] = {
        "id": job_id,
        "status": "queued",  # queued | running | done | error | cancelled
        "snapshot": str(snapshot_path) if snapshot_path is not None else None,
        "backlog": backlog,
        "mode": requested_mode,
        "only_new": only_new,
        "collect_new_posts": collect_new_posts,
        "csv_total": csv_total,
        "total": total,
        "done": 0,
        "current": "",
        "phase": "",
        "log_tail": [],
        "error": "",
        "started_at": None,
        "ended_at": None,
        "cancelled": False,
        "created_at": now,
        "updated_at": now,
        "new_posts": [],
        "oracle_online": oracle_online,
        "oracle_base_url": oracle_base_url,
        "oracle_status": "connecting" if oracle_online else "offline",
        "oracle_cursor": _load_oracle_cursor() if oracle_online else None,
        "oracle_poll_seconds": None,
        "oracle_idle_since": None,
        "oracle_interest_threshold": interest_threshold,
        "oracle_handles_council": bool(oracle_online),
        "oracle_progress_total": 0,
        "oracle_progress_index": 0,
        "oracle_progress_stage": "",
        "oracle_progress_message": "",
        "oracle_warmup_cutoff": warmup_cutoff_iso,
    }
    _job_save(job)

    worker = _worker_refresh_summaries
    t = threading.Thread(target=worker, args=(job_id,), daemon=True)
    t.start()

    return {"ok": True, "job_id": job_id, "total": total}

# ---------- Run stage(s) (single) ----------


def _round_table_kwargs(
    *,
    post_id: str,
    stages: List[str],
    refresh_from_csv: bool,
    echo_post: bool,
    title: str = "",
    text: str = "",
    log_callback: Optional[Callable[[str], None]] = None,
):
    return dict(
        post_id=post_id,
        title=title,
        text=text,
        platform="manual",
        source="manual",
        url="",
        model=ROUND_TABLE_MODEL,
        host=ROUND_TABLE_HOST,
        verbose=ROUND_TABLE_VERBOSE,
        dump_dir=ROUND_TABLE_DUMP_DIR,
        timeout=ROUND_TABLE_TIMEOUT,
        stages=stages,
        autofill_deps=ROUND_TABLE_AUTOFILL,
        evidence_csv=str(CSV_PATH),
        refresh_from_csv=refresh_from_csv,
        evidence_lookback_days=ROUND_TABLE_EVIDENCE_LOOKBACK,
        max_evidence_per_claim=ROUND_TABLE_MAX_EVIDENCE,
        pretty_print_stage_output=ROUND_TABLE_PRETTY,
        echo_post=echo_post,
        log_callback=log_callback,
    )


def _run_round_table(
    post_id: str,
    stages: List[str],
    refresh_from_csv: bool,
    echo_post: bool,
    *,
    title: str = "",
    text: str = "",
) -> Tuple[int, str, str]:
    if not callable(run_stages_for_post):
        hint = f": {ROUND_TABLE_IMPORT_ERROR}" if ROUND_TABLE_IMPORT_ERROR else ""
        raise RuntimeError(f"round-table-unavailable{hint}")

    log_lines: List[str] = []

    def _collect(line: str) -> None:
        clean = (line or "").rstrip()
        if clean:
            log_lines.append(clean)

    writer = _CallbackWriter(_collect)
    try:
        with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
            results = run_stages_for_post(**_round_table_kwargs(
                post_id=post_id,
                stages=stages,
                refresh_from_csv=refresh_from_csv,
                echo_post=echo_post,
                title=title,
                text=text,
                log_callback=_collect,
            ))
    except Exception as exc:
        writer.flush()
        logging.exception("round_table.run_stages_for_post failed for %s", post_id)
        log_lines.append(f"✗ round-table failed: {exc}")
        stderr = _trim("\n".join(log_lines), STDIO_TRIM)
        return 1, "", stderr

    writer.flush()
    stdout_payload = json.dumps({"post_id": post_id, "stages": list(results.keys())}, ensure_ascii=False)
    stdout = _trim(stdout_payload, STDIO_TRIM)
    stderr = _trim("\n".join(log_lines), STDIO_TRIM)
    return 0, stdout, stderr


def _run_research_pipeline(post_id: str) -> ResearchPayload:
    try:
        import researcher  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"researcher-import-failed: {exc}") from exc

    conversation_payload: Dict[str, Any] = {}
    with _connect() as conn:
        post_row = _q_one(conn, "SELECT * FROM posts WHERE post_id = ?", (post_id,))
        if not post_row:
            raise HTTPException(status_code=404, detail="post-not-found")

        summariser_payload = _latest_stage_payload(conn, post_id, "summariser")
        bull_payload = _latest_stage_payload(conn, post_id, "for") or {}
        bear_payload = _latest_stage_payload(conn, post_id, "against") or {}
        if not summariser_payload:
            raise HTTPException(status_code=409, detail="missing-prerequisites")

        claims_payload = _latest_stage_payload(conn, post_id, "claims") or {}
        context_payload = _latest_stage_payload(conn, post_id, "context") or {}
        direction_payload = _latest_stage_payload(conn, post_id, "direction") or {}
        entity_payload = _latest_stage_payload(conn, post_id, "entity") or {}
        conversation_payload = _latest_stage_payload(conn, post_id, "conversation_hub") or {}

        article_time = _normalize_article_time(post_row["posted_at"] or post_row["scraped_at"])
        summary_bullets = _clean_strings(summariser_payload.get("summary_bullets"))
        claims = _extract_claim_texts(claims_payload)
        context_bullets = _clean_strings(context_payload.get("context_bullets"))
        bull_points = _clean_strings(bull_payload.get("bull_points"))
        bear_points = _clean_strings(bear_payload.get("bear_points"))
        direction_est = _direction_estimate(direction_payload, summariser_payload)
        primary_ticker = _primary_ticker(summariser_payload, entity_payload)

        candidate_tickers: List[str] = []
        if primary_ticker:
            candidate_tickers.append(primary_ticker)
        convo_candidates = conversation_payload.get("tickers")
        if isinstance(convo_candidates, list):
            candidate_tickers.extend(str(t) for t in convo_candidates if isinstance(t, str))
        candidate_tickers.extend(_extract_convo_tickers(post_row, summariser_payload))

        normalized_candidates: List[str] = []
        seen_candidates: Set[str] = set()
        for candidate in candidate_tickers:
            symbol = (candidate or "").strip().upper()
            if not symbol or symbol in seen_candidates:
                continue
            seen_candidates.add(symbol)
            normalized_candidates.append(symbol)

        ordered_tickers = _filter_valid_tickers(candidate_tickers)
        if not ordered_tickers and primary_ticker:
            ordered_tickers = [primary_ticker]
        if not ordered_tickers:
            detail_payload: Dict[str, Any] = {"error": "ticker-not-found"}
            if primary_ticker:
                primary_clean = (primary_ticker or "").strip().upper()
                if primary_clean:
                    detail_payload["primary_ticker"] = primary_clean
            if normalized_candidates:
                detail_payload["candidates"] = normalized_candidates
                detail_payload["invalid_candidates"] = normalized_candidates
            else:
                detail_payload["reason"] = "no candidate tickers extracted"
            raise HTTPException(status_code=422, detail=detail_payload)

        base_input = {
            "article_time": article_time,
            "summary_bullets": summary_bullets,
            "claims": claims,
            "context_bullets": context_bullets,
            "direction_estimate": direction_est,
            "bull_points": bull_points,
            "bear_points": bear_points,
        }

    research_results: Dict[str, Dict[str, Any]] = {}
    research_stage_data: Dict[str, Dict[str, Any]] = {}
    technical_stage_data: Dict[str, Dict[str, Any]] = {}
    sentiment_stage_data: Dict[str, Dict[str, Any]] = {}

    ordered_symbols = [t.upper() for t in ordered_tickers]

    for symbol in ordered_symbols:
        researcher_input = dict(base_input)
        researcher_input["ticker"] = symbol
        log_buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(log_buffer), contextlib.redirect_stderr(log_buffer):
                stage1_raw, stage2_raw, plan_raw, session_id = researcher.run_two_stage(researcher_input)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"researcher-run-failed:{symbol}: {exc}") from exc

        stage1 = stage1_raw if isinstance(stage1_raw, dict) else {}
        plan = plan_raw if isinstance(plan_raw, dict) else {}

        # ── keep technical clean: remove any sentiment-only steps from the plan
        try:
            steps = plan.get("steps")
            if isinstance(steps, list):
                filtered = []
                for s in steps:
                    tool = (s.get("tool") or "").strip() if isinstance(s, dict) else ""
                    if tool and tool in SENTIMENT_TOOLS:
                        # drop it from the technical plan; sentiment is handled separately below
                        continue
                    filtered.append(s)
                if filtered is not steps:
                    plan = dict(plan)
                    plan["steps"] = filtered
        except Exception:
            # non-fatal: if plan is malformed, let the executor’s own guards handle it
            pass

        hypotheses = stage1.get("hypotheses") if isinstance(stage1.get("hypotheses"), list) else []
        rationale = stage1.get("rationale") if isinstance(stage1.get("rationale"), str) else ""

        try:
            technical_payload, technical_summary = _execute_technical_plan(plan)
        except Exception as exc:  # noqa: BLE001
            technical_payload = {
                "steps": plan.get("steps") if isinstance(plan, dict) else [],
                "results": [],
                "insights": [],
                "summary_lines": [f"Technical plan failed: {exc}"],
                "status": "error",
                "error": str(exc),
            }
            technical_summary = technical_payload["summary_lines"]

        try:
            sentiment_payload = _run_sentiment_block(
                symbol,
                article_time,
                channel="social",
                lookback_days=7,
                burst_hours=6,
                conversation_payload=conversation_payload,
            )
        except Exception as exc:  # noqa: BLE001
            sentiment_payload = {
                "error": str(exc),
                "ticker": symbol,
                "as_of": article_time,
            }

        sentiment_summary = _summarize_sentiment(sentiment_payload)
        summary_text = _build_research_summary_text(
            symbol,
            article_time,
            hypotheses if isinstance(hypotheses, list) else [],
            rationale,
            technical_summary,
            sentiment_summary,
        )

        run_at = _now_iso()
        log_text = log_buffer.getvalue()

        research_results[symbol] = {
            "ticker": symbol,
            "article_time": article_time,
            "hypotheses": hypotheses if isinstance(hypotheses, list) else [],
            "rationale": rationale,
            "plan": plan,
            "technical": technical_payload,
            "sentiment": sentiment_payload,
            "summary_text": summary_text,
            "updated_at": run_at,
            "session_id": session_id,
            "log": log_text,
        }

        research_stage_data[symbol] = {
            "input": researcher_input,
            "stage1": stage1,
            "plan": plan,
            "stage2_raw": stage2_raw,
            "session_id": session_id,
            "log": log_text,
            "run_at": run_at,
        }

        tech_entry = dict(technical_payload)
        tech_entry["run_at"] = run_at
        technical_stage_data[symbol] = tech_entry

        sentiment_stage_data[symbol] = {
            "ticker": symbol,
            "result": sentiment_payload,
            "run_at": run_at,
        }

    updated_at = _now_iso()
    research_payload = ResearchPayload(
        article_time=article_time,
        updated_at=updated_at,
        ordered_tickers=ordered_symbols,
        tickers=research_results,
    )

    with _connect() as conn:
        research_stage_payload = {
            "article_time": article_time,
            "tickers": research_stage_data,
            "ordered": ordered_symbols,
            "updated_at": updated_at,
        }
        _insert_stage_payload(conn, post_id, "researcher", research_stage_payload)

        technical_stage_payload = {
            "tickers": technical_stage_data,
            "ordered": ordered_symbols,
            "updated_at": updated_at,
        }
        _insert_stage_payload(conn, post_id, "technical_research", technical_stage_payload)

        sentiment_stage_payload = {
            "tickers": sentiment_stage_data,
            "ordered": ordered_symbols,
            "updated_at": updated_at,
        }
        _insert_stage_payload(conn, post_id, "sentiment_research", sentiment_stage_payload)

        extras = _load_extras_dict(conn, post_id)
        extras["research"] = research_payload.model_dump()
        _save_extras_dict(conn, post_id, extras)

    return research_payload

@app.post("/api/run-stage", response_model=RunStageResponse)
def run_stage(req: RunStageRequest):
    if not req.stages:
        raise HTTPException(status_code=400, detail="no-stages")

    try:
        code, out, err = _run_round_table(
            req.post_id,
            req.stages,
            req.refresh_from_csv,
            req.echo_post,
        )
    except Exception as ex:
        logging.exception("round_table execution crashed for %s", req.post_id)
        raise HTTPException(status_code=502, detail={"stdout": "", "stderr": f"runner-failed: {ex}"})

    ok = (code == 0)
    if not ok:
        # surface stderr for debugging in UI
        raise HTTPException(status_code=502, detail={"stderr": err, "stdout": out})

    if any(stage.lower() == "summariser" for stage in req.stages):
        try:
            with _connect() as conn:
                _ingest_conversation_hub(conn, req.post_id)
        except Exception as exc:  # noqa: BLE001
            logging.exception("conversation hub ingest failed after run_stage for %s", req.post_id)

    # round_table.run_stages_for_post writes back to SQL; we just report output
    return RunStageResponse(ok=True, post_id=req.post_id, stages_run=req.stages, stdout=out, stderr=err)


@app.post("/api/posts/{post_id:path}/research", response_model=ResearchResponse)
def run_research(post_id: str):
    try:
        payload = _run_research_pipeline(post_id)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logging.exception("research pipeline failed for %s", post_id)
        raise HTTPException(status_code=500, detail=f"research-failed: {exc}") from exc

    return ResearchResponse(ok=True, research=payload)

# ---------- Batch run ----------

def _select_post_ids(conn: sqlite3.Connection, flt: Optional[BatchRunFilter]) -> List[str]:
    where = ["1=1"]
    params: List[Any] = []
    if flt:
        if flt.query:
            where.append("(p.title LIKE ? OR p.text LIKE ?)")
            like = f"%{flt.query}%"
            params.extend([like, like])
        if flt.platform:
            where.append("p.platform = ?")
            params.append(flt.platform)
        if flt.source:
            where.append("p.source = ?")
            params.append(flt.source)
        if flt.date_from:
            where.append("(p.posted_at >= ? OR p.scraped_at >= ?)")
            params.extend([flt.date_from, flt.date_from])
        if flt.date_to:
            where.append("(p.posted_at < ? OR p.scraped_at < ?)")
            params.extend([flt.date_to, flt.date_to])

    sql = f"""
        SELECT p.post_id
        FROM posts p
        WHERE {" AND ".join(where)}
        ORDER BY datetime(COALESCE(p.scraped_at, p.posted_at)) DESC
        LIMIT ?
    """
    rows = _q_all(conn, sql, tuple(params + [BATCH_LIMIT]))
    return [r["post_id"] for r in rows]

@app.post("/api/run-stage/batch", response_model=BatchRunResponse)
def run_stage_batch(req: BatchRunRequest):
    if not req.stages:
        raise HTTPException(status_code=400, detail="no-stages")

    if req.post_ids:
        targets = list(dict.fromkeys(req.post_ids))[:BATCH_LIMIT]
    else:
        with _connect() as conn:
            targets = _select_post_ids(conn, req.filter)

    results: List[RunStageResponse] = []
    for pid in targets:
        try:
            code, out, err = _run_round_table(pid, req.stages, req.refresh_from_csv, echo_post=False)
            ok = (code == 0)
            if ok and any(stage.lower() == "summariser" for stage in req.stages):
                try:
                    with _connect() as conn:
                        _ingest_conversation_hub(conn, pid)
                except Exception as exc:  # noqa: BLE001
                    logging.exception("conversation hub ingest failed after batch run for %s", pid)
            results.append(
                RunStageResponse(
                    ok=ok,
                    post_id=pid,
                    stages_run=req.stages,
                    stdout=_trim(out, STDIO_TRIM),
                    stderr=_trim(err, STDIO_TRIM),
                )
            )
        except Exception as ex:
            logging.exception("round_table batch execution crashed for %s", pid)
            results.append(
                RunStageResponse(
                    ok=False,
                    post_id=pid,
                    stages_run=req.stages,
                    stdout="",
                    stderr=f"runner-failed: {ex}",
                )
            )

    return BatchRunResponse(ok=True, submitted=len(targets), results=results)

# ---------- Health ----------
@app.get("/api/health")
def health():
    return {
        "ok": True,
        "db": str(DB_PATH),
        "csv": str(CSV_PATH),
        "runner_model": ROUND_TABLE_MODEL,
        "runner_host": ROUND_TABLE_HOST,
        "runner_timeout": ROUND_TABLE_TIMEOUT,
        "tickers_dir": str(TICKERS_DIR),
        "time": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
