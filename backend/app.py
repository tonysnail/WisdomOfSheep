# backend/app.py

#  python -m uvicorn app:app --reload --port 8000

from __future__ import annotations

import contextlib
import csv
import io
import json
import logging
import os
import random
import re
import sqlite3
import sys
import threading
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Literal

from fastapi import Body, FastAPI, HTTPException, Query, Response
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
import requests
from pydantic import BaseModel, Field

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
PARENT_DIR_STR = str(PARENT_DIR)
if PARENT_DIR_STR not in sys.path:
    sys.path.insert(0, PARENT_DIR_STR)

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
PARENT_DIR_STR = str(PARENT_DIR)
if PARENT_DIR_STR not in sys.path:
    sys.path.insert(0, PARENT_DIR_STR)

from backend import database as _database_module
from backend import utils
from backend.council_time import CouncilTimeModel, approximate_token_count
from backend.config import (
    BATCH_LIMIT,
    CSV_PATH,
    DB_PATH,
    CONVO_STORE_PATH,
    JOB_LOG_KEEP,
    JOBS_DIR,
    ORACLE_BACKOFF_MAX,
    ORACLE_BACKOFF_MIN,
    ORACLE_BATCH_SIZE,
    ORACLE_BATCH_SLEEP,
    ORACLE_DEFAULT_BASE_URL,
    ORACLE_MAX_RETRIES,
    ORACLE_POLL_BASE,
    ORACLE_POLL_MAX,
    ORACLE_REQUEST_TIMEOUT,
    PAGE_SIZE_DEFAULT,
    ROUND_TABLE_AUTOFILL,
    ROUND_TABLE_DUMP_DIR,
    ROUND_TABLE_EVIDENCE_LOOKBACK,
    ROUND_TABLE_HOST,
    ROUND_TABLE_MAX_EVIDENCE,
    ROUND_TABLE_MODEL,
    ROUND_TABLE_PRETTY,
    ROUND_TABLE_TIMEOUT,
    ROUND_TABLE_VERBOSE,
    STDIO_TRIM,
    TIME_MODEL_PATH,
    ROOT,
)
from backend.conversation import (
    extract_convo_tickers as _extract_convo_tickers,
    fetch_recent_deltas as _fetch_recent_deltas,
    filter_valid_tickers as _filter_valid_tickers,
    get_conversation_hub as _get_conversation_hub,
    ingest_conversation_hub as _ingest_conversation_hub,
    load_ticker_universe as _load_ticker_universe,
)
from backend.database import (
    connect as _db_connect,
    ensure_database_ready as _db_ensure_database_ready,
    ensure_schema as _ensure_schema,
    execute as _exec,
    insert_stage_payload as _insert_stage_payload,
    latest_stage_payload as _latest_stage_payload,
    load_extras_dict as _load_extras_dict,
    query_all as _q_all,
    query_one as _q_one,
    save_extras_dict as _save_extras_dict,
    strip_research_from_extras as _strip_research_from_extras,
    upsert_extras as _upsert_extras,
    upsert_post_row as _upsert_post_row,
)
from backend.interest import InterestRecord, build_interest_record as _build_interest_record
from backend.jobs import (
    job_append_log as _job_append_log,
    job_increment as _job_increment,
    job_path as _job_path,
    job_update_fields as _job_update_fields,
    load_job as _load_job,
)
from backend.oracle import (
    ORACLE_CURSOR_PATH,
    ORACLE_RETRY_STATE_PATH,
    ORACLE_SKIPPED_PATH,
    ORACLE_UNSUMMARISED_PATH,
    append_skipped_article as _append_skipped_article,
    auth_tuple as _oracle_auth_tuple,
    clear_unsummarised as _clear_oracle_unsummarised,
    default_cursor as _default_oracle_cursor,
    default_retry_state as _default_oracle_retry_state,
    join_url as _oracle_join,
    load_cursor as _load_oracle_cursor,
    load_retry_state as _load_oracle_retry_state,
    save_cursor as _save_oracle_cursor,
    save_retry_state as _save_oracle_retry_state,
    save_unsummarised as _save_oracle_unsummarised,
)
from backend.research_tools import (
    SENTIMENT_TOOLS,
    TECH_TOOLS,
    build_research_summary_text as _build_research_summary_text,
    execute_technical_plan as _execute_technical_plan,
    run_sentiment_block as _run_sentiment_block,
    summarize_sentiment as _summarize_sentiment,
    summarize_technical_results as _summarize_technical_results,
)
from backend.schemas import (
    BatchRunFilter,
    BatchRunRequest,
    BatchRunResponse,
    CalendarDay,
    PostDetail,
    PostListItem,
    PostRow,
    PostsCalendarResponse,
    ResearchPayload,
    ResearchResponse,
    ResearchTickerPayload,
    RunStageRequest,
    RunStageResponse,
)

COUNCIL_FAILURES_PATH = ROOT / "council_failures.json"
COUNCIL_FAILURES_LOCK = threading.Lock()


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

try:  # pragma: no cover - optional interest score computation
    from council.interest_score import compute_interest_for_post
    INTEREST_SCORE_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001
    compute_interest_for_post = None  # type: ignore[assignment]
    INTEREST_SCORE_IMPORT_ERROR = exc


LOGGER = logging.getLogger(__name__)

_build_markets_from_entity = utils.build_markets_from_entity
_build_signal_from_dir_or_mod = utils.build_signal_from_dir_or_mod
_parse_spam_likelihood = utils.parse_spam_likelihood
_parse_spam_reason = utils.parse_spam_reason
_now_iso = utils.now_iso
_normalize_article_time = utils.normalize_article_time
_clean_strings = utils.clean_strings
_extract_claim_texts = utils.extract_claim_texts
_direction_estimate = utils.direction_estimate
_primary_ticker = utils.primary_ticker
_trim = utils.trim


def _sync_db_path() -> Path:
    """Ensure the database module honours the current DB_PATH."""
    path = Path(DB_PATH)
    _database_module.DB_PATH = path
    return path


def _ensure_database_ready() -> None:
    _sync_db_path()
    _db_ensure_database_ready()


def _connect() -> sqlite3.Connection:
    _sync_db_path()
    return _db_connect()


def _norm_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        text = str(value).strip()
    except Exception:  # noqa: BLE001
        return None
    return text or None


def _serialize_job_payload(job_id: str, job: Dict[str, Any]) -> Dict[str, Any]:
    """Return a serialisable view of a job with derived fields."""

    data = dict(job)
    data.setdefault("id", job_id)
    data.setdefault("job_id", job_id)

    total = 0
    done = 0
    try:
        total = int(data.get("total", 0) or 0)
    except (TypeError, ValueError):  # noqa: BLE001
        total = 0
    try:
        done = int(data.get("done", 0) or 0)
    except (TypeError, ValueError):  # noqa: BLE001
        done = 0

    phase = str(data.get("phase", "") or "").strip()
    current = str(data.get("current", "") or "").strip()

    message = ""
    if total > 0:
        message = f"{phase} {done}/{total}".strip()
        if current:
            message = f"{message} — {current}".strip()
    data["message"] = message

    log_tail = data.get("log_tail")
    if not isinstance(log_tail, list):
        data["log_tail"] = []

    return data

COUNCIL_TIME_MODEL = CouncilTimeModel(TIME_MODEL_PATH)

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
        if row:
            try:
                value = row["text"]
            except (KeyError, IndexError, TypeError):  # sqlite3.Row may not support mapping access
                try:
                    value = row[0]
                except Exception:  # noqa: BLE001
                    value = None
            if value:
                article_text = str(value or "")
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
    def _sort_key(item: Dict[str, Any]) -> Tuple[float, str, str]:
        scraped_raw = str(item.get("scraped_at") or "").strip()
        ts = float("inf")
        if scraped_raw:
            iso_txt = scraped_raw[:-1] + "+00:00" if scraped_raw.endswith("Z") else scraped_raw
            try:
                ts = datetime.fromisoformat(iso_txt).timestamp()
            except ValueError:
                ts = float("inf")
        return (ts, scraped_raw, str(item.get("post_id") or ""))

    ordered_entries = [
        item
        for item in sorted((entry for entry in entries if isinstance(entry, dict)), key=_sort_key)
    ]

    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cutoff": cutoff_iso,
        "count": len(ordered_entries),
        "articles": ordered_entries,
    }
    tmp_path = ORACLE_UNSUMMARISED_PATH.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, ORACLE_UNSUMMARISED_PATH)


def _clear_oracle_unsummarised() -> None:
    try:
        ORACLE_UNSUMMARISED_PATH.unlink(missing_ok=True)
    except Exception:
        pass
    try:
        ORACLE_UNSUMMARISED_PATH.with_suffix(".tmp").unlink(missing_ok=True)
    except Exception:
        pass


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

        def _reserve_progress_slots(min_remaining: int = 2) -> None:
            slack = max(int(min_remaining), 2)

            def mutate(job: Dict[str, Any]) -> None:
                done = int(job.get("done", 0) or 0)
                total = int(job.get("total", 0) or 0)
                desired = done + slack
                if desired > total:
                    job["total"] = desired

            _job_mutate(job_id, mutate)

        _reserve_progress_slots()

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

        def _sort_by_scraped(entry: Dict[str, Any]) -> Tuple[float, str, str]:
            scraped_txt = str(entry.get("scraped_at") or "")
            dt = _parse_scraped(scraped_txt)
            ts = dt.timestamp() if dt else float("inf")
            return (ts, scraped_txt, str(entry.get("post_id") or ""))

        def _normalize_article(article: Dict[str, Any]) -> Dict[str, Any]:
            def _clean(value: Any) -> str:
                return str(value or "").strip()

            platform = _clean(article.get("platform"))
            post_id = _clean(article.get("post_id"))
            scraped_at = _clean(article.get("scraped_at"))
            if not scraped_at:
                scraped_at = _clean(article.get("posted_at"))
            source = _clean(article.get("source"))
            title = _clean(article.get("title")) or _clean(article.get("headline"))

            normalized = {
                "platform": platform,
                "post_id": post_id,
                "scraped_at": scraped_at,
                "source": source,
                "title": title or (post_id if post_id else ""),
            }

            for key, value in normalized.items():
                if value:
                    article[key] = value

            return normalized

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
        warmup_snapshot_count = -1

        def _write_warmup_snapshot(force: bool = False) -> None:
            nonlocal warmup_snapshot_count
            if not force and len(warmup_records) == warmup_snapshot_count:
                return
            try:
                _save_oracle_unsummarised(warmup_records, cutoff_iso=warmup_cutoff_iso)
            except Exception as exc:  # noqa: BLE001
                logging.exception("failed to update Oracle backlog snapshot")
                _job_append_log(job_id, f"⚠ failed to update Oracle backlog snapshot: {exc}")
            else:
                warmup_snapshot_count = len(warmup_records)

        def _mark_backlog_processed(post_id: str) -> None:
            nonlocal warmup_records
            safe_pid = (post_id or "").strip()
            if not safe_pid or not warmup_records:
                return
            filtered = [
                entry
                for entry in warmup_records
                if (entry.get("post_id") or "").strip() != safe_pid
            ]
            if len(filtered) == len(warmup_records):
                return
            warmup_records = filtered
            _write_warmup_snapshot()

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

        if warmup_queue:
            warmup_queue.sort(key=_sort_by_scraped)
        warmup_records.sort(key=_sort_by_scraped)
        _write_warmup_snapshot(force=True)

        warmup_total = len(warmup_queue)
        if warmup_total:
            _reserve_progress_slots(warmup_total + 1)
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
                    _reserve_progress_slots(len(warmup_queue) + 2)
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
                    _reserve_progress_slots()

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
                if pending_stage == "backlog":
                    _mark_backlog_processed(pid)
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
                if pending_stage == "backlog":
                    _mark_backlog_processed(pid)
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
            if pending_stage == "backlog":
                _mark_backlog_processed(pid)

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

    def _thread_runner() -> None:
        global ACTIVE_COUNCIL_JOB_ID
        try:
            _worker_council_analysis(job_id)
        finally:
            with COUNCIL_JOB_LOCK:
                if ACTIVE_COUNCIL_JOB_ID == job_id:
                    ACTIVE_COUNCIL_JOB_ID = None

    thread = threading.Thread(target=_thread_runner, daemon=True)
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
    global ACTIVE_COUNCIL_JOB_ID
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
    global ACTIVE_COUNCIL_JOB_ID
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
        return Response(status_code=204)

    return _serialize_job_payload(job_id, job)


@app.get("/api/refresh-summaries/{job_id}")
def get_refresh_summaries(job_id: str):
    job = _load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job-not-found")
    return _serialize_job_payload(job_id, job)


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
            _clear_oracle_unsummarised()
        except Exception:
            pass
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
