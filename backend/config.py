from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOGGER = logging.getLogger("backend")


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def env_float(name: str, default: float | None) -> Optional[float]:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return value


def env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() not in {"", "0", "false", "no"}


DB_PATH = ROOT / "council" / "wisdom_of_sheep.sql"
DB_TEMPLATE_PATH = ROOT / "council" / "wisdom_of_sheep-empty.sql"
SCHEMA_PATH = ROOT / "council" / "council_schema.sql"
TIME_MODEL_PATH = ROOT / "council" / "council_time_model.json"
CSV_PATH = ROOT / "raw_posts_log.csv"
TICKERS_DIR = ROOT / "tickers"

DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

ORACLE_CURSOR_PATH = DATA_DIR / "oracle_cursor.json"
ORACLE_RETRY_STATE_PATH = DATA_DIR / "oracle_retry_state.json"
ORACLE_SKIPPED_PATH = DATA_DIR / "skipped_articles.json"
ORACLE_UNSUMMARISED_PATH = DATA_DIR / "oracle_unsummarised.json"
ORACLE_DEFAULT_BASE_URL = (os.getenv("WOS_ORACLE_BASE_URL", "") or "").strip()
ORACLE_USER = os.getenv("WOS_ORACLE_USER", "")
ORACLE_PASS = os.getenv("WOS_ORACLE_PASS", "")
ORACLE_BATCH_SIZE = max(1, env_int("WOS_ORACLE_BATCH_SIZE", 32))
_ORACLE_BATCH_SLEEP_RAW = env_float("WOS_ORACLE_BATCH_SLEEP", 0.2)
ORACLE_BATCH_SLEEP = 0.0 if _ORACLE_BATCH_SLEEP_RAW is None else max(0.0, _ORACLE_BATCH_SLEEP_RAW)
_ORACLE_POLL_BASE_RAW = env_float("WOS_ORACLE_POLL_BASE_SECONDS", 10.0)
ORACLE_POLL_BASE = 10.0 if _ORACLE_POLL_BASE_RAW is None else max(1.0, _ORACLE_POLL_BASE_RAW)
_ORACLE_POLL_MAX_RAW = env_float("WOS_ORACLE_POLL_MAX_SECONDS", 60.0)
ORACLE_POLL_MAX = 60.0 if _ORACLE_POLL_MAX_RAW is None else max(ORACLE_POLL_BASE, _ORACLE_POLL_MAX_RAW)
_ORACLE_BACKOFF_MIN_RAW = env_float("WOS_ORACLE_BACKOFF_MIN_SECONDS", 2.0)
ORACLE_BACKOFF_MIN = 2.0 if _ORACLE_BACKOFF_MIN_RAW is None else max(0.5, _ORACLE_BACKOFF_MIN_RAW)
_ORACLE_BACKOFF_MAX_RAW = env_float("WOS_ORACLE_BACKOFF_MAX_SECONDS", 30.0)
ORACLE_BACKOFF_MAX = 30.0 if _ORACLE_BACKOFF_MAX_RAW is None else max(ORACLE_BACKOFF_MIN, _ORACLE_BACKOFF_MAX_RAW)
ORACLE_REQUEST_TIMEOUT = env_float("WOS_ORACLE_TIMEOUT_SECONDS", 30.0)
ORACLE_MAX_RETRIES = max(0, env_int("WOS_ORACLE_MAX_RETRIES", 3))

PAGE_SIZE_DEFAULT = env_int("WOS_PAGE_SIZE", 50)
BATCH_LIMIT = env_int("WOS_BATCH_LIMIT", 200)
STDIO_TRIM = env_int("WOS_STDIO_TRIM", 100_000)

ROUND_TABLE_MODEL = os.getenv("WOS_MODEL_NAME", "mistral")
ROUND_TABLE_HOST = os.getenv("WOS_MODEL_HOST", "http://localhost:11434")
ROUND_TABLE_TIMEOUT_RAW = env_float("WOS_MODEL_TIMEOUT_SECS", None)
ROUND_TABLE_TIMEOUT = (
    None if ROUND_TABLE_TIMEOUT_RAW is None or ROUND_TABLE_TIMEOUT_RAW <= 0 else ROUND_TABLE_TIMEOUT_RAW
)
ROUND_TABLE_VERBOSE = env_flag("WOS_RUNNER_VERBOSE", "1")
ROUND_TABLE_AUTOFILL = env_flag("WOS_RUNNER_AUTOFILL", "1")
ROUND_TABLE_PRETTY = env_flag("WOS_RUNNER_PRETTY", "0")
ROUND_TABLE_DUMP_DIR = os.getenv("WOS_RUNNER_DUMP_DIR") or None
ROUND_TABLE_EVIDENCE_LOOKBACK = env_int("WOS_EVIDENCE_LOOKBACK_DAYS", 120)
ROUND_TABLE_MAX_EVIDENCE = env_int("WOS_MAX_EVIDENCE_PER_CLAIM", 3)

CONVO_STORE_PATH = ROOT / "convos" / "conversations.sqlite"
CONVO_MODEL = os.getenv("WOS_CONVO_MODEL", ROUND_TABLE_MODEL)

JOBS_DIR = ROOT / ".jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

JOB_LOG_KEEP = int(os.getenv("WOS_JOB_LOG_KEEP", "200"))
