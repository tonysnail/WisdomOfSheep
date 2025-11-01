from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .config import (
    ORACLE_BACKOFF_MAX,
    ORACLE_BACKOFF_MIN,
    ORACLE_BATCH_SIZE,
    ORACLE_BATCH_SLEEP,
    ORACLE_CURSOR_PATH,
    ORACLE_DEFAULT_BASE_URL,
    ORACLE_MAX_RETRIES,
    ORACLE_PASS,
    ORACLE_POLL_BASE,
    ORACLE_POLL_MAX,
    ORACLE_RETRY_STATE_PATH,
    ORACLE_SKIPPED_PATH,
    ORACLE_UNSUMMARISED_PATH,
    ORACLE_USER,
)


def default_cursor() -> Dict[str, str]:
    return {"platform": "", "post_id": "", "scraped_at": ""}


def load_cursor() -> Dict[str, str]:
    if not ORACLE_CURSOR_PATH.exists():
        return default_cursor()
    try:
        raw = json.loads(ORACLE_CURSOR_PATH.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return default_cursor()
        cursor = {
            "platform": str(raw.get("platform", "") or ""),
            "post_id": str(raw.get("post_id", "") or ""),
            "scraped_at": str(raw.get("scraped_at", "") or ""),
        }
        return cursor
    except Exception:
        return default_cursor()


def save_cursor(cursor: Dict[str, Any]) -> None:
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


def default_retry_state() -> Dict[str, Any]:
    return {
        "post_id": "",
        "attempts": 0,
        "scraped_at": "",
        "platform": "",
        "source": "",
        "title": "",
        "last_error": "",
    }


def load_retry_state() -> Dict[str, Any]:
    if not ORACLE_RETRY_STATE_PATH.exists():
        return default_retry_state()
    try:
        raw = json.loads(ORACLE_RETRY_STATE_PATH.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return default_retry_state()
        state = default_retry_state()
        state.update(raw)
        try:
            state["attempts"] = int(state.get("attempts", 0) or 0)
        except (TypeError, ValueError):
            state["attempts"] = 0
        return state
    except Exception:
        return default_retry_state()


def save_retry_state(state: Dict[str, Any]) -> None:
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


def save_unsummarised(entries: List[Dict[str, Any]], *, cutoff_iso: str) -> None:
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


def clear_unsummarised() -> None:
    try:
        ORACLE_UNSUMMARISED_PATH.unlink(missing_ok=True)
    except Exception:
        pass
    try:
        ORACLE_UNSUMMARISED_PATH.with_suffix(".tmp").unlink(missing_ok=True)
    except Exception:
        pass


def append_skipped_article(entry: Dict[str, Any], keep: int = 200) -> None:
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


def auth_tuple() -> Optional[Tuple[str, str]]:
    user = (ORACLE_USER or "").strip()
    pwd = (ORACLE_PASS or "").strip()
    if not (user and pwd):
        return None
    return user, pwd


def join_url(base_url: str, path: str) -> str:
    base = (base_url or "").strip()
    if not base:
        return path
    if not base.endswith("/"):
        base = base + "/"
    from urllib.parse import urljoin

    return urljoin(base, path.lstrip("/"))


__all__ = [
    "ORACLE_BACKOFF_MAX",
    "ORACLE_BACKOFF_MIN",
    "ORACLE_BATCH_SIZE",
    "ORACLE_BATCH_SLEEP",
    "ORACLE_DEFAULT_BASE_URL",
    "ORACLE_MAX_RETRIES",
    "ORACLE_POLL_BASE",
    "ORACLE_POLL_MAX",
    "append_skipped_article",
    "auth_tuple",
    "clear_unsummarised",
    "default_cursor",
    "default_retry_state",
    "join_url",
    "load_cursor",
    "load_retry_state",
    "save_cursor",
    "save_retry_state",
    "save_unsummarised",
]
