from __future__ import annotations

import csv
import logging
import re
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Sequence, Set

from .config import CONVO_MODEL, CONVO_STORE_PATH, TICKERS_DIR
from .database import (
    insert_stage_payload,
    latest_stage_payload,
    load_extras_dict,
    save_extras_dict,
)
from . import utils

try:  # pragma: no cover - optional dependency for conversation hub ingestion
    from ticker_conversation_hub import ConversationHub, SQLiteStore, compute_ticker_signal

    CONVO_HUB_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001
    ConversationHub = None  # type: ignore[assignment]
    SQLiteStore = None  # type: ignore[assignment]
    compute_ticker_signal = None  # type: ignore[assignment]
    CONVO_HUB_IMPORT_ERROR = exc

LOGGER = logging.getLogger("backend.conversation")

CONVO_HUB_LOCK = threading.Lock()
CONVO_HUB_INSTANCE: Optional["ConversationHub"] = None
_TICKER_UNIVERSE: Optional[Set[str]] = None

_CASHTAG_RE = re.compile(r"\$[A-Z]{1,5}\b")


def get_conversation_hub() -> Optional["ConversationHub"]:
    if not callable(ConversationHub) or SQLiteStore is None:
        return None
    global CONVO_HUB_INSTANCE
    with CONVO_HUB_LOCK:
        if CONVO_HUB_INSTANCE is None:
            try:
                store = SQLiteStore(str(CONVO_STORE_PATH))
                CONVO_HUB_INSTANCE = ConversationHub(store=store, model=CONVO_MODEL)
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("conversation hub init failed: %s", exc)
                CONVO_HUB_INSTANCE = None
        return CONVO_HUB_INSTANCE


def load_ticker_universe() -> Set[str]:
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
            LOGGER.warning("Failed to load ticker universe from %s: %s", tickers_path, exc)

    _TICKER_UNIVERSE = universe
    return universe


def filter_valid_tickers(candidates: Sequence[str]) -> List[str]:
    universe = load_ticker_universe()
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


def fetch_recent_deltas(store: Any, ticker: str, as_of_iso: str, limit: int = 5) -> List[Dict[str, Any]]:
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


def extract_convo_tickers(post_row: sqlite3.Row, summariser: Dict[str, Any]) -> List[str]:
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


def ingest_conversation_hub(conn: sqlite3.Connection, post_id: str) -> Optional[Dict[str, Any]]:
    hub = get_conversation_hub()
    if not hub:
        return None

    post_row = conn.execute("SELECT * FROM posts WHERE post_id = ?", (post_id,)).fetchone()
    if not post_row:
        return None

    summariser = latest_stage_payload(conn, post_id, "summariser")
    if not summariser:
        return None

    tickers = extract_convo_tickers(post_row, summariser)
    if not tickers:
        return None

    bullets = utils.clean_strings(summariser.get("summary_bullets"))
    if not bullets:
        fallback = summariser.get("summary") or summariser.get("summary_text")
        if isinstance(fallback, str) and fallback.strip():
            bullets = [fallback.strip()]

    extras = load_extras_dict(conn, post_id)
    ts = (
        post_row["posted_at"]
        or post_row["scraped_at"]
        or extras.get("posted_at")
        or extras.get("scraped_at")
        or utils.now_iso()
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
        "ingested_at": utils.now_iso(),
    }

    insert_stage_payload(conn, post_id, "conversation_hub", payload)
    extras["conversation_hub"] = payload
    save_extras_dict(conn, post_id, extras)
    return payload


__all__ = [
    "CONVO_HUB_IMPORT_ERROR",
    "ConversationHub",
    "SQLiteStore",
    "compute_ticker_signal",
    "extract_convo_tickers",
    "fetch_recent_deltas",
    "filter_valid_tickers",
    "get_conversation_hub",
    "ingest_conversation_hub",
    "load_ticker_universe",
]
