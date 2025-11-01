#!/usr/bin/env python3
# Single-arg CLI: compute Interest Score for a post-id using council/wisdom_of_sheep.sql
# Schema (from your dump):
#   posts(post_id TEXT PK, platform TEXT, source TEXT, text TEXT, ...)
#   stages(post_id TEXT, stage TEXT, created_at TEXT, payload JSON) -- summariser JSON lives here
#   tickers(post_id TEXT, ticker TEXT, market TEXT, confidence REAL) -- may be empty
#
# Strict ticker policy: use exact ticker from DB (no variants). If price fetch fails → error.

from __future__ import annotations

import argparse, json, math, os, sqlite3
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from yfinance_throttle import throttle_yfinance


class InterestScoreError(RuntimeError):
    """Exception raised when an interest score cannot be computed."""

    def __init__(self, code: str, message: str, *, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.details: Dict[str, Any] = details or {}

# --- make root import (technical_analyser.py is at repo root)
import sys
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
COUNCIL_DB_PATH = REPO_ROOT / "council" / "wisdom_of_sheep.sql"
CONVOS_DB_PATH = REPO_ROOT / "convos" / "conversations.sqlite"
try:
    import technical_analyser as ta
except Exception:
    # fallback if someone moves it under council/
    from . import technical_analyser as ta  # type: ignore


# ===== utils =====

def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def _json_out(obj: Dict[str, Any], pretty: bool = False) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2 if pretty else None))

def _open_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn

def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        f = float(x)
        return None if math.isnan(f) else f
    except Exception:
        return None


# ===== concrete fetchers for your schema =====

def _get_summary_payload(conn: sqlite3.Connection, post_id: str) -> Optional[Dict[str, Any]]:
    cur = conn.execute(
        "SELECT payload FROM stages WHERE post_id=? AND stage='summariser' ORDER BY created_at DESC LIMIT 1;",
        (post_id,),
    )
    row = cur.fetchone()
    if not row:
        return None
    try:
        payload = row["payload"]
        if isinstance(payload, str) and payload.strip():
            return json.loads(payload)
        if isinstance(payload, bytes):
            return json.loads(payload.decode("utf-8", errors="ignore"))
    except Exception:
        return None
    return None

def _get_post_row(conn: sqlite3.Connection, post_id: str) -> Optional[sqlite3.Row]:
    cur = conn.execute(
        "SELECT post_id, platform, source, title, scraped_at, posted_at, text FROM posts WHERE post_id=? LIMIT 1;",
        (post_id,),
    )
    return cur.fetchone()

def _get_best_ticker(conn: sqlite3.Connection, post_id: str, payload: Dict[str, Any]) -> Optional[str]:
    # Prefer tickers table (highest confidence) if present
    try:
        cur = conn.execute(
            "SELECT ticker FROM tickers WHERE post_id=? ORDER BY confidence DESC, ticker ASC LIMIT 1;",
            (post_id,),
        )
        row = cur.fetchone()
        if row:
            t = (row["ticker"] or "").strip()
            if t:
                return t
    except Exception:
        pass
    # Fallback: first assets_mentioned[].ticker from summariser payload
    for a in (payload.get("assets_mentioned") or []):
        if isinstance(a, dict):
            t = (a.get("ticker") or "").strip()
            if t:
                return t
    return None

def _extract_spam_pct(payload: Dict[str, Any]) -> int:
    v = payload.get("spam_likelihood_pct")
    try:
        if v is None:
            return 0
        return max(0, min(100, int(v)))
    except Exception:
        return 0


# ===== pricing helpers (strict, no variants) =====

def _can_fetch_prices(exact_symbol: str, days: int = 30) -> bool:
    try:
        t = yf.Ticker(exact_symbol)
        throttle_yfinance()
        df = t.history(period=f"{max(5, days)}d", interval="1d", auto_adjust=False, actions=False)
        return df is not None and not df.empty
    except Exception:
        return False

def _fetch_recent_returns(ticker: str, lookback_days: int = 90) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    try:
        t = yf.Ticker(ticker)
        throttle_yfinance()
        df = t.history(period=f"{max(lookback_days, 10)}d", interval="1d", auto_adjust=False, actions=False)
        if df is None or df.empty or len(df) < 7:
            return (None, None, None)
        close = df["Close"].astype(float)
        ret = close.pct_change()
        ret1d = float(ret.iloc[-1]) * 100.0 if not np.isnan(ret.iloc[-1]) else None
        ret5d = float((close.iloc[-1] / close.iloc[-6] - 1.0) * 100.0) if len(close) >= 6 else None
        ret_std = float(ret.rolling(60, min_periods=20).std(ddof=0).iloc[-1] * 100.0) if len(ret) >= 20 else None
        return (ret1d, ret5d, ret_std)
    except Exception:
        return (None, None, None)


# ===== scoring helpers (unchanged logic, concise) =====

def _unit_from_z(val: Optional[float], std: Optional[float], cap: float = 3.0) -> float:
    if val is None or std is None or std <= 0: return 0.0
    z = abs(val / std)
    return max(0.0, min(cap, z)) / cap

def _distance_interest_from_sr(sr: Dict[str, Any]) -> float:
    def _map(d):
        if d is None: return 0.0
        return math.exp(-0.5 * (max(0.0, d) / 5.0))
    ds = _safe_float(sr.get("distance_to_support_pct"))
    dr = _safe_float(sr.get("distance_to_resistance_pct"))
    return max(_map(ds), _map(dr))

def _event_flag(indicators: Dict[str, Any]) -> bool:
    crosses = (indicators.get("crosses") or {})
    rsi = _safe_float(indicators.get("rsi14"))
    return bool(crosses.get("golden_cross")) or (rsi is not None and (rsi >= 70.0 or rsi <= 30.0))

def _stance_value(stance: Optional[str]) -> float:
    if not stance: return 0.2
    s = str(stance).lower()
    if "bull" in s or "bear" in s: return 1.0
    if "neutral" in s: return 0.4
    return 0.2

def _source_score(platform: Optional[str]) -> int:
    p = (platform or "").lower()
    if p in ("sec","sedar","edgar"): return 10
    if p in ("news","bloomberg","reuters","ft","wsj","ap","cnbc"): return 9
    if p in ("company","pr","press","globenewswire","businesswire"): return 6
    if p in ("stocktwits","reddit"): return 5
    return 5

def _label(score: int) -> str:
    if score >= 85: return "urgent"
    if score >= 65: return "high"
    if score >= 40: return "medium"
    return "low"

def _iso_to_sqlite_dt(iso_ts: str) -> Optional[str]:
    """
    Convert ISO like '2025-09-30T22:15:47+00:00' → '2025-09-30 22:15:47' (SQLite datetime-friendly).
    Returns None if parse fails.
    """
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
        return dt.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        try:
            # last-resort: normalize by truncating + replacing T
            return iso_ts[:19].replace("T", " ")
        except Exception:
            return None


def _get_recent_texts_for_ticker_factory(conv_path: str) -> Optional[callable]:
    """
    conversations(
      id INTEGER PK, ticker TEXT, ts TEXT (ISO), kind TEXT, data TEXT, post_id TEXT
    )
    Returns callback: (ticker:str, end_ts_iso:str, days:int) -> List[str]
    Pulls rows with ts in [end - days, end], normalizing to SQLite datetime.
    """
    if not os.path.exists(conv_path):
        return None
    try:
        conn = sqlite3.connect(conv_path)
        conn.row_factory = sqlite3.Row
    except Exception:
        return None

    def _extract_text(data_val: Any) -> Optional[str]:
        if isinstance(data_val, (bytes, bytearray)):
            data_val = data_val.decode("utf-8", errors="ignore")
        if not isinstance(data_val, str) or not data_val.strip():
            return None
        s = data_val.strip()
        # Try JSON
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                for key in ("post_text", "text", "content", "body", "article_text", "summary"):
                    v = obj.get(key)
                    if isinstance(v, str) and v.strip():
                        return v.strip()
                # Handle your “delta” shape
                url = obj.get("url"); src = obj.get("src"); who = obj.get("who")
                parts = []
                if isinstance(url, str) and url:
                    slug = url.rstrip("/").split("/")[-1].replace("-", " ").replace("_", " ")
                    parts.append(slug)
                if isinstance(src, str) and src:
                    parts.append(src)
                if isinstance(who, list):
                    parts += [w for w in who if isinstance(w, str)]
                if parts:
                    return " | ".join(parts)
            elif isinstance(obj, list):
                strs = [x for x in obj if isinstance(x, str) and x.strip()]
                if strs:
                    return " ".join(strs[:20]).strip()
        except Exception:
            pass
        return s

    def _cb(ticker: str, end_ts_iso: str, days: int) -> List[str]:
        end_sql = _iso_to_sqlite_dt(end_ts_iso)
        if not end_sql:
            return []
        # start = end - days (in Python to avoid SQLite timezone quirks)
        from datetime import datetime, timedelta
        try:
            end_dt = datetime.fromisoformat(end_ts_iso.replace("Z", "+00:00"))
        except Exception:
            # fallback: parse the sqlite friendly string
            end_dt = datetime.strptime(end_sql, "%Y-%m-%d %H:%M:%S")
        start_dt = end_dt - timedelta(days=int(days))
        start_sql = start_dt.strftime("%Y-%m-%d %H:%M:%S")

        try:
            q = """
                SELECT data
                FROM conversations
                WHERE UPPER(ticker) = UPPER(?)
                  AND UPPER(kind) = 'DELTA'
                  AND datetime(replace(substr(ts,1,19),'T',' ')) BETWEEN ? AND ?
                ORDER BY ts ASC
                LIMIT 100;
            """
            rows = conn.execute(q, (ticker, start_sql, end_sql)).fetchall()
            out: List[str] = []
            for r in rows:
                t = _extract_text(r["data"])
                if t:
                    out.append(t)
            return out
        except Exception:
            return []

    return _cb


def _tokenize_words(s: str) -> List[str]:
    import re
    return re.findall(r"[a-z0-9]+", (s or "").lower())

def _shingles(words: List[str], k: int = 5) -> set:
    if not words or len(words) < k:
        return set()
    return {tuple(words[i:i+k]) for i in range(len(words)-k+1)}

def _novelty_unit(post_text: str, recent_texts: List[str]) -> float:
    """
    1 - max Jaccard similarity of 5-gram shingles across recent texts.
    Empty recent set => default mid novelty 0.7.
    """
    w0 = _tokenize_words(post_text)
    s0 = _shingles(w0, 5)
    if not s0:
        return 0.7
    best_sim = 0.0
    for t in recent_texts:
        w = _tokenize_words(t)
        s = _shingles(w, 5)
        if not s:
            continue
        inter = len(s0 & s)
        union = len(s0 | s)
        sim = (inter / union) if union else 0.0
        if sim > best_sim:
            best_sim = sim
    return max(0.0, min(1.0, 1.0 - best_sim))


def _is_mover_post(title: str, text: str) -> bool:
    t = f"{title or ''} {text or ''}".lower()
    keywords = ("after hrs", "after-hours", "after hours", "ah ", "ah:", "mover", "halt", "spike", "surge")
    return any(k in t for k in keywords)


def _compute_interest(
    *,
    ticker: str,
    post_text: str,
    summariser_json: Dict[str, Any],
    spam_pct: int,
    platform: str,
    title: str = "",
    novelty_cb: Optional[callable] = None,
    posted_at_iso: Optional[str] = None,
) -> Dict[str, Any]:
    # Technical snapshot
    indicators = ta.tool_compute_indicators(ticker, window_days=90)
    trend      = ta.tool_trend_strength(ticker, lookback_days=30)
    vol        = ta.tool_volatility_state(ticker, days=20, baseline_days=60)
    sr         = ta.tool_support_resistance_check(ticker, days=90)
    ret1d, ret5d, ret_std = _fetch_recent_returns(ticker, lookback_days=90)

    # Tech (0–45)
    ret_u, parts = 0.0, 0
    if ret1d is not None and ret_std: ret_u += _unit_from_z(ret1d, ret_std); parts += 1
    if ret5d is not None and ret_std: ret_u += _unit_from_z(ret5d, ret_std * math.sqrt(5)); parts += 1
    ret_u = (ret_u / parts) if parts > 0 else 0.0
    vol_ratio = _safe_float(vol.get("ratio"))
    vol_u = max(0.0, min(1.0, (vol_ratio - 1.0) / 1.5)) if vol_ratio is not None else 0.0
    strength = int(trend.get("strength") or 0)
    slope = _safe_float(trend.get("slope_pct_per_day")) or 0.0
    r2 = _safe_float(trend.get("r2")) or 0.0
    slope_u = max(0.0, min(1.0, abs(slope) / 0.8))
    trend_u = max(strength / 3.0, (0.5 * slope_u + 0.5 * max(0.0, min(1.0, r2))))
    dist_u = _distance_interest_from_sr(sr)
    event_u = 1.0 if _event_flag(indicators) else 0.0
    tech_pts = 10*ret_u + 10*vol_u + 10*trend_u + 10*dist_u + 5*event_u

    # Novelty & Source (0–20), relative to article time
    novelty_days = int(os.getenv("WOS_NOVELTY_DAYS", "30"))  # default 30d window
    novelty_unit = 0.7
    novelty_note = "fallback"
    novelty_samples: Optional[int] = None
    novelty_window_text: Optional[str] = None
    if novelty_cb and posted_at_iso:
        try:
            recent = novelty_cb(ticker, posted_at_iso, novelty_days)
            if recent:
                novelty_unit = _novelty_unit(post_text, recent)
                # window string: [end-days, end]
                from datetime import datetime, timedelta
                end_dt = datetime.fromisoformat(posted_at_iso.replace("Z", "+00:00"))
                start_dt = end_dt - timedelta(days=novelty_days)
                novelty_samples = len(recent)
                novelty_window_text = f"{start_dt.date()} → {end_dt.date()}"
                novelty_note = f"n={novelty_samples}; window={novelty_window_text}"
            else:
                novelty_unit = 0.7
                from datetime import datetime, timedelta
                end_dt = datetime.fromisoformat(posted_at_iso.replace("Z", "+00:00"))
                start_dt = end_dt - timedelta(days=novelty_days)
                novelty_samples = 0
                novelty_window_text = f"{start_dt.date()} → {end_dt.date()}"
                novelty_note = f"n=0; window={novelty_window_text}"
        except Exception:
            novelty_unit = 0.7
            novelty_note = "error"
    src_pts = _source_score(platform)
    nov_src_pts = min(10.0, 10.0*novelty_unit) + src_pts

    # Text (0–35)
    cats_list = list(summariser_json.get("claimed_catalysts") or [])
    nums = len(summariser_json.get("numbers_mentioned") or [])
    stance = summariser_json.get("author_stance")
    qf = summariser_json.get("quality_flags") or {}
    mover = _is_mover_post(title, post_text) or (ret1d is not None and abs(ret1d) >= 10.0)

    # Recognise "Catalysts: Yes" on movers as 2 catalysts
    cats = len(cats_list)
    if mover:
        if any(str(c).strip().lower() in ("yes", "y") for c in cats_list):
            cats = max(cats, 2)

    cats_u = min(cats, 4) / 4.0
    nums_u = min(nums, 5) / 5.0
    stance_u = _stance_value(stance)

    penalties = 0
    qf_l = {k.lower(): v for k, v in qf.items()}
    if qf_l.get("repetition_or_template"): penalties += 3
    if qf_l.get("vague_claims"): penalties += 3
    if qf_l.get("mentions_spread_or_liquidity"): penalties += 2
    penalties = min(penalties, 8)
    if mover:
        penalties = min(penalties, 3)   # soften penalties for short mover posts

    text_pts = max(0.0, min(35.0, (10*cats_u + 5*nums_u + 10*stance_u) - penalties))

    raw = tech_pts + text_pts + nov_src_pts

    # Spam dampener
    s = max(0, min(100, int(spam_pct)))
    mult = 1.0 - 0.7 * (s / 100.0)
    score = max(0.0, min(100.0, raw * mult))
    label = _label(int(round(score)))

    # Safety rails / floors
    score_after_spam = score
    if ret1d is not None:
        if abs(ret1d) >= 20.0:
            score = max(score, 55.0)
        elif abs(ret1d) >= 10.0:
            score = max(score, 40.0)
    if (event_u == 1.0) and (strength >= 2) and (ret1d is not None and abs(ret1d) >= 7.0):
        score = max(score, 40.0)

    score_i = int(round(max(0.0, min(100.0, score))))
    label = _label(score_i)

    why_bits = []
    if ret_u > 0.66 or vol_u > 0.66:           why_bits.append("abnormal price/volatility")
    if event_u == 1.0:                          why_bits.append("technical event (RSI/cross)")
    if cats >= 2:                                why_bits.append("multiple catalysts")
    if s >= 20:                                  why_bits.append(f"{s}% spam dampened")
    if novelty_unit >= 0.8:                      why_bits.append("high novelty")
    if mover:                                    why_bits.append("mover heuristics")
    if not why_bits:                             why_bits.append("baseline factors")
    why = ", ".join(sorted(set(why_bits)))

    metrics = {
        "ret1d_pct": ret1d,
        "ret5d_pct": ret5d,
        "ret_std_pct": ret_std,
        "ret_unit": ret_u,
        "vol_ratio": vol_ratio,
        "vol_unit": vol_u,
        "trend_strength": strength,
        "slope_pct_per_day": slope,
        "trend_r2": r2,
        "trend_unit": trend_u,
        "distance_unit": dist_u,
        "event_flag": bool(event_u),
        "novelty_unit": novelty_unit,
        "novelty_note": novelty_note,
        "novelty_days": novelty_days,
        "novelty_window": novelty_window_text,
        "novelty_samples": novelty_samples,
        "mover": mover,
        "spam_pct": s,
        "cats_count": cats,
        "nums_count": nums,
        "stance_value": stance_u,
        "penalties": penalties,
        "tech_points": tech_pts,
        "text_points": text_pts,
        "novelty_source_points": nov_src_pts,
        "raw_score_pre_spam": raw,
        "score_after_spam": score_after_spam,
        "score_final": score_i,
        "source_points": src_pts,
    }

    return {
        "interest_score": score_i,
        "interest_label": label,
        "interest_why": why,
        "council_recommended": bool(score_i >= 65 or platform.lower() in ("sec","sedar","edgar")),
        "council_priority": "urgent" if score_i >= 85 else "normal",
        "metrics": metrics,
    }


def _persist_interest_result(conn_path: str, result: dict) -> None:
    """Insert or update interest score results into the existing council DB."""
    import sqlite3
    import json
    import datetime

    conn = sqlite3.connect(conn_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS council_stage_interest (
                post_id TEXT PRIMARY KEY,
                ticker TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'ok',
                interest_score REAL NOT NULL,
                interest_label TEXT NOT NULL,
                interest_why TEXT,
                council_recommended INTEGER NOT NULL DEFAULT 0,
                council_priority TEXT NOT NULL DEFAULT 'normal',
                error_code TEXT,
                error_message TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                debug_json TEXT
            );
            """
        )

        # Backfill new columns if the table pre-dates this schema.
        try:
            rows = conn.execute("PRAGMA table_info(council_stage_interest)").fetchall()
            existing = {row[1] for row in rows}
        except Exception:
            existing = set()
        alters = []
        if "status" not in existing:
            alters.append("ALTER TABLE council_stage_interest ADD COLUMN status TEXT DEFAULT 'ok'")
        if "error_code" not in existing:
            alters.append("ALTER TABLE council_stage_interest ADD COLUMN error_code TEXT")
        if "error_message" not in existing:
            alters.append("ALTER TABLE council_stage_interest ADD COLUMN error_message TEXT")
        for sql in alters:
            try:
                conn.execute(sql)
            except Exception:
                continue

        status = (result.get("status") or "ok").lower().strip() or "ok"
        ticker = result.get("ticker") or result.get("ticker_used") or ""
        score = result.get("interest_score")
        if score is None:
            score = 0.0
        label = result.get("interest_label") or ("error" if status != "ok" else "unknown")
        why = result.get("interest_why") or ("interest score unavailable" if status != "ok" else "")
        created_at = result.get("calculated_at") or result.get("asof")
        if not created_at:
            created_at = datetime.datetime.utcnow().isoformat()
        debug_payload = result.get("debug") or result.get("metrics") or {}
        if not isinstance(debug_payload, dict):
            debug_payload = {"_value": debug_payload}

        conn.execute(
            """
            INSERT INTO council_stage_interest
              (post_id, ticker, status, interest_score, interest_label, interest_why,
               council_recommended, council_priority, error_code, error_message, created_at, debug_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(post_id) DO UPDATE SET
               ticker=excluded.ticker,
               status=excluded.status,
               interest_score=excluded.interest_score,
               interest_label=excluded.interest_label,
               interest_why=excluded.interest_why,
               council_recommended=excluded.council_recommended,
               council_priority=excluded.council_priority,
               error_code=excluded.error_code,
               error_message=excluded.error_message,
               created_at=excluded.created_at,
               debug_json=excluded.debug_json
            ;
            """,
            (
                result["post_id"],
                ticker,
                status,
                float(score),
                label,
                why,
                1 if result.get("council_recommended") else 0,
                result.get("council_priority") or "normal",
                result.get("error_code"),
                result.get("error_message"),
                created_at,
                json.dumps(debug_payload, ensure_ascii=False),
            ),
        )
        conn.commit()
    finally:
        conn.close()


# ===== CLI =====

def compute_interest_for_post(
    post_id: str,
    *,
    db_path: Optional[str | Path] = None,
    conv_db_path: Optional[str | Path] = None,
    persist: bool = True,
) -> Dict[str, Any]:
    """Compute and optionally persist an interest score for ``post_id``.

    Parameters
    ----------
    post_id:
        Identifier of the post/article in the council database.
    db_path:
        Path to the SQLite database. Defaults to ``council/wisdom_of_sheep.sql``.
    conv_db_path:
        Optional path to the conversation hub SQLite store used for novelty checks.
    persist:
        When ``True`` (default) the result is written to ``council_stage_interest``.
    """

    resolved_db_path = Path(db_path) if db_path is not None else COUNCIL_DB_PATH
    db_path_str = str(resolved_db_path)

    if conv_db_path is None:
        conv_path_candidate: Optional[Path] = CONVOS_DB_PATH
    else:
        conv_text = str(conv_db_path).strip()
        conv_path_candidate = Path(conv_text) if conv_text else None
    conv_path_str = str(conv_path_candidate) if conv_path_candidate else None

    ticker_value: Optional[str] = None
    spam_pct: int = 0
    platform_value: str = ""
    source_value: str = ""
    post_title: str = ""
    posted_at_iso: Optional[str] = None
    post_text: str = ""

    debug_context: Dict[str, Any] = {}

    try:
        if not os.path.exists(db_path_str):
            raise InterestScoreError("db_not_found", f"Database not found at {db_path_str}")

        conn = _open_db(db_path_str)
        try:
            summariser_payload = _get_summary_payload(conn, post_id)
            if not summariser_payload:
                raise InterestScoreError(
                    "summary_not_found",
                    f"No summariser payload found in stages for post_id={post_id}",
                )

            post_row = _get_post_row(conn, post_id)
            if post_row:
                platform_value = (post_row["platform"] or "") or ""
                source_value = (post_row["source"] or "") or ""
                post_title = post_row["title"] or ""
                post_text = post_row["text"] or ""
                posted_at_iso = post_row["posted_at"] or post_row["scraped_at"] or None
            else:
                platform_value = ""
                source_value = ""
                post_title = ""
                post_text = ""
                posted_at_iso = None

            spam_pct = _extract_spam_pct(summariser_payload)
            ticker_value = _get_best_ticker(conn, post_id, summariser_payload)
        finally:
            conn.close()

        if not ticker_value:
            raise InterestScoreError(
                "no_ticker_detected",
                "No ticker detected (tickers table empty and assets_mentioned missing).",
            )

        if not _can_fetch_prices(ticker_value, days=30):
            raise InterestScoreError(
                "price_data_unavailable",
                "Price data unavailable for detected ticker.",
                details={"ticker": ticker_value},
            )

        novelty_cb = None
        if conv_path_str:
            novelty_cb = _get_recent_texts_for_ticker_factory(conv_path_str)

        try:
            result = _compute_interest(
                ticker=ticker_value,
                post_text=post_text,
                summariser_json=summariser_payload,
                spam_pct=spam_pct,
                platform=source_value or platform_value or "reddit",
                title=post_title,
                novelty_cb=novelty_cb,
                posted_at_iso=posted_at_iso,
            )
        except InterestScoreError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise InterestScoreError(
                "technical_failure",
                f"Interest score computation failed: {exc}",
                details={"ticker": ticker_value},
            ) from exc

        metrics = result.get("metrics") or {}
        calculated_at = _utcnow_iso()
        debug_context = {
            "metrics": metrics,
            "spam_pct": spam_pct,
            "platform": platform_value,
            "source": source_value,
            "posted_at": posted_at_iso,
        }

        payload: Dict[str, Any] = {
            "status": "ok",
            "post_id": post_id,
            "ticker": ticker_value,
            "interest_score": result["interest_score"],
            "interest_label": result["interest_label"],
            "interest_why": result["interest_why"],
            "council_recommended": result["council_recommended"],
            "council_priority": result["council_priority"],
            "calculated_at": calculated_at,
            "metrics": metrics,
        }

        if persist:
            _persist_interest_result(
                db_path_str,
                {
                    **payload,
                    "debug": debug_context,
                },
            )

        payload["debug"] = debug_context
        return payload

    except InterestScoreError as exc:
        error_payload: Dict[str, Any] = {
            "status": "error",
            "post_id": post_id,
            "ticker": ticker_value or exc.details.get("ticker"),
            "interest_score": None,
            "interest_label": "error",
            "interest_why": "",
            "council_recommended": False,
            "council_priority": "normal",
            "calculated_at": _utcnow_iso(),
            "error_code": exc.code,
            "error_message": str(exc),
            "metrics": {},
            "error_details": exc.details,
        }
        error_debug = {"details": exc.details, **debug_context}
        if persist:
            _persist_interest_result(
                db_path_str,
                {
                    **error_payload,
                    "debug": error_debug,
                },
            )
        error_payload["debug"] = error_debug
        return error_payload

    except Exception as exc:  # noqa: BLE001
        message = f"Unexpected interest score failure: {exc}"
        error_payload = {
            "status": "error",
            "post_id": post_id,
            "ticker": ticker_value,
            "interest_score": None,
            "interest_label": "error",
            "interest_why": "",
            "council_recommended": False,
            "council_priority": "normal",
            "calculated_at": _utcnow_iso(),
            "error_code": "internal_error",
            "error_message": message,
            "metrics": {},
            "error_details": {"exception": repr(exc)},
        }
        error_debug = {"exception": repr(exc), **debug_context}
        if persist:
            _persist_interest_result(
                db_path_str,
                {
                    **error_payload,
                    "debug": error_debug,
                },
            )
        error_payload["debug"] = error_debug
        return error_payload


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute Interest Score for a post-id from council/wisdom_of_sheep.sql (strict ticker, no variants)"
    )
    p.add_argument("post_id", help="Post ID (e.g., t3_1nuo12h)")
    p.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    return p

def main() -> None:
    args = _build_argparser().parse_args()
    post_id = args.post_id
    result = compute_interest_for_post(
        post_id,
        db_path=COUNCIL_DB_PATH,
        conv_db_path=CONVOS_DB_PATH,
        persist=True,
    )

    _json_out(result, pretty=args.pretty)
    raise SystemExit(0 if (result.get("status") == "ok") else 1)


if __name__ == "__main__":
    main()
