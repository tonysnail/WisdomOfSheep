#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
portfolio_probe.py — Wisdom Of Sheep (Portfolio Manager probe / backtester / GA optimizer)


Usage:


Run Genetic Algorithm across this date range to calibrate trading algorithm.

    python portfolio_probe.py \
      --db council/wisdom_of_sheep.sql \
      --start 2025-09-20 --end 2025-10-21 \
      --ga --pop 120 --gens 160 --tournament-k 5 --cx 0.9 --mut 0.25 \
      --interval auto --intraday-limit-days 59 --verbose 1

Perform Backtest with default values:

    python portfolio_probe.py \
      --db council/wisdom_of_sheep.sql \
      --start 2025-09-20 --end 2025-10-21 \
      --interval auto --intraday-limit-days 59 --verbose 2





Warm the Cache between a Date Range so the Stable Island GA Detector can work without needing to pull yfinance data:

    python portfolio_probe.py \
      --db council/wisdom_of_sheep.sql \
      --start 2025-09-20 --end 2025-10-22 \
      --interval auto --intraday-limit-days 59 \
      --warm-only --verbose 2


Full GA Island Detection after Cache is warmed:     [ Writes verbose output to log files in portfolio/   -  going easy on the Terminal Window ]

    (venv) python portfolio_ga_marathon.py \
      --db council/wisdom_of_sheep.sql \
      --start 2025-09-20 --end 2025-10-22 \
      --pop 80 --gens 120 --tournament-k 5 --cx 0.9 --mut 0.25 \
      --interval auto --intraday-limit-days 59 \
      --initial-equity 100000 --verbose 1 \
      --runs 6 --seeds 42,1337,7,11,23,101 --name oct_window \
      --console key --gen-goal 120 \
      | tee "logs/ga_marathon_$(date -u +%Y%m%dT%H%M%SZ)_oct_window.log"




This version:
  - Treats Chairman verdict fields as first-class GA genes & decision gates.
  - Adds catalyst/watchout scoring and consistency in the fitness.
  - Null-safe imputers so missing upstream fields don't break learning.
  - Uses a TEMP VIEW (not persistent) so a read-only DB file is fine.
  - Writes all artifacts to ./portfolio/ (trades CSV, GA logs, best genes, metrics JSON).

Requires:
  - pandas, numpy
  - yfinance (optional; pip install yfinance)
  - Your local technical_analyser.fetch_indicator_series()
  - Your yfinance_throttle.throttle_yfinance()
"""

import argparse
import json
import math
import os
import random
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Project-local modules you already have
from technical_analyser import fetch_indicator_series
from yfinance_throttle import throttle_yfinance

# ============================================================
# Globals / Environment
# ============================================================

# Verbosity: 0 = silent, 1 = normal, 2 = per-signal, 3 = per-bar
_VERB = int(os.getenv("WOS_VERBOSE", "1"))

def vprint(level: int, *args, **kwargs):
    """Conditional print based on verbosity level."""
    if _VERB >= level:
        print(*args, **kwargs)

# Paths
PORTFOLIO_DIR = Path("portfolio").resolve()
PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)

# yfinance throttle
os.environ.setdefault("WOS_YF_THROTTLE_SECONDS", "0.5")

# Price cache
os.environ.setdefault("WOS_PRICE_CACHE_DIR", ".cache/prices")
os.environ.setdefault("WOS_PRICE_CACHE_TTL_HOURS", "24")
os.environ.setdefault("WOS_PRICE_CACHE_FMT", "parquet")

_PRICE_CACHE_DIR = Path(os.getenv("WOS_PRICE_CACHE_DIR", ".cache/prices")).resolve()
_PRICE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_PRICE_CACHE_TTL_HOURS = float(os.getenv("WOS_PRICE_CACHE_TTL_HOURS", "24"))
_PRICE_CACHE_FMT = os.getenv("WOS_PRICE_CACHE_FMT", "parquet").lower()
_PRICE_CACHE_LOCK = threading.Lock()

# Optional enriched tickers file
_ENRICHED_CSV = os.getenv("WOS_TICKERS_ENRICHED", "tickers/tickers_enriched.csv")

# Exchange → yfinance suffix map
_YF_EX_SUFFIX = {
    # US
    "NYSE": "", "NASDAQ": "", "NYQ": "", "NMS": "", "NCM": "", "NQSM": "", "ARCA": "", "NYSEARCA": "", "CBOE": "", "BATS": "", "PCX": "",
    # UK / Europe
    "LSE": ".L", "LONDON": ".L",
    "EURONEXT PARIS": ".PA", "PARIS": ".PA", "EPA": ".PA",
    "EURONEXT AMSTERDAM": ".AS", "AMSTERDAM": ".AS",
    "EURONEXT BRUSSELS": ".BR", "BRUSSELS": ".BR",
    "FRANKFURT": ".DE", "XETRA": ".DE",
    "MILAN": ".MI", "BIT": ".MI",
    "MADRID": ".MC",
    "COPENHAGEN": ".CO",
    "STOCKHOLM": ".ST",
    "HELSINKI": ".HE",
    "OSLO": ".OL",
    "SIX": ".SW", "SWISS": ".SW",
    # Americas
    "TSX": ".TO", "TORONTO": ".TO",
    "TSXV": ".V", "VENTURE": ".V",
    "MEXICO": ".MX", "BOLSA MEXICANA": ".MX",
    "SÃO PAULO": ".SA", "SAO PAULO": ".SA", "B3": ".SA",
    # APAC
    "ASX": ".AX",
    "TSE": ".T", "TOKYO": ".T",
    "HONG KONG": ".HK", "HKEX": ".HK",
    "NSE": ".NS", "NATIONAL STOCK EXCHANGE OF INDIA": ".NS",
    "BSE": ".BO", "BOMBAY": ".BO",
    "KOREA": ".KS", "KOSPI": ".KS", "KOSDAQ": ".KQ",
    "TAIWAN": ".TW", "TWSE": ".TW", "TWO": ".TWO",
}

# Probe order for bare equities if not found in enriched.csv
_YF_SUFFIX_PROBE = [
    "", ".L", ".TO", ".V", ".PA", ".AS", ".BR", ".DE", ".MI", ".MC", ".CO", ".ST", ".HE", ".OL", ".SW",
    ".AX", ".T", ".HK", ".NS", ".BO", ".KS", ".KQ", ".TW", ".TWO", ".SA", ".MX",
]

# Ad-hoc aliases
_YF_ALIAS: Dict[str, Optional[str]] = {
    "ASCX": None, "ATT": None, "AV": None, "AVNRL": None,
}

_resolve_cache_ok: Dict[str, str] = {}
_resolve_cache_bad: set[str] = set()
_enriched_by_key: Dict[str, str] = {}
_enriched_loaded = False

# ============================================================
# Utilities
# ============================================================

def load_genes_file(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Genes file not found: {p}")
    try:
        return json.loads(p.read_text())
    except Exception as exc:
        raise RuntimeError(f"Failed to parse genes JSON at {p}: {exc}") from exc

def run_backtest_from_genes(
    conn: sqlite3.Connection,
    genes: dict,
    *,
    start: str,
    end: str,
    initial_equity: float,
    ticker_filter: Optional[str],
    interval: str,
    intraday_limit_days: int
) -> dict:
    # Convert GA gene → backtest params + gates
    params = genes_to_params(genes)
    gates  = genes_to_gates(genes)

    vprint(1, "\n[RUN] Backtest from genes:")
    vprint(1, f"      start={start} end={end} initial_equity={initial_equity:,.2f}")
    vprint(2, f"      params={params}")
    vprint(2, f"      gates={gates}")

    res = backtest(
        conn,
        params,
        start=start,
        end=end,
        initial_equity=initial_equity,
        ticker_filter=ticker_filter,
        cache={},  # fresh cache for clarity; reuse if you prefer
        gates=gates,
        interval=interval,
        intraday_limit_days=intraday_limit_days,
    )

    metrics = res.get("metrics", {})
    vprint(1, "\n[BT] Metrics:")
    vprint(1, json.dumps(metrics, indent=2))
    vprint(1, f"[BT] Trades CSV: {res.get('trades_csv')}")
    vprint(1, "[BT] Done.")

    return res

def _map_interval_from_timeframe(tf: str) -> str:
    tf = (tf or "").lower()
    if tf == "intraday":
        return "15m"
    if tf == "swing_days":
        return "1h"
    if tf == "swing_weeks":
        return "4h"
    return "1d"

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)

def safe_div(num, den, eps=1e-9):
    d = den if abs(den) > eps else (eps if den >= 0 else -eps)
    return num / d

def _decide_row_interval(row_tf: Optional[str], global_interval: str) -> str:
    if global_interval and global_interval != "auto":
        return global_interval
    return _map_interval_from_timeframe(row_tf or "")

def _interval_is_intraday(interval: str) -> bool:
    return interval in ("4h","1h","30m","15m","5m")

def _nz(x, default):
    """Return default for None/NaN, else x."""
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return default
    except Exception:
        if x is None:
            return default
    return x

def _parse_json_array(txt) -> List[str]:
    try:
        a = json.loads(txt) if isinstance(txt, str) else (txt or [])
        return [str(x) for x in (a or [])]
    except Exception:
        return []

def _tag_any_contains(items: List[str], needles: Tuple[str, ...]) -> bool:
    s = " ".join(items).lower()
    return any(n.lower() in s for n in needles)

def fnum(x, nd=2):
    try:
        if x is None:
            return None
        return round(float(x), nd)
    except Exception:
        return None

def _normalize_key(s: Any) -> str:
    try:
        if s is None or (isinstance(s, float) and pd.isna(s)) or (isinstance(s, str) and not s.strip()):
            return ""
    except Exception:
        if s is None:
            return ""
    return str(s).strip().upper()

def _suffix_for_row(row: dict) -> str:
    probes = [
        _normalize_key(row.get("market")),
        _normalize_key(row.get("fullExchangeName")),
        _normalize_key(row.get("exchange")),
        _normalize_key(row.get("country")),
    ]
    for p in probes:
        if not p:
            continue
        for k, sfx in _YF_EX_SUFFIX.items():
            if k in p:
                return sfx
    return ""

def _load_enriched_csv():
    """Load tickers_enriched.csv (optional) for exchange suffix + aliases."""
    global _enriched_loaded, _enriched_by_key
    if _enriched_loaded:
        return
    try:
        df = pd.read_csv(_ENRICHED_CSV)
        vprint(1, f"[RESOLVE] Loading enriched tickers: {_ENRICHED_CSV}")
    except Exception:
        _enriched_loaded = True
        vprint(1, "[RESOLVE] No enriched tickers file found (optional).")
        return
    for _, r in df.iterrows():
        sym = _normalize_key(r.get("Symbol"))
        if not sym:
            continue
        sfx = _suffix_for_row(r.to_dict())
        base = sym if any(ch in sym for ch in ".=-^") else (sym + sfx)
        _enriched_by_key.setdefault(sym, base)
        aliases_raw = r.get("aliases")
        aliases_str = "" if aliases_raw is None or (isinstance(aliases_raw, float) and pd.isna(aliases_raw)) else str(aliases_raw)
        if aliases_str:
            for a in [x.strip() for x in aliases_str.split(",") if x.strip()]:
                _enriched_by_key.setdefault(_normalize_key(a), base)
    _enriched_loaded = True
    vprint(2, f"[RESOLVE] Enriched map loaded: {len(_enriched_by_key)} keys")

def _strip_dollar(sym: str) -> str:
    s = (sym or "").strip().upper()
    if s.startswith("$"):
        s = s[1:]
    return s

def _looks_bare_equity(sym: str) -> bool:
    return (("." not in sym) and ("=" not in sym) and ("^" not in sym) and (len(sym) <= 5))

def _pk_cache_key(ticker: str, start: Optional[str], end: Optional[str], period_days: Optional[int], interval: str) -> str:
    payload = {
        "t": (ticker or "").upper(),
        "start": start,
        "end": end,
        "period_days": period_days,
        "interval": interval,
    }
    s = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    import hashlib
    return hashlib.sha1(s).hexdigest()

def _pk_paths(key: str) -> Tuple[Path, Path]:
    ext = ".parquet" if _PRICE_CACHE_FMT == "parquet" else ".pkl"
    return (_PRICE_CACHE_DIR / f"{key}{ext}", _PRICE_CACHE_DIR / f"{key}.json")

def _pk_is_fresh(meta: Path) -> bool:
    if not meta.exists():
        return False
    if _PRICE_CACHE_TTL_HOURS <= 0:
        return True
    try:
        ts = float(json.loads(meta.read_text()).get("created_at", 0))
        return (time.time() - ts) <= _PRICE_CACHE_TTL_HOURS * 3600.0
    except Exception:
        return False

def _pk_load(data: Path):
    if not data.exists():
        return None
    try:
        if _PRICE_CACHE_FMT == "parquet":
            return pd.read_parquet(data)
        return pd.read_pickle(data)
    except Exception:
        return None

def _pk_save(data: Path, meta: Path, df: pd.DataFrame) -> None:
    tmp = data.with_suffix(data.suffix + ".tmp")
    with _PRICE_CACHE_LOCK:
        try:
            if _PRICE_CACHE_FMT == "parquet":
                df.to_parquet(tmp, index=True)
            else:
                df.to_pickle(tmp)
            os.replace(tmp, data)
            meta.write_text(json.dumps({"created_at": time.time()}, indent=2))
        finally:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass

def cached_fetch_indicator_series(ticker: str, start: Optional[str] = None, end: Optional[str] = None,
                                  period_days: Optional[int] = None, interval: str = "1d") -> pd.DataFrame:
    """Price/indicator fetch with a persistent on-disk cache (freshness via TTL) and interval support."""
    tkr = (ticker or "").strip().upper()
    key = _pk_cache_key(
        tkr,
        start or None,
        end or None,
        None if (start and end) else (period_days or None),
        interval or "1d",
    )
    data_path, meta_path = _pk_paths(key)
    if _pk_is_fresh(meta_path):
        df = _pk_load(data_path)
        if df is not None and not df.empty:
            try:
                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC")
                else:
                    df.index = df.index.tz_convert("UTC")
            except Exception:
                pass
            vprint(3, f"[CACHE] hit {tkr} {start}→{end} @{interval}")
            return df

    vprint(2, f"[CACHE] miss {tkr} {start}→{end} @{interval} — fetching")

    # throttle only on actual fetch (miss path)
    try:
        throttle_yfinance()
    except Exception:
        pass

    # Gracefully pass interval through; fallback if your function doesn't accept it yet
    try:
        df = fetch_indicator_series(tkr, start=start, end=end, period_days=period_days, interval=interval)
    except TypeError:
        df = fetch_indicator_series(tkr, start=start, end=end, period_days=period_days)

    if df is None or df.empty:
        return pd.DataFrame()
    try:
        _pk_save(data_path, meta_path, df)
    except Exception:
        pass
    return df

def _probe_yf_has_data(sym: str, start: str, end: str) -> bool:
    try:
        df = cached_fetch_indicator_series(sym, start=start, end=end)
        return (df is not None) and (not df.empty)
    except Exception:
        return False

def resolve_yf_symbol(raw_sym: Optional[str], start: str, end: str) -> Optional[str]:
    """Resolve bare symbols to yfinance symbols using enriched.csv and suffix probes."""
    if not raw_sym or not _normalize_key(raw_sym):
        return None
    _load_enriched_csv()
    raw = _strip_dollar(raw_sym)
    if raw in _resolve_cache_ok:
        return _resolve_cache_ok[raw]
    if raw in _resolve_cache_bad:
        return None

    # 1) Enriched map
    mapped = _enriched_by_key.get(_normalize_key(raw))
    if mapped:
        if _probe_yf_has_data(mapped, start, end):
            _resolve_cache_ok[raw] = mapped
            vprint(2, f"[RESOLVE] {raw} → {mapped} (enriched)")
            return mapped
        _resolve_cache_bad.add(raw)
        vprint(2, f"[RESOLVE] {raw} enriched→{mapped} but no data; marking bad")
        return None

    # 2) Ad-hoc alias
    if raw in _YF_ALIAS:
        aliased = _YF_ALIAS[raw]
        if not aliased:
            _resolve_cache_bad.add(raw)
            return None
        if _probe_yf_has_data(aliased, start, end):
            _resolve_cache_ok[raw] = aliased
            vprint(2, f"[RESOLVE] {raw} → {aliased} (alias)")
            return aliased
        _resolve_cache_bad.add(raw)
        return None

    # 3) Suffix probe for bare equities
    candidates: List[str] = []
    if not _looks_bare_equity(raw):
        candidates.append(raw)
    else:
        for sfx in _YF_SUFFIX_PROBE:
            candidates.append(raw + sfx)

    seen = set()
    for cand in [c for c in candidates if not (c in seen or seen.add(c))]:
        if _probe_yf_has_data(cand, start, end):
            _resolve_cache_ok[raw] = cand
            vprint(2, f"[RESOLVE] {raw} → {cand} (probe)")
            return cand
    _resolve_cache_bad.add(raw)
    vprint(1, f"[RESOLVE] Failed to resolve: {raw}")
    return None


# ============================================================
# Cache warming helpers
# ============================================================

def _effective_intraday_window(start: str, end: str, intraday_limit_days: int) -> Tuple[str, str]:
    """Clamp the lookback for sub-hourly bars (vendor limits)."""
    try:
        end_ts = pd.Timestamp(end).normalize() + pd.Timedelta(days=1)
        start_ts = end_ts - pd.Timedelta(days=intraday_limit_days)
        orig_start = pd.Timestamp(start)
        eff_start = max(start_ts, orig_start).strftime("%Y-%m-%d")
        eff_end = pd.Timestamp(end).strftime("%Y-%m-%d")
        return eff_start, eff_end
    except Exception:
        return start, end


def warm_cache(conn: sqlite3.Connection, start: str, end: str, interval: str, intraday_limit_days: int) -> None:
    """
    Prefetch all (ticker, interval) series needed for the window,
    mirroring the same interval bucketing logic used in backtest().
    """
    vprint(1, f"[WARM] Starting cache warm — window {start}→{end}  interval={interval}  intraday_limit_days={intraday_limit_days}")

    q = """
    SELECT ticker, timeframe, article_date
    FROM chairman_flat_v3
    WHERE ticker IS NOT NULL AND TRIM(ticker) <> ''
      AND implied_direction IS NOT NULL
      AND date(article_date) BETWEEN date(?) AND date(?)
    ORDER BY ticker, article_date
    """
    rows = conn.execute(q, (start, end)).fetchall()
    if not rows:
        vprint(1, "[WARM] No rows found to warm.")
        return

    from collections import defaultdict
    groups: Dict[str, Dict[str, bool]] = defaultdict(dict)

    for raw_tkr, tf, _ts in rows:
        eff_int = _decide_row_interval(tf, interval)
        groups[raw_tkr][eff_int] = True

    warmed = 0
    skipped_resolve = 0
    for raw_tkr, ints in groups.items():
        yf_sym = resolve_yf_symbol(raw_tkr, start, end)
        if not yf_sym:
            vprint(2, f"[WARM] Resolve failed for {raw_tkr}; skip")
            skipped_resolve += 1
            continue
        for row_interval in ints.keys():
            use_start, use_end = (start, end)
            if _interval_is_intraday(row_interval):
                use_start, use_end = _effective_intraday_window(start, end, intraday_limit_days)
            vprint(2, f"[WARM] {raw_tkr}→{yf_sym} @{row_interval} {use_start}→{use_end}")
            df = cached_fetch_indicator_series(yf_sym, start=use_start, end=use_end, interval=row_interval)
            if df is not None and not df.empty:
                warmed += 1

    vprint(1, f"[WARM] Finished. Series warmed: {warmed}. Unresolved tickers: {skipped_resolve}.")


# ============================================================
# TEMP VIEW (read-only DB compat)
# ============================================================

VIEW_SQL = """
DROP VIEW IF EXISTS temp.chairman_flat_v3;

CREATE TEMP VIEW chairman_flat_v3 AS
WITH chairman_union AS (
  SELECT s.post_id,
         s.created_at            AS stage_created_at,
         s.payload               AS result_json
  FROM stages s
  WHERE s.stage='chairman'
    AND s.payload IS NOT NULL
    AND TRIM(s.payload) <> ''
),
joined AS (
  SELECT
    u.post_id,
    /* Use the article’s own timestamp (NOT the stage time) */
    COALESCE(p.posted_at, p.scraped_at) AS article_date,
    u.stage_created_at,
    u.result_json
  FROM chairman_union u
  JOIN posts p ON p.post_id = u.post_id
)
SELECT
  j.post_id,
  j.article_date,
  j.stage_created_at,
  json_extract(j.result_json, '$.plain_english_result')              AS plain_english_result,

  json_extract(j.result_json, '$.final_metrics.ticker')              AS ticker,
  json_extract(j.result_json, '$.final_metrics.timeframe')           AS timeframe,
  json_extract(j.result_json, '$.final_metrics.implied_direction')   AS implied_direction,
  json_extract(j.result_json, '$.final_metrics.direction_strength')  AS direction_strength,
  json_extract(j.result_json, '$.final_metrics.conviction_0to100')   AS conviction_0to100,
  json_extract(j.result_json, '$.final_metrics.risk_level')          AS risk_level,
  json_extract(j.result_json, '$.final_metrics.tradability_score_0to100') AS tradability_0to100,
  json_extract(j.result_json, '$.final_metrics.uncertainty_0to3')    AS uncertainty_0to3,
  json_extract(j.result_json, '$.final_metrics.stale_risk_0to3')     AS stale_risk_0to3,
  json_extract(j.result_json, '$.final_metrics.why')                 AS why_text,

  json_extract(j.result_json, '$.final_metrics.technical.close')     AS close,
  json_extract(j.result_json, '$.final_metrics.technical.price_window_close_pct') AS price_window_close_pct,
  json_extract(j.result_json, '$.final_metrics.technical.rsi14')     AS rsi14,
  json_extract(j.result_json, '$.final_metrics.technical.mfi')       AS mfi,
  json_extract(j.result_json, '$.final_metrics.technical.macd_hist') AS macd_hist,
  json_extract(j.result_json, '$.final_metrics.technical.macd_line') AS macd_line,
  json_extract(j.result_json, '$.final_metrics.technical.macd_signal') AS macd_signal,
  json_extract(j.result_json, '$.final_metrics.technical.trend_direction') AS trend_direction,
  json_extract(j.result_json, '$.final_metrics.technical.trend_strength')  AS trend_strength,
  json_extract(j.result_json, '$.final_metrics.technical.trend_slope_pct_per_day') AS trend_slope_pct_per_day,
  json_extract(j.result_json, '$.final_metrics.technical.sma20')     AS sma20,
  json_extract(j.result_json, '$.final_metrics.technical.sma50')     AS sma50,
  json_extract(j.result_json, '$.final_metrics.technical.sma200')    AS sma200,
  json_extract(j.result_json, '$.final_metrics.technical.golden_cross') AS golden_cross,
  json_extract(j.result_json, '$.final_metrics.technical.price_above_sma20') AS price_above_sma20,
  json_extract(j.result_json, '$.final_metrics.technical.price_above_sma50') AS price_above_sma50,
  json_extract(j.result_json, '$.final_metrics.technical.price_above_sma200') AS price_above_sma200,
  json_extract(j.result_json, '$.final_metrics.technical.vol_ratio') AS vol_ratio,
  json_extract(j.result_json, '$.final_metrics.technical.vol_state') AS vol_state,
  json_extract(j.result_json, '$.final_metrics.technical.distance_to_support_pct') AS distance_to_support_pct,
  json_extract(j.result_json, '$.final_metrics.technical.distance_to_resistance_pct') AS distance_to_resistance_pct,

  json_extract(j.result_json, '$.final_metrics.sentiment.des_raw')   AS des_raw,
  json_extract(j.result_json, '$.final_metrics.sentiment.conf')      AS des_conf,

  json_extract(j.result_json, '$.final_metrics.catalysts')           AS catalysts_json,
  json_extract(j.result_json, '$.final_metrics.watchouts')           AS watchouts_json
FROM joined j;
"""

def create_temp_view(conn: sqlite3.Connection):
    vprint(1, "[SQL] Creating TEMP VIEW chairman_flat_v3 …")
    conn.executescript(VIEW_SQL)
    vprint(2, "[SQL] TEMP VIEW ready.")

# ============================================================
# Strategy & Council Gates
# ============================================================

@dataclass
class StrategyParams:
    # Entry confirmation fallback (legacy path, still useful)
    min_tradability: float = 45.0
    max_uncertainty: float = 2.5
    bear_rsi_max: float = 45.0
    bear_macd_hist_max: float = 0.0
    require_below_sma20_for_short: bool = True
    bull_rsi_min: float = 55.0
    bull_macd_hist_min: float = 0.0
    bull_require_price_above_sma20: bool = True
    bull_confirm_count_needed: int = 3

    # Execution / risk
    stop_type: str = "percent"          # "percent" | "atr"
    stop_loss_pct: float = 0.08
    take_profit_pct: float = 0.16
    trailing_stop_pct: Optional[float] = 0.10
    time_stop_days: int = 15            # NOTE: interpreted as "bars" in our sim
    atr_period: int = 14
    atr_multiple: float = 2.5
    risk_per_trade_pct: float = 0.5
    slippage_bps: float = 10.0
    fees_per_trade: float = 0.0
    partial_take_profit_at: float = 0.10
    partial_size_pct: float = 50.0
    risk_parity_target_vol: Optional[float] = None

    # Meta behavior
    allow_neutral_bias: bool = True
    min_conviction: float = 55.0
    min_strength: float = 2.0
    use_rsi_exit: bool = True
    rsi_exit_overbought: float = 70.0
    rsi_exit_oversold: float = 30.0
    use_macd_exit: bool = True
    use_sma_exit: bool = True
    one_position_per_ticker: bool = True

@dataclass
class CouncilGates:
    # Direction & conviction
    need_implied_up: bool = True
    need_implied_down: bool = False      # for shorts if you choose
    min_direction_strength: float = 1.0
    min_conviction: float = 55.0

    # Risk / tradability
    max_uncertainty: float = 2.5
    max_stale_risk: float = 2.5
    min_tradability: float = 45.0
    allow_risk_levels: Tuple[str, ...] = ("low", "medium")

    # Sentiment
    use_sentiment_gate: bool = False
    min_des_conf: float = 0.35
    min_des_raw: float = -0.02

    # Technical gates (long bias shown)
    require_above_sma20: bool = True
    require_above_sma50: bool = False
    allow_below_sma200: bool = True
    min_trend_strength: float = 1.0
    min_trend_slope_pct_per_day: float = 0.5
    min_rsi: float = 52.0
    max_rsi: float = 78.0
    mfi_max_for_longs: float = 82.0
    min_macd_hist_for_longs: float = 0.0
    max_price_to_resistance_pct: float = 2.5
    min_distance_to_support_pct: float = 2.0
    vol_state_allowed: Tuple[str, ...] = ("normal", "calm")
    min_vol_ratio: float = 0.6
    max_vol_ratio: float = 1.8

    # Catalysts/Watchouts weighting
    use_catalyst_bonus: bool = True
    use_watchout_penalty: bool = True
    w_reg_approval: float = 1.0
    w_overbought_penalty: float = 0.7
    w_delisting: float = 1.0
    min_pretrade_score: float = 0.0

# ============================================================
# Council gates & scoring
# ============================================================

def passes_council_gates_row(r: Dict[str, Any], g: CouncilGates, want_long: bool=True) -> bool:
    implied = (r.get("implied_direction") or "").lower()
    if want_long and g.need_implied_up and implied != "up":
        return False
    if (not want_long) and g.need_implied_down and implied != "down":
        return False

    if _nz(r.get("direction_strength"), 0) < g.min_direction_strength: return False
    if _nz(r.get("conviction_0to100"), 0) < g.min_conviction: return False

    if _nz(r.get("uncertainty_0to3"), 3) > g.max_uncertainty: return False
    if _nz(r.get("stale_risk_0to3"), 3) > g.max_stale_risk: return False
    if _nz(r.get("tradability_0to100"), 0) < g.min_tradability: return False
    if (r.get("risk_level") or "").lower() not in g.allow_risk_levels: return False

    if g.use_sentiment_gate:
        if _nz(r.get("des_conf"), 0.0) < g.min_des_conf: return False
        if _nz(r.get("des_raw"), 0.0) < g.min_des_raw: return False

    pa20 = r.get("price_above_sma20")
    pa50 = r.get("price_above_sma50")
    pa200 = r.get("price_above_sma200")
    if want_long:
        if g.require_above_sma20 and pa20 is False: return False
        if g.require_above_sma50 and pa50 is False: return False
        if (not g.allow_below_sma200) and pa200 is False: return False

        if _nz(r.get("trend_strength"), 0) < g.min_trend_strength: return False
        if _nz(r.get("trend_slope_pct_per_day"), 0.0) < g.min_trend_slope_pct_per_day: return False

        rsi = r.get("rsi14")
        if rsi is not None and (rsi < g.min_rsi or rsi > g.max_rsi): return False

        mfi = r.get("mfi")
        if mfi is not None and mfi > g.mfi_max_for_longs: return False

        mh = r.get("macd_hist")
        if mh is not None and mh < g.min_macd_hist_for_longs: return False

        d2r = r.get("distance_to_resistance_pct")
        if d2r is not None and d2r < g.max_price_to_resistance_pct: return False
        d2s = r.get("distance_to_support_pct")
        if d2s is not None and d2s < g.min_distance_to_support_pct: return False
    else:
        # For shorts you could mirror the logic; we keep it permissive unless implied_down
        pass

    vs = (r.get("vol_state") or "").lower()
    if g.vol_state_allowed and vs and vs not in g.vol_state_allowed: return False
    vr = r.get("vol_ratio")
    if vr is not None and (vr < g.min_vol_ratio or vr > g.max_vol_ratio): return False

    return True

def pretrade_score_row(r: Dict[str, Any], g: CouncilGates) -> float:
    score = 0.0
    c = min(100.0, _nz(r.get("conviction_0to100"), 0.0))
    t = min(100.0, _nz(r.get("tradability_0to100"), 0.0))
    score += 0.01 * c + 0.01 * t

    catalysts = _parse_json_array(r.get("catalysts_json"))
    watchouts = _parse_json_array(r.get("watchouts_json"))

    if g.use_catalyst_bonus and _tag_any_contains(catalysts, ("regulatory", "approval", "fda", "ema")):
        score += g.w_reg_approval
    if g.use_watchout_penalty and _tag_any_contains(watchouts, ("nasdaq", "delist", "reverse split")):
        score -= g.w_delisting

    mfi = r.get("mfi")
    if mfi is not None and mfi >= 80.0:
        score -= g.w_overbought_penalty

    return score

# ============================================================
# Decisions (wrap legacy with Council gates)
# ============================================================

def decide_no_position(r: Dict[str, Any], params: StrategyParams,
                       gates: Optional[CouncilGates]=None) -> Dict[str, Any]:
    """Return a soft decision dict: action, confidence, why, key_metrics."""
    if gates is not None:
        implied = (r.get("implied_direction") or "").lower()
        want_long = (implied == "up") or (params.allow_neutral_bias and implied not in ("up", "down") and (r.get("trend_direction") or "").lower()=="up")
        ok = passes_council_gates_row(r, gates, want_long=want_long)
        pts = pretrade_score_row(r, gates)
        if ok and pts >= gates.min_pretrade_score:
            act = "consider_buy" if want_long else ("consider_short" if implied=="down" else "watch_for_breakout")
            return {"action": act, "confidence": 0.7, "why": f"Council gates passed; score={pts:.2f}",
                    "key_metrics": {"pretrade_score": pts, "passed_gates": True}}
        # else: fall through to legacy guidance (“watch/hold” style)

    # Legacy soft guidance
    implied = (r.get("implied_direction") or "").lower()
    strength = fnum(r.get("direction_strength"), 0) or 0
    conviction = fnum(r.get("conviction_0to100"), 0) or 0
    risk = (r.get("risk_level") or "").lower()
    rsi = fnum(r.get("rsi14"))
    macd_hist = fnum(r.get("macd_hist"))
    trend_dir = (r.get("trend_direction") or "").lower()
    tradability = fnum(r.get("tradability_0to100"), 0) or 0
    des_raw = fnum(r.get("des_raw"))

    if (tradability) < params.min_tradability or _nz(r.get("uncertainty_0to3"), 3) >= params.max_uncertainty:
        return {"action":"avoid","confidence":0.4,"why":"Low tradability or high uncertainty.","key_metrics":{}}

    min_strength = params.min_strength
    min_conviction = params.min_conviction

    bear_bias = (implied == "down" and strength >= min_strength and conviction >= min_conviction and risk in ("low","medium"))
    bull_bias = (implied == "up" and strength >= min_strength and conviction >= min_conviction and risk in ("low","medium"))

    if params.allow_neutral_bias and implied not in ("up","down"):
        if trend_dir == "up":
            bull_bias = True
        elif trend_dir == "down":
            bear_bias = True

    if bear_bias:
        confirms = [trend_dir == "down"]
        if rsi is not None:        confirms.append(rsi <= params.bear_rsi_max)
        if macd_hist is not None:  confirms.append(macd_hist <= params.bear_macd_hist_max)
        if params.require_below_sma20_for_short and (r.get("price_above_sma20") is not None):
            confirms.append(r.get("price_above_sma20") is False)
        if all(x is True for x in confirms if x is not None):
            if rsi is not None and rsi < (params.bear_rsi_max - 10):
                return {"action":"watch_for_relief_then_short","confidence":0.65,"why":"Oversold; short bounces.","key_metrics":{}}
            else:
                why = "Bear bias confirmed by trend/momentum/MA filters."
                if des_raw is not None and des_raw >= 0.6:
                    why += " DES positive — squeeze risk."
                return {"action":"consider_short","confidence":0.7,"why":why,"key_metrics":{}}

    if bull_bias:
        sigs = [trend_dir == "up"]
        if rsi is not None:       sigs.append(rsi >= params.bull_rsi_min)
        if macd_hist is not None: sigs.append(macd_hist >= params.bull_macd_hist_min)
        pa20_ok = (not params.bull_require_price_above_sma20) or (r.get("price_above_sma20") is True)
        confirms_met = sum(1 for s in sigs if s is True)
        if pa20_ok and confirms_met >= params.bull_confirm_count_needed:
            return {"action":"consider_buy","confidence":0.7,"why":"Bull bias with confirmations.","key_metrics":{}}
        else:
            return {"action":"watch_for_breakout","confidence":0.55,"why":"Bias up but confirms insufficient.","key_metrics":{}}

    return {"action":"watch","confidence":0.5,"why":"Insufficient confirmation.","key_metrics":{}}

# ============================================================
# Price / ATR / Position sizing
# ============================================================

def try_import_yf():
    try:
        import yfinance as yf  # type: ignore
        return yf
    except Exception:
        return None

def compute_atr(df, period=14):
    import numpy as np
    high = df["High"]; low = df["Low"]; close_prev = df["Close"].shift(1)
    if getattr(high, "ndim", 1) > 1: high = high.iloc[:, 0]
    if getattr(low, "ndim", 1) > 1: low = low.iloc[:, 0]
    if getattr(close_prev, "ndim", 1) > 1: close_prev = close_prev.iloc[:, 0]
    high = high.values; low = low.values; close_prev = close_prev.values
    tr = np.maximum(high - low, np.maximum(abs(high - close_prev), abs(low - close_prev)))
    atr = pd.Series(tr, index=df.index).rolling(period).mean()
    return atr.values

def derive_initial_stops(entry_price: float, direction: str, params: StrategyParams,
                         atr_value: Optional[float]=None) -> Tuple[float,float]:
    if params.stop_type == "atr" and atr_value is not None and atr_value > 0:
        if direction == "long":
            stop = entry_price - params.atr_multiple * atr_value
            target = entry_price + params.atr_multiple * atr_value
        else:
            stop = entry_price + params.atr_multiple * atr_value
            target = entry_price - params.atr_multiple * atr_value
        return stop, target
    if direction == "long":
        return entry_price * (1 - params.stop_loss_pct), entry_price * (1 + params.take_profit_pct)
    return entry_price * (1 + params.stop_loss_pct), entry_price * (1 - params.take_profit_pct)

def update_trailing_stop(current_price: float, trail_anchor: float, direction: str, params: StrategyParams):
    if params.trailing_stop_pct is None:
        return trail_anchor, None
    ts = params.trailing_stop_pct
    if direction == "long":
        new_anchor = max(trail_anchor, current_price)
        new_stop = new_anchor * (1 - ts)
    else:
        new_anchor = min(trail_anchor, current_price)
        new_stop = new_anchor * (1 + ts)
    return new_anchor, new_stop

def annualized_vol(returns: List[float]) -> Optional[float]:
    if not returns:
        return None
    import statistics as stats
    sd = stats.pstdev(returns)
    return sd * math.sqrt(252.0)

def compute_position_size(account_equity: float, entry_price: float, stop_price: float,
                          params: StrategyParams, ann_vol: Optional[float]=None) -> float:
    if entry_price <= 0:
        return 0.0
    if params.risk_parity_target_vol and ann_vol and ann_vol > 0:
        exposure = (params.risk_parity_target_vol / ann_vol) * account_equity
        shares = exposure / entry_price
        return max(0.0, shares)
    risk_dollars = account_equity * params.risk_per_trade_pct / 100.0
    per_share_risk = max(abs(entry_price - stop_price), 0.01)
    shares = risk_dollars / per_share_risk
    return max(0.0, shares)

def _cli_clear_cache() -> None:
    n = 0
    for p in _PRICE_CACHE_DIR.glob("*"):
        try:
            if p.is_file():
                p.unlink(); n += 1
        except Exception:
            pass
    print(f"Cleared {n} cached files from {_PRICE_CACHE_DIR}")

# ============================================================
# Backtest core
# ============================================================

def load_price_history(yf, ticker: str, start: str, end: str, interval: str) -> Optional[pd.DataFrame]:
    tick = (ticker or "").strip()
    if not tick:
        vprint(1, "⚠️  Skipping empty ticker.")
        return None
    tries, backoff, last_exc = 3, 0.75, None
    for i in range(tries):
        try:
            df = cached_fetch_indicator_series(tick, start=start, end=end, interval=interval)
            if df is None or df.empty:
                vprint(1, f"⚠️  No price data for {tick} @{interval}; skipping.")
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            vprint(2, f"[DATA] Loaded {tick} @{interval}: {len(df)} bars")
            return df
        except Exception as exc:
            last_exc = exc
            msg = str(exc).lower()
            permanent = any(s in msg for s in (
                #"possibly delisted", "no price data", "not found", "quote not found",
                #"no timezone found", "yftzmissingerror"
            ))
            transient = any(s in msg for s in (
                "too many requests", "timed out", "temporarily unavailable",
                "read timed out", "connection aborted", "connection reset",
                "ssl", "remote end closed", "rate limit"
            ))
            if permanent or not transient:
                vprint(1, f"⚠️  {tick}: {exc}. Not retrying.")
                return None
            sleep_s = backoff * (2 ** i)
            vprint(1, f"⏳  Retry {i+1}/{tries} {tick}@{interval}: {exc} (sleep {sleep_s:.2f}s)")
            time.sleep(sleep_s)
    if last_exc:
        vprint(1, f"⚠️  Failed to load {tick}@{interval} after retries: {last_exc}")
    return None

def simulate_trade_stream(df: pd.DataFrame, side: str, entry_time: pd.Timestamp,
                          params: StrategyParams, use_atr: bool) -> Dict[str, Any]:
    """Simple forward simulation from first bar >= signal time to an exit condition."""
    future = df[df.index > entry_time]
    if future.empty:
        return {"filled": False}
    first_idx = future.index[0]
    entry = float(future.loc[first_idx, "Open"])
    atr_val = None
    if use_atr and "ATR" in df.columns:
        prefix = df.loc[:first_idx]
        if not prefix.empty:
            atr_val = float(prefix["ATR"].iloc[-1])

    stop, target = derive_initial_stops(entry, "long" if side=="long" else "short", params, atr_val)
    trail_anchor = entry
    trail_stop = None
    partial_done = False

    exit_price = None
    exit_time = None
    direction = 1 if side=="long" else -1
    start_idx_loc = df.index.get_loc(first_idx)
    max_idx_loc = len(df.index) - 1

    for loc in range(start_idx_loc, max_idx_loc+1):
        ts = df.index[loc]
        o = float(df.loc[ts, "Open"])
        h = float(df.loc[ts, "High"])
        l = float(df.loc[ts, "Low"])
        c = float(df.loc[ts, "Close"])

        if _VERB >= 3 and ((loc - start_idx_loc) % 25 == 0):
            vprint(3, f"[SIM] ts={ts.isoformat()} stop={stop:.4f} target={target:.4f} trail={trail_stop}")

        trail_anchor, new_trail = update_trailing_stop(c, trail_anchor, "long" if side=="long" else "short", params)
        if new_trail is not None:
            trail_stop = new_trail

        if params.partial_take_profit_at and not partial_done:
            tp_level = entry * (1 + params.partial_take_profit_at) if side=="long" else entry * (1 - params.partial_take_profit_at)
            if (side=="long" and h >= tp_level) or (side=="short" and l <= tp_level):
                partial_done = True

        c_rsi = float(df.loc[ts, "RSI14"]) if "RSI14" in df.columns and pd.notna(df.loc[ts, "RSI14"]) else None
        c_macd = float(df.loc[ts, "MACD_LINE"]) if "MACD_LINE" in df.columns and pd.notna(df.loc[ts, "MACD_LINE"]) else None
        c_sig = float(df.loc[ts, "MACD_SIGNAL"]) if "MACD_SIGNAL" in df.columns and pd.notna(df.loc[ts, "MACD_SIGNAL"]) else None
        c_sma20 = float(df.loc[ts, "SMA20"]) if "SMA20" in df.columns and pd.notna(df.loc[ts, "SMA20"]) else None

        # Momentum exits
        if params.use_rsi_exit and c_rsi is not None:
            if side == "long" and c_rsi >= params.rsi_exit_overbought:
                exit_price, exit_time = c, ts; break
            if side == "short" and c_rsi <= params.rsi_exit_oversold:
                exit_price, exit_time = c, ts; break

        if params.use_macd_exit and (c_macd is not None) and (c_sig is not None):
            cross_down = (c_macd < c_sig); cross_up = (c_macd > c_sig)
            if side == "long" and cross_down:
                exit_price, exit_time = c, ts; break
            if side == "short" and cross_up:
                exit_price, exit_time = c, ts; break

        if params.use_sma_exit and c_sma20 is not None:
            if side == "long" and c < c_sma20:
                exit_price, exit_time = c, ts; break
            if side == "short" and c > c_sma20:
                exit_price, exit_time = c, ts; break

        # Stop/Target checks (includes trailing if present)
        eff_stop = min(stop, trail_stop) if (side=="long" and trail_stop) else max(stop, trail_stop) if (side=="short" and trail_stop) else stop
        hit_stop = (side=="long" and l <= eff_stop) or (side=="short" and h >= eff_stop)
        hit_target = (side=="long" and h >= target) or (side=="short" and l <= target)

        if hit_stop and hit_target:
            exit_price, exit_time = eff_stop, ts; break
        elif hit_stop:
            exit_price, exit_time = eff_stop, ts; break
        elif hit_target:
            exit_price, exit_time = target, ts; break

        # Time stop (bars)
        bars_held = loc - start_idx_loc + 1
        if bars_held >= max(1, params.time_stop_days):
            exit_price, exit_time = c, ts; break

    # If no exit hit, close at last bar
    if exit_price is None:
        ts = df.index[-1]
        exit_price = float(df.loc[ts, "Close"])
        exit_time = ts

    partial_px = None
    if partial_done:
        partial_px = entry * (1 + params.partial_take_profit_at) if side=="long" else entry * (1 - params.partial_take_profit_at)

    if partial_px is not None:
        remainder_fraction = max(0.0, 1.0 - (params.partial_size_pct / 100.0))
        pnl = (params.partial_size_pct/100.0) * (partial_px - entry) * direction + remainder_fraction * (exit_price - entry) * direction
    else:
        pnl = (exit_price - entry) * direction

    return {
        "filled": True,
        "entry_time": first_idx, "entry_price": entry,
        "exit_time": exit_time, "exit_price": exit_price,
        "side": side, "pnl_per_share": pnl,
        "partial_taken": partial_done, "partial_px": partial_px,
        "stop": stop, "target": target, "trail_stop": trail_stop,
    }

def _max_drawdown(equity_curve: List[float], start_equity: float) -> float:
    peak = start_equity
    max_dd = 0.0
    for e in equity_curve:
        peak = max(peak, e)
        dd = (peak - e) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)
    return max_dd

def _equity_to_returns(equity_curve: List[float]) -> List[float]:
    rets = []
    for i in range(1, len(equity_curve)):
        if equity_curve[i-1] > 0:
            rets.append((equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1])
    return rets

def _safe_cagr(initial_equity: float, end_equity: float, t0, t1) -> float:
    try:
        if initial_equity <= 0 or t0 is None or t1 is None:
            return 0.0
        span_sec = max((pd.Timestamp(t1) - pd.Timestamp(t0)).total_seconds(), 86400.0)
        years = span_sec / (365.25 * 86400.0)
        ratio = max(float(end_equity) / float(initial_equity), 1e-12)
        log_ratio = math.log(ratio)
        exponent = max(min(log_ratio / years, 20.0), -20.0)
        return math.expm1(exponent)
    except Exception:
        return 0.0

def compute_metrics(trades_df: pd.DataFrame, initial_equity: float) -> Dict[str, float]:
    # Empty-safe defaults
    empty = {
        "trades": 0, "total_pnl": 0.0, "win_rate": 0.0,
        "cagr": 0.0, "sharpe": 0.0, "sortino": 0.0, "profit_factor": 0.0,
        "max_dd": 0.0, "calmar": 0.0, "vol": 0.0, "expectancy": 0.0,
        "end_equity": float(initial_equity),
        "mean_pretrade_score": 0.0, "watchout_hits": 0.0,
        "score": 0.0,
    }
    if trades_df is None or trades_df.empty:
        return empty

    # Core tallies
    equity = trades_df["equity_after"].tolist()
    total_pnl = float(trades_df["pnl"].sum())
    wins = int((trades_df["pnl"] > 0).sum())
    win_rate = wins / len(trades_df) if len(trades_df) > 0 else 0.0

    # Timeline / CAGR
    t0 = trades_df["entry_time"].min()
    t1 = trades_df["exit_time"].max()
    end_equity = float(equity[-1]) if equity else float(initial_equity)
    cagr_raw = _safe_cagr(initial_equity, end_equity, t0, t1)

    # Return series → Sharpe/Sortino/Vol
    trade_rets = _equity_to_returns(equity)
    if trade_rets:
        s = pd.Series(trade_rets)
        avg = float(s.mean())
        std = float(s.std(ddof=0))
        sharpe_raw = (avg / std) * math.sqrt(252) if std > 0 else 0.0
        downside = pd.Series([min(0.0, r) for r in trade_rets])
        dd_std = float(downside.std(ddof=0))
        sortino_raw = (avg / dd_std) * math.sqrt(252) if dd_std > 0 else 0.0
        vol_raw = std * math.sqrt(252)
    else:
        avg = 0.0; std = 0.0; sharpe_raw = 0.0; sortino_raw = 0.0; vol_raw = 0.0

    # Profit factor
    gross_profit = float(trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum())
    gross_loss = float(-trades_df.loc[trades_df["pnl"] < 0, "pnl"].sum())
    pf_raw = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

    # Max DD / Calmar
    max_dd_raw = _max_drawdown([initial_equity] + equity, initial_equity)
    calmar_raw = (cagr_raw / max_dd_raw) if max_dd_raw > 1e-9 else float("inf")

    # Expectancy
    expectancy = float(trades_df["pnl"].mean()) if len(trades_df) > 0 else 0.0

    # Optional diagnostics from signal_info
    mean_pretrade_score = 0.0
    watchout_hits = 0.0
    if "signal_info" in trades_df.columns:
        try:
            s = trades_df["signal_info"].apply(lambda x: x.get("pretrade_score") if isinstance(x, dict) else None).dropna()
            if len(s) > 0:
                mean_pretrade_score = float(pd.Series(s).mean())
            w = trades_df["signal_info"].apply(lambda x: 1.0 if (isinstance(x, dict) and x.get("watchout_hit")) else 0.0)
            watchout_hits = float(pd.Series(w).mean())
        except Exception:
            pass

    # === Stabilise & normalise (bounded metrics) ===
    pf       = clamp(safe_div(gross_profit, abs(gross_loss)), 0.0, 10.0)
    sharpe   = clamp(sharpe_raw, -5.0, 5.0)
    sortino  = clamp(sortino_raw, -5.0, 5.0)
    max_dd   = clamp(max_dd_raw, 0.0, 1.0)           # 0..1
    cagr     = clamp(cagr_raw, -1.0, 3.0)            # -100%..+200%/yr
    vol      = clamp(vol_raw, 0.0, 1.0)
    win_rate = clamp(win_rate, 0.0, 1.0)

    # Normalised 0..1
    cagr_n    = (cagr + 1.0) / 4.0                   # [-1,3] → [0,1]
    maxdd_n   = 1.0 - max_dd
    pf_n      = pf / 10.0
    sharpe_n  = (sharpe + 5.0) / 10.0
    sortino_n = (sortino + 5.0) / 10.0
    win_n     = win_rate
    vol_n     = 1.0 - min(vol, 1.0)                  # prefer lower vol

    # Gates / regularisation (soft penalties)
    trades_gate = 1.0 if len(trades_df) >= 30 else (len(trades_df) / 30.0)
    # Optional hold-time regulariser if present on rows
    if "bars_held" in trades_df.columns:
        target_hold_bars = 20
        avg_hold_bars = float(pd.to_numeric(trades_df["bars_held"], errors="coerce").fillna(0).mean() or 0.0)
        hold_gate = clamp(avg_hold_bars / target_hold_bars, 0.0, 1.0)
    else:
        hold_gate = 1.0

    composite_score = (
        1.00 * cagr_n +
        0.40 * sharpe_n +
        0.25 * sortino_n +
        0.35 * pf_n +
        0.30 * maxdd_n +
        0.10 * win_n +
        0.10 * vol_n
    ) * trades_gate * hold_gate

    # Guard against NaN/Inf
    if any(map(lambda x: isinstance(x, float) and (math.isnan(x) or math.isinf(x)),
               [pf, sharpe, sortino, max_dd, cagr, vol, composite_score])):
        composite_score = 0.0

    return {
        "trades": int(len(trades_df)),
        "total_pnl": total_pnl,
        "win_rate": win_rate,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "profit_factor": pf,
        "max_dd": max_dd,
        "calmar": safe_div(cagr, max_dd) if max_dd > 0 else float("inf"),
        "vol": vol,
        "expectancy": expectancy,
        "end_equity": end_equity,
        "mean_pretrade_score": mean_pretrade_score,
        "watchout_hits": watchout_hits,
        "score": composite_score,  # <— bounded score for GA
    }

def compute_fitness(metrics: Dict[str, float],
                    w_cagr=1.0, w_sharpe=0.5, w_sortino=0.3,
                    w_pf=0.2, w_win=0.0, w_exp=0.1, w_maxdd=-0.5, w_vol=0.0,
                    w_cons=0.3, w_trades=0.2, min_trades_required=10) -> float:
    if metrics.get("trades", 0) < min_trades_required:
        return -1.0
    pf = metrics.get("profit_factor", 0.0)
    if pf == float("inf"): pf = 50.0
    pf_term   = math.log1p(min(pf, 50.0))
    cagr_term = math.log1p(max(metrics.get("cagr", 0.0), 0.0))
    dd_term   = 1.0 / (1.0 + max(0.0, metrics.get("max_dd", 0.0)))
    cons_term = metrics.get("mean_pretrade_score", 0.0)
    trades_term = math.log1p(max(0.0, metrics.get("trades", 0)))

    score  = w_cagr   * cagr_term
    score += w_sharpe * metrics.get("sharpe", 0.0)
    score += w_sortino* metrics.get("sortino", 0.0)
    score += w_pf     * pf_term
    score += w_win    * metrics.get("win_rate", 0.0)
    score += w_exp    * (metrics.get("expectancy", 0.0) / max(1.0, metrics.get("end_equity", 1.0)) * 1000.0)
    score += w_vol    * (-metrics.get("vol", 0.0))
    score += w_maxdd  * metrics.get("max_dd", 0.0)
    score += w_cons   * cons_term
    score += w_trades * trades_term
    return score

# ============================================================
# Backtest runner
# ============================================================

def _to_jsonable_trades(df: pd.DataFrame, limit=0) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    rename = {}
    if "entry" in df.columns and "entry_price" not in df.columns:
        rename["entry"] = "entry_price"
    if "exit" in df.columns and "exit_price" not in df.columns:
        rename["exit"] = "exit_price"
    if "shares" in df.columns and "qty" not in df.columns:
        rename["shares"] = "qty"

    out = df.rename(columns=rename).copy()
    keep = ["ticker","yf_ticker","side","signal_time","entry_time","entry_price",
            "exit_time","exit_price","qty","pnl","pnl_pct","status","signal_info",
            "equity_after","run_id"]
    keep = [c for c in keep if c in out.columns]
    out = out[keep]
    for c in ("signal_time","entry_time","exit_time"):
        if c in out.columns:
            out[c] = out[c].apply(lambda x: x.isoformat() if hasattr(x, "isoformat") else x)
    for c in ("entry_price","exit_price","pnl","pnl_pct","qty","equity_after"):
        if c in out.columns:
            out[c] = out[c].astype(float)
    records = out.to_dict(orient="records")
    if limit and limit > 0:
        records = records[:limit]
    return records

def _group_by(rows, key):
    bucket = {}
    for r in rows:
        k = key(r)
        bucket.setdefault(k, []).append(r)
    for k,v in bucket.items():
        yield k,v

def backtest(conn: sqlite3.Connection, params: StrategyParams, start: str, end: str, initial_equity: float,
             ticker_filter: Optional[str]=None, cache: Optional[Dict[str, pd.DataFrame]]=None,
             sql_filter: Optional[str] = None, gates: Optional[CouncilGates]=None,
             interval: str = "auto", intraday_limit_days: int = 59) -> Dict[str, Any]:

    yf = try_import_yf()
    if yf is None:
        print("⚠️  yfinance not installed — skipping backtest. pip install yfinance")
        return {}

    # Respect vendor intraday limits by clamping the effective window
    eff_start, eff_end = start, end
    if _interval_is_intraday(interval):
        try:
            end_ts = pd.Timestamp(end).normalize() + pd.Timedelta(days=1)
            start_ts = end_ts - pd.Timedelta(days=intraday_limit_days)
            orig_start = pd.Timestamp(start)
            eff_start = max(start_ts, orig_start).strftime("%Y-%m-%d")
            eff_end = pd.Timestamp(end).strftime("%Y-%m-%d")
        except Exception:
            pass

    q = """
    SELECT * FROM chairman_flat_v3
    WHERE ticker IS NOT NULL AND TRIM(ticker) <> ''
      AND implied_direction IS NOT NULL
      AND date(article_date) BETWEEN date(?) AND date(?)
    """
    args = [start, end]
    if ticker_filter:
        q += " AND ticker = ?"; args.append(ticker_filter)
    if sql_filter:
        q += f" AND ({sql_filter})"
    q += " ORDER BY ticker, article_date"

    cols = [d[1] for d in conn.execute("PRAGMA table_info(chairman_flat_v3);")]
    rows = [dict(zip(cols, r)) for r in conn.execute(q, args).fetchall()]
    if not rows:
        vprint(1, "No rows to backtest.")
        metrics = compute_metrics(pd.DataFrame(), initial_equity)
        (PORTFOLIO_DIR / "last_metrics.json").write_text(json.dumps(metrics, indent=2))
        return {"metrics": metrics, "trades": [], "equity_curve": [initial_equity], "trades_csv": str(PORTFOLIO_DIR / "backtest_trades.csv")}

    vprint(1, f"[BT] Rows: {len(rows)} | Window: {start}→{end} | Equity: {initial_equity:.2f}")

    equity = initial_equity
    equity_curve = [equity]
    trades: List[Dict[str, Any]] = []
    price_cache = cache if cache is not None else {}
    open_until: Dict[str, pd.Timestamp] = {} if params.one_position_per_ticker else {}

    for raw_tkr, group in _group_by(rows, key=lambda x: x.get("ticker")):
        yf_sym = resolve_yf_symbol(raw_tkr, eff_start, eff_end)
        if not yf_sym:
            vprint(1, f"⚠️  Resolve failed: '{raw_tkr}' → skip")
            continue

        # Bucket by interval if using auto
        sub_buckets: Dict[str, List[Dict[str, Any]]] = {}
        for r in group:
            row_int = _decide_row_interval(r.get("timeframe"), interval)
            sub_buckets.setdefault(row_int, []).append(r)

        for row_interval, subrows in sub_buckets.items():
            cache_key = f"{yf_sym}@{row_interval}"
            if cache_key not in price_cache:
                use_start, use_end = eff_start, eff_end
                if _interval_is_intraday(row_interval):
                    try:
                        end_ts = pd.Timestamp(end).normalize() + pd.Timedelta(days=1)
                        start_ts = end_ts - pd.Timedelta(days=intraday_limit_days)
                        orig_start = pd.Timestamp(start)
                        use_start = max(start_ts, orig_start).strftime("%Y-%m-%d")
                        use_end = pd.Timestamp(end).strftime("%Y-%m-%d")
                    except Exception:
                        pass

                df = load_price_history(yf, yf_sym, use_start, use_end, row_interval)
                if df is None:
                    vprint(1, f"⚠️  No price data for {yf_sym} @{row_interval}; skip")
                    continue

                df.index = pd.to_datetime(df.index, utc=True)
                if params.stop_type == "atr":
                    df["ATR"] = pd.Series(compute_atr(df, params.atr_period), index=df.index).ffill()

                price_cache[cache_key] = df

            df = price_cache[cache_key]
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            close = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
            ret = close.pct_change().dropna()
            ann_vol = annualized_vol(ret.to_list())

            for r in subrows:
                ts_str = r.get("article_date")
                try:
                    ts = pd.Timestamp(ts_str).tz_convert("UTC") if pd.Timestamp(ts_str).tzinfo else pd.Timestamp(ts_str, tz="UTC")
                except Exception:
                    ts = pd.Timestamp(ts_str, tz="UTC")

                if params.one_position_per_ticker and raw_tkr in open_until:
                    if ts <= open_until[raw_tkr]:
                        vprint(2, f"[BT] skip {raw_tkr}@{ts} — active until {open_until[raw_tkr]}")
                        continue

                no_pos = decide_no_position(r, params, gates=gates)
                action = no_pos["action"]
                side = "long" if action == "consider_buy" else ("short" if action in ("consider_short", "watch_for_relief_then_short") else None)
                vprint(2, f"[BT]  {raw_tkr}@{ts.isoformat()} @{row_interval} → action={action} side={side} why={no_pos.get('why')}")
                if side is None:
                    continue

                snap = df[df.index >= ts]
                if snap.empty:
                    vprint(2, "[BT]   no future bars; skip"); 
                    continue

                snap_ts = snap.index[0]
                approx_entry = float(snap.loc[snap_ts, "Open"])
                atr_val = float(df.loc[:snap_ts, "ATR"].iloc[-1]) if ("ATR" in df.columns and not df.loc[:snap_ts, "ATR"].empty) else None
                stop_approx, _ = derive_initial_stops(approx_entry, side, params, atr_val)
                shares = compute_position_size(equity, approx_entry, stop_approx, params, ann_vol)

                sim = simulate_trade_stream(df, side, ts, params, use_atr=(params.stop_type == "atr"))
                if not sim.get("filled"):
                    vprint(2, "[BT]   not filled; skip")
                    continue

                bps = params.slippage_bps / 10000.0
                pnl_per_share = sim["pnl_per_share"] - approx_entry * bps
                pnl = pnl_per_share * shares - params.fees_per_trade
                equity += pnl
                equity_curve.append(equity)

                pre_score = pretrade_score_row(r, gates) if gates else 0.0
                watchout_hit = _tag_any_contains(_parse_json_array(r.get("watchouts_json")), ("nasdaq", "delist", "reverse split"))

                trades.append({
                    "ticker": raw_tkr,
                    "yf_ticker": yf_sym,
                    "side": side,
                    "signal_time": ts,
                    "entry_time": sim["entry_time"], "exit_time": sim["exit_time"],
                    "entry": sim["entry_price"], "exit": sim["exit_price"],
                    "shares": shares, "pnl": pnl, "equity_after": equity,
                    "partial": sim["partial_taken"], "stop": sim["stop"], "target": sim["target"],
                    "signal_info": {"pretrade_score": pre_score, "watchout_hit": watchout_hit},
                    "interval": row_interval,
                })

                vprint(2, f"[BT]   filled {side} {raw_tkr}@{row_interval}: entry={sim['entry_price']:.4f}, exit={sim['exit_price']:.4f}, pnl={pnl:,.2f}, equity={equity:,.2f}")

                if params.one_position_per_ticker:
                    open_until[raw_tkr] = sim["exit_time"]

    # Wrap up
    out_csv = PORTFOLIO_DIR / "backtest_trades.csv"
    df_tr = pd.DataFrame(trades)
    if df_tr.empty:
        vprint(1, "No executed trades.")
        metrics = compute_metrics(df_tr, initial_equity)
        (PORTFOLIO_DIR / "last_metrics.json").write_text(json.dumps(metrics, indent=2))
        return {"metrics": metrics, "trades": [], "equity_curve": [initial_equity], "trades_csv": str(out_csv)}

    df_tr.to_csv(out_csv, index=False)
    total_pnl = df_tr["pnl"].sum()
    win_rate = (df_tr["pnl"] > 0).mean()
    max_dd = _max_drawdown([initial_equity] + df_tr["equity_after"].tolist(), initial_equity)

    print("\nBacktest summary")
    print(f"  Trades: {len(df_tr)}")
    print(f"  Total PnL: {total_pnl:,.2f}")
    print(f"  Win rate: {win_rate*100:.1f}%")
    print(f"  Max drawdown: {max_dd*100:.1f}%")
    print(f"  Trades CSV: {out_csv}")

    metrics = compute_metrics(df_tr, initial_equity)  # <-- includes bounded composite 'score'
    (PORTFOLIO_DIR / "last_metrics.json").write_text(json.dumps(metrics, indent=2))
    return {
        "trades_csv": str(out_csv),
        "trades": trades,
        "metrics": metrics,
        "equity_curve": [initial_equity] + df_tr["equity_after"].tolist()
    }

# ============================================================
# GA — search space (Chairman-aware)
# ============================================================

SPACE: Dict[str, Tuple[Any, Any]] = {
    # Legacy confirmations & execution
    "min_tradability":    (20.0, 90.0),
    "max_uncertainty":    (1.0, 3.0),
    "bear_rsi_max":       (35.0, 55.0),
    "bear_macd_hist_max": (-0.2, 0.2),
    "require_below_sma20_for_short": (0, 1),
    "bull_rsi_min":       (45.0, 65.0),
    "bull_macd_hist_min": (-0.05, 0.2),
    "bull_require_price_above_sma20": (0, 1),
    "bull_confirm_count_needed": (1, 3),

    "stop_type":          (0, 1),  # 0=percent,1=atr
    "stop_loss_pct":      (0.03, 0.15),
    "take_profit_pct":    (0.06, 0.30),
    "trailing_stop_pct":  (-1.0, 0.2),  # <0 disables
    "time_stop_days":     (5, 30),
    "atr_period":         (10, 30),
    "atr_multiple":       (1.5, 4.0),
    "risk_per_trade_pct": (0.1, 2.0),
    "slippage_bps":       (0.0, 30.0),
    "fees_per_trade":     (0.0, 2.0),
    "partial_take_profit_at": (0.0, 0.2),
    "partial_size_pct":   (0.0, 80.0),
    "risk_parity_target_vol": (0.0, 0.6),  # 0 disables

    "allow_neutral_bias": (0, 1),
    "min_conviction": (45.0, 80.0),
    "min_strength": (1.0, 3.0),
    "use_rsi_exit": (0, 1),
    "rsi_exit_overbought": (60.0, 85.0),
    "rsi_exit_oversold": (15.0, 45.0),
    "use_macd_exit": (0, 1),
    "use_sma_exit": (0, 1),
    "one_position_per_ticker": (0, 1),

    # Council gates (Chairman-aware)
    "need_implied_up": (0, 1),
    "min_direction_strength_gene": (0.0, 3.0),  # name distinct to avoid clash with params
    "max_stale_risk": (1.0, 3.0),
    "cg_min_tradability": (40.0, 90.0),
    "use_sentiment_gate": (0, 1),
    "min_des_conf": (0.2, 0.6),
    "min_des_raw": (-0.02, 0.06),
    "require_above_sma20": (0, 1),
    "require_above_sma50": (0, 1),
    "allow_below_sma200": (0, 1),
    "min_trend_strength": (0.0, 3.0),
    "min_trend_slope_pct_per_day": (0.0, 2.0),
    "min_rsi": (48.0, 60.0),
    "max_rsi": (70.0, 85.0),
    "mfi_max_for_longs": (70.0, 85.0),
    "min_macd_hist_for_longs": (-0.05, 0.2),
    "max_price_to_resistance_pct": (1.0, 5.0),
    "min_distance_to_support_pct": (1.0, 6.0),
    "min_vol_ratio": (0.5, 0.9),
    "max_vol_ratio": (1.4, 2.5),
    "use_catalyst_bonus": (0, 1),
    "use_watchout_penalty": (0, 1),
    "w_reg_approval": (0.3, 1.5),
    "w_overbought_penalty": (0.3, 1.2),
    "w_delisting": (0.5, 1.8),
    "min_pretrade_score": (-1.0, 1.5),

    # Fitness shaping (weights are kept constant by default; genes allow slight tuning)
    "w_cons": (0.0, 0.8),
    "w_trades": (0.0, 0.8),
    "min_trades_required": (5, 20),
}

def random_individual(rng: random.Random) -> Dict[str, Any]:
    g = {}
    for k,(lo,hi) in SPACE.items():
        if k in ("bull_confirm_count_needed","time_stop_days","atr_period","min_trades_required"):
            g[k] = rng.randint(int(lo), int(hi))
        elif k in ("require_below_sma20_for_short","bull_require_price_above_sma20","stop_type",
                   "allow_neutral_bias","use_rsi_exit","use_macd_exit","use_sma_exit","one_position_per_ticker",
                   "need_implied_up","use_sentiment_gate","require_above_sma20","require_above_sma50",
                   "allow_below_sma200","use_catalyst_bonus","use_watchout_penalty"):
            g[k] = rng.randint(int(lo), int(hi))
        else:
            g[k] = rng.uniform(lo, hi)
    return g

def clip_gene(k, v):
    lo,hi = SPACE[k]
    if k in ("bull_confirm_count_needed","time_stop_days","atr_period","min_trades_required"):
        return int(max(lo, min(hi, round(v))))
    if k in ("require_below_sma20_for_short","bull_require_price_above_sma20","stop_type",
             "allow_neutral_bias","use_rsi_exit","use_macd_exit","use_sma_exit","one_position_per_ticker",
             "need_implied_up","use_sentiment_gate","require_above_sma20","require_above_sma50",
             "allow_below_sma200","use_catalyst_bonus","use_watchout_penalty"):
        return int(max(lo, min(hi, round(v))))
    return max(lo, min(hi, v))

def crossover(p1: Dict[str,Any], p2: Dict[str,Any], rng: random.Random) -> Dict[str,Any]:
    child = {}
    for k in SPACE.keys():
        a = p1[k]; b = p2[k]
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            alpha = rng.random()
            w = 0.5 + (alpha - 0.5) * 1.5 if (not isinstance(a, int) or not isinstance(b, int)) else 0.5
            val = a*w + b*(1-w)
            child[k] = clip_gene(k, val)
        else:
            child[k] = rng.choice([a,b])
    return child

def mutate(ind: Dict[str,Any], rng: random.Random, pm: float=0.2, sigma: float=0.2) -> Dict[str,Any]:
    ch = dict(ind)
    for k in SPACE.keys():
        if rng.random() < pm:
            lo,hi = SPACE[k]; span = hi - lo
            if k in ("bull_confirm_count_needed","time_stop_days","atr_period","min_trades_required"):
                step = rng.randint(-2, 2); ch[k] = clip_gene(k, ch[k] + step)
            elif k in ("require_below_sma20_for_short","bull_require_price_above_sma20","stop_type",
                       "allow_neutral_bias","use_rsi_exit","use_macd_exit","use_sma_exit","one_position_per_ticker",
                       "need_implied_up","use_sentiment_gate","require_above_sma20","require_above_sma50",
                       "allow_below_sma200","use_catalyst_bonus","use_watchout_penalty"):
                ch[k] = 1 - int(ch[k])
            else:
                jitter = rng.gauss(0, sigma * span); ch[k] = clip_gene(k, ch[k] + jitter)
    return ch

def genes_to_params(g: Dict[str,Any]) -> StrategyParams:
    return StrategyParams(
        min_tradability=float(g.get("min_tradability", 45.0)),
        max_uncertainty=float(g.get("max_uncertainty", 2.5)),
        bear_rsi_max=float(g.get("bear_rsi_max", 45.0)),
        bear_macd_hist_max=float(g.get("bear_macd_hist_max", 0.0)),
        require_below_sma20_for_short=bool(int(g.get("require_below_sma20_for_short", 1))),
        bull_rsi_min=float(g.get("bull_rsi_min", 55.0)),
        bull_macd_hist_min=float(g.get("bull_macd_hist_min", 0.0)),
        bull_require_price_above_sma20=bool(int(g.get("bull_require_price_above_sma20", 1))),
        bull_confirm_count_needed=int(g.get("bull_confirm_count_needed", 3)),
        stop_type="atr" if int(g.get("stop_type", 0))==1 else "percent",
        stop_loss_pct=float(g.get("stop_loss_pct", 0.08)),
        take_profit_pct=float(g.get("take_profit_pct", 0.16)),
        trailing_stop_pct=None if float(g.get("trailing_stop_pct", 0.10)) < 0 else float(g.get("trailing_stop_pct", 0.10)),
        time_stop_days=int(g.get("time_stop_days", 15)),
        atr_period=int(g.get("atr_period", 14)),
        atr_multiple=float(g.get("atr_multiple", 2.5)),
        risk_per_trade_pct=float(g.get("risk_per_trade_pct", 0.5)),
        slippage_bps=float(g.get("slippage_bps", 10.0)),
        fees_per_trade=float(g.get("fees_per_trade", 0.0)),
        partial_take_profit_at=float(g.get("partial_take_profit_at", 0.10)),
        partial_size_pct=float(g.get("partial_size_pct", 50.0)),
        risk_parity_target_vol=(None if float(g.get("risk_parity_target_vol", 0.0)) <= 0 else float(g.get("risk_parity_target_vol"))),
        allow_neutral_bias=bool(int(g.get("allow_neutral_bias", 1))),
        min_conviction=float(g.get("min_conviction", 55.0)),
        min_strength=float(g.get("min_strength", 2.0)),
        use_rsi_exit=bool(int(g.get("use_rsi_exit", 1))),
        rsi_exit_overbought=float(g.get("rsi_exit_overbought", 70.0)),
        rsi_exit_oversold=float(g.get("rsi_exit_oversold", 30.0)),
        use_macd_exit=bool(int(g.get("use_macd_exit", 1))),
        use_sma_exit=bool(int(g.get("use_sma_exit", 1))),
        one_position_per_ticker=bool(int(g.get("one_position_per_ticker", 1))),
    )

def genes_to_gates(g: Dict[str,Any]) -> CouncilGates:
    return CouncilGates(
        need_implied_up=bool(int(g.get("need_implied_up", 1))),
        min_direction_strength=float(g.get("min_direction_strength_gene", 1.0)),
        min_conviction=float(g.get("min_conviction", 55.0)),
        max_uncertainty=float(g.get("max_uncertainty", 2.5)),
        max_stale_risk=float(g.get("max_stale_risk", 2.5)),
        min_tradability=float(g.get("cg_min_tradability", 45.0)),
        use_sentiment_gate=bool(int(g.get("use_sentiment_gate", 0))),
        min_des_conf=float(g.get("min_des_conf", 0.35)),
        min_des_raw=float(g.get("min_des_raw", -0.02)),
        require_above_sma20=bool(int(g.get("require_above_sma20", 1))),
        require_above_sma50=bool(int(g.get("require_above_sma50", 0))),
        allow_below_sma200=bool(int(g.get("allow_below_sma200", 1))),
        min_trend_strength=float(g.get("min_trend_strength", 1.0)),
        min_trend_slope_pct_per_day=float(g.get("min_trend_slope_pct_per_day", 0.5)),
        min_rsi=float(g.get("min_rsi", 52.0)),
        max_rsi=float(g.get("max_rsi", 78.0)),
        mfi_max_for_longs=float(g.get("mfi_max_for_longs", 82.0)),
        min_macd_hist_for_longs=float(g.get("min_macd_hist_for_longs", 0.0)),
        max_price_to_resistance_pct=float(g.get("max_price_to_resistance_pct", 2.5)),
        min_distance_to_support_pct=float(g.get("min_distance_to_support_pct", 2.0)),
        min_vol_ratio=float(g.get("min_vol_ratio", 0.6)),
        max_vol_ratio=float(g.get("max_vol_ratio", 1.8)),
        use_catalyst_bonus=bool(int(g.get("use_catalyst_bonus", 1))),
        use_watchout_penalty=bool(int(g.get("use_watchout_penalty", 1))),
        w_reg_approval=float(g.get("w_reg_approval", 1.0)),
        w_overbought_penalty=float(g.get("w_overbought_penalty", 0.7)),
        w_delisting=float(g.get("w_delisting", 1.0)),
        min_pretrade_score=float(g.get("min_pretrade_score", 0.0)),
    )

def tournament_select(pop: List[Tuple[Dict[str,Any], float]], k: int, rng: random.Random) -> Dict[str,Any]:
    sample = rng.sample(pop, k=min(k, len(pop)))
    best = max(sample, key=lambda x: x[1])
    return dict(best[0])

def ga_optimize(conn: sqlite3.Connection, start: str, end: str, initial_equity: float,
                pop_size: int, generations: int, tournament_k: int, crossover_rate: float,
                mutation_rate: float, seed: Optional[int],
                weights: Dict[str,float], ticker_filter: Optional[str]=None) -> Dict[str,Any]:

    rng = random.Random(seed)
    price_cache: Dict[str, pd.DataFrame] = {}
    population = [random_individual(rng) for _ in range(pop_size)]
    scored: List[Tuple[Dict[str,Any], float, Dict[str,float]]] = []

    def score_ind(gene: Dict[str,Any]) -> Tuple[float, Dict[str,float]]:
        params = genes_to_params(gene)
        gates  = genes_to_gates(gene)
        bt = backtest(
            conn, params,
            start=start, end=end, initial_equity=initial_equity,
            ticker_filter=ticker_filter, cache=price_cache, gates=gates,
            interval=weights.get("_interval", "auto"),
            intraday_limit_days=int(weights.get("_intraday_limit_days", 59)),
        )
        metrics = bt.get("metrics", {"trades": 0, "score": 0.0})
        # Prefer the bounded score computed inside compute_metrics
        score = float(metrics.get("score", 0.0))
        if math.isnan(score) or math.isinf(score):
            score = -1.0  # hard penalty for pathological metrics
        return score, metrics

    # Seed scoring
    for i, g in enumerate(population, start=1):
        vprint(1, f"[GA] seed {i}/{len(population)}")
        s, m = score_ind(g)
        scored.append((g, s, m))

    elite_keep = max(1, pop_size // 10)
    all_log_rows = []

    for gen in range(generations):
        scored.sort(key=lambda x: x[1], reverse=True)
        best_gene, best_score, best_metrics = scored[0]

        scores_only = [s for (_, s, _) in scored]
        mean_s = float(pd.Series(scores_only).mean()) if scores_only else 0.0
        med_s  = float(pd.Series(scores_only).median()) if scores_only else 0.0
        std_s  = float(pd.Series(scores_only).std(ddof=0)) if scores_only else 0.0
        trades_counts = [m.get("trades", 0) for (_, _, m) in scored]
        mean_tr = float(pd.Series(trades_counts).mean()) if trades_counts else 0.0
        smell = (best_score / med_s) if med_s else float("inf")

        print(f"[GA] Gen {gen} | best={best_score:.4f} (trades={best_metrics.get('trades',0)}, "
              f"CAGR={best_metrics.get('cagr',0):.2%}, PF={best_metrics.get('profit_factor',0):.2f}, "
              f"MaxDD={best_metrics.get('max_dd',0):.2%})")
        print(f"     mean={mean_s:.4f}  median={med_s:.4f}  std={std_s:.4f}  "
              f"avg_trades={mean_tr:.1f}  smell={smell:.2f}")

        all_log_rows.append({
            "gen": gen,
            "best_score": best_score,
            "mean_score": mean_s,
            "median_score": med_s,
            "std_score": std_s,
            "avg_trades": mean_tr,
            "best_trades": best_metrics.get("trades", 0),
            "best_cagr": best_metrics.get("cagr", 0.0),
            "best_sharpe": best_metrics.get("sharpe", 0.0),
            "best_sortino": best_metrics.get("sortino", 0.0),
            "best_pf": best_metrics.get("profit_factor", 0.0),
            "best_maxdd": best_metrics.get("max_dd", 0.0),
            "best_calmar": best_metrics.get("calmar", 0.0),
            "best_expectancy": best_metrics.get("expectancy", 0.0),
        })

        # --- evolve ---
        next_pop: List[Tuple[Dict[str,Any], float, Dict[str,float]]] = []

        # Elitism
        elites = scored[:elite_keep]
        if _VERB >= 1:
            top_s = ", ".join(f"{i}:{s:.3f}" for i,(_,s,_) in enumerate(elites))
            print(f"     elites[{elite_keep}]: {top_s}")
        next_pop.extend(elites)

        # Breed the rest
        while len(next_pop) < pop_size:
            p1 = tournament_select(scored, tournament_k, rng)
            p2 = tournament_select(scored, tournament_k, rng)
            child = crossover(p1, p2, rng) if rng.random() < crossover_rate else dict(rng.choice([p1, p2]))
            child = mutate(child, rng, pm=mutation_rate, sigma=0.18)
            s, m = score_ind(child)
            next_pop.append((child, s, m))

        scored = next_pop

        # Persist rolling artifacts each gen
        best_gene, best_score, best_metrics = max(scored, key=lambda x: x[1])
        (PORTFOLIO_DIR / "ga_progress.json").write_text(json.dumps({
            "gen": gen,
            "best_score": best_score,
            "best_metrics": best_metrics,
            "best_gene": best_gene
        }, indent=2))
        pd.DataFrame(all_log_rows).to_csv(PORTFOLIO_DIR / "ga_log.csv", index=False)

    # Finished GA — dump final artifacts
    scored.sort(key=lambda x: x[1], reverse=True)
    best_gene, best_score, best_metrics = scored[0]
    print("\n[GA] Finished.")
    print(f"     best_score={best_score:.4f}  trades={best_metrics.get('trades',0)}  "
          f"CAGR={best_metrics.get('cagr',0):.2%}  PF={best_metrics.get('profit_factor',0):.2f}  "
          f"Sharpe={best_metrics.get('sharpe',0):.2f}  MaxDD={best_metrics.get('max_dd',0):.2%}")

    (PORTFOLIO_DIR / "best_genes.json").write_text(json.dumps(best_gene, indent=2))
    (PORTFOLIO_DIR / "best_metrics.json").write_text(json.dumps(best_metrics, indent=2))
    pd.DataFrame(all_log_rows).to_csv(PORTFOLIO_DIR / "ga_log.csv", index=False)

    return {
        "best_gene": best_gene,
        "best_score": best_score,
        "best_metrics": best_metrics,
        "ga_log_csv": str(PORTFOLIO_DIR / "ga_log.csv")
    }

# ============================================================
# CLI
# ============================================================

def _default_weights():
    # Defaults — you said “just start with the default weights”
    return {
        "w_cagr": 1.0,
        "w_sharpe": 0.5,
        "w_sortino": 0.3,
        "w_pf": 0.2,
        "w_win": 0.0,
        "w_exp": 0.1,
        "w_maxdd": -0.5,
        "w_vol": 0.0,
    }

def _pretty_kv(d: Dict[str, Any]) -> str:
    lines = []
    for k in sorted(d.keys()):
        v = d[k]
        if isinstance(v, float):
            lines.append(f"  {k}: {v:.6g}")
        else:
            lines.append(f"  {k}: {v}")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser(
        description="Portfolio Manager probe / backtester / GA optimizer (Chairman-aware)."
    )
    ap.add_argument("--db", required=True, help="Path to the SQLite database (read-only is fine).")
    ap.add_argument("--start", required=True, help="Start date (YYYY-MM-DD).")
    ap.add_argument("--end", required=True, help="End date (YYYY-MM-DD).")
    ap.add_argument("--initial-equity", type=float, default=100000.0, help="Starting equity.")
    ap.add_argument("--ticker", default=None, help="Restrict to a single Chairman ticker (optional).")
    ap.add_argument("--sql-filter", default=None, help="Extra SQL WHERE fragment for chairman_flat_v3 (optional).")

    # GA knobs
    ap.add_argument("--ga", action="store_true", help="Run GA optimization. If not set, runs a single backtest with defaults.")
    ap.add_argument("--pop", type=int, default=60, help="Population size.")
    ap.add_argument("--gens", type=int, default=80, help="Generations.")
    ap.add_argument("--tournament-k", type=int, default=4, help="Tournament selection size.")
    ap.add_argument("--cx", type=float, default=0.9, help="Crossover rate.")
    ap.add_argument("--mut", type=float, default=0.25, help="Mutation rate.")
    ap.add_argument("--seed", type=int, default=None, help="Random seed (optional).")

    # Bar Sizes
    ap.add_argument(
        "--interval",
        default="auto",
        choices=["auto","1d","4h","1h","30m","15m","5m"],
        help="Bar size for price data. 'auto' maps from Chairman timeframe: "
             "intraday→15m, swing_days→1h, swing_weeks→4h, else 1d."
    )
    ap.add_argument(
        "--intraday-limit-days",
        type=int,
        default=59,
        help="Max lookback window (days) for sub-hourly bars (Yahoo often ~60d)."
    )

    # Using precalculated GA results
    ap.add_argument(
        "--use-best",
        action="store_true",
        help="Load genes from a JSON file (default: portfolio/best_genes.json) and run a single backtest with them."
    )
    ap.add_argument(
        "--genes-file",
        default=str(PORTFOLIO_DIR / "best_genes.json"),
        help="Path to a genes JSON produced by GA (default: portfolio/best_genes.json)."
    )

    # Cache warmer
    ap.add_argument("--warm-cache", action="store_true",
                    help="Warm the price cache for all (ticker,interval) combos needed by this window.")
    ap.add_argument("--warm-only", action="store_true",
                    help="Perform cache warm then exit without running GA/backtest.")


    # Misc
    ap.add_argument("--clear-cache", action="store_true", help="Clear price cache and exit.")
    ap.add_argument("--verbose", type=int, default=None, help="Override WOS_VERBOSE (0..3).")

    args = ap.parse_args()

    # Verbosity override
    global _VERB
    if args.verbose is not None:
        _VERB = int(args.verbose)

    if args.clear_cache:
        _cli_clear_cache()
        return

    # Connect DB and create TEMP VIEW
    if not os.path.exists(args.db):
        print(f"❌ DB not found: {args.db}")
        sys.exit(2)

    vprint(1, f"[BOOT] portfolio_probe starting — db={args.db} window={args.start}→{args.end} equity={args.initial_equity}")
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    create_temp_view(conn)

    # Optional cache warming pass
    if args.warm_cache or args.warm_only:
        warm_cache(
            conn=conn,
            start=args.start,
            end=args.end,
            interval=args.interval,
            intraday_limit_days=int(args.intraday_limit_days),
        )
        if args.warm_only:
            return

    # --------------------------------------------
    # Mode 1: Backtest from saved GA genes
    # --------------------------------------------
    if getattr(args, "use_best", False):
        genes_path = getattr(args, "genes_file", str(PORTFOLIO_DIR / "best_genes.json"))
        try:
            genes = load_genes_file(genes_path)
        except Exception as exc:
            print(f"❌ Failed to load genes from {genes_path}: {exc}")
            sys.exit(3)

        vprint(1, f"[RUN] Using genes file: {genes_path}")
        res = run_backtest_from_genes(
            conn=conn,
            genes=genes,
            start=args.start,
            end=args.end,
            initial_equity=args.initial_equity,
            ticker_filter=args.ticker,
            interval=args.interval,
            intraday_limit_days=int(args.intraday_limit_days),
        )
        # If you want to honor extra SQL filter even in genes mode, re-run with sql_filter:
        if args.sql_filter:
            vprint(1, f"[RUN] Re-running with extra SQL filter: {args.sql_filter}")
            # Convert genes→params/gates directly and call backtest to pass sql_filter through.
            params = genes_to_params(genes)
            gates  = genes_to_gates(genes)
            res = backtest(
                conn=conn,
                params=params,
                start=args.start,
                end=args.end,
                initial_equity=args.initial_equity,
                ticker_filter=args.ticker,
                sql_filter=args.sql_filter,
                gates=gates,
                interval=args.interval,
                intraday_limit_days=args.intraday_limit_days,
            )
            print("\n[BT] Metrics:")
            print(json.dumps(res.get("metrics", {}), indent=2))
            if res.get("trades_csv"):
                print("[BT] Trades CSV:", res["trades_csv"])
            print("[BT] Done.")
        return

    # --------------------------------------------
    # Mode 2: GA Optimization
    # --------------------------------------------
    if args.ga:
        # GA run with default weights
        weights = _default_weights()
        # Carry CLI interval/intraday clamp through GA scoring via weights shim
        weights["_interval"] = args.interval
        weights["_intraday_limit_days"] = args.intraday_limit_days

        print("[GA] Using default weights:")
        print(_pretty_kv(weights))
        print(f"[GA] Search space: {len(SPACE)} genes | pop={args.pop} gens={args.gens} "
              f"k={args.tournament_k} cx={args.cx} mut={args.mut}")
        res = ga_optimize(
            conn=conn,
            start=args.start,
            end=args.end,
            initial_equity=args.initial_equity,
            pop_size=args.pop,
            generations=args.gens,
            tournament_k=args.tournament_k,
            crossover_rate=args.cx,
            mutation_rate=args.mut,
            seed=args.seed,
            weights=weights,
            ticker_filter=args.ticker,
        )
        print("\n[GA] Best genes saved to:", PORTFOLIO_DIR / "best_genes.json")
        print("[GA] Best metrics saved to:", PORTFOLIO_DIR / "best_metrics.json")
        print("[GA] Log CSV:", PORTFOLIO_DIR / "ga_log.csv")
        print("[GA] Done.")
        return

    # --------------------------------------------
    # Mode 3: Single backtest with stock defaults
    # --------------------------------------------
    print("[BT] Single-run backtest with default params & default council gates.")
    params = StrategyParams()
    gates = CouncilGates()
    if args.sql_filter:
        vprint(1, f"[BT] Extra SQL filter: {args.sql_filter}")
    bt = backtest(
        conn=conn,
        params=params,
        start=args.start,
        end=args.end,
        initial_equity=args.initial_equity,
        ticker_filter=args.ticker,
        sql_filter=args.sql_filter,
        gates=gates,
        interval=args.interval,
        intraday_limit_days=args.intraday_limit_days,
    )
    print("\n[BT] Metrics:")
    print(json.dumps(bt.get("metrics", {}), indent=2))
    if bt.get("trades_csv"):
        print("[BT] Trades CSV:", bt["trades_csv"])
    print("[BT] Done.")


if __name__ == "__main__":
    main()
