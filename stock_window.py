#!/usr/bin/env python3
"""
stock_window.py — Fetch a ticker's price window centered on a specific datetime.

Features
- Importable function get_stock_window(...) for backend calls.
- CLI with args: ticker, center, before, after, (optional) interval, csv output.
- Auto interval selection by total span; gracefully degrades if data too old for minute bars.
- Adds 'seconds_from_center' and 'return_from_t0' columns for quick visual/backtest use.
- Includes pre/post-market where available (prepost=True).

Dependencies: yfinance, pandas, numpy, python-dateutil (for robust datetime parsing)
    pip install yfinance pandas numpy python-dateutil
"""

from __future__ import annotations

import sys
import json
import argparse
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from dateutil import parser as dtparse
import yfinance as yf


# ---------- Helpers ----------
def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_center(center_str: Optional[str]) -> datetime:
    """
    Parse center datetime. Accepts ISO8601 (with or without 'Z'), or the keywords:
    - None / "now" / "today" → current UTC time
    """
    if not center_str or center_str.strip().lower() in {"now", "today"}:
        return _now_utc()
    dt = dtparse.parse(center_str)
    # Make timezone-aware in UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def _parse_span(span_str: Optional[str], default: str) -> pd.Timedelta:
    """
    Parse span like '6h', '2d', '15m', '1w'. Falls back to default if None.
    """
    s = span_str or default
    try:
        # Pandas understands strings like '6h', '2d', '15m', '1w'
        td = pd.Timedelta(s)
        if td <= pd.Timedelta(0):
            raise ValueError
        return td
    except Exception:
        raise ValueError(f"Invalid span '{span_str}'. Use forms like 6h, 2d, 15m, 1w.")


def _auto_interval(total_span: pd.Timedelta) -> str:
    """
    Pick a reasonable default interval based on total span (before+after).
    """
    seconds = total_span.total_seconds()
    if seconds <= 3 * 3600:      # ≤ 3 hours
        return "1m"
    if seconds <= 2 * 86400:     # ≤ 2 days
        return "5m"
    if seconds <= 14 * 86400:    # ≤ 14 days
        return "15m"
    if seconds <= 60 * 86400:    # ≤ 60 days
        return "1h"
    if seconds <= 2 * 365 * 86400:  # ≤ ~2 years
        return "1d"
    return "1wk"


# Limits commonly observed via Yahoo Finance:
# - 1m: ~30 days lookback
# - 5m: ~60 days lookback
# (15m/1h/1d generally fine for longer)
_INTERVAL_DEGRADE_ORDER = ["1m", "5m", "15m", "1h", "1d", "1wk"]
_INTERVAL_MAX_AGE = {
    "1m": pd.Timedelta(days=30),
    "5m": pd.Timedelta(days=60),
    # Others left "unbounded" for practical purposes
}


def _needs_degrade(interval: str, start_utc: datetime, now_utc: datetime) -> bool:
    max_age = _INTERVAL_MAX_AGE.get(interval)
    if not max_age:
        return False
    return (now_utc - start_utc) > max_age


def _degrade_interval(interval: str) -> str:
    try:
        idx = _INTERVAL_DEGRADE_ORDER.index(interval)
        return _INTERVAL_DEGRADE_ORDER[min(idx + 1, len(_INTERVAL_DEGRADE_ORDER) - 1)]
    except ValueError:
        return "1d"


def _closest_bar_at_or_before(df: pd.DataFrame, center_dt: datetime) -> Optional[int]:
    """
    Return the index (integer) of the last row with timestamp <= center_dt.
    Assumes df['timestamp'] is tz-aware UTC.
    """
    if df.empty:
        return None
    # Boolean mask where rows are at or before center
    mask = df["timestamp"] <= center_dt
    if not mask.any():
        return None
    return np.where(mask)[0][-1]


@dataclass
class StockWindowResult:
    ticker: str
    interval: str
    start_utc: str
    end_utc: str
    center_utc: str
    t0_timestamp_utc: Optional[str]
    t0_close: Optional[float]
    rows: int
    note: Optional[str] = None
    data: Optional[List[Dict[str, Any]]] = None  # can be omitted for large payloads


def get_stock_window(
    ticker: str,
    center: Optional[str] = None,
    before: Optional[str] = "3d",
    after: Optional[str] = "3d",
    interval: Optional[str] = None,
    include_data: bool = True,
    auto_adjust: bool = False,
    prepost: bool = True,
) -> StockWindowResult:
    """
    Core backend function.
    """
    center_dt = _parse_center(center)
    before_td = _parse_span(before, "3d")
    after_td = _parse_span(after, "3d")
    start_dt = (center_dt - before_td).astimezone(timezone.utc)
    end_dt = (center_dt + after_td).astimezone(timezone.utc)
    total_span = before_td + after_td

    chosen_interval = interval or _auto_interval(total_span)

    # Graceful degradation for old minute bars
    now = _now_utc()
    degrade_notes: List[str] = []
    while _needs_degrade(chosen_interval, start_dt, now):
        new_interval = _degrade_interval(chosen_interval)
        if new_interval == chosen_interval:
            break
        degrade_notes.append(f"{chosen_interval} too old → {new_interval}")
        chosen_interval = new_interval

    # ---- IMPORTANT FIXES START ----
    # yfinance sometimes fails when start/end include seconds in strings.
    # Use tz-aware datetimes directly, snapped to minute precision for intraday.
    def _to_min(dt: datetime) -> datetime:
        return dt.replace(second=0, microsecond=0)

    start_dt_for_fetch = _to_min(start_dt)
    end_dt_for_fetch = _to_min(end_dt)

    # Small guard: ensure end > start at least by 1 minute to avoid empty windows
    if end_dt_for_fetch <= start_dt_for_fetch:
        end_dt_for_fetch = start_dt_for_fetch + pd.Timedelta(minutes=1)

    yf_tkr = yf.Ticker(ticker)

    try:
        # Pass datetime objects (not strings) so yfinance/pandas handle parsing robustly.
        df = yf_tkr.history(
            start=start_dt_for_fetch,
            end=end_dt_for_fetch,
            interval=chosen_interval,
            auto_adjust=auto_adjust,
            prepost=prepost,
            actions=False,
        )
    except Exception as e:
        # Retry once with seconds already zeroed shouldn't be necessary, but in case of
        # other parsing hiccups return a structured error.
        return StockWindowResult(
            ticker=ticker,
            interval=chosen_interval,
            start_utc=start_dt.isoformat(),
            end_utc=end_dt.isoformat(),
            center_utc=center_dt.isoformat(),
            t0_timestamp_utc=None,
            t0_close=None,
            rows=0,
            note=f"Fetch error: {e}",
            data=[] if include_data else None,
        )
    # ---- IMPORTANT FIXES END ----

    if df is None or df.empty:
        return StockWindowResult(
            ticker=ticker,
            interval=chosen_interval,
            start_utc=start_dt.isoformat(),
            end_utc=end_dt.isoformat(),
            center_utc=center_dt.isoformat(),
            t0_timestamp_utc=None,
            t0_close=None,
            rows=0,
            note="No data returned for the requested window (market closed, illiquid, or interval too coarse/fine).",
            data=[] if include_data else None,
        )

    # Normalize index to tz-aware UTC
    idx = df.index
    if idx.tz is None:
        ts = idx.tz_localize(timezone.utc)
    else:
        ts = idx.tz_convert(timezone.utc)

    df = df.copy()
    df.insert(0, "timestamp", ts.to_pydatetime())

    # Find t0 (bar at or before center)
    t0_idx = _closest_bar_at_or_before(df, center_dt)
    t0_ts_iso = None
    t0_close = None
    if t0_idx is not None:
        t0_ts = df.iloc[t0_idx]["timestamp"]
        t0_ts_iso = t0_ts.isoformat()
        t0_close = float(df.iloc[t0_idx]["Close"])

    # Derived columns
    df["seconds_from_center"] = (df["timestamp"] - center_dt).dt.total_seconds()
    if t0_close and t0_close != 0.0:
        df["return_from_t0"] = df["Close"] / t0_close - 1.0
    else:
        df["return_from_t0"] = np.nan

    # Build records
    out_records = None
    if include_data:
        cols = [
            "timestamp", "Open", "High", "Low", "Close", "Volume",
            "seconds_from_center", "return_from_t0",
        ]
        have = [c for c in cols if c in df.columns]
        recs = df[have].copy()
        recs["timestamp"] = recs["timestamp"].map(lambda d: d.isoformat())
        out_records = recs.to_dict(orient="records")

    note = "; ".join(degrade_notes) if degrade_notes else None

    return StockWindowResult(
        ticker=ticker,
        interval=chosen_interval,
        start_utc=start_dt.isoformat(),
        end_utc=end_dt.isoformat(),
        center_utc=center_dt.isoformat(),
        t0_timestamp_utc=t0_ts_iso,
        t0_close=t0_close,
        rows=len(df),
        note=note,
        data=out_records,
    )


# ---------- CLI ----------
def _cli() -> None:
    p = argparse.ArgumentParser(
        description="Fetch a centered stock-price window for backtesting/inspection."
    )
    p.add_argument("ticker", help="Ticker symbol, e.g., AAPL")
    p.add_argument(
        "--center",
        default=None,
        help="Center datetime (ISO8601). Examples: '2025-09-30T14:30:00Z', 'now', 'today'. "
             "Defaults to now UTC.",
    )
    p.add_argument("--before", default="3d", help="Span before center, e.g., '6h', '2d'. Default: 3d")
    p.add_argument("--after", default="3d", help="Span after center, e.g., '6h', '2d'. Default: 3d")
    p.add_argument(
        "--interval",
        default=None,
        choices=["1m", "5m", "15m", "1h", "1d", "1wk"],
        help="Explicit Yahoo interval. If omitted, it's chosen automatically.",
    )
    p.add_argument(
        "--no-data",
        action="store_true",
        help="If set, omit the data array (metadata only).",
    )
    p.add_argument(
        "--csv",
        default=None,
        help="Optional path to write the window data as CSV.",
    )
    p.add_argument(
        "--auto-adjust",
        action="store_true",
        help="Use yfinance auto-adjusted OHLC (splits/dividends). Default: False",
    )
    p.add_argument(
        "--no-prepost",
        action="store_true",
        help="Exclude pre/post-market. Default includes it when available.",
    )

    args = p.parse_args()

    res = get_stock_window(
        ticker=args.ticker,
        center=args.center,
        before=args.before,
        after=args.after,
        interval=args.interval,
        include_data=not args.no_data,
        auto_adjust=args.auto_adjust,
        prepost=not args.no_prepost,
    )

    # Optional CSV dump of data rows
    if args.csv and res.data is not None:
        # Write timestamped rows to CSV
        df = pd.DataFrame(res.data)
        df.to_csv(args.csv, index=False)

    # Print JSON to stdout (metadata + possibly data)
    print(json.dumps(asdict(res), ensure_ascii=False))


if __name__ == "__main__":
    _cli()
