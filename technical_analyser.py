#!/usr/bin/env python3
"""
technical_analyser.py — Token‑lean technical analysis toolkit for Wisdom of Sheep

Implements the TECHNICAL tools used by the two‑stage researcher:
- price_window                 {"ticker":"T","from":"YYYY-MM-DD","to":"YYYY-MM-DD","interval":"1d"}
- compute_indicators           {"ticker":"T","window_days":N}
- trend_strength               {"ticker":"T","lookback_days":N}
- volatility_state             {"ticker":"T","days":N,"baseline_days":M}
- support_resistance_check     {"ticker":"T","days":N}
- bollinger_breakout_scan      {"ticker":"T","days":N}
- obv_trend                    {"ticker":"T","lookback_days":N}
- mfi_flow                     {"ticker":"T","period":N}

Notes
- Uses yfinance for price data. If present, you may also import stock_window.get_stock_window for advanced windows.
- All functions return compact, JSON‑serialisable dicts.
- Includes a CLI: run a single tool or a small JSON test harness (`run-plan`).

Dependencies
    pip install yfinance pandas numpy python-dateutil
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from dateutil import parser as dtparse

from yfinance_throttle import throttle_yfinance

# --------- Optional: use stock_window if available (not required) ---------
try:
    from stock_window import get_stock_window  # noqa: F401
    _HAS_STOCK_WINDOW = True
except Exception:
    _HAS_STOCK_WINDOW = False

# ===============
# Utils
# ===============
DATE_FMT = "%Y-%m-%d"


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_date(d: str) -> datetime:
    return _to_utc(dtparse.parse(d))


def _json(obj: Any) -> None:
    print(json.dumps(obj, ensure_ascii=False))


def _fetch_daily_history(ticker: str, days: int) -> pd.DataFrame:
    """Fetch last N days of DAILY OHLCV (auto_adjust=False)."""
    period = f"{max(days, 5)}d"
    t = yf.Ticker(ticker)
    throttle_yfinance()
    df = t.history(period=period, interval="1d", auto_adjust=False, actions=False)
    if df is None or df.empty:
        return pd.DataFrame()
    # Normalize index to tz‑aware UTC timestamps
    idx = df.index
    if idx.tz is None:
        ts = idx.tz_localize(timezone.utc)
    else:
        ts = idx.tz_convert(timezone.utc)
    df = df.copy()
    df.insert(0, "timestamp", ts.to_pydatetime())
    return df


# ===============
# Indicators
# ===============

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd = ema_fast - ema_slow
    sig = _ema(macd, signal)
    hist = macd - sig
    return pd.DataFrame({"macd": macd, "signal": sig, "hist": hist})


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


# ===============
# Tools
# ===============

def tool_price_window(ticker: str, from_date: str, to_date: str, interval: str = "1d") -> Dict[str, Any]:
    if interval != "1d":
        note = "Only interval=1d supported here; using 1d."
    else:
        note = None
    start = _parse_date(from_date)
    end = _parse_date(to_date) + pd.Timedelta(days=1)  # inclusive end
    t = yf.Ticker(ticker)
    throttle_yfinance()
    try:
        df = t.history(start=start, end=end, interval="1d", auto_adjust=False, actions=False)
    except Exception as e:
        return {
            "tool": "price_window",
            "ticker": ticker,
            "from": from_date,
            "to": to_date,
            "interval": "1d",
            "rows": 0,
            "note": f"fetch_error: {e}",
            "data": [],
        }
    if df is None or df.empty:
        return {
            "tool": "price_window",
            "ticker": ticker,
            "from": from_date,
            "to": to_date,
            "interval": "1d",
            "rows": 0,
            "note": "no_data",
            "data": [],
        }
    idx = df.index
    ts = idx.tz_localize(timezone.utc) if idx.tz is None else idx.tz_convert(timezone.utc)
    df = df.copy()
    df.insert(0, "timestamp", ts.to_pydatetime())
    rows = df[["timestamp", "Open", "High", "Low", "Close", "Volume"]].copy()
    rows["timestamp"] = rows["timestamp"].map(lambda d: d.strftime("%Y-%m-%dT%H:%M:%SZ"))
    return {
        "tool": "price_window",
        "ticker": ticker,
        "from": from_date,
        "to": to_date,
        "interval": "1d",
        "rows": len(rows),
        "note": note,
        "data": rows.to_dict(orient="records"),
    }


def tool_compute_indicators(ticker: str, window_days: int) -> Dict[str, Any]:
    df = _fetch_daily_history(ticker, window_days + 220)
    if df.empty:
        return {"tool": "compute_indicators", "ticker": ticker, "window_days": window_days, "rows": 0, "note": "no_data"}
    close = df["Close"].astype(float)
    rsi14 = _rsi(close, 14)
    macd = _macd(close)
    sma20 = _sma(close, 20)
    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)

    last = df.iloc[-1]
    out = {
        "tool": "compute_indicators",
        "ticker": ticker,
        "asof": last["timestamp"].strftime("%Y-%m-%dT%H:%M:%SZ"),
        "close": float(last["Close"]),
        "rsi14": float(rsi14.iloc[-1]) if not np.isnan(rsi14.iloc[-1]) else None,
        "macd": {
            "line": float(macd["macd"].iloc[-1]) if not np.isnan(macd["macd"].iloc[-1]) else None,
            "signal": float(macd["signal"].iloc[-1]) if not np.isnan(macd["signal"].iloc[-1]) else None,
            "hist": float(macd["hist"].iloc[-1]) if not np.isnan(macd["hist"].iloc[-1]) else None,
        },
        "sma": {
            "sma20": float(sma20.iloc[-1]) if not np.isnan(sma20.iloc[-1]) else None,
            "sma50": float(sma50.iloc[-1]) if not np.isnan(sma50.iloc[-1]) else None,
            "sma200": float(sma200.iloc[-1]) if not np.isnan(sma200.iloc[-1]) else None,
        },
        "crosses": {
            "golden_cross": bool(sma50.iloc[-1] > sma200.iloc[-1]) if not np.isnan(sma50.iloc[-1]) and not np.isnan(sma200.iloc[-1]) else False,
            "price_above_sma20": bool(last["Close"] > sma20.iloc[-1]) if not np.isnan(sma20.iloc[-1]) else False,
            "price_above_sma50": bool(last["Close"] > sma50.iloc[-1]) if not np.isnan(sma50.iloc[-1]) else False,
            "price_above_sma200": bool(last["Close"] > sma200.iloc[-1]) if not np.isnan(sma200.iloc[-1]) else False,
        },
        "rows": int(len(df)),
    }
    return out


def tool_trend_strength(ticker: str, lookback_days: int) -> Dict[str, Any]:
    df = _fetch_daily_history(ticker, lookback_days + 5)
    if df.empty or len(df) < max(lookback_days // 2, 10):
        return {"tool": "trend_strength", "ticker": ticker, "lookback_days": lookback_days, "rows": int(len(df)), "note": "insufficient_data"}
    close = df["Close"].astype(float).tail(lookback_days)
    y = np.log(close.values)
    x = np.arange(len(y))
    # Linear regression via polyfit
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = ((y - y_hat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    # Daily pct slope approximation
    slope_pct_per_day = (np.exp(slope) - 1.0) * 100.0

    # Classify
    direction = "up" if slope_pct_per_day > 0 else ("down" if slope_pct_per_day < 0 else "flat")
    strength = 0
    abs_s = abs(slope_pct_per_day)
    if abs_s >= 0.30 and r2 >= 0.40:
        strength = 3
    elif abs_s >= 0.15 and r2 >= 0.25:
        strength = 2
    elif abs_s >= 0.05 and r2 >= 0.10:
        strength = 1
    else:
        strength = 0

    last = df.iloc[-1]
    return {
        "tool": "trend_strength",
        "ticker": ticker,
        "asof": last["timestamp"].strftime("%Y-%m-%dT%H:%M:%SZ"),
        "lookback_days": lookback_days,
        "slope_pct_per_day": round(float(slope_pct_per_day), 4),
        "r2": round(float(r2), 4),
        "direction": direction,
        "strength": strength,
        "rows": int(len(close)),
    }


def tool_volatility_state(ticker: str, days: int, baseline_days: int) -> Dict[str, Any]:
    need = days + baseline_days + 5
    df = _fetch_daily_history(ticker, need)
    if df.empty or len(df) < max(days + baseline_days // 2, 20):
        return {"tool": "volatility_state", "ticker": ticker, "rows": int(len(df)), "note": "insufficient_data"}
    close = df["Close"].astype(float)
    ret = close.pct_change().dropna()
    curr = ret.tail(days)
    base = ret.tail(days + baseline_days).head(baseline_days) if len(ret) >= days + baseline_days else ret.head(0)
    rv_curr = curr.std(ddof=0) * np.sqrt(252) * 100.0 if len(curr) > 1 else None
    rv_base = base.std(ddof=0) * np.sqrt(252) * 100.0 if len(base) > 1 else None

    ratio = (rv_curr / rv_base) if (rv_curr and rv_base and rv_base > 0) else None
    state = None
    if ratio is not None:
        if ratio >= 1.2:
            state = "elevated"
        elif ratio <= 0.8:
            state = "compressed"
        else:
            state = "normal"

    last = df.iloc[-1]
    return {
        "tool": "volatility_state",
        "ticker": ticker,
        "asof": last["timestamp"].strftime("%Y-%m-%dT%H:%M:%SZ"),
        "days": days,
        "baseline_days": baseline_days,
        "realized_vol_annual_pct": round(float(rv_curr), 3) if rv_curr is not None else None,
        "baseline_vol_annual_pct": round(float(rv_base), 3) if rv_base is not None else None,
        "ratio": round(float(ratio), 3) if ratio is not None else None,
        "state": state,
        "rows": int(len(close)),
    }


def _local_extrema_levels(df: pd.DataFrame, k: int = 3) -> Dict[str, List[Dict[str, Any]]]:
    # Use centered rolling to find local max/min over window=2k+1
    highs = df["High"].astype(float)
    lows = df["Low"].astype(float)
    roll_max = highs.rolling(2 * k + 1, center=True).max()
    roll_min = lows.rolling(2 * k + 1, center=True).min()
    piv_hi = df[(highs == roll_max) & roll_max.notna()][["timestamp", "High"]]
    piv_lo = df[(lows == roll_min) & roll_min.notna()][["timestamp", "Low"]]
    hi_levels = [
        {"timestamp": r["timestamp"].strftime("%Y-%m-%d"), "level": float(r["High"])} for _, r in piv_hi.iterrows()
    ]
    lo_levels = [
        {"timestamp": r["timestamp"].strftime("%Y-%m-%d"), "level": float(r["Low"])} for _, r in piv_lo.iterrows()
    ]
    return {"resistance_levels": hi_levels, "support_levels": lo_levels}


def tool_support_resistance_check(ticker: str, days: int) -> Dict[str, Any]:
    df = _fetch_daily_history(ticker, days + 200)
    if df.empty:
        return {"tool": "support_resistance_check", "ticker": ticker, "rows": 0, "note": "no_data"}
    last_close = float(df.iloc[-1]["Close"])
    levels = _local_extrema_levels(df.tail(days))

    # Find nearest levels around last close
    supports = sorted([lv["level"] for lv in levels["support_levels"] if lv["level"] <= last_close], reverse=True)
    resistances = sorted([lv["level"] for lv in levels["resistance_levels"] if lv["level"] >= last_close])
    near_support = supports[0] if supports else None
    near_resist = resistances[0] if resistances else None

    def _pct(a: Optional[float], b: float) -> Optional[float]:
        if a is None or b == 0:
            return None
        return round(100.0 * (b - a) / b, 2) if a <= b else round(100.0 * (a - b) / b, 2)

    return {
        "tool": "support_resistance_check",
        "ticker": ticker,
        "asof": df.iloc[-1]["timestamp"].strftime("%Y-%m-%dT%H:%M:%SZ"),
        "last_close": last_close,
        "nearest_support": near_support,
        "nearest_resistance": near_resist,
        "distance_to_support_pct": _pct(near_support, last_close),
        "distance_to_resistance_pct": _pct(near_resist, last_close),
        "levels": levels,
        "rows": int(len(df)),
    }


def tool_bollinger_breakout_scan(ticker: str, days: int, period: int = 20, num_std: float = 2.0) -> Dict[str, Any]:
    df = _fetch_daily_history(ticker, days + period + 5)
    if df.empty:
        return {"tool": "bollinger_breakout_scan", "ticker": ticker, "rows": 0, "note": "no_data"}
    close = df["Close"].astype(float)
    ma = close.rolling(period, min_periods=period).mean()
    sd = close.rolling(period, min_periods=period).std(ddof=0)
    upper = ma + num_std * sd
    lower = ma - num_std * sd

    recent = df.tail(days).copy()
    recent = recent.assign(
        ma=ma.tail(days).values,
        upper=upper.tail(days).values,
        lower=lower.tail(days).values,
        close=close.tail(days).values,
    )
    above = recent["close"] > recent["upper"]
    below = recent["close"] < recent["lower"]

    last_event = None
    last_date = None
    if above.any():
        idx = np.where(above.values)[0][-1]
        last_event = "upper_breakout"
        last_date = recent.iloc[idx]["timestamp"].strftime("%Y-%m-%d")
    if below.any() and (last_date is None or recent.index[-1] >= recent.index[idx]):
        idx2 = np.where(below.values)[0][-1]
        if last_date is None or idx2 >= idx:
            last_event = "lower_breakout"
            last_date = recent.iloc[idx2]["timestamp"].strftime("%Y-%m-%d")

    # %b and bandwidth at last bar
    last = recent.iloc[-1]
    bw = (last["upper"] - last["lower"]) / last["ma"] if last["ma"] else np.nan
    pct_b = (last["close"] - last["lower"]) / (last["upper"] - last["lower"]) if (last["upper"] - last["lower"]) else np.nan

    return {
        "tool": "bollinger_breakout_scan",
        "ticker": ticker,
        "asof": last["timestamp"].strftime("%Y-%m-%dT%H:%M:%SZ"),
        "period": period,
        "num_std": num_std,
        "last_event": last_event,
        "last_event_date": last_date,
        "bandwidth": round(float(bw), 4) if not np.isnan(bw) else None,
        "%b": round(float(pct_b), 4) if not np.isnan(pct_b) else None,
        "rows": int(len(df)),
    }


def tool_obv_trend(ticker: str, lookback_days: int) -> Dict[str, Any]:
    df = _fetch_daily_history(ticker, lookback_days + 5)
    if df.empty or len(df) < max(lookback_days // 2, 10):
        return {"tool": "obv_trend", "ticker": ticker, "rows": int(len(df)), "note": "insufficient_data"}
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float)
    direction = np.sign(close.diff().fillna(0.0))
    obv = (direction * vol).cumsum()
    y = obv.tail(lookback_days).values
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = ((y - y_hat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    trend = "up" if slope > 0 else ("down" if slope < 0 else "flat")
    last = df.iloc[-1]
    return {
        "tool": "obv_trend",
        "ticker": ticker,
        "asof": last["timestamp"].strftime("%Y-%m-%dT%H:%M:%SZ"),
        "lookback_days": lookback_days,
        "trend": trend,
        "slope": round(float(slope), 4),
        "r2": round(float(r2), 4),
        "rows": int(len(df)),
    }


def tool_mfi_flow(ticker: str, period: int = 14) -> Dict[str, Any]:
    df = _fetch_daily_history(ticker, period + 60)
    if df.empty or len(df) < period + 5:
        return {"tool": "mfi_flow", "ticker": ticker, "period": period, "rows": int(len(df)), "note": "insufficient_data"}
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float)
    tp = (high + low + close) / 3.0
    rmf = tp * vol
    pos = rmf.where(tp > tp.shift(1), 0.0)
    neg = rmf.where(tp < tp.shift(1), 0.0)
    pos_n = pos.rolling(period, min_periods=period).sum()
    neg_n = neg.rolling(period, min_periods=period).sum()
    mr = pos_n / neg_n.replace(0, np.nan)
    mfi = 100 - (100 / (1 + mr))
    last_mfi = float(mfi.iloc[-1]) if not np.isnan(mfi.iloc[-1]) else None
    state = None
    if last_mfi is not None:
        if last_mfi >= 80:
            state = "overbought"
        elif last_mfi <= 20:
            state = "oversold"
        else:
            state = "neutral"

    last = df.iloc[-1]
    return {
        "tool": "mfi_flow",
        "ticker": ticker,
        "asof": last["timestamp"].strftime("%Y-%m-%dT%H:%M:%SZ"),
        "period": period,
        "mfi": round(last_mfi, 2) if last_mfi is not None else None,
        "state": state,
        "rows": int(len(df)),
    }


# ===============
# CLI & Harness
# ===============

def _cli_single() -> None:
    p = argparse.ArgumentParser(description="Technical analysis toolkit (single tool or run-plan)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # price_window
    spw = sub.add_parser("price-window", help="Fetch daily OHLCV between dates (inclusive)")
    spw.add_argument("--ticker", required=True)
    spw.add_argument("--from", dest="from_date", required=True)
    spw.add_argument("--to", dest="to_date", required=True)
    spw.add_argument("--interval", default="1d")

    # compute_indicators
    sci = sub.add_parser("compute-indicators", help="Compute RSI/MACD/SMA crosses")
    sci.add_argument("--ticker", required=True)
    sci.add_argument("--window-days", type=int, default=60)

    # trend_strength
    sts = sub.add_parser("trend-strength", help="Slope+R2 of log price over lookback")
    sts.add_argument("--ticker", required=True)
    sts.add_argument("--lookback-days", type=int, default=30)

    # volatility_state
    sv = sub.add_parser("volatility-state", help="Realized vol vs baseline")
    sv.add_argument("--ticker", required=True)
    sv.add_argument("--days", type=int, default=20)
    sv.add_argument("--baseline-days", type=int, default=60)

    # support_resistance_check
    ssr = sub.add_parser("support-resistance-check", help="Nearest support/resistance using local extrema")
    ssr.add_argument("--ticker", required=True)
    ssr.add_argument("--days", type=int, default=60)

    # bollinger_breakout_scan
    sbb = sub.add_parser("bollinger-breakout-scan", help="Detect last breakout vs Bollinger bands")
    sbb.add_argument("--ticker", required=True)
    sbb.add_argument("--days", type=int, default=60)

    # obv_trend
    sobv = sub.add_parser("obv-trend", help="OBV slope trend over lookback")
    sobv.add_argument("--ticker", required=True)
    sobv.add_argument("--lookback-days", type=int, default=30)

    # mfi_flow
    smfi = sub.add_parser("mfi-flow", help="Money Flow Index state")
    smfi.add_argument("--ticker", required=True)
    smfi.add_argument("--period", type=int, default=14)

    # run-plan (mini harness): expects JSON with steps[] (tool+args)
    rp = sub.add_parser("run-plan", help="Run a compact JSON plan {steps:[{tool,args}]} and emit results")
    rp.add_argument("--plan-json", required=True, help="Inline JSON or @path/to/file.json")

    args = p.parse_args()

    if args.cmd == "price-window":
        _json(tool_price_window(args.ticker, args.from_date, args.to_date, args.interval))
        return
    if args.cmd == "compute-indicators":
        _json(tool_compute_indicators(args.ticker, args.window_days))
        return
    if args.cmd == "trend-strength":
        _json(tool_trend_strength(args.ticker, args.lookback_days))
        return
    if args.cmd == "volatility-state":
        _json(tool_volatility_state(args.ticker, args.days, args.baseline_days))
        return
    if args.cmd == "support-resistance-check":
        _json(tool_support_resistance_check(args.ticker, args.days))
        return
    if args.cmd == "bollinger-breakout-scan":
        _json(tool_bollinger_breakout_scan(args.ticker, args.days))
        return
    if args.cmd == "obv-trend":
        _json(tool_obv_trend(args.ticker, args.lookback_days))
        return
    if args.cmd == "mfi-flow":
        _json(tool_mfi_flow(args.ticker, args.period))
        return
    if args.cmd == "run-plan":
        plan_raw = args.plan_json
        if plan_raw.startswith("@"):
            with open(plan_raw[1:], "r", encoding="utf-8") as f:
                plan = json.load(f)
        else:
            plan = json.loads(plan_raw)
        steps: List[Dict[str, Any]] = plan.get("steps", []) if isinstance(plan, dict) else []
        results: List[Dict[str, Any]] = []
        for i, st in enumerate(steps):
            tool = (st or {}).get("tool")
            a = (st or {}).get("args", {})
            try:
                if tool == "price_window":
                    res = tool_price_window(a["ticker"], a["from"], a["to"], a.get("interval", "1d"))
                elif tool == "compute_indicators":
                    res = tool_compute_indicators(a["ticker"], int(a["window_days"]))
                elif tool == "trend_strength":
                    res = tool_trend_strength(a["ticker"], int(a["lookback_days"]))
                elif tool == "volatility_state":
                    res = tool_volatility_state(a["ticker"], int(a["days"]), int(a["baseline_days"]))
                elif tool == "support_resistance_check":
                    res = tool_support_resistance_check(a["ticker"], int(a["days"]))
                elif tool == "bollinger_breakout_scan":
                    res = tool_bollinger_breakout_scan(a["ticker"], int(a["days"]))
                elif tool == "obv_trend":
                    res = tool_obv_trend(a["ticker"], int(a["lookback_days"]))
                elif tool == "mfi_flow":
                    res = tool_mfi_flow(a["ticker"], int(a.get("period", 14)))
                else:
                    res = {"error": f"unknown_tool: {tool}"}
                results.append({"index": i, "tool": tool, "result": res})
            except Exception as e:
                results.append({"index": i, "tool": tool, "error": str(e)})
        _json({"results": results})
        return


if __name__ == "__main__":
    _cli_single()
