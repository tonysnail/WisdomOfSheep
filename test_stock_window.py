#!/usr/bin/env python3
"""
test_harness_stock_window.py — quick visual check for get_stock_window()

Usage examples:
  python test_harness_stock_window.py NVDA --center "2025-06-12T14:00:00Z" --before 3d --after 5d
  python test_harness_stock_window.py AAPL --center now --before 6h --after 18h --interval 5m
  python test_harness_stock_window.py TSLA --center "2025-01-15 15:30:00" --save tsla_window.png

Dependencies: matplotlib, pandas, (and whatever stock_window.py needs)
    pip install matplotlib pandas
"""

import argparse
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt

from stock_window import get_stock_window  # import your module


def _parse_iso(dt_str: str) -> datetime:
    # Lightweight ISO-ish parser; center we pass to stock_window can be free-form
    # but for plotting/shading it's nice to have a datetime object
    # Stock_window already does robust parsing, so here we handle only common forms.
    try:
        # Try pandas (handles lots of formats)
        ts = pd.to_datetime(dt_str, utc=True)
        return ts.to_pydatetime()
    except Exception:
        # Fallback to 'now'
        return datetime.now(timezone.utc)


def main():
    ap = argparse.ArgumentParser(description="Plot a centered price window for a ticker.")
    ap.add_argument("ticker", help="Ticker symbol, e.g., AAPL")
    ap.add_argument(
        "--center",
        default="now",
        help="Center datetime (ISO8601 or 'now'/'today'). Default: now (UTC).",
    )
    ap.add_argument("--before", default="3d", help="Span before center, e.g., '6h', '2d'. Default: 3d")
    ap.add_argument("--after", default="3d", help="Span after center, e.g., '6h', '2d'. Default: 3d")
    ap.add_argument(
        "--interval",
        default=None,
        choices=["1m", "5m", "15m", "1h", "1d", "1wk"],
        help="Explicit interval; otherwise auto-picked.",
    )
    ap.add_argument(
        "--save",
        default=None,
        help="If provided, save the chart to this PNG path instead of only showing it.",
    )
    ap.add_argument(
        "--no-prepost",
        action="store_true",
        help="Exclude pre/post-market from the window (default includes it).",
    )
    ap.add_argument(
        "--auto-adjust",
        action="store_true",
        help="Use auto-adjusted OHLC from yfinance (default False).",
    )
    args = ap.parse_args()

    res = get_stock_window(
        ticker=args.ticker,
        center=args.center,
        before=args.before,
        after=args.after,
        interval=args.interval,
        include_data=True,
        auto_adjust=args.auto_adjust,
        prepost=not args.no_prepost,
    )

    # Console summary
    print(f"Ticker: {res.ticker}")
    print(f"Interval: {res.interval} | Window: {res.start_utc} → {res.end_utc}")
    print(f"Center: {res.center_utc} | t0: {res.t0_timestamp_utc} | t0_close: {res.t0_close}")
    if res.note:
        print(f"Note: {res.note}")
    print(f"Rows: {res.rows}")

    # Build dataframe for plotting
    df = pd.DataFrame(res.data or [])
    if df.empty:
        print("No data to plot for this window.")
        return

    # Timestamp to pandas datetime (UTC)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Plot
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(df["timestamp"], df["Close"], linewidth=1.5)
    ax.set_title(
        f"{res.ticker}  |  Close around {res.center_utc}  "
        f"(interval {res.interval}, rows {res.rows})",
        fontsize=12,
    )
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Close")

    # Mark center and t0
    center_dt = _parse_iso(res.center_utc)
    ax.axvline(center_dt, linestyle="--", linewidth=1.2)

    if res.t0_timestamp_utc:
        t0_dt = _parse_iso(res.t0_timestamp_utc)
        # mark t0 point
        t0_row = df.loc[df["timestamp"] == t0_dt]
        if not t0_row.empty:
            ax.scatter(t0_dt, float(t0_row["Close"].iloc[0]), s=36, zorder=3)

    # Shade before/after regions for quick visual context
    start_dt = _parse_iso(res.start_utc)
    end_dt = _parse_iso(res.end_utc)
    ax.axvspan(start_dt, center_dt, alpha=0.08)
    ax.axvspan(center_dt, end_dt, alpha=0.12)

    # Annotate end return if available
    if "return_from_t0" in df.columns and res.t0_close:
        try:
            last_ret = float(df["return_from_t0"].iloc[-1])
            pct = last_ret * 100.0
            ax.text(
                0.99,
                0.02,
                f"Last bar vs t0: {pct:+.2f}%",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
            )
        except Exception:
            pass

    ax.grid(True, linewidth=0.4, alpha=0.5)
    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"Saved chart → {args.save}")

    plt.show()


if __name__ == "__main__":
    main()
