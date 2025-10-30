#!/usr/bin/env python3
"""
Wisdom of Sheep — Oracle Receiver DB CLI

Quick diagnostics for /var/wos/posts.sqlite

Features
- Menu & one-shot flags
- Count by date (YYYY-MM-DD)
- Count in the last hour / 24h
- Recent rows (pretty-printed)
- Top platforms over a window
- First/last timestamps
- DB size + NDJSON size
- WAL checkpoint & simple VACUUM tools (safe)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

DATA_DIR = Path("/var/wos")
DB_PATH = DATA_DIR / "posts.sqlite"
NDJSON_PATH = DATA_DIR / "raw_posts_log.ndjson"

def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    return conn

def fmt(n: int) -> str:
    return f"{n:,}"

def human_bytes(n: int) -> str:
    for unit in ("B","KB","MB","GB","TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"

def ensure_db_exists() -> None:
    if not DB_PATH.exists():
        print(f"[!] DB not found: {DB_PATH}", file=sys.stderr)
        sys.exit(2)

def count_by_date(date_str: str) -> int:
    ensure_db_exists()
    # scraped_at stored as ISO strings; constrain by prefix match
    conn = connect()
    cur = conn.execute(
        "SELECT COUNT(*) AS n FROM posts WHERE substr(scraped_at,1,10)=?",
        (date_str,)
    )
    n = cur.fetchone()["n"]
    conn.close()
    return int(n)

def count_last_hours(hours: int) -> int:
    ensure_db_exists()
    now = dt.datetime.utcnow()
    since = (now - dt.timedelta(hours=hours)).isoformat(timespec="seconds") + "Z"
    # scraped_at may be with +00:00 or Z; compare lexically where possible
    # Fallback: >= since by string compare is OK if consistently formatted
    conn = connect()
    cur = conn.execute(
        "SELECT COUNT(*) AS n FROM posts WHERE scraped_at >= ?",
        (since,)
    )
    n = cur.fetchone()["n"]
    conn.close()
    return int(n)

def recent(limit: int = 10) -> List[Dict[str, Any]]:
    ensure_db_exists()
    conn = connect()
    cur = conn.execute(
        "SELECT scraped_at, platform, post_id, json FROM posts "
        "ORDER BY scraped_at DESC, id DESC LIMIT ?",
        (limit,)
    )
    rows = []
    for r in cur.fetchall():
        try:
            js = json.loads(r["json"])
        except Exception:
            js = {}
        rows.append({
            "scraped_at": r["scraped_at"],
            "platform": r["platform"],
            "post_id": r["post_id"],
            "title": (js.get("title") or "")[:140],
            "url": js.get("url") or "",
        })
    conn.close()
    return rows

def top_platforms(hours: int = 24, limit: int = 10) -> List[Tuple[str,int]]:
    ensure_db_exists()
    now = dt.datetime.utcnow()
    since = (now - dt.timedelta(hours=hours)).isoformat(timespec="seconds") + "Z"
    conn = connect()
    cur = conn.execute(
        "SELECT platform, COUNT(*) AS n FROM posts "
        "WHERE scraped_at >= ? GROUP BY platform "
        "ORDER BY n DESC LIMIT ?",
        (since, limit)
    )
    out = [(row["platform"], row["n"]) for row in cur.fetchall()]
    conn.close()
    return out

def first_last_ts() -> Tuple[Optional[str], Optional[str], int]:
    ensure_db_exists()
    conn = connect()
    cur = conn.execute("SELECT MIN(scraped_at) AS min_ts, MAX(scraped_at) AS max_ts, COUNT(*) AS n FROM posts")
    row = cur.fetchone()
    conn.close()
    return row["min_ts"], row["max_ts"], int(row["n"])

def storage_info() -> Dict[str, Any]:
    info = {"db": None, "ndjson": None, "fs_free": None}
    if DB_PATH.exists():
        info["db"] = {"path": str(DB_PATH), "size_bytes": DB_PATH.stat().st_size}
    if NDJSON_PATH.exists():
        info["ndjson"] = {"path": str(NDJSON_PATH), "size_bytes": NDJSON_PATH.stat().st_size}
    usage = shutil.disk_usage(DATA_DIR)
    info["fs_free"] = {"total": usage.total, "used": usage.used, "free": usage.free}
    return info

def wal_checkpoint() -> str:
    ensure_db_exists()
    conn = connect()
    cur = conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
    row = cur.fetchone()
    conn.close()
    return f"PRAGMA wal_checkpoint(TRUNCATE) -> {tuple(row)}"

def vacuum() -> str:
    ensure_db_exists()
    conn = connect()
    conn.execute("VACUUM;")
    conn.close()
    return "VACUUM complete."

def print_rows_table(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("(no rows)")
        return
    cols = ["scraped_at","platform","post_id","title"]
    widths = {c: max(len(c), max(len(str(r.get(c,""))) for r in rows)) for c in cols}
    sep = " | "
    header = sep.join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print(sep.join(str(r.get(c,"")).ljust(widths[c]) for c in cols))

def interactive_menu() -> None:
    while True:
        print("\n=== WOS Receiver DB CLI ===")
        print("1) Count by date (YYYY-MM-DD)")
        print("2) Count in last hour")
        print("3) Count in last 24 hours")
        print("4) Top platforms (last 24h)")
        print("5) Show recent N rows")
        print("6) First/Last timestamps & total rows")
        print("7) Storage info (DB/NDJSON/disk)")
        print("8) WAL checkpoint (truncate)")
        print("9) VACUUM (safe, may take time)")
        print("0) Quit")
        choice = input("> ").strip()

        if choice == "1":
            d = input("Enter date (YYYY-MM-DD): ").strip()
            try:
                dt.date.fromisoformat(d)
            except Exception:
                print("Invalid date.")
                continue
            n = count_by_date(d)
            print(f"{d}: {fmt(n)} rows")
        elif choice == "2":
            n = count_last_hours(1)
            print(f"Last hour: {fmt(n)} rows")
        elif choice == "3":
            n = count_last_hours(24)
            print(f"Last 24h: {fmt(n)} rows")
        elif choice == "4":
            items = top_platforms(24, 20)
            for plat, n in items:
                print(f"{plat or '(blank)'}: {fmt(n)}")
        elif choice == "5":
            try:
                lim = int(input("How many rows (default 10)? ").strip() or "10")
            except Exception:
                lim = 10
            rows = recent(lim)
            print_rows_table(rows)
        elif choice == "6":
            lo, hi, n = first_last_ts()
            print(f"First: {lo or '-'}")
            print(f"Last : {hi or '-'}")
            print(f"Total: {fmt(n)}")
        elif choice == "7":
            info = storage_info()
            if info["db"]:
                print(f"DB     : {info['db']['path']} ({human_bytes(info['db']['size_bytes'])})")
            else:
                print("DB     : (missing)")
            if info["ndjson"]:
                print(f"NDJSON : {info['ndjson']['path']} ({human_bytes(info['ndjson']['size_bytes'])})")
            else:
                print("NDJSON : (missing)")
            fs = info["fs_free"]
            print(f"Disk   : total {human_bytes(fs['total'])} | used {human_bytes(fs['used'])} | free {human_bytes(fs['free'])}")
        elif choice == "8":
            print(wal_checkpoint())
        elif choice == "9":
            confirm = input("Type 'yes' to VACUUM now: ").strip().lower()
            if confirm == "yes":
                print(vacuum())
            else:
                print("Skipped.")
        elif choice == "0":
            print("Bye.")
            return
        else:
            print("Unknown option.")

def build_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="WOS Receiver DB CLI")
    sub = p.add_subparsers(dest="cmd")

    s1 = sub.add_parser("count-date", help="Count rows for a specific date (YYYY-MM-DD)")
    s1.add_argument("date", help="Date in YYYY-MM-DD")

    s2 = sub.add_parser("count-last", help="Count rows in last N hours")
    s2.add_argument("--hours", type=int, default=1, help="Hours back (default 1)")

    s3 = sub.add_parser("recent", help="Show recent rows")
    s3.add_argument("--limit", type=int, default=10, help="How many (default 10)")

    s4 = sub.add_parser("top-platforms", help="Top platforms in last N hours")
    s4.add_argument("--hours", type=int, default=24)
    s4.add_argument("--limit", type=int, default=10)

    sub.add_parser("first-last", help="Show first/last timestamps and total count")
    sub.add_parser("storage", help="Show DB/NDJSON/disk usage")
    sub.add_parser("checkpoint", help="Run PRAGMA wal_checkpoint(TRUNCATE)")
    sub.add_parser("vacuum", help="VACUUM the database (safe)")

    return p

def main() -> int:
    args = build_args().parse_args()

    if not args.cmd:
        interactive_menu()
        return 0

    if args.cmd == "count-date":
        try:
            dt.date.fromisoformat(args.date)
        except Exception:
            print("Invalid date format; expected YYYY-MM-DD", file=sys.stderr)
            return 2
        print(count_by_date(args.date))
        return 0

    if args.cmd == "count-last":
        print(count_last_hours(args.hours))
        return 0

    if args.cmd == "recent":
        rows = recent(args.limit)
        print_rows_table(rows)
        return 0

    if args.cmd == "top-platforms":
        for plat, n in top_platforms(args.hours, args.limit):
            print(f"{plat or '(blank)'} {n}")
        return 0

    if args.cmd == "first-last":
        lo, hi, n = first_last_ts()
        print(json.dumps({"first": lo, "last": hi, "total": n}, indent=2))
        return 0

    if args.cmd == "storage":
        info = storage_info()
        # humanize sizes:
        if info.get("db"):
            info["db"]["size_h"] = human_bytes(info["db"]["size_bytes"])
        if info.get("ndjson"):
            info["ndjson"]["size_h"] = human_bytes(info["ndjson"]["size_bytes"])
        fs = info["fs_free"]
        info["fs_free_h"] = {
            "total": human_bytes(fs["total"]),
            "used": human_bytes(fs["used"]),
            "free": human_bytes(fs["free"]),
        }
        print(json.dumps(info, indent=2))
        return 0

    if args.cmd == "checkpoint":
        print(wal_checkpoint())
        return 0

    if args.cmd == "vacuum":
        print(vacuum())
        return 0

    return 0

if __name__ == "__main__":
    sys.exit(main())
