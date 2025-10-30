#!/usr/bin/env python3

"""

cd ~/sheep_harvester
source .venv/bin/activate
export WOS_ORACLE_USER="carlhudson83"
export WOS_ORACLE_PASS="B*********68-"

python backfill_to_oracle.py \
  --csv raw_posts_log.csv \
  --base-url http://130.162.168.45:8000 \
  --sleep 0.2 \
  --timeout 45 \
  --checkpoint .backfill_checkpoint.json
  
"""

from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
from typing import Any, Dict
import pandas as pd
import requests

DEFAULT_SLEEP = 0.02
DEFAULT_TIMEOUT = 10
HEADERS = {"Content-Type": "application/json"}

def load_checkpoint(path: Path) -> int:
    if not path.exists(): return 0
    try: return int(json.loads(path.read_text())["index"])
    except Exception: return 0

def save_checkpoint(path: Path, idx: int) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"index": idx}), encoding="utf-8")
    tmp.replace(path)

def server_ready(base: str, auth, timeout: int) -> bool:
    try:
        r = requests.get(base + "/wos/ready", auth=auth, timeout=timeout)
        r.raise_for_status()
        return bool(r.json().get("ready", True))
    except Exception:
        # if we can't read it, be conservative and wait
        return False

def exists_on_server(base: str, auth, timeout: int, platform: str, post_id: str) -> bool:
    try:
        r = requests.get(base + "/wos/exists", params={"platform": platform, "post_id": post_id}, auth=auth, timeout=timeout)
        r.raise_for_status()
        return bool(r.json().get("exists", False))
    except Exception:
        # if unsure, let POST handle dedupe
        return False

def post_row(base: str, auth, row: Dict[str, Any], timeout: int) -> str:
    payload = {"rows": [{
        "scraped_at": str(row.get("scraped_at","")),
        "platform":   str(row.get("platform","")),
        "source":     str(row.get("source","")),
        "post_id":    str(row.get("post_id","")),
        "url":        str(row.get("url","")),
        "title":      str(row.get("title","")),
        "text":       str(row.get("text","")),
    }]}
    r = requests.post(base + "/wos/raw-posts", headers=HEADERS, json=payload, auth=auth, timeout=timeout)
    if r.status_code == 409:
        return "busy"
    r.raise_for_status()
    js = r.json()
    if js.get("duplicates", 0) > 0 and js.get("accepted", 0) == 0:
        return "duplicate"
    return "ok"

def main() -> int:
    ap = argparse.ArgumentParser(description="Backfill CSV to Oracle receiver, one row at a time (oldest first) with readiness + existence checks.")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--base-url", required=True, help="Base URL, e.g. http://130.162.168.45:8000")
    ap.add_argument("--sleep", type=float, default=DEFAULT_SLEEP)
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    ap.add_argument("--checkpoint", default=".backfill_checkpoint.json")
    ap.add_argument("--max-busy-wait", type=float, default=30.0, help="Max seconds to wait while server reports busy")
    args = ap.parse_args()

    user = os.getenv("WOS_ORACLE_USER","").strip()
    pwd  = os.getenv("WOS_ORACLE_PASS","").strip()
    auth = (user, pwd) if user and pwd else None

    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}", file=sys.stderr); return 2

    df = pd.read_csv(csv_path, dtype=str).fillna("")
    if "scraped_at" in df.columns:
        try:
            df["_srt"] = pd.to_datetime(df["scraped_at"], errors="coerce")
            df = df.sort_values(by=["_srt","scraped_at"]).drop(columns=["_srt"])
        except Exception:
            df = df.sort_values(by=["scraped_at"])
    df = df.reset_index(drop=True)

    total = len(df)
    ckp = Path(args.checkpoint).resolve()
    idx = min(max(0, load_checkpoint(ckp)), total)

    print(f"Backfilling {total} rows from {csv_path}")
    print(f"Starting at index {idx}; posting to {args.base-url}/wos/raw-posts")
    if not auth:
        print("WARN: No Basic Auth set; set WOS_ORACLE_USER/WOS_ORACLE_PASS", file=sys.stderr)

    sent = 0
    while idx < total:
        row = df.iloc[idx].to_dict()
        plat = str(row.get("platform","")).strip()
        pid  = str(row.get("post_id","")).strip()

        # Existence check (fast, and saves a POST)
        if plat and pid and exists_on_server(args.base_url, auth, args.timeout, plat, pid):
            idx += 1
            save_checkpoint(ckp, idx)
            continue

        # Wait until server says it's ready (with max wait)
        start_wait = time.time()
        while not server_ready(args.base_url, auth, args.timeout):
            if time.time() - start_wait > args.max_busy_wait:
                print("Server remained busy too long; aborting.", file=sys.stderr)
                save_checkpoint(ckp, idx)
                return 3
            time.sleep(0.5)

        # Try to post; handle transient busy (409) quickly
        while True:
            result = post_row(args.base_url, auth, row, args.timeout)
            if result == "busy":
                time.sleep(0.5)
                continue
            break

        # Advance
        idx += 1
        sent += 1
        save_checkpoint(ckp, idx)
        if args.sleep > 0:
            time.sleep(args.sleep)

        if sent % 100 == 0:
            print(f"Sent {sent} rows (idx {idx}/{total})")

    print(f"Done. Sent {sent} rows.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
