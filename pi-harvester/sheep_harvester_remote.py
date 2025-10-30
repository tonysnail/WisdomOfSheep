#!/usr/bin/env python3
"""
Wisdom of Sheep — Raw Article Harvester (Pi producer + Oracle uploader)

Design
- Producer (main loop): scrape sources, write to local CSV, ENQUEUE each new row to a disk-backed outbox.
- Consumer (uploader thread): drain the outbox to Oracle with backoff; never blocks the producer.
- If Oracle is down, queue grows and resumes later; no missed articles.

Auth via env (recommended):

cd ~/sheep_harvester
source .venv/bin/activate
export WOS_ORACLE_USER="carlhudson83"
export WOS_ORACLE_PASS="B*********68-"

Examples
  python sheep_harvester_remote.py \
  --remote-url http://130.162.168.45:8000 \
  --interval 60 \
  --max-items 16 \
  --batch-size 8 \
  --per-post-sleep 0.4 \
  --remote-timeout 45


Troubleshooting:
Check Oracle Server is active:  
    curl -u 'carlhudson83:Briandavidson68-' http://130.162.168.45:8000/healthz


This program runs as a SERVICE at startup on the Pi.

To view the CLI Log:    journalctl --user u sheep-harvester -f


To Restart or Stop Service:

systemctl --user restart sheep-harvester
systemctl --user stop sheep-harvester 



Notes
- Local CSV backup is ON by default (use --no-local to disable).
- Queue is SQLite at ./outbox.sqlite by default (configurable).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import queue
import signal
import random
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from requests.auth import HTTPBasicAuth

# --- your scrapers ---
from reddit_scraper import DEFAULT_SUBS, scrape_sub_new
from rss_parser import DEFAULT_RSS_FEEDS, fetch_rss_feed
from x_scraper import DEFAULT_X_HANDLES, scrape_x_feed
from stocktwits_scraper import scrape_stocktwits_news

# ==================== Paths & Defaults ====================
REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT
RAW_POSTS_CSV = DATA_DIR / "raw_posts_log.csv"

# We'll accept any CSV columns; these are the canonical ones we try to include when sending:
CORE_COLS = ["scraped_at", "platform", "source", "post_id", "url", "title", "text"]

DEFAULT_INTERVAL_SEC = 90
DEFAULT_MAX_ITEMS = 16
DEFAULT_BATCH_SIZE = 8
DEFAULT_PER_POST_SLEEP = 1.0

DEFAULT_BACKFILL = True
DEFAULT_MAX_BACKFILL_BATCHES = 10

# Startup sync behavior
READY_STARTUP_TIMEOUT = 300       # seconds max to wait at startup for /wos/ready (5 min)
READY_POLL_INTERVAL   = 2.0       # seconds between readiness polls at startup
STARTUP_BACKOFF_MAX   = 10.0      # seconds

# Env Auth
ORACLE_BASIC_USER = os.getenv("WOS_ORACLE_USER", "")
ORACLE_BASIC_PASS = os.getenv("WOS_ORACLE_PASS", "")


# ==================== Utilities ====================
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_csv_exists():
    if not RAW_POSTS_CSV.exists():
        RAW_POSTS_CSV.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=CORE_COLS).to_csv(RAW_POSTS_CSV, index=False)


def read_csv_safely() -> pd.DataFrame:
    ensure_csv_exists()
    try:
        return pd.read_csv(RAW_POSTS_CSV, dtype=str).fillna("")
    except Exception:
        return pd.DataFrame(columns=CORE_COLS)


def load_seen_sets() -> Tuple[set, set]:
    seen_keys, seen_urls = set(), set()
    df = read_csv_safely()
    for _, r in df.iterrows():
        plat = (r.get("platform") or "").strip() or "reddit"
        pid  = (r.get("post_id") or "").strip()
        url  = (r.get("url") or "").strip()
        if pid:
            seen_keys.add(f"{plat}:{pid}")
        if url:
            seen_urls.add(url)
    return seen_keys, seen_urls


def append_rows_local(rows: List[Dict[str, Any]]):
    if not rows:
        return
    exist = read_csv_safely()
    new_df = pd.DataFrame(rows)
    # keep any columns we already have + any new ones that appear
    ordered = list(dict.fromkeys(list(exist.columns) + list(new_df.columns)))
    for col in ordered:
        if col not in exist.columns:
            exist[col] = ""
        if col not in new_df.columns:
            new_df[col] = ""
    out = pd.concat([exist[ordered], new_df[ordered]], ignore_index=True)
    tmp = RAW_POSTS_CSV.with_suffix(".tmp.csv")
    out.to_csv(tmp, index=False)
    tmp.replace(RAW_POSTS_CSV)


# ==================== Remote Client ====================
class RemoteClient:
    def __init__(
        self,
        base_url: str,
        timeout: int = 15,
        verify_tls: bool = True,
        extra_headers: Optional[Dict[str, str]] = None,
        basic_user: Optional[str] = None,
        basic_pass: Optional[str] = None,
    ):
        self.base = base_url.rstrip("/")
        self.timeout = int(timeout)
        self.verify = bool(verify_tls)
        self.headers = {"User-Agent": "WOS-Harvester/1.0"}
        if extra_headers:
            self.headers.update(extra_headers)
        self.auth = HTTPBasicAuth(basic_user, basic_pass) if (basic_user and basic_pass) else None

        self.url_ready  = f"{self.base}/wos/ready"
        self.url_post   = f"{self.base}/wos/raw-posts"
        self.url_exists = f"{self.base}/wos/exists"

    def is_ready(self) -> bool:
        try:
            r = requests.get(self.url_ready, headers=self.headers, timeout=self.timeout,
                             verify=self.verify, auth=self.auth)
            if r.status_code != 200:
                return False
            with contextlib.suppress(Exception):
                data = r.json()
                return bool(data.get("ready", True))
            return True
        except Exception:
            return False

    def wait_until_ready(self, max_wait_s: int = READY_STARTUP_TIMEOUT) -> bool:
        start = time.time()
        backoff = 1.0
        while time.time() - start < max_wait_s:
            if self.is_ready():
                return True
            logging.info("[startup] Oracle not ready; retrying in %.1fs", READY_POLL_INTERVAL)
            time.sleep(READY_POLL_INTERVAL)
            backoff = min(STARTUP_BACKOFF_MAX, backoff * 1.5)
        return self.is_ready()

    def exists(self, platform: str, post_id: str) -> bool:
        try:
            params = {"platform": platform, "post_id": post_id}
            r = requests.get(self.url_exists, params=params, headers=self.headers,
                             timeout=self.timeout, verify=self.verify, auth=self.auth)
            if r.status_code != 200:
                return False
            data = r.json()
            return bool(data.get("exists", False))
        except Exception:
            return False

    def post_row_blocking(self, row: Dict[str, Any]):
        payload = {"rows": [row]}
        headers = dict(self.headers)
        headers["Content-Type"] = "application/json"
        r = requests.post(self.url_post, headers=headers, json=payload,
                          timeout=self.timeout, verify=self.verify, auth=self.auth)
        if r.status_code >= 300:
            raise RuntimeError(f"POST {r.status_code}: {r.text[:200]}")


# ==================== Background Uploader ====================
class RemoteUploader:
    def __init__(
        self,
        client: RemoteClient,
        outbox_dir: Path,
        queue_maxsize: int = 5000,
        min_interval: float = 0.05,
        backoff_max: float = 30.0,
    ):
        self.client = client
        self.outbox = Path(outbox_dir)
        self.outbox.mkdir(parents=True, exist_ok=True)
        self.q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=queue_maxsize)
        self._stop = threading.Event()
        self._t: Optional[threading.Thread] = None
        self.min_interval = float(min_interval)
        self.backoff_max = float(backoff_max)
        self._last_post_ts = 0.0
        self._backoff = 1.0

    def configured(self) -> bool:
        return bool(self.client and self.client.base)

    def enqueue(self, row: Dict[str, Any]):
        try:
            self.q.put_nowait(row)
        except queue.Full:
            self._spool(row)
            logging.warning("[uploader] queue full; spooled to outbox")

    def start(self):
        if not self.configured():
            logging.info("[uploader] disabled (no remote)")
            return
        self._t = threading.Thread(target=self._run, name="wos-uploader", daemon=True)
        self._t.start()

    def stop(self, join_timeout: float = 5.0):
        self._stop.set()
        if self._t and self._t.is_alive():
            self._t.join(timeout=join_timeout)

    # ---- internals
    def _spool(self, row: Dict[str, Any]):
        ts = int(time.time() * 1000)
        path = self.outbox / f"row_{ts}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _flush_outbox(self):
        files = sorted(self.outbox.glob("row_*.jsonl"))
        if not files:
            return
        logging.info("[uploader] flushing outbox (%d file(s))", len(files))
        for file in files:
            try:
                with file.open("r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        self._post_one_with_ready(row)
                file.unlink(missing_ok=True)
                logging.info("[uploader] flushed %s", file.name)
            except Exception as e:
                logging.warning("[uploader] flush failed on %s: %s", file.name, e)
                break

    def _sleep_min_interval(self):
        dt = time.time() - self._last_post_ts
        if dt < self.min_interval:
            time.sleep(self.min_interval - dt)

    def _post_one_with_ready(self, row: Dict[str, Any]):
        # wait until ready (short loop; uploader keeps running forever)
        while not self.client.is_ready():
            if self._stop.is_set():
                raise RuntimeError("stopping while waiting for ready()")
            logging.info("[uploader] remote not ready; sleeping %.1fs", self._backoff)
            time.sleep(self._backoff)
            self._backoff = min(self.backoff_max, self._backoff * 2)
        # rate limit
        self._sleep_min_interval()
        # post
        self.client.post_row_blocking(row)
        self._last_post_ts = time.time()
        self._backoff = 1.0

    def _run(self):
        logging.info("[uploader] started; remote=%s", self.client.base)
        self._flush_outbox()
        while not self._stop.is_set():
            try:
                row = self.q.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                self._post_one_with_ready(row)
                logging.info("[uploader] sent 1 row")
                self.q.task_done()
            except Exception as e:
                logging.warning("[uploader] POST failed: %s; spooling & backoff", e)
                self._spool(row)
                self.q.task_done()
                time.sleep(self._backoff)
                self._backoff = min(self.backoff_max, self._backoff * 2)

        # drain queue to outbox
        drained = 0
        while True:
            try:
                row = self.q.get_nowait()
            except queue.Empty:
                break
            self._spool(row)
            drained += 1
            self.q.task_done()
        if drained:
            logging.info("[uploader] stopped; spooled %d queued row(s)", drained)


# ==================== Harvester (producer) ====================
class Harvester:
    def __init__(
        self,
        subs: List[str],
        rss_feeds: List[Dict[str, str]],
        x_handles: List[str],
        include_stocktwits: bool,
        max_items: int,
        interval_sec: int,
        batch_size: int,
        per_post_sleep: float,
        backfill: bool,
        max_backfill_batches: int,
        uploader: Optional[RemoteUploader],
        no_local: bool,
    ):
        self.subs = subs
        self.rss_feeds = rss_feeds
        self.x_handles = x_handles
        self.include_stocktwits = include_stocktwits
        self.max_items = int(max_items)
        self.interval_sec = int(interval_sec)
        self.batch_size = int(batch_size)
        self.per_post_sleep = float(per_post_sleep)
        self.backfill = bool(backfill)
        self.max_backfill_batches = int(max_backfill_batches)

        self.uploader = uploader
        self.no_local = bool(no_local)

        self.seen_keys, self.seen_urls = load_seen_sets()
        self.sources = self._build_sources()
        self.source_idx = 0

        self.current_posts: List[Dict[str, Any]] = []
        self.post_idx = 0

        self.backfill_seen_streak: Dict[str, int] = {}
        self.backfill_batches_for_source: Dict[str, int] = {}

        self.cooldown_until: Dict[str, float] = {}
        self.platform_min_interval_sec: Dict[str, float] = {
            "reddit": 2.0,   # at least 2s between any reddit fetches
            "rss": 0.0,
            "x":  1.0,
            "stocktwits": 1.0,
        }
        self.last_platform_fetch_ts: Dict[str, float] = {}

        self._stop = False

    def stop(self):
        self._stop = True

    def _build_sources(self) -> List[Dict[str, str]]:
        s: List[Dict[str, str]] = []
        for sub in self.subs:
            s.append({"platform": "reddit", "identifier": sub, "label": f"r/{sub}"})
        for feed in self.rss_feeds:
            s.append({"platform": "rss", "identifier": feed["url"], "label": feed["name"]})
        for handle in self.x_handles:
            s.append({"platform": "x", "identifier": handle, "label": f"@{handle}"})
        if self.include_stocktwits:
            s.append({"platform": "stocktwits", "identifier": "news-articles", "label": "Stocktwits News"})
        return s

    def _is_seen(self, platform: str, post_id: str, url: str) -> bool:
        return (post_id and f"{platform}:{post_id}" in self.seen_keys) or (url and url in self.seen_urls)

    # NEW: tiny platform limiter + a dash of jitter to avoid rigid patterns
    def _respect_platform_min_interval(self, platform: str):
        now = time.time()
        last = self.last_platform_fetch_ts.get(platform, 0.0)
        min_gap = self.platform_min_interval_sec.get(platform, 0.0)
        # jitter ~0–150ms so calls don’t look machine-perfect
        jitter = random.uniform(0.0, 0.15)
        wait = (last + min_gap + jitter) - now
        if wait > 0:
            time.sleep(wait)

    # NEW: simple helper to put a source on cooldown
    def _bump_cooldown(self, key: str, seconds: float):
        until = time.time() + max(0.0, seconds)
        prev = self.cooldown_until.get(key, 0.0)
        self.cooldown_until[key] = max(prev, until)

    def _fetch_for_source(self, src: Dict[str, str]) -> List[Dict[str, Any]]:
        plat, ident, label = src["platform"], src["identifier"], src["label"]

        # NEW: honour cooldown (e.g., after a 429)
        key = f"{plat}:{ident}"
        until = self.cooldown_until.get(key, 0.0)
        if until and time.time() < until:
            remaining = int(until - time.time())
            logging.info("[%s] on cooldown %ss (skipping this round)", label, remaining)
            return []

        # NEW: platform limiter
        self._respect_platform_min_interval(plat)

        try:
            if plat == "reddit":
                # Optional: clamp reddit pulls a bit if you want:
                max_posts = min(self.max_items, 12)   # soften bursts slightly
                posts = scrape_sub_new(ident, max_posts=max_posts)
            elif plat == "rss":
                posts = fetch_rss_feed(ident, name=label, max_items=self.max_items, platform="rss")
            elif plat == "x":
                posts = scrape_x_feed(ident, max_posts=min(self.max_items, 12))
            elif plat == "stocktwits":
                posts = scrape_stocktwits_news(max_items=min(self.max_items, 12))
            else:
                posts = []
        except requests.HTTPError as e:
            # CHANGED: special handling for 429 (Too Many Requests)
            status = getattr(e.response, "status_code", None)
            if status == 429:
                # Try to honour Retry-After header; otherwise back off sensibly
                retry_after = 0
                try:
                    hdr = e.response.headers.get("Retry-After")
                    if hdr:
                        retry_after = int(hdr)
                except Exception:
                    pass
                cooldown = retry_after if retry_after > 0 else 120  # 2 minutes default
                # Exponential-ish bump if repeated
                prev_until = self.cooldown_until.get(key, 0.0)
                if prev_until and prev_until > time.time():
                    # double the remaining time, capped at 30 min
                    remaining = prev_until - time.time()
                    cooldown = min(1800, max(cooldown, remaining * 2))
                logging.warning("[%s] error: 429 Too Many Requests; cooling down for %ss", label, int(cooldown))
                self._bump_cooldown(key, cooldown)
                posts = []
            else:
                logging.warning("[%s] HTTP error: %s", label, e)
                posts = []
        except Exception as e:
            logging.warning("[%s] error: %s", label, e)
            posts = []

        # record last fetch timestamp per platform
        self.last_platform_fetch_ts[plat] = time.time()

        for p in posts:
            p.setdefault("platform", plat)
            if not p.get("source"):
                p["source"] = label
        logging.info("Fetched %d from %s (%s)", len(posts), label, plat)
        return posts

    def _handle_posts(self, posts: List[Dict[str, Any]]) -> int:
        new_rows: List[Dict[str, Any]] = []
        for p in posts:
            platform = p.get("platform", "reddit")
            source   = p.get("source") or platform
            text     = p.get("text") or ""
            post_identifier = p.get("id") or p.get("url") or p.get("title")
            post_id  = str(post_identifier) if post_identifier else f"{platform}-{int(time.time()*1000)}"
            url      = str(p.get("url") or "")
            scraped  = p.get("scraped_at") or now_iso()

            if self._is_seen(platform, post_id, url):
                continue

            row = {
                "scraped_at": scraped,
                "platform": platform,
                "source": source,
                "post_id": post_id,
                "url": url,
                "title": str(p.get("title") or ""),
                "text": text,
            }

            # mark seen to avoid duplicates locally
            self.seen_keys.add(f"{platform}:{post_id}")
            if url:
                self.seen_urls.add(url)

            new_rows.append(row)

            # enqueue for remote
            if self.uploader and self.uploader.configured():
                self.uploader.enqueue(row)

            if self.per_post_sleep > 0:
                time.sleep(self.per_post_sleep)

        if new_rows:
            if not self.no_local:
                append_rows_local(new_rows)
                logging.info("  -> appended %d new row(s) locally", len(new_rows))
            else:
                logging.info("  -> skipping local write (--no-local)")
            return len(new_rows)
        return 0

    def _advance_source(self):
        if not self.sources:
            return
        self.source_idx = (self.source_idx + 1) % len(self.sources)
        self.current_posts = []
        self.post_idx = 0

    def step_once(self):
        if not self.sources:
            logging.warning("No sources configured; sleeping.")
            time.sleep(self.interval_sec)
            return

        src = self.sources[self.source_idx]
        # NEW: if the current source is on cooldown, advance immediately
        key = f"{src['platform']}:{src['identifier']}"
        until = self.cooldown_until.get(key, 0.0)
        if until and time.time() < until:
            self._advance_source()
            return

        if not self.current_posts:
            logging.info("Scraping %s …", src["label"])
            self.current_posts = self._fetch_for_source(src)
            self.post_idx = 0
            if not self.current_posts:
                self._advance_source()
                return

        start_idx = self.post_idx
        end_idx = min(len(self.current_posts), self.post_idx + self.batch_size)
        batch = self.current_posts[start_idx:end_idx]
        got = self._handle_posts(batch)
        self.post_idx = end_idx

        key = f"{src['platform']}:{src['identifier']}"
        if self.backfill:
            self.backfill_batches_for_source.setdefault(key, 0)
            self.backfill_batches_for_source[key] += 1

            if got == 0:
                self._advance_source()
            else:
                if self.post_idx >= len(self.current_posts):
                    self.current_posts = []
                    self.post_idx = 0

            if self.backfill_batches_for_source[key] >= self.max_backfill_batches:
                self._advance_source()
        else:
            if self.post_idx >= len(self.current_posts):
                self._advance_source()

    def run(self, once: bool = False):
        if self.uploader and self.uploader.configured():
            self.uploader.start()

        backoff = 1.0
        while not self._stop:
            try:
                self.step_once()
                backoff = 1.0
            except KeyboardInterrupt:
                logging.info("Interrupted.")
                break
            except Exception as e:
                logging.exception("Step failed: %s", e)
                sleep_for = min(300, int(backoff))
                logging.warning("Backing off %ss…", sleep_for)
                time.sleep(sleep_for)
                backoff *= 2.0

            if once:
                break
            time.sleep(self.interval_sec)

        if self.uploader and self.uploader.configured():
            self.uploader.stop()


# ==================== Startup Sync ====================
def make_row_from_series(s: pd.Series) -> Dict[str, Any]:
    """Include all columns present; ensure core keys exist."""
    row = {k: (s.get(k) if k in s else "") for k in s.index}
    for k in CORE_COLS:
        row.setdefault(k, "")
    return {k: ("" if pd.isna(v) else v) for k, v in row.items()}


def startup_sync(client: RemoteClient, min_interval: float = 0.05) -> None:
    """
    1) Ensure /wos/ready is reachable (wait up to READY_STARTUP_TIMEOUT).
    2) Find newest local row that already exists remotely by scanning from tail backwards.
    3) Backfill forward from the first missing row to the end (oldest missing → newest), 1-by-1.
    """
    logging.info("[startup] waiting for Oracle receiver readiness…")
    if not client.wait_until_ready(READY_STARTUP_TIMEOUT):
        logging.warning("[startup] Oracle not ready after timeout; continuing (uploader will spool).")
        return

    df = read_csv_safely()
    if df.empty:
        logging.info("[startup] no local rows; nothing to sync.")
        return

    # Find the newest existing row on the server
    newest_existing_idx = None
    logging.info("[startup] scanning local CSV (newest→older) to locate sync boundary…")
    for idx in range(len(df) - 1, -1, -1):
        s = df.iloc[idx]
        platform = (s.get("platform") or "reddit").strip()
        post_id  = (s.get("post_id") or "").strip()
        if not post_id:
            continue
        if client.exists(platform, post_id):
            newest_existing_idx = idx
            logging.info("[startup] found newest existing at local index %d (platform=%s, post_id=%s)",
                         idx, platform, post_id)
            break

    # Decide backfill start
    start_idx = 0 if newest_existing_idx is None else newest_existing_idx + 1
    if start_idx >= len(df):
        logging.info("[startup] remote is already up-to-date.")
        return

    # Backfill forward from start_idx → end (respect ready + min_interval)
    logging.info("[startup] backfilling %d row(s) (idx %d → %d)…",
                 len(df) - start_idx, start_idx, len(df) - 1)
    last_post_ts = 0.0
    for idx in range(start_idx, len(df)):
        s = df.iloc[idx]
        row = make_row_from_series(s)

        # if Oracle has it already (race / previous run), skip
        if client.exists((row.get("platform") or "reddit").strip(), (row.get("post_id") or "").strip()):
            continue

        # wait until /ready
        while not client.is_ready():
            logging.info("[startup] remote not ready; sleeping 1.0s")
            time.sleep(1.0)

        # rate limit
        dt = time.time() - last_post_ts
        if dt < min_interval:
            time.sleep(min_interval - dt)

        # send row
        client.post_row_blocking(row)
        last_post_ts = time.time()

        if (idx - start_idx + 1) % 100 == 0:
            logging.info("[startup] backfilled %d/%d",
                         idx - start_idx + 1, len(df) - start_idx)

    logging.info("[startup] backfill complete; server now in sync.")


# ==================== CLI / Main ====================
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="WOS Harvester (with startup sync to Oracle)")
    p.add_argument("--interval", type=int, default=DEFAULT_INTERVAL_SEC, help="Seconds between loop steps (default 90)")
    p.add_argument("--max-items", type=int, default=DEFAULT_MAX_ITEMS, help="Max items per source fetch (default 16)")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Posts per refresh/batch (default 8)")
    p.add_argument("--per-post-sleep", type=float, default=DEFAULT_PER_POST_SLEEP, help="Seconds to sleep per post")

    p.add_argument("--subs", type=str, default=",".join(DEFAULT_SUBS), help="Comma-separated Reddit subs")
    p.add_argument("--rss-file", type=str, default="", help="File with 'Name|URL' lines")
    p.add_argument("--rss-text", type=str, default="", help="Inline RSS 'Name|URL\\n…' (overrides defaults if set)")
    p.add_argument("--x-file", type=str, default="", help="File with @handles (comma/line separated)")
    p.add_argument("--x-handles", type=str, default="", help="Inline @handles, comma separated")
    p.add_argument("--no-stocktwits", action="store_true", help="Disable Stocktwits News")

    p.add_argument("--once", action="store_true", help="Run a single loop step then exit")
    p.add_argument("--no-backfill", action="store_true", help="Disable harvester backfill mode (default ON)")
    p.add_argument("--max-backfill-batches", type=int, default=DEFAULT_MAX_BACKFILL_BATCHES,
                   help="Max batches per source while backfilling (default 10)")
    p.add_argument("--log-level", type=str, default="INFO",
                   help="Logging level (DEBUG, INFO, WARNING, ERROR)")

    # Remote / uploader
    p.add_argument("--remote-url", type=str, default="",
                   help="Oracle receiver base, e.g. http://130.162.168.45:8000")
    p.add_argument("--remote-timeout", type=int, default=15, help="HTTP timeout seconds (default 15)")
    p.add_argument("--remote-no-verify-tls", action="store_true", help="Disable TLS verification")
    p.add_argument("--remote-header", action="append", default=[],
                   help="Extra header(s) 'Key: Value'. Use multiple times.")
    p.add_argument("--uploader-min-interval", type=float, default=0.05,
                   help="Minimum seconds between POSTs.")
    p.add_argument("--uploader-queue-size", type=int, default=5000, help="Max queued rows for uploader.")
    p.add_argument("--outbox-dir", type=str, default=str(DATA_DIR / ".remote_outbox"),
                   help="Directory to spool rows on failure.")
    return p


def main():
    args = build_argparser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
    )

    ensure_csv_exists()

    # Build sources config
    subs = [s.strip() for s in (args.subs or "").split(",") if s.strip()]

    if args.rss_text:
        rss_feeds = []
        for line in args.rss_text.splitlines():
            ln = line.strip()
            if not ln:
                continue
            name, url = (ln.split("|", 1)+[ln])[:2] if "|" in ln else (ln, ln)
            rss_feeds.append({"name": name.strip(), "url": url.strip()})
    elif args.rss_file:
        rss_text = ""
        with contextlib.suppress(Exception):
            rss_text = Path(args.rss_file).read_text(encoding="utf-8")
        rss_feeds = []
        for line in (rss_text or "").splitlines():
            ln = line.strip()
            if not ln:
                continue
            name, url = (ln.split("|", 1)+[ln])[:2] if "|" in ln else (ln, ln)
            rss_feeds.append({"name": name.strip(), "url": url.strip()})
    else:
        rss_feeds = DEFAULT_RSS_FEEDS[:]

    if args.x_handles:
        x_handles = [h.lstrip("@").strip() for h in args.x_handles.replace("\n", ",").split(",") if h.strip()]
    elif args.x_file:
        text = ""
        with contextlib.suppress(Exception):
            text = Path(args.x_file).read_text(encoding="utf-8")
        x_handles = [h.lstrip("@").strip() for h in text.replace("\n", ",").split(",") if h.strip()]
    else:
        x_handles = DEFAULT_X_HANDLES[:]

    # Remote client + uploader
    uploader = None
    if args.remote_url:
        verify_tls = not args.remote_no_verify_tls
        extra_headers: Dict[str, str] = {}
        for h in (args.remote_header or []):
            if ":" in h:
                k, v = h.split(":", 1)
                extra_headers[k.strip()] = v.strip()
        client = RemoteClient(
            base_url=args.remote_url.rstrip("/"),
            timeout=args.remote_timeout,
            verify_tls=verify_tls,
            extra_headers=extra_headers,
            basic_user=(ORACLE_BASIC_USER or None),
            basic_pass=(ORACLE_BASIC_PASS or None),
        )

        # 1) STARTUP SYNC (blocking, best-effort)
        try:
            startup_sync(client, min_interval=args.uploader_min_interval)
        except Exception as e:
            logging.exception("[startup] sync failed (continuing with uploader): %s", e)

        # 2) Background uploader for new content
        uploader = RemoteUploader(
            client=client,
            outbox_dir=Path(args.outbox_dir),
            queue_maxsize=args.uploader_queue_size,
            min_interval=args.uploader_min_interval,
        )
    else:
        logging.warning("Remote URL not configured; running local-only mode.")

    # Harvester
    harv = Harvester(
        subs=subs,
        rss_feeds=rss_feeds,
        x_handles=x_handles,
        include_stocktwits=True,
        max_items=args.max_items,
        interval_sec=args.interval,
        batch_size=args.batch_size,
        per_post_sleep=args.per_post_sleep,
        backfill=(not args.no_backfill),
        max_backfill_batches=args.max_backfill_batches,
        uploader=uploader,
        no_local=False,
    )

    def _sigterm(_sig, _frm):
        logging.info("SIGTERM received — stopping…")
        harv.stop()
    signal.signal(signal.SIGTERM, _sigterm)

    logging.info(
        "Harvester starting | reddit=%d rss=%d x=%d | interval=%ss max_items=%d batch=%d per_post=%.2fs | remote=%s",
        len(subs), len(rss_feeds), len(x_handles),
        args.interval, args.max_items, args.batch_size, args.per_post_sleep,
        (args.remote_url or "off"),
    )

    harv.run(once=args.once)
    logging.info("Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)