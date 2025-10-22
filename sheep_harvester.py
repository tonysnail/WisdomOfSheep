#!/usr/bin/env python3
"""
Wisdom of Sheep — Raw Article Harvester (headless, backfill-first)

Scrapes raw articles/posts from multiple sources and appends them to raw_posts_log.csv.
No Streamlit, no LLM. Tuned for Raspberry Pi / always-on.

python sheep_harvester.py \
  --interval 90 \
  --max-items 16 \
  --batch-size 8 \
  --per-post-sleep 1.0

  (replaces previous 'streamlit run wisdom_of_sheep.py')

Defaults match your Streamlit scrape loop:
- posts per source: 16
- loop interval: 90s
- posts per refresh (batch): 8
- pacing per post: 1.0s
- backfill: ON at startup

Sources (via your modules):
- reddit_scraper: DEFAULT_SUBS, scrape_sub_new(sub, max_posts)
- rss_parser:     DEFAULT_RSS_FEEDS, fetch_rss_feed(url, name=..., max_items=..., platform="rss")
- x_scraper:      DEFAULT_X_HANDLES, scrape_x_feed(handle, max_posts)
- stocktwits_scraper: scrape_stocktwits_news(max_items)

Appends to raw_posts_log.csv with columns:
  ["scraped_at","platform","source","post_id","url","title","text"]
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd
import requests

# --- your scrapers ---
from reddit_scraper import DEFAULT_SUBS, scrape_sub_new
from rss_parser import DEFAULT_RSS_FEEDS, fetch_rss_feed
from x_scraper import DEFAULT_X_HANDLES, scrape_x_feed
from stocktwits_scraper import scrape_stocktwits_news

# ==================== Config ====================
REPO_ROOT = Path(__file__).resolve().parent
RAW_POSTS_CSV = REPO_ROOT / "raw_posts_log.csv"
RAW_POST_COLUMNS = ["scraped_at", "platform", "source", "post_id", "url", "title", "text"]

# Defaults = your Streamlit settings
DEFAULT_INTERVAL_SEC = 90
DEFAULT_MAX_ITEMS = 16
DEFAULT_BATCH_SIZE = 8
DEFAULT_PER_POST_SLEEP = 1.0

# Backfill ON by default
DEFAULT_BACKFILL = True
DEFAULT_MAX_BACKFILL_BATCHES = 10


# ==================== Utils ====================
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_rss_config(text: str) -> List[Dict[str, str]]:
    feeds: List[Dict[str, str]] = []
    for line in (text or "").splitlines():
        ln = line.strip()
        if not ln:
            continue
        if "|" in ln:
            name, url = ln.split("|", 1)
            name = name.strip()
            url = url.strip()
        else:
            name, url = ln, ln
        if not url:
            continue
        feeds.append({"name": name or url, "url": url})
    return feeds


def parse_x_handles(text: str) -> List[str]:
    out: List[str] = []
    for chunk in (text or "").replace("\n", ",").split(","):
        h = chunk.strip()
        if not h:
            continue
        if h.startswith("http"):
            continue
        h = h.lstrip("@")
        if not h:
            continue
        if h not in out:
            out.append(h)
    return out


def read_text_file(path: Optional[str]) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        logging.warning("File not found: %s", path)
        return ""
    try:
        return p.read_text(encoding="utf-8")
    except Exception as e:
        logging.error("Failed reading %s: %s", path, e)
        return ""


def ensure_raw_csv():
    if not RAW_POSTS_CSV.exists():
        pd.DataFrame(columns=RAW_POST_COLUMNS).to_csv(RAW_POSTS_CSV, index=False)


def load_seen_sets() -> Tuple[set, set]:
    """Return (seen_keys, seen_urls) from existing CSV for cross-run de-dupe."""
    seen_keys, seen_urls = set(), set()
    if not RAW_POSTS_CSV.exists():
        return seen_keys, seen_urls
    try:
        df = pd.read_csv(RAW_POSTS_CSV, dtype=str).fillna("")
        for _, r in df.iterrows():
            platform = r.get("platform", "").strip() or "reddit"
            pid = r.get("post_id", "").strip()
            url = r.get("url", "").strip()
            if pid:
                seen_keys.add(f"{platform}:{pid}")
            if url:
                seen_urls.add(url)
    except Exception as e:
        logging.error("Failed to load seen sets: %s", e)
    return seen_keys, seen_urls


def append_raw_posts(rows: List[Dict[str, Any]]):
    if not rows:
        return
    ensure_raw_csv()

    try:
        exist = pd.read_csv(RAW_POSTS_CSV)
    except Exception:
        exist = pd.DataFrame(columns=RAW_POST_COLUMNS)

    new_df = pd.DataFrame(rows)

    ordered = list(dict.fromkeys(RAW_POST_COLUMNS + exist.columns.tolist() + new_df.columns.tolist()))
    for col in ordered:
        if col not in exist.columns:
            exist[col] = ""
        if col not in new_df.columns:
            new_df[col] = ""

    exist = exist[ordered]
    new_df = new_df[ordered]
    out = pd.concat([exist, new_df], ignore_index=True)

    tmp = RAW_POSTS_CSV.with_suffix(".tmp.csv")
    out.to_csv(tmp, index=False)
    tmp.replace(RAW_POSTS_CSV)


# ==================== Harvester ====================
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

        self.seen_keys, self.seen_urls = load_seen_sets()
        self.sources = self._build_sources()
        self.source_idx = 0

        # per-source work buffer (to honor batch_size & pacing)
        self.current_posts: List[Dict[str, Any]] = []
        self.post_idx = 0

        # backfill counters per source key
        self.backfill_seen_streak: Dict[str, int] = {}
        self.backfill_batches_for_source: Dict[str, int] = {}

        self._stop = False

    def stop(self):
        self._stop = True

    def _build_sources(self) -> List[Dict[str, str]]:
        sources: List[Dict[str, str]] = []
        for sub in self.subs:
            sources.append({"platform": "reddit", "identifier": sub, "label": f"r/{sub}"})
        for feed in self.rss_feeds:
            sources.append({"platform": "rss", "identifier": feed["url"], "label": feed["name"]})
        for handle in self.x_handles:
            sources.append({"platform": "x", "identifier": handle, "label": f"@{handle}"})
        if self.include_stocktwits:
            sources.append({"platform": "stocktwits", "identifier": "news-articles", "label": "Stocktwits News"})
        return sources

    def _is_seen(self, platform: str, post_id: str, url: str) -> bool:
        return (post_id and f"{platform}:{post_id}" in self.seen_keys) or (url and url in self.seen_urls)

    def _fetch_for_source(self, src: Dict[str, str]) -> List[Dict[str, Any]]:
        plat, ident, label = src["platform"], src["identifier"], src["label"]
        try:
            if plat == "reddit":
                posts = scrape_sub_new(ident, max_posts=self.max_items)
            elif plat == "rss":
                posts = fetch_rss_feed(ident, name=label, max_items=self.max_items, platform="rss")
            elif plat == "x":
                posts = scrape_x_feed(ident, max_posts=self.max_items)
            elif plat == "stocktwits":
                posts = scrape_stocktwits_news(max_items=self.max_items)
            else:
                posts = []
        except requests.HTTPError as e:
            logging.warning("[%s] HTTP error: %s", label, e)
            posts = []
        except Exception as e:
            logging.warning("[%s] error: %s", label, e)
            posts = []

        # Normalize platform/source and compute latest scraped_at for logging
        latest_ts = None
        for p in posts:
            p.setdefault("platform", plat)
            if not p.get("source"):
                p["source"] = label
            s_at = p.get("scraped_at")
            if s_at:
                try:
                    ts = datetime.fromisoformat(s_at.replace("Z", "+00:00"))
                    if latest_ts is None or ts > latest_ts:
                        latest_ts = ts
                except Exception:
                    pass

        logging.info("Fetched %d from %s (%s)", len(posts), label, plat)
        if latest_ts is not None:
            logging.info("LATEST SCRAPED_AT: %s", latest_ts.isoformat(timespec="seconds"))
        else:
            logging.info("LATEST SCRAPED_AT: (unknown)")

        return posts

    def _append_new_rows(self, posts: List[Dict[str, Any]]) -> int:
        new_rows: List[Dict[str, Any]] = []
        for p in posts:
            platform = p.get("platform", "reddit")
            source = p.get("source") or platform
            text_full = p.get("text") or ""
            post_identifier = p.get("id") or p.get("url") or p.get("title")
            post_id = str(post_identifier) if post_identifier else f"{platform}-{int(time.time()*1000)}"
            url = str(p.get("url") or "")
            snippet = (text_full[:160] + "…") if len(text_full) > 160 else text_full

            if self._is_seen(platform, post_id, url):
                # mirror Streamlit log line for already-seen items
                logging.info("  Analyzing [%s] %s", platform, snippet)
                continue

            scraped_ts = p.get("scraped_at") or now_iso()

            logging.info("  Analyzing [%s] %s", platform, snippet)

            new_rows.append(
                {
                    "scraped_at": scraped_ts,
                    "platform": platform,
                    "source": source,
                    "post_id": post_id,
                    "url": url,
                    "title": str(p.get("title") or ""),
                    "text": text_full,
                }
            )
            self.seen_keys.add(f"{platform}:{post_id}")
            if url:
                self.seen_urls.add(url)

            # pacing per post (like Streamlit’s “Pacing per post”)
            if self.per_post_sleep > 0:
                time.sleep(self.per_post_sleep)

        if new_rows:
            append_raw_posts(new_rows)
            logging.info("  -> appended %d new row(s)", len(new_rows))
        else:
            logging.info("  -> no new rows")
        return len(new_rows)

    def _source_key(self, src: Dict[str, str]) -> str:
        return f"{src['platform']}:{src['identifier']}"

    def _advance_source(self):
        if not self.sources:
            return
        self.source_idx = (self.source_idx + 1) % len(self.sources)
        self.current_posts = []
        self.post_idx = 0

    def step_once(self):
        """One loop step: fetch/continue current source, process up to batch_size, log like Streamlit."""
        if not self.sources:
            logging.warning("No sources configured; sleeping.")
            time.sleep(self.interval_sec)
            return

        src = self.sources[self.source_idx]
        label = src["label"]
        plat = src["platform"]

        # Fetch posts when buffer is empty
        if not self.current_posts:
            logging.info("Scraping %s (%s) …", label, plat)
            self.current_posts = self._fetch_for_source(src)
            self.post_idx = 0
            if not self.current_posts:
                # move on if empty
                self._advance_source()
                return

        # Process up to batch_size items from the buffer
        start_idx = self.post_idx
        end_idx = min(len(self.current_posts), self.post_idx + self.batch_size)
        batch = self.current_posts[start_idx:end_idx]
        new_items_this_batch = self._append_new_rows(batch)
        self.post_idx = end_idx

        # Backfill logic (matches your Streamlit behavior):
        key = self._source_key(src)
        if self.backfill:
            self.backfill_seen_streak.setdefault(key, 0)
            self.backfill_batches_for_source.setdefault(key, 0)
            self.backfill_batches_for_source[key] += 1

            if new_items_this_batch == 0:
                self.backfill_seen_streak[key] += 1
                logging.info("[backfill] %s: no new items; streak=%d", label, self.backfill_seen_streak[key])
                self._advance_source()
            else:
                logging.info("[backfill] %s: +%d new; continuing…", label, new_items_this_batch)
                # If we consumed all buffered posts, clear so we fetch fresh from same source next cycle
                if self.post_idx >= len(self.current_posts):
                    self.current_posts = []
                    self.post_idx = 0

            # safety cap so one source doesn't hog
            if self.backfill_batches_for_source[key] >= self.max_backfill_batches:
                logging.info("[backfill] %s: reached cap (%d); moving on.", label, self.max_backfill_batches)
                self._advance_source()
        else:
            # Round-robin: if we exhausted the buffer, advance
            if self.post_idx >= len(self.current_posts):
                self._advance_source()

    def run(self, once: bool = False):
        backoff = 1.0
        while not self._stop:
            try:
                self.step_once()
                backoff = 1.0
            except KeyboardInterrupt:
                logging.info("Interrupted — exiting.")
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


# ==================== CLI ====================
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Wisdom of Sheep — Raw Article Harvester (headless)")
    p.add_argument("--interval", type=int, default=DEFAULT_INTERVAL_SEC, help="Seconds between loop steps (default 90)")
    p.add_argument("--max-items", type=int, default=DEFAULT_MAX_ITEMS, help="Max items per source fetch (default 16)")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Posts per refresh/batch (default 8)")
    p.add_argument("--per-post-sleep", type=float, default=DEFAULT_PER_POST_SLEEP, help="Seconds to sleep per post (default 1.0)")

    p.add_argument("--subs", type=str, default=",".join(DEFAULT_SUBS), help="Comma-separated Reddit subs")
    p.add_argument("--rss-file", type=str, default="", help="File with 'Name|URL' lines")
    p.add_argument("--rss-text", type=str, default="", help="Inline RSS 'Name|URL\\n…' (overrides defaults if set)")
    p.add_argument("--x-file", type=str, default="", help="File with @handles (comma/line separated)")
    p.add_argument("--x-handles", type=str, default="", help="Inline @handles, comma separated")
    p.add_argument("--no-stocktwits", action="store_true", help="Disable Stocktwits News")

    p.add_argument("--once", action="store_true", help="Run a single loop step then exit")
    p.add_argument("--no-backfill", action="store_true", help="Disable backfill mode (default is ON)")
    p.add_argument("--max-backfill-batches", type=int, default=DEFAULT_MAX_BACKFILL_BATCHES,
                   help="Max consecutive batches per source while backfilling (default 10)")
    p.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    return p


def main():
    args = build_argparser().parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
    )

    ensure_raw_csv()

    # Build source lists from args/defaults
    subs = [s.strip() for s in (args.subs or "").split(",") if s.strip()]

    if args.rss_text:
        rss_feeds = parse_rss_config(args.rss_text)
    elif args.rss_file:
        rss_feeds = parse_rss_config(read_text_file(args.rss_file))
    else:
        rss_feeds = DEFAULT_RSS_FEEDS[:]  # from your module

    if args.x_handles:
        x_handles = parse_x_handles(args.x_handles)
    elif args.x_file:
        x_handles = parse_x_handles(read_text_file(args.x_file))
    else:
        x_handles = DEFAULT_X_HANDLES[:]  # from your module

    harv = Harvester(
        subs=subs,
        rss_feeds=rss_feeds,
        x_handles=x_handles,
        include_stocktwits=(not args.no_stocktwits),
        max_items=args.max_items,
        interval_sec=args.interval,
        batch_size=args.batch_size,
        per_post_sleep=args.per_post_sleep,
        backfill=(not args.no_backfill),  # backfill ON by default
        max_backfill_batches=args.max_backfill_batches,
    )

    def _sigterm(_sig, _frm):
        logging.info("SIGTERM received — stopping…")
        harv.stop()
    signal.signal(signal.SIGTERM, _sigterm)

    logging.info(
        "Starting harvester | reddit=%d rss=%d x=%d stocktwits=%s | interval=%ss max_items=%d batch_size=%d per_post_sleep=%.2fs | backfill=%s",
        len(subs), len(rss_feeds), len(x_handles),
        "on" if not args.no_stocktwits else "off",
        args.interval, args.max_items, args.batch_size, args.per_post_sleep,
        "on" if not args.no_backfill else "off",
    )

    harv.run(once=args.once)
    logging.info("Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
