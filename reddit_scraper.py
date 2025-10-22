"""Utility functions for scraping Reddit posts for trading signals."""
from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (reddit-crowd-signals/1.0)"}
DEFAULT_SUBS = [
    "stocks",
    "StockMarket",
    "options",
    "pennystocks",
    "shortsqueeze",
    "daytrading",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fetch_post_full_text(permalink_or_url: str) -> str:
    """Fetch the post page (old.reddit.com) and return the OP's full selftext + title."""
    url = permalink_or_url
    if permalink_or_url.startswith("/"):
        url = "https://old.reddit.com" + permalink_or_url

    try:
        res = requests.get(url, headers=HEADERS, timeout=30)
        res.raise_for_status()
    except Exception:
        return ""

    soup = BeautifulSoup(res.text, "html.parser")

    title_el = soup.find("a", class_="title") or soup.find("p", class_="title")
    title = title_el.get_text(" ", strip=True) if title_el else ""

    op = soup.select_one("div.thing")
    body_texts: List[str] = []
    if op:
        for sel in [
            "div.expando div.usertext-body",
            "div.usertext-body",
            "div.expando .md",
            "div.selftext .md",
        ]:
            for el in op.select(sel):
                txt = el.get_text(" ", strip=True)
                if txt:
                    body_texts.append(txt)

    body = " ".join(body_texts)
    full = (title + " " + body).strip() if body else title
    full = re.sub(r"\bloading\.\.\.$", "", full, flags=re.IGNORECASE).strip()
    return full[:6000]


def scrape_sub_new(sub: str, max_posts: int = 12) -> List[Dict[str, Any]]:
    """Scrape the newest posts for a subreddit using old.reddit.com."""
    url = f"https://old.reddit.com/r/{sub}/new/"
    res = requests.get(url, headers=HEADERS, timeout=30)
    if res.status_code == 429:
        raise RuntimeError("429 Too Many Requests")
    res.raise_for_status()

    soup = BeautifulSoup(res.text, "html.parser")
    posts = soup.select("div.thing")
    out: List[Dict[str, Any]] = []
    for post in posts[:max_posts]:
        title = post.find("a", class_="title").get_text(strip=True)
        permalink = post.get("data-permalink") or ""
        full_text = fetch_post_full_text(permalink) if permalink else title
        snippet = (full_text[:160] + "…") if len(full_text) > 160 else full_text
        scraped_at = _now_iso()
        out.append(
            {
                "id": post.get("data-fullname") or post.get("id") or "",
                "url": "https://reddit.com" + permalink if permalink else (post.get("data-url") or ""),
                "title": title,
                "text": full_text,
                "snippet": snippet,
                "subreddit": sub,
                "source": sub,
                "platform": "reddit",
                "scraped_at": scraped_at,
            }
        )
        time.sleep(0.15)
    return out
