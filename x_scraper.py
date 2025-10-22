"""Helpers to retrieve public X (Twitter) posts via Nitter RSS feeds."""
from __future__ import annotations

from typing import Any, Dict, List

from rss_parser import fetch_rss_feed

DEFAULT_X_HANDLES = [
    "CNBC",
    "ritholtz",
    "wsjmarkets",
    "the_real_fly",
    "TraderTVLive",
    "LizAnnSonders",
    "Bespoke",
]
NITTER_BASE = "https://nitter.net"


def scrape_x_feed(handle: str, max_posts: int = 10) -> List[Dict[str, Any]]:
    """Fetch recent posts for an X handle using Nitter RSS."""
    if not handle:
        return []
    handle_clean = handle.strip().lstrip("@")
    if not handle_clean:
        return []
    rss_url = f"{NITTER_BASE}/{handle_clean}/rss"
    posts = fetch_rss_feed(rss_url, name=f"@{handle_clean}", max_items=max_posts, platform="x")
    for post in posts:
        post["platform"] = "x"
        post["source"] = f"@{handle_clean}"
    return posts
