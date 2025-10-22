"""Utility functions for scraping Stocktwits News articles for trading signals."""
from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://stocktwits.com"
LIST_URL = f"{BASE_URL}/news-articles"

# Match your existing convention
HEADERS = {"User-Agent": "Mozilla/5.0 (wisdom-of-sheep/1.0; +https://stocktwits.com/news-articles)"}

DEFAULT_SECTIONS = [
    # Listing page already aggregates "All" news; keeping this for parity with other scrapers
    "news-articles",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _clean_text(s: str) -> str:
    # Collapse whitespace and strip
    return re.sub(r"\s+", " ", s or "").strip()


def _article_id_from_url(url: str) -> str:
    # Use the last path segment if present; otherwise fall back to the whole URL
    try:
        p = urlparse(url)
        # e.g. /news-articles/markets/equity/<slug>/<short-id>
        segs = [seg for seg in p.path.split("/") if seg]
        return segs[-1] if segs else url
    except Exception:
        return url


def _extract_article_body(soup: BeautifulSoup) -> str:
    """
    Stocktwits article pages (as of 2025-09-25) render the body as text nodes under the main article container.
    We gather paragraphs under the article content region, skipping nav/ads.
    """
    parts: List[str] = []

    # Try a few reasonable containers
    candidates = [
        # main article area usually holds the h1 and paragraphs
        soup.select_one("main") or soup,
    ]

    for root in candidates:
        # Prefer paragraphs under the main content; ignore footers/related links
        for p in root.select("article p"):
            txt = _clean_text(p.get_text(" ", strip=True))
            if txt:
                parts.append(txt)

        if not parts:
            # Fallback: any <p> between the H1 and "Latest News" section
            # (crude but resilient)
            h1 = root.find("h1")
            if h1:
                cur = h1.find_next("p")
                while cur:
                    # stop at Latest News / footer cues
                    if cur.name in ("h2", "h3") and "Latest News" in cur.get_text("", strip=True):
                        break
                    txt = _clean_text(cur.get_text(" ", strip=True))
                    if txt:
                        parts.append(txt)
                    cur = cur.find_next_sibling()
            # If still nothing, last resort: grab all <p>
            if not parts:
                for p in root.find_all("p"):
                    txt = _clean_text(p.get_text(" ", strip=True))
                    if txt:
                        parts.append(txt)

    body = " ".join(parts)
    # Keep to a sane size for your LLM extractor (same policy as reddit)
    return body[:6000]


def fetch_article_full_text(url: str) -> Dict[str, Any]:
    """Fetch a single Stocktwits News article and return normalized fields."""
    full_url = urljoin(BASE_URL, url)
    try:
        res = requests.get(full_url, headers=HEADERS, timeout=30)
        res.raise_for_status()
    except Exception:
        # Return minimal structure so caller can skip gracefully
        return {
            "id": _article_id_from_url(full_url),
            "url": full_url,
            "title": "",
            "text": "",
            "snippet": "",
            "source": "Stocktwits News",
            "platform": "stocktwits",
            "scraped_at": _now_iso(),
        }

    soup = BeautifulSoup(res.text, "html.parser")

    # Title
    h1 = soup.find("h1")
    title = _clean_text(h1.get_text(" ", strip=True)) if h1 else ""

    # Body
    text = _extract_article_body(soup)

    # Snippet
    snippet = (text[:160] + "…") if len(text) > 160 else text

    return {
        "id": _article_id_from_url(full_url),
        "url": full_url,
        "title": title or snippet or full_url,
        "text": text or title,
        "snippet": snippet or title,
        "source": "Stocktwits News",
        "platform": "stocktwits",
        "scraped_at": _now_iso(),
    }


def scrape_stocktwits_news(max_items: int = 12) -> List[Dict[str, Any]]:
    """
    Scrape the main Stocktwits News listing and return up to `max_items` normalized posts.
    Strategy: find all "Full Article" links or article card links under /news-articles/, then fetch each article.
    """
    try:
        res = requests.get(LIST_URL, headers=HEADERS, timeout=30)
        # Stocktwits sometimes blocks bots or redirects; raise on non-200
        res.raise_for_status()
    except Exception as e:
        # Surface a throttling-style error if needed, to be consistent with your reddit scraper semantics
        if hasattr(res, "status_code") and res.status_code == 429:
            raise RuntimeError("429 Too Many Requests")
        raise

    soup = BeautifulSoup(res.text, "html.parser")

    # Collect article hrefs:
    hrefs: List[str] = []

    # 1) Explicit "Full Article" anchors
    for a in soup.find_all("a", string=lambda s: s and "Full Article" in s):
        href = a.get("href")
        if href and "/news-articles/" in href and href not in hrefs:
            hrefs.append(href)

    # 2) Fallback: any anchor under the News listing that points into /news-articles/
    if not hrefs:
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/news-articles/" in href and href not in hrefs:
                hrefs.append(href)

    # Normalize absolute URLs and de-dup while keeping order
    normed: List[str] = []
    seen = set()
    for href in hrefs:
        url = urljoin(BASE_URL, href)
        if url not in seen:
            seen.add(url)
            normed.append(url)

    posts: List[Dict[str, Any]] = []
    for url in normed[:max_items]:
        item = fetch_article_full_text(url)
        posts.append(item)
        time.sleep(0.15)  # be polite

    return posts
