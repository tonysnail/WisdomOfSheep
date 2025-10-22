"""Utilities for parsing RSS feeds into normalized post dictionaries (with full-article extraction)."""
from __future__ import annotations

import html
import json
import re
import time
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple
import xml.etree.ElementTree as ET
from urllib.parse import urljoin, urlparse
from pathlib import Path
import http.cookiejar as cookiejar

import requests
from bs4 import BeautifulSoup, Tag


# -------------------- Config --------------------
RSS_HEADERS = {"User-Agent": "Mozilla/5.0 (rss-crowd-signals/1.0)"}

# Enable SA full-text attempts by default (override with ALLOW_SA_FULLTEXT=0 if needed)
ALLOW_SA_FULLTEXT = os.getenv("ALLOW_SA_FULLTEXT", "1") == "1"

PAGE_HEADERS_PRIMARY = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36 (news-fulltext/1.0)"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
}
PAGE_HEADERS_MOBILE = {
    "User-Agent": (
        "Mozilla/5.0 (iPhone; CPU iPhone OS 16_2 like Mac OS X) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.2 Mobile/15E148 Safari/604.1"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
    "Referer": "https://news.google.com/",
}
PAGE_HEADERS_BOT = {
    "User-Agent": "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
}

# Domains where we always skip full-text (hard paywalls/heavy JS)
FULLTEXT_BLOCKLIST = {
    "wsj.com",
    "bloomberg.com",
}

# Connection pooling: reuse TLS handshakes, speed up repeated domain hits
_SESSION = requests.Session()
_adapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=40, max_retries=0)
_SESSION.mount("https://", _adapter)
_SESSION.mount("http://", _adapter)


def _apply_seekingalpha_auth(session: requests.Session) -> None:
    """
    If SA auth is available (env var SA_COOKIE or cookies file), attach it.
    Prefers a real cookie jar over a raw Cookie header.
    - Place a Netscape-format cookies file as 'sa_cookies.txt' next to this script, OR
    - export SA_COOKIE="keyA=...; keyB=...; ..."
    """
    # 1) cookies.txt file (Netscape/Mozilla format)
    ck_path = Path(__file__).with_name("sa_cookies.txt")
    if ck_path.exists():
        jar = cookiejar.MozillaCookieJar(str(ck_path))
        try:
            jar.load(ignore_discard=True, ignore_expires=True)
            session.cookies.update(jar)
            return
        except Exception:
            pass

    # 2) raw Cookie header from env
    raw = os.getenv("SA_COOKIE", "").strip()
    if raw:
        # store temporarily; injected only for seekingalpha.com requests
        session.headers["X-SA-Raw-Cookie"] = raw


_apply_seekingalpha_auth(_SESSION)

REQUEST_TIMEOUT = 15
RETRIES = 2
BACKOFF = 1.7
FETCH_FULL_ARTICLE = True
_BRIDGE_CACHE: Dict[str, str] = {}  # maps IR URL -> www.thomsonreuters.com PR URL

DEFAULT_RSS_FEEDS: List[Dict[str, str]] = [
    {"name": "CNBC Markets", "url": "https://www.cnbc.com/id/15839135/device/rss/rss.xml"},
    {"name": "Yahoo Finance Top News", "url": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=yhoo&region=US&lang=en-US"},
    {"name": "MarketWatch Top Stories", "url": "https://feeds.marketwatch.com/marketwatch/topstories/"},
    {"name": "Nasdaq Market News", "url": "https://www.nasdaq.com/feed/rssoutbound?category=Stock-Market-News"},
    # {"name": "Investing.com Market News", "url": "https://www.investing.com/rss/news_25.rss"},
    {"name": "Wall Street Journal Markets", "url": "https://feeds.a.dj.com/rss/RSSMarketsMain"},
    {"name": "Federal Reserve Press Releases", "url": "https://www.federalreserve.gov/feeds/press_all.xml"},
    {"name": "Thomson Reuters Investor Relations", "url": "https://ir.thomsonreuters.com/rss/news-releases.xml"},
    {"name": "Central Banking News", "url": "https://www.centralbanking.com/rss"},
    {"name": "Top Financial News (Feedspot curated)", "url": "https://www.feedspot.com/infiniterss.php?_src=feed_title&followfeedid=5204005"},
    {"name": "Seeking Alpha — All News", "url": "https://seekingalpha.com/market_currents.xml"},
    {"name": "Seeking Alpha — Wall Street Breakfast", "url": "https://seekingalpha.com/tag/wall-st-breakfast.xml"},
    {"name": "CNBCTV18 — Markets", "url": "https://www.cnbctv18.com/commonfeeds/v1/cne/rss/market.xml"},

    # --- Nasdaq Trader (official) ---
    {"name": "Nasdaq Trader — Equity Alerts (All)", 
     "url": "https://www.nasdaqtrader.com/rss.aspx?categorylist=2,6,7&feed=currentheadlines"},
    {"name": "Nasdaq Trader — Options Alerts (All)", 
     "url": "https://www.nasdaqtrader.com/rss.aspx?categorylist=11,12,13&feed=currentheadlines"},
    {"name": "Nasdaq Trader — Data & Technical Alerts (All)", 
     "url": "https://www.nasdaqtrader.com/rss.aspx?categorylist=5,48&feed=currentheadlines"},
    {"name": "Nasdaq Trader — NFN News", 
     "url": "https://www.nasdaqtrader.com/rss.aspx?categorylist=19&feed=currentheadlines"},
    {"name": "Nasdaq Trader — Futures Alerts (All)", 
     "url": "https://www.nasdaqtrader.com/rss.aspx?categorylist=51,52,53&feed=currentheadlines"},
]

_NAV_MENU_LINE = re.compile(
    r"^U\.S\. Market.*NASDAQ Commodities Europe.*$", re.IGNORECASE
)

# ----- NasdaqTrader post-slice anchors -----
_NASDAQTRADER_START_RE = re.compile(
    r"^(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]*,?\s+[\w\u00A0]+\s+\d{1,2},\s+\d{4}$",
    re.IGNORECASE,
)

_NASDAQTRADER_HEADER_RE = re.compile(
    r"^Nasdaq[\s\u00A0]+Fund[\s\u00A0]+Network[\s\u00A0]*#\d{4}\s*-\s*\d+",
    re.IGNORECASE,
)

_NASDAQTRADER_END_RE = re.compile(
    r"^(Email Alert Subscriptions:|"
    r"View NASDAQTrader\.com Mobile|"
    r"Nasdaq Trader Popular Sections:|"
    r"© Copyright|Privacy Statement|Contact Us|Help|Feedback)$",
    re.IGNORECASE,
)

# A single-line “bulletin code” like ETA 2025-72, UTP Vendor Alert 2025-10, etc.
_NTR_CODE_RE = re.compile(
    r"^(ETA|OTA|RTA|NOM|PHLX|BX|ISE|GEMX|MRX|UTP(?: Vendor Alert)?)\s*\d{4}-\d+\b",
    re.IGNORECASE,
)


_NTR_DATE_RE = re.compile(
    r"^(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]*,?\s+"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}$",
    re.IGNORECASE,
)


# words common in the big menu slabs; used to penalize/stop
_NAV_TOKENS = {
    "u.s.", "market", "equities", "nasdaq", "bx", "psx",
    "exchange", "traded", "funds", "nextshares", "options",
    "nom", "phlx", "ise", "gemx", "mrx", "membership",
    "etf", "home", "indexes", "dlp", "tradeinfo", "imbalance",
    "first", "north", "nordic", "baltic", "commodities", "europe",
    "specifications", "price", "reports", "testing", "protocols", "connectivity",
}

# -------------------- Helpers --------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _extract_text(nodes: Iterable[ET.Element]) -> str:
    pieces: List[str] = []
    for node in nodes:
        if node is None:
            continue
        text = "".join(node.itertext())
        if text:
            pieces.append(text)
    return " ".join(pieces)


def _clean_html(raw: str) -> str:
    if not raw:
        return ""
    txt = BeautifulSoup(html.unescape(raw), "html.parser").get_text(" ", strip=True)
    return txt.strip()


def _domain(url: str) -> str:
    return urlparse(url).netloc.lower()

def _is_nav_line(s: str) -> bool:
    """Line-level heuristic: drop very nav-like lines."""
    tokens = {t.lower() for t in re.findall(r"[A-Za-z][A-Za-z\-]+", s or "")}
    return len(tokens & _NAV_TOKENS) >= 10

def _normalize_ws_line(s: str) -> str:
    """Collapse whitespace and NBSP to a single ASCII space."""
    return re.sub(r"[\s\u00A0]+", " ", (s or "").strip())


def _normalize_ws(s: str) -> str:
    """Normalize whitespace for blocks."""
    return re.sub(r"[\s\u00A0]+", " ", (s or "").strip())

def _find_date_node(soup: BeautifulSoup) -> Optional[Tag]:
    """Find the DOM node whose text matches the bulletin date line."""
    root = soup.select_one("#mainContent") or soup.select_one("[id*='MainContent']") or soup
    # Likely date spots first
    for sel in (
        "#mainContent time", "#mainContent .date", "#mainContent .dateline",
        "[id*='MainContent'] time", "[id*='MainContent'] .date", "[id*='MainContent'] .dateline",
    ):
        for el in root.select(sel):
            t = _normalize_ws(el.get_text(" ", strip=True))
            if _NTR_DATE_RE.match(t):
                return el
    # General scan near top
    for el in root.find_all(["h1", "h2", "h3", "p", "div", "span", "time"], limit=400):
        t = _normalize_ws(el.get_text(" ", strip=True))
        if _NTR_DATE_RE.match(t):
            return el
    return None

def _collect_following_from(node: Tag) -> List[Tag]:
    """
    From the given node (date/title), collect following siblings that look like content.
    Skip link-dense/nav blocks. Stop when footer anchors appear.
    """
    # climb to a reasonable container where siblings are meaningful
    container = node.parent if isinstance(node.parent, Tag) else None
    for _ in range(6):
        if not container or not isinstance(container, Tag):
            break
        children = [c for c in container.children if isinstance(c, Tag)]
        if len(children) >= 3:
            break
        container = container.parent if isinstance(container.parent, Tag) else None
    if not container or not isinstance(container, Tag):
        container = node.parent if isinstance(node.parent, Tag) else node

    parts: List[Tag] = []
    seen_start = False
    for ch in container.children:
        if not isinstance(ch, Tag):
            continue
        if not seen_start:
            if (ch is node) or (node in getattr(ch, "descendants", [])):
                seen_start = True
            continue
        # hard stop on obvious section breaks/footers
        line = _normalize_ws(ch.get_text(" ", strip=True))
        if not line:
            continue
        if _NASDAQTRADER_END_RE.match(line):
            break
        if ch.name in ("header", "h1", "h2"):
            break
        if _is_nav_like_block(ch):
            continue
        parts.append(ch)
        if sum(len(_normalize_ws(p.get_text(' ', strip=True))) for p in parts) > 15000:
            break
    return parts

def _render_blocks(parts: List[Tag]) -> str:
    frag = BeautifulSoup("", "html.parser")
    for p in parts:
        frag.append(p)
    _strip_junk(frag)
    return _join_paragraphs(frag)

def _nasdaqtrader_line_slice(text: str) -> str:
    """
    Last-resort line-based slicer: start at first date/header line, stop at footer anchors.
    Removes nav-like lines anywhere.
    """
    lines: List[str] = []
    for raw in (text or "").split("\n"):
        ln = _normalize_ws_line(raw)
        if ln and not _is_nav_line(ln):
            lines.append(ln)
    if not lines:
        return ""
    # start at date or NFN header
    start = 0
    for i, ln in enumerate(lines):
        if _NASDAQTRADER_START_RE.match(ln) or _NASDAQTRADER_HEADER_RE.match(ln):
            start = i
            break
    # stop at first footer token
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if _NASDAQTRADER_END_RE.match(lines[j]):  # now matches "starts with"
            end = j
            break
    out = "\n\n".join(lines[start:end]).strip()
    cleaned = [ln for ln in out.split("\n") if ln and not _is_nav_line(ln)]
    return "\n\n".join(cleaned).strip()

def _find_date_node(soup: BeautifulSoup) -> Optional[Tag]:
    """Find the DOM node whose text matches the bulletin date line."""
    root = soup.select_one("#mainContent") or soup.select_one("[id*='MainContent']") or soup
    # Likely date containers first
    for sel in ("#mainContent time", "#mainContent .date", "[id*='MainContent'] time", "[id*='MainContent'] .date"):
        for el in root.select(sel):
            t = _normalize_ws(el.get_text(" ", strip=True))
            if _NTR_DATE_RE.match(t):
                return el
    # General scan near top
    for el in root.find_all(["h1", "h2", "h3", "p", "div", "time"], limit=300):
        t = _normalize_ws(el.get_text(" ", strip=True))
        if _NTR_DATE_RE.match(t):
            return el
    return None

def _collect_following_from(node: Tag) -> List[Tag]:
    """
    Starting from the node (date/title), collect following siblings that look like content.
    Skip nav-like/link-dense blocks. Stop at footer anchors or new sections.
    """
    # climb to a reasonable container where siblings are meaningful
    container = node.parent if isinstance(node.parent, Tag) else None
    for _ in range(6):
        if not container or not isinstance(container, Tag):
            break
        blocks = [c for c in container.children if isinstance(c, Tag)]
        if len(blocks) >= 3:
            break
        container = container.parent if isinstance(container.parent, Tag) else None
    if not container or not isinstance(container, Tag):
        container = node.parent if isinstance(node.parent, Tag) else node

    parts: List[Tag] = []
    seen_start = False
    for ch in container.children:
        if not isinstance(ch, Tag):
            continue
        if not seen_start:
            if (ch is node) or (node in getattr(ch, "descendants", [])):
                seen_start = True
            continue
        text_line = _normalize_ws(ch.get_text(" ", strip=True))
        if not text_line:
            continue
        if _NASDAQTRADER_END_RE.match(text_line):
            break
        if ch.name in ("header", "h1", "h2"):
            break
        if _is_nav_like_block(ch):
            continue
        parts.append(ch)
        # Safety bound
        if sum(len(_normalize_ws(p.get_text(' ', strip=True))) for p in parts) > 15000:
            break
    return parts

def _render_blocks(parts: List[Tag]) -> str:
    frag = BeautifulSoup("", "html.parser")
    for p in parts:
        frag.append(p)
    _strip_junk(frag)
    text = _join_paragraphs(frag)
    return _nasdaqtrader_postslice_lines(text)

def _nasdaqtrader_postslice_lines(text: str) -> str:
    """Line-based slice: start at date/header, stop at footer anchors; drop nav-like lines."""
    lines = []
    for raw in (text or "").split("\n"):
        ln = _normalize_ws_line(raw)
        if ln and not _is_nav_line(ln):
            lines.append(ln)
    if not lines:
        return ""
    # find start: date or header
    start = 0
    for i, ln in enumerate(lines):
        if _NASDAQTRADER_START_RE.match(ln) or _NASDAQTRADER_HEADER_RE.match(ln):
            start = i
            break
    # find end: first footer/boilerplate
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if _NASDAQTRADER_END_RE.match(lines[j]):
            end = j
            break
    out = "\n\n".join(lines[start:end]).strip()
    cleaned = [ln for ln in out.split("\n") if ln and not _is_nav_line(ln)]
    return "\n\n".join(cleaned).strip()



def _is_nav_like_block(el: Tag) -> bool:
    """Heuristic: lots of short links/labels = nav."""
    try:
        text = el.get_text(" ", strip=True)
    except Exception:
        text = ""
    if _textlen(text) < 20:
        return False
    links = el.find_all("a")
    if links:
        link_text = " ".join(a.get_text(" ", strip=True) for a in links)
        # ratio of link chars to total chars
        if _textlen(link_text) / max(1, _textlen(text)) > 0.55:
            return True
        # many very short links is nav-y
        short_links = sum(1 for a in links if _textlen(a.get_text().strip()) <= 12)
        if short_links >= 12:
            return True
    # many nav-ish tokens?
    tokens = {t.lower() for t in re.findall(r"[A-Za-z][A-Za-z\-]+", text)}
    if len(tokens & _NAV_TOKENS) >= 8:
        return True
    return False

def _is_fulltext_allowed(link_url: str) -> bool:
    """
    Decide if we should try to fetch and extract full text from the linked page.
    - Always block hard paywalls (e.g., WSJ, Bloomberg).
    - Seeking Alpha controlled by env flag (ALLOW_SA_FULLTEXT).
    - For nasdaqtrader.com, allow full-text only for article pages (e.g., TraderNews.aspx),
      but skip when the link itself is an RSS endpoint (rss.aspx?...) — those are bulletins.
    """
    try:
        d = _domain(link_url)
        p = urlparse(link_url).path.lower()
    except Exception:
        return True

    # Skip full-text attempts for bulletin/RSS endpoints, but still keep the items.
    if d.endswith("nasdaqtrader.com") and p.endswith("/rss.aspx"):
        return False

    if d.endswith("seekingalpha.com"):
        # Allow SA full text only if explicitly enabled
        return ALLOW_SA_FULLTEXT

    # Hard paywalls / heavy JS
    return not any(d.endswith(bad) for bad in FULLTEXT_BLOCKLIST)


def _fetch_html_with_headers(url: str, headers: Dict[str, str]) -> Tuple[str, Optional[str]]:
    try:
        # Inject raw Cookie for seekingalpha.com if provided (only for that domain)
        h = dict(headers)
        dom = _domain(url)
        raw_cookie = _SESSION.headers.get("X-SA-Raw-Cookie")
        if dom.endswith("seekingalpha.com") and raw_cookie and "Cookie" not in h:
            h["Cookie"] = raw_cookie

        resp = _SESSION.get(url, headers=h, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        resp.raise_for_status()
        final_url = resp.url
        body = resp.text or ""
        if len(body) < 200:
            return final_url, None
        return final_url, body
    except Exception:
        return url, None


def _fetch_html(url: str) -> Tuple[str, Optional[str]]:
    final_url, html_text = url, None
    # Desktop
    for i in range(RETRIES + 1):
        final_url, html_text = _fetch_html_with_headers(url, PAGE_HEADERS_PRIMARY)
        if html_text:
            return final_url, html_text
        time.sleep(min(5, BACKOFF ** i))
    # Mobile
    for i in range(RETRIES + 1):
        final_url, html_text = _fetch_html_with_headers(url, PAGE_HEADERS_MOBILE)
        if html_text:
            return final_url, html_text
        time.sleep(min(5, BACKOFF ** i))
    # Googlebot
    for i in range(RETRIES + 1):
        final_url, html_text = _fetch_html_with_headers(url, PAGE_HEADERS_BOT)
        if html_text:
            return final_url, html_text
        time.sleep(min(5, BACKOFF ** i))
    return final_url, None

def _fetch_html_sa(url: str) -> Tuple[str, Optional[str]]:
    """
    Seeking Alpha tends to return a longer SSR body if the request looks like
    it came via a news referrer. Try desktop UA with a Google News referrer,
    then mobile UA with the same referrer.
    """
    sa_headers_1 = dict(PAGE_HEADERS_PRIMARY)
    sa_headers_1["Referer"] = "https://news.google.com/"
    sa_headers_2 = dict(PAGE_HEADERS_MOBILE)  # already has the Referer

    final_url, html_text = _fetch_html_with_headers(url, sa_headers_1)
    if html_text:
        return final_url, html_text
    final_url, html_text = _fetch_html_with_headers(url, sa_headers_2)
    if html_text:
        return final_url, html_text
    # fall back to your normal rotations
    return _fetch_html(url)

def _fix_mojibake(s: str) -> str:
    """
    Common UTF-8-as-latin1 mojibake fixer.
    If round-trip fails, return original.
    """
    if not s:
        return s
    try:
        # e.g., “â” -> “’”
        return bytes(s, "latin-1").decode("utf-8")
    except Exception:
        return s

def _as_soup_fragment(node: Tag) -> BeautifulSoup:
    return BeautifulSoup(str(node), "html.parser")


def _textlen(s: Optional[str]) -> int:
    return len((s or "").strip())


def _safe_select_one(soup: BeautifulSoup, selectors: List[str]) -> Optional[Tag]:
    for sel in selectors:
        try:
            n = soup.select_one(sel)
        except Exception:
            n = None
        if n:
            return n
    return None

def _textlen(s: Optional[str]) -> int:
    return len((s or "").strip())


def _get_title_node_and_text(soup: BeautifulSoup) -> Tuple[Optional[Tag], Optional[str]]:
    title_sel = [
        "#mainContent h1", "[id$='_MainContent'] h1", "[id*='MainContent'] h1",
        "main h1", "article h1", "h1.pageTitle", "h1.title", "header h1", "h1",
    ]
    for sel in title_sel:
        h = soup.select_one(sel)
        if h:
            t = h.get_text(" ", strip=True)
            if _textlen(t) > 5:
                return h, t
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content"):
        return None, og["content"].strip()
    return None, None

def _pick_ancestor_container(node: Tag, max_levels: int = 6) -> Optional[Tag]:
    """
    Walk up ancestors to find a container likely to hold the bulletin body.
    Prefer <article>, then <section>, then <div>.
    """
    if not node:
        return None
    cur = node
    best = None
    level = 0
    while cur and level <= max_levels:
        if isinstance(cur, Tag) and cur.name in ("article", "section", "div", "main"):
            best = cur
            # Stop early if it's an article/section
            if cur.name in ("article", "section"):
                break
        cur = cur.parent
        level += 1
    return best

def _sa_jsonstring_unescape(s: str) -> str:
    """
    Safely unescape a JSON string fragment (without surrounding quotes).
    Handles NNN, escaped quotes, etc.
    """
    try:
        return json.loads(f'"{s}"')
    except Exception:
        # last resort: HTML-unescape then strip
        return html.unescape(s or "").strip()

def _count_p_text(container: Tag) -> int:
    try:
        return sum(_textlen(p.get_text(" ", strip=True)) for p in container.find_all("p"))
    except Exception:
        return 0

# ----- NasdaqTrader post-slice anchors (more tolerant to NBSP etc.) -----
_NASDAQTRADER_START_RE = re.compile(
    r"^(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]*,?\s+[\w\u00A0]+\s+\d{1,2},\s+\d{4}$",
    re.IGNORECASE,
)

_NASDAQTRADER_HEADER_RE = re.compile(
    r"^Nasdaq[\s\u00A0]+Fund[\s\u00A0]+Network[\s\u00A0]*#\d{4}\s*-\s*\d+",
    re.IGNORECASE,
)

# Stop markers (match line *starts with* these; not full-line equals)
_NASDAQTRADER_END_RE = re.compile(
    r"^(?:Email Alert Subscriptions\b|"
    r"Please follow Nasdaq\b|"
    r"View NASDAQTrader\.com Mobile\b|"
    r"Nasdaq Trader Popular Sections\b|"
    r"© Copyright\b|Disclaimer\b|Trademarks\b|Privacy Statement\b|Contact Us\b|Help\b|Feedback\b)",
    re.IGNORECASE,
)

# words common in the big menu slabs; used to penalize/stop
_NAV_TOKENS = {
    "u.s.", "market", "equities", "nasdaq", "bx", "psx",
    "exchange", "traded", "funds", "nextshares", "options",
    "nom", "phlx", "ise", "gemx", "mrx", "membership",
    "etf", "home", "indexes", "dlp", "tradeinfo", "imbalance",
    "first", "north", "nordic", "baltic", "commodities", "europe",
    "specifications", "price", "reports", "testing", "protocols", "connectivity",
}

def _is_nav_line(s: str) -> bool:
    """Line-level heuristic: drop very nav-like lines."""
    tokens = {t.lower() for t in re.findall(r"[A-Za-z][A-Za-z\-]+", s or "")}
    return len(tokens & _NAV_TOKENS) >= 10

def _normalize_ws_line(s: str) -> str:
    """Collapse whitespace and NBSP to a single ASCII space."""
    return re.sub(r"[\s\u00A0]+", " ", (s or "").strip())

def _nasdaqtrader_postslice(text: str) -> str:
    """
    Given cleaned text for a TraderNews page, retain only the bulletin content:
    - Start: date line OR 'Nasdaq Fund Network #YYYY - N'
    - End: before subscriptions/boilerplate/footer section.
    """
    if not text:
        return text

    # Normalize lines (NBSP → space, collapse whitespace)
    raw_lines = [ln for ln in text.split("\n")]
    lines = []
    for ln in raw_lines:
        ln_norm = _normalize_ws_line(ln)
        if not ln_norm:
            continue
        # Drop obvious nav-ish lines up front
        if _is_nav_line(ln_norm):
            continue
        lines.append(ln_norm)

    n = len(lines)
    if n == 0:
        return ""

    # find start anchor (date or 'Nasdaq Fund Network #…')
    start = 0
    for i, ln in enumerate(lines):
        if _NASDAQTRADER_START_RE.match(ln) or _NASDAQTRADER_HEADER_RE.match(ln):
            start = i
            break

    # find end anchor:
    #  1) first footer/boilerplate heading, OR
    #  2) the *next* date line (beginning of another bulletin), OR
    #  3) a strong new-bulletin code on its own line (ETA 2025-xx, UTP Vendor Alert yyyy-n)
    end = n
    for j in range(start + 1, n):
        ln = lines[j]
        if _NASDAQTRADER_END_RE.match(ln) or _NASDAQTRADER_START_RE.match(ln) or _NTR_CODE_RE.match(ln):
            end = j
            break

    sliced = "\n\n".join(lines[start:end]).strip()

    # safety: drop any remaining mega-nav line
    if sliced:
        rows = [r for r in (r.strip() for r in sliced.split("\n")) if r]
        rows = [r for r in rows if not _is_nav_line(r)]
        sliced = "\n\n".join(rows).strip()

    # Optional: fix spaced ordinals like "1 st" → "1st"
    if sliced:
        sliced = re.sub(r"\b(\d+)\s+(st|nd|rd|th)\b", r"\1\2", sliced, flags=re.IGNORECASE)

    return sliced

# -------------------- Extraction rules --------------------
SITE_SELECTORS: Dict[str, List[str]] = {
    "www.investing.com": [
        "main article",
        "article .WYSIWYG",
        "article .article__content",
        "div#articleContent",
        "div.articlePage",
    ],
    "ir.thomsonreuters.com": [
        "main article",
        "div.wd_text_block",
        "div.module_body",
        "div.news-release",
        "main article .wd_text_block",
        "main article .text",
        "main .module_body",
    ],
    "www.thomsonreuters.com": [
        "main [data-component='ArticleBody']",
        "article [data-component='ArticleBody']",
        "main article",
        "article",
    ],
    "www.marketwatch.com": [
        "article .article__content",
        "article .paywall",
        "article .js-article__body",
        "div.article__body",
    ],
    "www.cnbc.com": [
        "div.ArticleBody-articleBody",
        "article .ArticleBodyWrapper",
        "article .articleBody",
        "div.group",
    ],
    "finance.yahoo.com": [
        "div.caas-body",
        "article div[data-test-locator='storycontent']",
    ],
    "www.wsj.com": [
        "article .article-content",
        "article section[data-type='article']",
    ],
    "www.nasdaq.com": [
        "main article",
        "article [data-module='article-body']",
        "article .body__content",
        "div#two-column-main-content article",
        "div#two-column-main-content",
    ],
    # Nasdaq Trader (TraderNews articles)
    "www.nasdaqtrader.com": [
        "#mainContent .nw_content",
        "#mainContent .newsContainer",
        "#mainContent .content",
        "#mainContent article",
        "[id$='_MainContent'] article",
        "[id$='_MainContent'] .content",
        "article",
    ],
    "www.federalreserve.gov": [
        "main article",
        "main #article",
        "main #content",
        "main #press",
        "main .col-xs-12.col-md-8",
        "div#main",
        "div.col-md-8",
        "div.col-sm-8",
    ],
    "www.centralbanking.com": [
        "article .content-body",
        "article .text",
        "article",
    ],
    # Seeking Alpha
    "seekingalpha.com": [
        # News briefs variants
        "section[data-test-id='post-content']",
        "div[data-test-id='post-content']",
        "div[data-test-id='content-container']",
        "article [data-test-id='post-content']",
        "article [itemprop='articleBody']",
        # Additional fallbacks observed in SA news pages
        "article .sa-art__content",
        "article .sa-article-content",
        "main article",
        "article",
    ],
    "www.cnbctv18.com": [
        # Standard templates
        "article [itemprop='articleBody']",
        "article .contentarea",
        "article .article-content",
        "article .story-content",
        "article .storypage-content",
        "article .storypage-article-content",
        "article .story__content",
        "article .post-content",
        "main article",
        "article",
        # AMP-ish / fallbacks seen on variants
        ".amp-article-content",
        ".articleBody",
        ".storyBody",
        ".content-body",
        "#storyBody",
    ],
    "cnbctv18.com": [
        "article [itemprop='articleBody']",
        "article .contentarea",
        "article .article-content",
        "article .story-content",
        "article .storypage-content",
        "article .storypage-article-content",
        "article .story__content",
        "article .post-content",
        "main article",
        "article",
        ".amp-article-content",
        ".articleBody",
        ".storyBody",
        ".content-body",
        "#storyBody",
    ],
}

# Strip chrome/ads/paywalls
JUNK_SELECTORS = [
    # ----- Global chrome -----
    "nav", "header", "footer", "aside",
    "form", "noscript", "template",

    # ----- Ads / promos / paywalls -----
    ".promo", ".promotions", ".advert", ".advertisement", ".ad", ".ads",
    "[aria-label='advertisement']", "[class*='ad-']", "[id*='ad-']",
    ".paywall", ".paywall-overlay", "#paywall", ".meteredContent",
    ".subscription", ".subscribe", ".subscribe-cta",
    ".piano-offer", ".tp-modal", ".tp-backdrop", ".overlay", "#offer-overlay",

    # ----- Social / engagement / comments -----
    ".social", ".social-bar", ".share", ".share-bar", "[data-test-id='share-bar']",
    "[data-test-id='engagement']", ".engagement",
    ".comment", ".comments", ".comment-thread", ".discussion", ".disqus",

    # ----- Related / recirculation / newsletters / widgets -----
    ".related", ".related-articles", ".recommended", ".more", ".more-stories",
    ".newsletter", ".outbrain", ".trending", ".recirc", ".sidebar", ".widget",

    # ----- Meta / author / breadcrumbs / timestamps -----
    ".byline", ".author-bio", ".author-info", ".contributor",
    ".dateline", ".meta", ".tags", ".tag-list",
    ".breadcrumbs", ".breadcrumb", "#BreadCrumb", "[id*='BreadCrumb']",
    ".publication-date", ".timestamp",

    # ----- Media wrappers (keep plain text instead) -----
    "figure", "figcaption", ".inline-video", ".video", ".jwplayer", ".vjs-player",
    ".inline-audio", ".podcast-player", ".gallery", ".slideshow",
    "iframe", "embed", "object", "canvas",

    # ----- Legal / utility / notices -----
    ".cookie", ".cookie-banner", ".gdpr", ".consent", ".disclaimer",
    ".legal", ".caption", ".note", ".footnotes", ".endnote",

    # ----- Generic UI junk -----
    "button", "[role='button']", "svg", "path",

    # ----- Site-specific: Investing.com reg/paywall -----
    "#regWall", ".reg-wall", ".reg-overlay", ".reg-page", ".login-overlay",
    "#userAccount", ".signupForm", ".js-modal", ".js-login",

    # ----- Site-specific: Seeking Alpha -----
    "div.sa-art__actions",
    "div.sa-article-actions",
    "div[data-test-id='action-bar']",
    "div[data-test-id='audio-player']",
    "[aria-label='audio player']",
    ".inline-media",

    # ----- Site-specific: NasdaqTrader / broad nav blocks -----
    "#leftCol", ".leftCol", ".left-column",
    "#rightCol", ".rightCol", ".right-column",
    "#topnav", ".topnav", "#mainNav", ".mainNav", "#siteNav", ".siteNav",
    "[id*='TopNavigation']",

    "[data-test-id='comments']",
    "[data-test-id='comment']",
    "[data-test-id='engagement']",
]


# Boilerplate lines to drop
PARA_DROP_PATTERNS = [
    r"^CONTACTS?:", r"^Contact:", r"^Investor Relations", r"^Media Relations",
    r"^Forward-?looking statements", r"^Safe Harbor", r"^\(?c\)? ?\d{4}",
    r"^SOURCE: ", r"^View (?:the )?source version on", r"^For further information",
    r"^Subscribe", r"^Follow us on", r"^About (?:us|the company)",
    r"^(?:\d+\s*)?Share$",
    r"^Save$",
    r"^Play(?: \(\s*[\d<]+ ?min\s*\))?$",
    r"^Listen (?:now|to this podcast)$",
    r"^Read (?:more|the full story)$",
    # SA crumbs
    r"^See also\b",
    r"^More on\b",
    r"^Editor'?s Note\b",
    # NasdaqTrader-specific boilerplate/crumbs
    r"^Email Alert Subscriptions\b",
    r"^Please follow Nasdaq\b",
    r"^View NASDAQTrader\.com Mobile\b",
    r"^Nasdaq Trader Popular Sections\b",
    r"^© Copyright\b|^Disclaimer$|^Trademarks$|^Privacy Statement$|^Contact Us$|^Help$|^Feedback$",
    r"^Performance Statistics$", r"^Email Sign-Up$",
    # Stand-alone cross-refs to other bulletins (keep if they’re inline, drop if the whole line)
    r"^(?:ETA|OTA|RTA|NOM|PHLX|BX|ISE|GEMX|MRX|UTP(?: Vendor Alert)?)\s*\d{4}-\d+\s*$",
    # Exchange-name crumbs and utility links that appear as standalone lines
    r"^(?:The )?Nasdaq Stock Market(?: \(Nasdaq\))?$",
    r"^Nasdaq BX$", r"^Nasdaq PSX$", r"^Nasdaq Options Market$",
    r"^Nasdaq PHLX$", r"^Nasdaq BX Options$", r"^Nasdaq ISE$",
    r"^Nasdaq GEMX$", r"^Nasdaq MRX$",
    r"^U\.S\. Market Operations\b", r"^U\.S\. Market Sales\b",
    r"^Equities Trading Services\b",
    r"^Saturday Testing Policy Page$", r"^Regulation SCI BCP DR Information$",
    r"^Price List$", r"^Specifications Page$",
    # Corporate “About Nasdaq” boilerplate (first words vary a bit)
    r"^Nasdaq\s*\(Nasdaq:\s*NDAQ\)\s+is a leading global provider\b",
    r"^Comments?\s*\(\d+\)\s*$",
    r"^(See also|More on|Editor'?s Note)\b",
]
PARA_DROP_RE = re.compile("|".join(PARA_DROP_PATTERNS), re.IGNORECASE)


def _strip_junk(node: Tag) -> None:
    if not hasattr(node, "select"):
        return
    for sel in JUNK_SELECTORS:
        try:
            for el in node.select(sel):
                el.decompose()
        except Exception:
            continue


def _join_paragraphs(container: Tag) -> str:
    parts: List[str] = []
    for el in container.find_all(["p", "li"]):
        txt = el.get_text(separator=" ", strip=True)
        if not txt:
            continue
        if PARA_DROP_RE.search(txt):
            continue
        parts.append(txt)
    if not parts:
        txt = container.get_text(separator="\n", strip=True)
        if txt and not PARA_DROP_RE.search(txt):
            parts = [txt]
    text = "\n\n".join(parts)
    text = re.sub(r"[ \t\u00A0]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _readability_extract(html_text: str) -> Optional[str]:
    try:
        from readability import Document
    except Exception:
        return None
    try:
        doc = Document(html_text)
        content_html = doc.summary(html_partial=True)
        soup = BeautifulSoup(content_html, "html.parser")
        _strip_junk(soup)
        return _join_paragraphs(soup)
    except Exception:
        return None


def _jsonld_extract(html_text: str) -> Optional[str]:
    try:
        soup = BeautifulSoup(html_text, "html.parser")
        blocks = soup.find_all("script", type="application/ld+json")
        for b in blocks:
            try:
                data = json.loads(b.string or "")
            except Exception:
                continue
            items = data if isinstance(data, list) else [data]
            for obj in items:
                if not isinstance(obj, dict):
                    continue
                # @graph form
                if "@graph" in obj and isinstance(obj["@graph"], list):
                    for g in obj["@graph"]:
                        if isinstance(g, dict) and (g.get("@type") in ["Article", "NewsArticle", "Report", "PressRelease"]):
                            body = g.get("articleBody") or g.get("description")
                            if body and len(body.strip()) > 150:
                                return body.strip()
                if obj.get("@type") in ["Article", "NewsArticle", "Report", "PressRelease"]:
                    body = obj.get("articleBody") or obj.get("description")
                    if body and len(body.strip()) > 150:
                        return body.strip()
    except Exception:
        pass
    return None


def _fetch_html_once(url: str, headers: Dict[str, str], timeout: int = 10) -> Tuple[str, Optional[str]]:
    try:
        # Inject raw Cookie for seekingalpha.com if provided (only for that domain)
        h = dict(headers)
        dom = _domain(url)
        raw_cookie = _SESSION.headers.get("X-SA-Raw-Cookie")
        if dom.endswith("seekingalpha.com") and raw_cookie and "Cookie" not in h:
            h["Cookie"] = raw_cookie

        resp = _SESSION.get(url, headers=h, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
        body = resp.text or ""
        return resp.url, (body if len(body) >= 200 else None)
    except Exception:
        return url, None


def _nasdaqtrader_article_text(soup: BeautifulSoup) -> Optional[str]:
    """
    Date-first; then title-anchored; then root clean + strict line slice.
    """
    # 1) Date-anchored (best)
    date_node = _find_date_node(soup)
    if date_node:
        parts = _collect_following_from(date_node)
        body = _render_blocks(parts)
        if body:
            return _nasdaqtrader_line_slice(body) or None

    # 2) Title-anchored fallback
    root = soup.select_one("#mainContent") or soup.select_one("[id*='MainContent']") or soup
    title_node, title_text = _get_title_node_and_text(root)
    if title_node:
        parts = _collect_following_from(title_node)
        body = _render_blocks(parts)
        if body:
            if title_text and title_text not in body[:200]:
                body = f"{title_text}\n\n{body}"
            return _nasdaqtrader_line_slice(body) or None

    # 3) Root clean + line slice
    frag = _as_soup_fragment(root)
    _strip_junk(frag)
    txt = _join_paragraphs(frag)
    return _nasdaqtrader_line_slice(txt) or None


def _site_specific_extract(dom: str, soup: BeautifulSoup) -> Optional[str]:
    min_len = 300
    if dom.endswith("seekingalpha.com"):
        min_len = 80
    elif dom.endswith("www.investing.com") or dom.endswith("www.nasdaq.com"):
        min_len = 220
    elif dom.endswith("www.federalreserve.gov"):
        min_len = 180
    elif dom.endswith("cnbctv18.com") or dom.endswith("www.cnbctv18.com"):
        min_len = 160
    elif dom.endswith("www.nasdaqtrader.com"):
        min_len = 80  # bulletins can be very concise

    # NasdaqTrader: title-anchored extractor first
    if dom.endswith("www.nasdaqtrader.com"):
        # Use only the custom extractor to avoid grabbing mega-nav containers.
        t = _nasdaqtrader_article_text(soup)
        return t if t and len(t) >= min_len else None

    # SeekingAlpha: prefer rich JSON body if present
    if dom.endswith("seekingalpha.com") or dom == "seekingalpha.com":
        rich = _seekingalpha_rich_extract(str(soup), soup)
        if rich and len(rich) >= 80:
            return rich

    # Other sites: normal selector path
    for site, selectors in SITE_SELECTORS.items():
        if dom.endswith(site):
            node = _safe_select_one(soup, selectors)
            if node:
                frag = _as_soup_fragment(node)
                _strip_junk(frag)
                text = _join_paragraphs(frag)
                if text and len(text) > min_len:
                    return text
    return None



def _fetch_html_tr_fast(url: str) -> Tuple[str, Optional[str]]:
    try:
        # Inject raw Cookie for seekingalpha.com if provided (only for that domain)
        h = dict(PAGE_HEADERS_PRIMARY)
        dom = _domain(url)
        raw_cookie = _SESSION.headers.get("X-SA-Raw-Cookie")
        if dom.endswith("seekingalpha.com") and raw_cookie and "Cookie" not in h:
            h["Cookie"] = raw_cookie

        resp = _SESSION.get(url, headers=h, timeout=8, allow_redirects=True)
        resp.raise_for_status()
        body = resp.text or ""
        return resp.url, (body if len(body) >= 200 else None)
    except Exception:
        return url, None


def _heuristic_extract(soup: BeautifulSoup) -> Optional[str]:
    node = _safe_select_one(soup, ["article", "main", "[role='main']", ".story-body", ".content__article-body"])
    if node:
        frag = _as_soup_fragment(node)
        _strip_junk(frag)
        t = _join_paragraphs(frag)
        if t:
            return t
    best = None
    best_len = 0
    for div in soup.find_all("div"):
        if not isinstance(div, Tag):
            continue
        try:
            txt = div.get_text(" ", strip=True)
        except Exception:
            txt = ""
        if txt and len(txt) > best_len:
            best = div
            best_len = len(txt)
    if best:
        frag = _as_soup_fragment(best)
        _strip_junk(frag)
        return _join_paragraphs(frag)
    return None


def _pick_best_text(candidates: List[str], dom: Optional[str] = None) -> Optional[str]:
    best = None
    best_len = 0
    min_chars = 400
    min_paras = 3

    if dom:
        if dom.endswith("www.investing.com") or dom.endswith("www.nasdaq.com"):
            min_chars = 220; min_paras = 2
        elif dom.endswith("www.federalreserve.gov"):
            min_chars = 200; min_paras = 2
        elif dom.endswith("seekingalpha.com"):
            # SA news briefs can be one short paragraph
            min_chars = 80; min_paras = 1
        elif dom.endswith("cnbctv18.com") or dom.endswith("www.cnbctv18.com"):
            min_chars = 160; min_paras = 2
        elif dom.endswith("www.nasdaqtrader.com"):
            min_chars = 160; min_paras = 2

    # normal pass with thresholds
    for t in candidates:
        if not t:
            continue
        tl = t.strip()
        if len(tl) < min_chars:
            continue
        paras = [p for p in tl.split("\n\n") if p.strip()]
        if len(paras) < min_paras:
            continue
        if len(tl) > 100_000:
            tl = tl[:100_000]
        if len(tl) > best_len:
            best = tl; best_len = len(tl)

    if best:
        return best

    # CNBCTV18 fallback: pick the longest non-empty candidate anyway
    if dom and (dom.endswith("cnbctv18.com") or dom.endswith("www.cnbctv18.com")):
        longest = None
        longest_len = 0
        for t in candidates:
            if not t:
                continue
            tl = t.strip()
            if len(tl) > longest_len:
                longest = tl
                longest_len = len(tl)
        return longest

    return None

def _extract_json_value_by_key(raw: str, key: str) -> List[str]:
    """
    Scan raw HTML/JS for `"key": <json_value>` and return decoded strings.
    Handles values that are JSON strings, arrays, or objects using a simple
    balanced-brace/bracket/string parser (no full JSON parse required).
    Returns *text* strings (HTML-stripped if value is HTML).
    """
    results: List[str] = []
    pat = re.compile(rf'"{re.escape(key)}"\s*:\s*', re.IGNORECASE)
    for m in pat.finditer(raw):
        i = m.end()
        if i >= len(raw):
            continue
        c = raw[i]

        def parse_json_string(j: int) -> Tuple[int, Optional[str]]:
            # raw[j] is the opening quote
            j += 1
            out = []
            esc = False
            while j < len(raw):
                ch = raw[j]
                if esc:
                    out.append(ch)
                    esc = False
                else:
                    if ch == '\\':
                        esc = True
                    elif ch == '"':
                        # end of string
                        return j + 1, "".join(out)
                    else:
                        out.append(ch)
                j += 1
            return j, None  # unterminated

        def parse_balanced(j: int, open_ch: str, close_ch: str) -> Tuple[int, Optional[str]]:
            # raw[j] is the opening bracket/brace
            depth = 0
            in_str = False
            esc = False
            while j < len(raw):
                ch = raw[j]
                if in_str:
                    if esc:
                        esc = False
                    else:
                        if ch == '\\':
                            esc = True
                        elif ch == '"':
                            in_str = False
                else:
                    if ch == '"':
                        in_str = True
                    elif ch == open_ch:
                        depth += 1
                    elif ch == close_ch:
                        depth -= 1
                        if depth == 0:
                            return j + 1, raw[m.end():j + 1]
                j += 1
            return j, None  # unbalanced

        # Decide by first non-space char
        if c == '"':
            end, s = parse_json_string(i)
            if s is not None:
                # decode JSON string escapes properly
                try:
                    s_decoded = json.loads('"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"')
                except Exception:
                    # lenient fallback
                    s_decoded = bytes(s, "utf-8", "ignore").decode("unicode_escape", "ignore")
                # turn \u003C p \u003E etc into text
                txt = BeautifulSoup(s_decoded, "html.parser").get_text(" ", strip=True)
                if txt:
                    results.append(txt)
            continue

        if c in "[{":
            open_ch, close_ch = ("[", "]") if c == "[" else ("{", "}")
            end, blob = parse_balanced(i, open_ch, close_ch)
            if blob:
                try:
                    data = json.loads(blob)
                    # If array/object, try common text-y keys inside
                    strings = []
                    if isinstance(data, str):
                        strings = [data]
                    else:
                        strings = _deep_find_strings(
                            data,
                            keys=("articleBody","renderedBody","contentHtml","content","body","text","value","html","rawHtml")
                        )
                    parts = []
                    for s in strings:
                        parts.append(BeautifulSoup(s, "html.parser").get_text(" ", strip=True))
                    txt = "\n\n".join([p for p in parts if p.strip()])
                    if txt:
                        results.append(txt)
                except Exception:
                    # lenient: try to strip HTML out of raw blob
                    txt = BeautifulSoup(blob, "html.parser").get_text(" ", strip=True)
                    if txt:
                        results.append(txt)
            continue

        # Unexpected char – skip this occurrence
        continue

    # De-dup and keep the longest first
    uniq = []
    seen = set()
    for t in sorted(results, key=len, reverse=True):
        k = t.strip()
        if k and k not in seen:
            uniq.append(k); seen.add(k)
    return uniq

def _cnbctv18_try_variants(orig_url: str, final_url: str, chosen: Optional[str]) -> Tuple[str, Optional[str]]:
    """
    CNBCTV18: aggressively try AMP/mobile variants when the main page renders via JS.
    We try:
      - ?amp=1
      - /amp
      - m. subdomain
    Accepts the first variant with >= 180 chars; if none meet thresholds,
    returns the longest non-empty candidate across variants.
    """
    def _variants(u: str) -> List[str]:
        base = u.rstrip("/")
        out = [
            u + (("&" if "?" in u else "?") + "amp=1"),
            base + "/amp",
            u.replace("://www.", "://m."),
        ]
        # de-dupe preserving order
        seen, vlist = set(), []
        for x in out:
            if x not in seen:
                vlist.append(x); seen.add(x)
        return vlist

    # If we already have decent text, keep it; otherwise force AMP attempts
    need_variant = not chosen or len(chosen.strip()) < 300

    best_cand = None
    best_final = final_url

    if need_variant:
        for vurl in _variants(final_url):
            alt_final, alt_html = _fetch_html(vurl)
            if not alt_html:
                continue
            alt_soup = BeautifulSoup(alt_html, "html.parser")
            site_text   = _site_specific_extract(_domain(alt_final), alt_soup)
            jsonld_text = _jsonld_extract(alt_html)
            read_text   = _readability_extract(alt_html)
            heur_text   = _heuristic_extract(alt_soup)
            cand = _pick_best_text([site_text, jsonld_text, read_text, heur_text], dom=_domain(alt_final))

            # Track the longest non-empty candidate in case we don't meet thresholds
            if cand and (not best_cand or len(cand) > len(best_cand)):
                best_cand = cand
                best_final = alt_final

            # Early accept if good enough
            if cand and len(cand.strip()) >= 180:
                return alt_final, cand

        # If no variant crossed thresholds, return the best we saw
        if best_cand and len(best_cand.strip()) >= 80:
            return best_final, best_cand

    return final_url, chosen

def _sa_html_to_text(s: str) -> str:
    return BeautifulSoup(s or "", "html.parser").get_text(" ", strip=True)

def _sa_sentence_count(s: str) -> int:
    return len(re.findall(r"[.!?](?:\s|$)", s or ""))

def _sa_is_good(body: Optional[str]) -> bool:
    if not body:
        return False
    t = (body or "").strip()
    if len(t) >= 600:
        return True
    if _sa_sentence_count(t) >= 2 and len(t) >= 220:
        return True
    return False

def _sa_url_strip_amp(u: str) -> str:
    u = re.sub(r"([?&])amp=1(&|$)", r"\1", u).rstrip("?&")
    if u.rstrip("/").endswith("/amp"):
        u = u.rstrip("/")[:-4]
    return u

def _sa_id_from_url(u: str) -> Optional[str]:
    m = re.search(r"/news/(\d+)-", u)
    return m.group(1) if m else None


def _sa_collect_from_obj(o, out: List[str]) -> None:
    """
    DFS over SA Next.js + Apollo shapes.
    Collect paragraph-like strings from common keys and block arrays.
    """
    PARA_KEYS = {
        "articlebody", "body", "cleanedbody", "content", "summary",
        "description", "text", "value", "html", "rawhtml", "rendered"
    }

    def add(s: str):
        s = (s or "").strip()
        if not s:
            return
        # must look like actual prose (avoid UI crumbs)
        if len(s) < 40 and not re.search(r"[.!?](?:\s|$)", s):
            return
        out.append(s)

    if isinstance(o, dict):
        # (a) direct content-ish keys
        for k, v in o.items():
            kl = str(k).lower()

            # normalized blocks lists: [{"type":"paragraph","text":...}, ...]
            if isinstance(v, list) and kl in {"blocks", "paragraphs", "children", "items", "content"}:
                for b in v:
                    if isinstance(b, str):
                        add(_sa_html_to_text(b))
                    elif isinstance(b, dict):
                        # common SA subkeys
                        for sk in ("text", "content", "value", "html", "rawHtml", "rendered"):
                            if sk in b and isinstance(b[sk], str):
                                add(_sa_html_to_text(b[sk]))
                        if "children" in b and isinstance(b["children"], list):
                            _sa_collect_from_obj(b["children"], out)

            # direct rich-text keys
            if kl in PARA_KEYS:
                if isinstance(v, str):
                    add(_sa_html_to_text(v))
                elif isinstance(v, list):
                    for it in v:
                        if isinstance(it, str):
                            add(_sa_html_to_text(it))
                        elif isinstance(it, dict):
                            for sk in ("text", "content", "value", "html", "rawHtml", "rendered"):
                                if sk in it and isinstance(it[sk], str):
                                    add(_sa_html_to_text(it[sk]))

            # recurse
            if isinstance(v, (dict, list)):
                _sa_collect_from_obj(v, out)

    elif isinstance(o, list):
        for it in o:
            _sa_collect_from_obj(it, out)

def _deep_find_strings(obj, keys=("articleBody","renderedBody","contentHtml","content","body")) -> List[str]:
    """Walk an arbitrary JSON structure and collect strings from known content keys."""
    out: List[str] = []
    try:
        if isinstance(obj, dict):
            for k,v in obj.items():
                kl = str(k).lower()
                if any(kl == key.lower() for key in keys):
                    if isinstance(v, str):
                        out.append(v)
                out.extend(_deep_find_strings(v, keys))
        elif isinstance(obj, list):
            for it in obj:
                out.extend(_deep_find_strings(it, keys))
    except Exception:
        pass
    return out

def _seekingalpha_rich_extract(html_text: str, soup: BeautifulSoup) -> Optional[str]:
    """
    Seeking Alpha often embeds the full body in Next.js data (script#__NEXT_DATA__)
    or other inline JSON. Extract and normalize that first.
    """
    # 1) __NEXT_DATA__ (canonical for Next.js apps)
    try:
        nd = soup.find("script", id="__NEXT_DATA__")
        if nd and (nd.string or nd.text):
            data = json.loads(nd.string or nd.text)
            cand_strs = _deep_find_strings(data)
            # Prefer the longest plausible content block
            cand_strs = [s for s in cand_strs if isinstance(s, str) and len(s.strip()) >= 40]
            if cand_strs:
                best = max(cand_strs, key=lambda s: len(s))
                txt = _sa_html_to_text(best) or BeautifulSoup(best, "html.parser").get_text(" ", strip=True)
                if txt and len(txt.strip()) >= 80:
                    return txt.strip()
    except Exception:
        pass

    # 2) Any other JSON script blocks
    try:
        for sc in soup.find_all("script", type=lambda t: t and "json" in t.lower()):
            raw = sc.string or sc.text or ""
            if not raw or "{" not in raw:
                continue
            # Be lenient: some scripts include trailing semicolons/comments
            m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if not m:
                continue
            obj = json.loads(m.group(0))
            cand_strs = _deep_find_strings(obj)
            cand_strs = [s for s in cand_strs if isinstance(s, str) and len(s.strip()) >= 40]
            if cand_strs:
                best = max(cand_strs, key=lambda s: len(s))
                txt = _sa_html_to_text(best) or BeautifulSoup(best, "html.parser").get_text(" ", strip=True)
                if txt and len(txt.strip()) >= 80:
                    return txt.strip()
    except Exception:
        pass

    # 3) Nothing rich found
    return None

def _sa_extract_from_nextdata(soup: BeautifulSoup) -> Optional[str]:
    """
    Pull the biggest article/news body from Next.js __NEXT_DATA__ (SEO preloaded).
    We traverse all nested dict/list structures and collect paragraph-like strings.
    """
    nd = soup.find("script", id="__NEXT_DATA__", type="application/json") or soup.find("script", id="__NEXT_DATA__")
    if not nd or not (nd.string or nd.text):
        return None
    try:
        data = json.loads(nd.string or nd.text)
    except Exception:
        return None

    paras: List[str] = []

    # Walk arbitrary structures, collecting likely body fields.
    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                kl = str(k).lower()
                # Direct body-ish keys
                if isinstance(v, str) and any(tok in kl for tok in ["articlebody","renderedbody","body","cleanedbody","content","html","rawhtml"]):
                    txt = _sa_html_to_text(v)
                    if len(txt) >= 40:
                        paras.append(txt)
                # Blocks/paragraph arrays
                if isinstance(v, list) and any(tok in kl for tok in ["blocks","paragraphs","content","children","items"]):
                    for it in v:
                        if isinstance(it, str):
                            t = _sa_html_to_text(it)
                            if len(t) >= 40:
                                paras.append(t)
                        elif isinstance(it, dict):
                            for sk in ("text","content","value","html","rawHtml","rendered"):
                                if sk in it and isinstance(it[sk], str):
                                    t = _sa_html_to_text(it[sk])
                                    if len(t) >= 40:
                                        paras.append(t)
                            walk(it)
                if isinstance(v, (dict, list)):
                    walk(v)
        elif isinstance(o, list):
            for it in o:
                walk(it)

    walk(data)

    if not paras:
        return None
    body = "\n\n".join(paras).strip()
    return body if len(body) >= 120 else None

def _sa_extract_from_ldjson(soup: BeautifulSoup) -> Optional[str]:
    """
    Many SA pages include NewsArticle with articleBody in ld+json.
    """
    for sc in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(sc.string or "")
        except Exception:
            continue
        items = data if isinstance(data, list) else [data]
        for obj in items:
            if not isinstance(obj, dict):
                continue
            if obj.get("@type") in {"NewsArticle","Article","Report","PressRelease"}:
                body = obj.get("articleBody") or obj.get("description")
                if isinstance(body, str) and len(body.strip()) >= 200:
                    return _sa_html_to_text(body)
                if isinstance(body, list):
                    parts = []
                    for it in body:
                        if isinstance(it, str):
                            parts.append(it.strip())
                        elif isinstance(it, dict):
                            for sk in ("text","content","value","html","rawHtml"):
                                if sk in it and isinstance(it[sk], str):
                                    parts.append(_sa_html_to_text(it[sk]))
                    if parts:
                        t = "\n\n".join(parts).strip()
                        if len(t) >= 200:
                            return t
    return None

def _sa_extract_from_dom(soup: BeautifulSoup) -> Optional[str]:
    """
    Fallback to DOM selectors that often hold the fully rendered body for SA News.
    """
    node = _safe_select_one(soup, [
        "section[data-test-id='post-content']",
        "div[data-test-id='post-content']",
        "div[data-test-id='content-container']",
        "article [data-test-id='post-content']",
        "article [itemprop='articleBody']",
        "article",
        "main article",
    ])
    if not node:
        return None
    frag = _as_soup_fragment(node)
    # aggressively drop comments/engagement blocks
    for sel in ["[data-test-id='engagement']","[data-test-id='comments']",".comments",".comment-thread"]:
        for el in frag.select(sel):
            el.decompose()
    _strip_junk(frag)
    t = _join_paragraphs(frag)
    return t if len(t) >= 120 else None

def _sa_api_fetch_by_id(news_id: str, referer_url: str) -> Optional[str]:
    """
    Try Seeking Alpha's JSON endpoints that often return full text for Market Currents.
    (These are public for many pages; works best with a valid Cookie if provided.)
    """
    api_headers = {
        **PAGE_HEADERS_PRIMARY,
        "Accept": "application/json, text/plain, */*",
        "Referer": _sa_url_strip_amp(referer_url),
        "X-Requested-With": "XMLHttpRequest",
    }
    candidates = [
        f"https://seekingalpha.com/api/v3/news/{news_id}",
        f"https://seekingalpha.com/api/v3/market_currents/{news_id}",
        f"https://seekingalpha.com/api/v3/news?id={news_id}",
    ]
    for api_url in candidates:
        try:
            r = _SESSION.get(api_url, headers=api_headers, timeout=REQUEST_TIMEOUT, allow_redirects=True)
            if r.status_code != 200:
                continue
            data = r.json()
        except Exception:
            continue

        # Look for common body fields
        strings: List[str] = []
        def dig(o):
            if isinstance(o, dict):
                for k,v in o.items():
                    kl = str(k).lower()
                    if isinstance(v, str) and any(tok in kl for tok in ["articlebody","body","content","renderedbody","html","rawhtml"]):
                        s = _sa_html_to_text(v)
                        if len(s) >= 40: strings.append(s)
                    elif isinstance(v, (dict, list)):
                        dig(v)
            elif isinstance(o, list):
                for it in o:
                    dig(it)
        dig(data)

        if strings:
            body = "\n\n".join(strings).strip()
            if _sa_is_good(body):
                return body
    return None




def _sanitize_sa_text(text: str) -> str:
    if not text:
        return text
    txt = html.unescape(text)
    lines = [ln.strip() for ln in re.split(r"\r?\n", txt)]

    drop_ui = re.compile(r"^(comments?(?:\s*\(\d+\))?|most popular|trending|share|save|play)\b", re.IGNORECASE)

    kept = []
    prev = None
    for ln in lines:
        if not ln:
            continue
        if drop_ui.match(ln):
            continue
        if len(ln) < 6 and not re.search(r"[A-Za-z]\.", ln):
            continue
        if prev is not None and ln == prev:
            continue
        kept.append(ln)
        prev = ln

    out = "\n".join(kept)
    out = re.sub(r"\n{3,}", "\n\n", out)
    out = re.sub(r"[ \t\u00A0]{2,}", " ", out)
    return out.strip()


def _sa_strip_amp(u: str) -> str:
    # remove &amp params and /amp suffixes
    u = re.sub(r"([?&])amp=1(&|$)", r"\1", u).rstrip("?&")
    if u.rstrip("/").endswith("/amp"):
        u = u.rstrip("/")[:-4]
    return u

def _seekingalpha_try_variants(orig_url: str, final_url: str, chosen: Optional[str]) -> Tuple[str, Optional[str]]:
    base = _sa_url_strip_amp(final_url)
    best_url, best_text = base, (chosen or "")

    def try_one(u: str) -> Tuple[str, Optional[str]]:
        f_url, html_text = _fetch_html(u)
        if not html_text:
            return u, None
        soup = BeautifulSoup(html_text, "html.parser")

        # 1) JSON (Next.js + ld+json) first
        body = _sa_extract_from_nextdata(soup)
        if not _sa_is_good(body):
            ld = _sa_extract_from_ldjson(soup)
            if ld and (not body or len(ld) > len(body)):
                body = ld

        # 2) DOM fallback
        if not _sa_is_good(body):
            dom_body = _sa_extract_from_dom(soup)
            if dom_body and (not body or len(dom_body) > len(body)):
                body = dom_body

        # 3) API fallback if still short
        if not _sa_is_good(body):
            nid = _sa_id_from_url(u)
            if nid:
                api_body = _sa_api_fetch_by_id(nid, u)
                if _sa_is_good(api_body):
                    body = api_body

        if body:
            body = _sanitize_sa_text(body)
            body = _fix_mojibake(body)
        return f_url, body

    # Try canonical, then AMP, then /amp, then mobile
    for candidate in [
        base,
        base + (("&" if "?" in base else "?") + "amp=1"),
        base.rstrip("/") + "/amp",
        base.replace("://seekingalpha.com", "://m.seekingalpha.com"),
    ]:
        u, t = try_one(candidate)
        if _sa_is_good(t):
            return u, t
        if t and len(t) > len(best_text):
            best_url, best_text = u, t

    return best_url, (best_text or None)


def _sa_walk_for_text(obj) -> List[str]:
    """
    Recursively walk a JSON object looking for plausible article bodies.
    Returns a list of text chunks; caller will join and score.
    """
    found: List[str] = []

    def push(s: str):
        s = (s or "").strip()
        # keep only reasonably sentence-like chunks
        if len(s) >= 40 and re.search(r"[.!?]\s", s):
            found.append(s)

    if isinstance(obj, dict):
        # Common keys observed in SA __NEXT_DATA__/ld+json
        for k in list(obj.keys()):
            v = obj[k]
            kl = str(k).lower()
            # direct string bodies
            if isinstance(v, str) and any(t in kl for t in [
                "articlebody", "body", "content", "cleaned", "text"
            ]):
                push(v)
            # arrays of blocks/paragraphs
            if isinstance(v, list) and any(t in kl for t in [
                "blocks", "paragraphs", "content", "children", "items"
            ]):
                for it in v:
                    if isinstance(it, str):
                        push(it)
                    elif isinstance(it, dict):
                        # common nested forms
                        for ckey in ("text", "content", "html", "value"):
                            if ckey in it and isinstance(it[ckey], str):
                                push(it[ckey])
                        # recurse deeper
                        found.extend(_sa_walk_for_text(it))
            # generic recurse
            found.extend(_sa_walk_for_text(v))
    elif isinstance(obj, list):
        for it in obj:
            found.extend(_sa_walk_for_text(it))

    return found


def _sa_collect_paragraphs_from_json(obj) -> List[str]:
    """
    Walk SA's Next.js / ld+json structures and collect ordered paragraph-like text.
    We try common shapes seen on Seeking Alpha News pages.
    """
    out: List[str] = []

    def add(s: str):
        s = (s or "").strip()
        if not s:
            return
        # must look like prose, not UI
        if len(s) < 40 and not re.search(r"[.!?]\s", s):
            return
        out.append(s)

    def from_blocks(blocks):
        # Typical shapes: [{"type":"paragraph","text":"..."}, {"type":"html","html":"..."}]
        for b in (blocks or []):
            if isinstance(b, str):
                add(b)
            elif isinstance(b, dict):
                if "text" in b and isinstance(b["text"], str):
                    add(b["text"])
                elif "content" in b and isinstance(b["content"], str):
                    add(b["content"])
                elif "value" in b and isinstance(b["value"], str):
                    add(b["value"])
                elif "html" in b and isinstance(b["html"], str):
                    add(BeautifulSoup(b["html"], "html.parser").get_text(" ", strip=True))
                # nested children
                if "children" in b and isinstance(b["children"], list):
                    from_blocks(b["children"])

    def walk(o):
        if isinstance(o, dict):
            # Direct body keys
            for k, v in o.items():
                kl = str(k).lower()
                if isinstance(v, str) and kl in {"articlebody", "body", "cleanedbody", "content"}:
                    add(BeautifulSoup(v, "html.parser").get_text(" ", strip=True))
                if isinstance(v, list) and kl in {"blocks", "paragraphs", "content", "children", "items"}:
                    from_blocks(v)
                # Special: common SA paths
                # e.g. props.pageProps.post.content.blocks OR props.pageProps.article.body.blocks
                if isinstance(v, (dict, list)):
                    walk(v)
        elif isinstance(o, list):
            for it in o:
                walk(it)

    walk(obj)
    return out


def _seekingalpha_json_articlebody(html_text: str) -> Optional[str]:
    """
    Robust SA extractor:
      - __NEXT_DATA__ (regardless of type attr)
      - window.__APOLLO_STATE__ inline JSON
      - ld+json NewsArticle/articleBody
    Returns the longest plausible body if >= ~220 chars.
    """
    try:
        soup = BeautifulSoup(html_text, "html.parser")
        candidates: List[str] = []

        def _collect_strings_from_json(obj) -> List[str]:
            # collect likely body strings from arbitrary JSON shapes
            found: List[str] = []
            KEYS = {"articleBody","renderedBody","cleanedBody","body","content","contentHtml","html","value","text","description","summary"}
            def walk(o):
                if isinstance(o, dict):
                    for k, v in o.items():
                        kl = str(k).lower()
                        if isinstance(v, str) and any(kl == key.lower() for key in KEYS):
                            s = BeautifulSoup(v, "html.parser").get_text(" ", strip=True)
                            if s:
                                found.append(s)
                        elif isinstance(v, (dict, list)):
                            walk(v)
                elif isinstance(o, list):
                    for it in o:
                        walk(it)
            walk(obj)
            # prefer sentence-y chunks
            return [s for s in found if len(s.strip()) >= 80 and re.search(r"[.!?]\s", s)]
        
        # 1) __NEXT_DATA__ (some pages omit type attr)
        nd = soup.find("script", id="__NEXT_DATA__")
        if nd and (nd.string or nd.text):
            try:
                data = json.loads(nd.string or nd.text)
                candidates += _collect_strings_from_json(data)
            except Exception:
                pass

        # 2) window.__APOLLO_STATE__ inline payload
        #    Looks like: <script>window.__APOLLO_STATE__ = {...};</script>
        for sc in soup.find_all("script"):
            raw = sc.string or sc.text or ""
            if "__APOLLO_STATE__" in raw:
                m = re.search(r"__APOLLO_STATE__\s*=\s*(\{.*?\})\s*;", raw, flags=re.DOTALL)
                if m:
                    try:
                        apollo = json.loads(m.group(1))
                        candidates += _collect_strings_from_json(apollo)
                    except Exception:
                        continue

        # 3) ld+json NewsArticle / Article
        for sc in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(sc.string or "")
            except Exception:
                continue
            blocks = data if isinstance(data, list) else [data]
            for obj in blocks:
                if not isinstance(obj, dict):
                    continue
                if obj.get("@type") in {"NewsArticle", "Article", "Report", "PressRelease"}:
                    body = obj.get("articleBody") or obj.get("description")
                    if isinstance(body, str) and len(body.strip()) >= 120:
                        candidates.append(BeautifulSoup(body, "html.parser").get_text(" ", strip=True))
                    elif isinstance(body, list):
                        parts = []
                        for it in body:
                            if isinstance(it, str):
                                parts.append(BeautifulSoup(it, "html.parser").get_text(" ", strip=True))
                            elif isinstance(it, dict):
                                for sk in ("text","content","value","html","rawHtml"):
                                    if sk in it and isinstance(it[sk], str):
                                        parts.append(BeautifulSoup(it[sk], "html.parser").get_text(" ", strip=True))
                        if parts:
                            candidates.append("\n\n".join(parts))

        # Choose best
        best, best_len = None, 0
        for c in candidates:
            t = html.unescape(BeautifulSoup(c, "html.parser").get_text(" ", strip=True))
            if len(t) > best_len and len(t) >= 220:
                best, best_len = t, len(t)
        return best
    except Exception:
        return None




def _find_tr_press_release_link(original_html: str, base_url: str) -> Optional[str]:
    """
    On ir.thomsonreuters.com pages, find the real press-release link on www.thomsonreuters.com.
    """
    soup = BeautifulSoup(original_html, "html.parser")

    # canonical
    link = soup.find("link", rel=lambda v: v and "canonical" in v.lower())
    if link and link.get("href"):
        href = link["href"].strip()
        if "thomsonreuters.com" in href and "/press-releases/" in href:
            return href

    # og:url
    og = soup.find("meta", attrs={"property": "og:url"})
    if og and og.get("content"):
        href = og["content"].strip()
        if "thomsonreuters.com" in href and "/press-releases/" in href:
            return href

    # any anchor to thomsonreuters.com press releases
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        abs_url = urljoin(base_url, href)
        if "thomsonreuters.com" in abs_url and "/press-releases/" in abs_url:
            return abs_url

    return None


def _maybe_follow_tr_ir_bridge(final_url: str, html_text: str) -> Tuple[str, Optional[str]]:
    """
    If we are on ir.thomsonreuters.com and there's a bridge link to the main
    www.thomsonreuters.com press release, fetch that page and return (new_url, new_html).
    """
    bridge = _find_tr_press_release_link(html_text, final_url)
    if not bridge:
        return final_url, None
    new_url, new_html = _fetch_html(bridge)
    if new_html:
        return new_url, new_html
    return final_url, None


def _investing_try_variants(orig_url: str, final_url: str, chosen: Optional[str]) -> Tuple[str, Optional[str]]:
    """
    Investing.com: try a suite of AMP/mobile/print variants if body is short/empty.
    """
    def _variants(u: str) -> List[str]:
        vs: List[str] = []
        # Proper AMP params/suffix (no HTML-escaped &amp)
        vs.append(u + (("&" if "?" in u else "?") + "amp=1"))
        vs.append(u.rstrip("/") + "/amp")
        # AMP path forms
        if "/news/" in u:
            vs.append(u.replace("/news/", "/news/amp/"))
            vs.append(u.replace("/news/", "/amp/news/"))
        # mobile subdomain
        vs.append(u.replace("://www.", "://m."))
        # print-ish params
        for q in ("print=1", "view=print", "output=1"):
            vs.append(u + (("&" if "?" in u else "?") + q))
        # dedupe preserving order
        out, seen = [], set()
        for x in vs:
            if x not in seen:
                out.append(x); seen.add(x)
        return out

    text_len = len((chosen or "").strip())
    if text_len >= 220:
        return final_url, chosen

    for v in _variants(final_url):
        alt_final, alt_html = _fetch_html(v)
        if not alt_html:
            continue
        alt_soup = BeautifulSoup(alt_html, "html.parser")
        site_text = _site_specific_extract(_domain(alt_final), alt_soup)
        jsonld_text = _jsonld_extract(alt_html)
        read_text = _readability_extract(alt_html)
        heur_text = _heuristic_extract(alt_soup)
        cand = _pick_best_text([site_text, jsonld_text, read_text, heur_text], dom=_domain(alt_final))
        if cand and len(cand.strip()) >= 200:
            return alt_final, cand
    return final_url, chosen


def extract_article_fulltext(url: str) -> Dict[str, Optional[str]]:
    """
    Fetch a news article page and return { 'final_url', 'title', 'text' }.

    Fast path for ir.thomsonreuters.com:
      - single quick fetch to read the bridge link
      - follow to www.thomsonreuters.com press release
      - then extract (with a fast TR fetch if needed)

    For Seeking Alpha:
      - Prefer __NEXT_DATA__ / window.__APOLLO_STATE__ JSON bodies
      - Then site selectors / ld+json / readability / heuristic
      - Try AMP/mobile variants if initial body seems short
      - Clean up UI crumbs + mojibake
    """
    dom0 = _domain(url)

    # ---------- Fast path for IR hub pages ----------
    if dom0.endswith("ir.thomsonreuters.com"):
        # If we've bridged this IR URL before, use the cached PR URL
        cached_bridge = _BRIDGE_CACHE.get(url)
        if cached_bridge:
            final_url, html_text = _fetch_html_tr_fast(cached_bridge)
            if not html_text:
                final_url, html_text = _fetch_html(cached_bridge)
            if not html_text:
                return {"final_url": final_url, "title": None, "text": None}
        else:
            # Single quick fetch (no rotations) just to read the bridge link
            final_url, html_text = _fetch_html_once(url, PAGE_HEADERS_PRIMARY, timeout=8)
            if not html_text:
                # Try a single mobile pass before giving up
                final_url, html_text = _fetch_html_once(url, PAGE_HEADERS_MOBILE, timeout=8)
            if not html_text:
                return {"final_url": final_url, "title": None, "text": None}

            # Try to hop to the real press release on www.thomsonreuters.com
            curr_dom = _domain(final_url)
            if curr_dom.endswith("ir.thomsonreuters.com"):
                bridged_url, bridged_html = _maybe_follow_tr_ir_bridge(final_url, html_text)
                if bridged_html:
                    _BRIDGE_CACHE[url] = bridged_url  # remember for next time
                    final_url, html_text = bridged_url, bridged_html

        # If we’re on www.thomsonreuters.com but HTML is tiny, do a fast refetch
        if _domain(final_url).endswith("www.thomsonreuters.com") and (not html_text or len(html_text) < 200):
            final_url, html_text = _fetch_html_tr_fast(final_url)
            if not html_text:
                final_url, html_text = _fetch_html(final_url)
            if not html_text:
                return {"final_url": final_url, "title": None, "text": None}

    else:
        # ---------- Normal path for other domains ----------
        final_url, html_text = _fetch_html(url)
        if not html_text:
            return {"final_url": final_url, "title": None, "text": None}

    # From here on, we have the destination page HTML (IR-bridged or original)
    soup = BeautifulSoup(html_text, "html.parser")
    dom = _domain(final_url)

    # Title
    title = None
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content"):
        title = og["content"].strip()
    if not title and soup.title and soup.title.string:
        title = soup.title.string.strip()

    # ---------- Build body candidates ----------
    # SA first: pull from __NEXT_DATA__ / window.__APOLLO_STATE__ / ld+json
    sa_json_body = _seekingalpha_json_articlebody(html_text) if (
        dom.endswith("seekingalpha.com") or dom == "seekingalpha.com"
    ) else None

    # Site-specific DOM selectors
    site_text   = _site_specific_extract(dom, soup)
    # Structured ld+json
    jsonld_text = _jsonld_extract(html_text)
    # Generic readability & heuristics
    read_text   = _readability_extract(html_text)
    heur_text   = _heuristic_extract(soup)

    # SA: prefer JSON body > site selectors > json-ld > readability > heuristics
    # Others: this order is still sensible (structured -> DOM -> generic)
    candidates = [sa_json_body, site_text, jsonld_text, read_text, heur_text]
    chosen = _pick_best_text(candidates, dom=dom)

    # SA-specific cleanup (UI crumbs / duplicates / mojibake)
    if chosen and (dom.endswith("seekingalpha.com") or dom == "seekingalpha.com"):
        chosen = _sanitize_sa_text(chosen)
        chosen = _fix_mojibake(chosen)

    # ---------- Domain-specific fallbacks (AMP/Mobile/etc.) ----------
    if dom.endswith("www.investing.com"):
        final_url, chosen = _investing_try_variants(url, final_url, chosen)
    elif dom.endswith("seekingalpha.com") or dom == "seekingalpha.com":
        final_url, chosen = _seekingalpha_try_variants(url, final_url, chosen)
        if chosen:
            chosen = _sanitize_sa_text(_fix_mojibake(chosen))
    elif dom.endswith("cnbctv18.com") or dom.endswith("www.cnbctv18.com"):
        final_url, chosen = _cnbctv18_try_variants(url, final_url, chosen)

    # Tail-trim boilerplate commonly found on PR/IR pages
    if chosen:
        tail_cut = re.split(
            r"\n\n(?:View (?:the )?source version on|For further information|SOURCE:|Contacts?:)\b",
            chosen, maxsplit=1, flags=re.IGNORECASE
        )
        chosen = (tail_cut[0] or "").strip()

    # NasdaqTrader safety: run slicer once more to strip any stray nav that leaked in
    if chosen and dom.endswith("www.nasdaqtrader.com"):
        chosen = _nasdaqtrader_line_slice(chosen)

    return {"final_url": final_url, "title": title, "text": chosen}




# -------------------- RSS fetch (with full-article upgrade) --------------------
def fetch_rss_feed(
    url: str,
    name: Optional[str] = None,
    max_items: int = 10,
    platform: str = "rss",
) -> List[Dict[str, Any]]:
    try:
        res = _SESSION.get(url, headers=RSS_HEADERS, timeout=30)
        res.raise_for_status()
    except Exception:
        return []

    try:
        root = ET.fromstring(res.content)
    except ET.ParseError:
        return []

    channel_title = ""
    channel = root.find("channel")
    if channel is not None:
        channel_title = (channel.findtext("title") or "").strip()
        items = list(channel.findall("item"))
    else:
        items = list(root.findall(".//{http://www.w3.org/2005/Atom}entry"))
        channel_title = (root.findtext("{http://www.w3.org/2005/Atom}title") or "").strip()

    label = name or channel_title or url
    posts: List[Dict[str, Any]] = []

    for item in items[:max_items]:
        title = (item.findtext("title") or item.findtext("{http://www.w3.org/2005/Atom}title") or "").strip()

        link = (item.findtext("link") or item.findtext("{http://www.w3.org/2005/Atom}link") or "").strip()
        if not link:
            link_el = item.find("{http://www.w3.org/2005/Atom}link")
            if link_el is not None:
                link = link_el.attrib.get("href", "")

        guid = (
            item.findtext("guid")
            or item.findtext("id")
            or item.findtext("{http://www.w3.org/2005/Atom}id")
            or link
            or title
        )

        # Feed summary fallback
        desc_parts: List[str] = []
        for tag in [
            "description",
            "summary",
            "{http://www.w3.org/2005/Atom}summary",
            "{http://purl.org/rss/1.0/modules/content/}encoded",
        ]:
            txt = item.findtext(tag)
            if txt:
                desc_parts.append(txt)
        if not desc_parts:
            desc_parts.append(_extract_text(item.findall("description")))
        desc = " ".join(desc_parts)
        clean_desc = _clean_html(desc)
        feed_text = (title + " " + clean_desc).strip() if clean_desc else title

        final_url = link
        article_text = None

        if FETCH_FULL_ARTICLE and link and _is_fulltext_allowed(link):
            full = extract_article_fulltext(link)
            final_url = full.get("final_url") or link
            if full.get("title") and not title:
                title = full["title"] or title
            if full.get("text"):
                article_text = full["text"]

        # For bulletin-style feeds (e.g., Nasdaq Trader RSS endpoints), the feed content is already valuable.
        chosen_text = article_text or feed_text or clean_desc or title
        snippet = (chosen_text[:160] + "…") if len(chosen_text) > 160 else chosen_text

        posts.append(
            {
                "id": guid,
                "url": link or url,
                "final_url": final_url,
                "title": title,
                "text": chosen_text,
                "snippet": snippet,
                "source": label,
                "platform": platform,
                "scraped_at": _now_iso(),
            }
        )

    return posts
