#!/usr/bin/env python3
# streamlit run app.py

import io, re, json, time, math, logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple

import requests
import pandas as pd
import yfinance as yf
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from reddit_scraper import DEFAULT_SUBS, HEADERS, scrape_sub_new
from rss_parser import DEFAULT_RSS_FEEDS, fetch_rss_feed
from x_scraper import DEFAULT_X_HANDLES, scrape_x_feed
from stocktwits_scraper import scrape_stocktwits_news

RSS_DEFAULT_TEXT = "\n".join(f"{item['name']}|{item['url']}" for item in DEFAULT_RSS_FEEDS)
X_DEFAULT_TEXT = ", ".join(f"@{handle}" for handle in DEFAULT_X_HANDLES)

# ==================== Logging ====================
logging.getLogger("yfinance").setLevel(logging.ERROR)

# ==================== Config ====================
TICKER_STOP = {
    "THE","AND","FOR","WITH","LONG","SHORT","CALL","CALLS","PUT","PUTS",
    "A","TO","ON","AT","IN","OF","IS","IT","IMO","DD","BUY","SELL","UNDER","ABOUT"
}
CONF_MAP = {"low": 0.3, "medium": 0.6, "high": 0.9}

TICKER_CACHE_PATH = Path("tickers_all.csv")
TICKER_CACHE_MAX_AGE = timedelta(days=7)

# --- Trade log (append-only events) ---
TRADES_CSV = Path("trades_log.csv")
# event_type ∈ {"PLACED","CLOSE"}
TRADE_LOG_COLUMNS = [
    "event_type","trade_id","post_id","ticker","direction",
    "placed_at","signal_ts",
    "entry_time","entry_price","exit_time","exit_price","exit_reason","pnl",
    "confidence","target","subreddit","post_url","title","text_snippet","post_text","llm_json",
    "params_notional","params_tp_pct","params_sl_pct","params_horizon_min",
]

# --- Raw post archive (append-only) ---
RAW_POSTS_CSV = Path("raw_posts_log.csv")
RAW_POST_COLUMNS = [
    "scraped_at",
    "platform",
    "source",
    "post_id",
    "url",
    "title",
    "text",
]

# ==================== Time utils ====================
def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def now_iso() -> str:
    return now_utc().isoformat(timespec="seconds")

def as_utc_naive(ts) -> pd.Timestamp:
    """Return tz-naive UTC timestamp for comparisons with yfinance (we normalize indices to UTC-naive)."""
    return pd.to_datetime(ts, utc=True).tz_localize(None)

def crop_last_minutes(df: pd.DataFrame, minutes: int = 120) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    now_naive_utc = pd.to_datetime(datetime.utcnow()).tz_localize(None)
    cutoff = now_naive_utc - timedelta(minutes=int(minutes))
    return df.loc[df.index >= cutoff]

# ==================== Text utils ====================
def normalize_direction(s: Optional[str]) -> str:
    if not s: return "neutral"
    v = s.strip().lower()
    if v in ("call","calls","bull","bullish","long","buy","up"): return "long"
    if v in ("put","puts","bear","bearish","short","sell","down"): return "short"
    return "neutral"

def normalize_conf(val) -> Optional[float]:
    if val is None: return None
    if isinstance(val, (int, float)):
        f = float(val); return max(0.0, min(1.0, f if f <= 1 else f/100.0))
    if isinstance(val, str):
        v = val.strip().lower()
        if v in CONF_MAP: return CONF_MAP[v]
        m = re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*%", v)
        if m: return max(0.0, min(1.0, float(m[0]) / 100.0))
        try:
            f = float(v); return max(0.0, min(1.0, f if f <= 1 else f/100.0))
        except: return 0.5
    return None

def normalize_tickers(tickers: List[str]) -> List[str]:
    out = []
    for t in (tickers or []):
        if t is None:
            continue
        if not isinstance(t, str):
            # Some extractors may emit numbers or other objects; skip them.
            continue
        t = re.sub(r"[^A-Za-z]", "", t).upper()
        if 1 < len(t) <= 5 and t not in TICKER_STOP:
            out.append(t)
    seen, uniq = set(), []
    for t in out:
        if t not in seen:
            seen.add(t); uniq.append(t)
    return uniq


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
    handles: List[str] = []
    for chunk in re.split(r"[\n,]", text or ""):
        h = chunk.strip()
        if not h:
            continue
        if h.startswith("http"):
            continue
        h = h.lstrip("@")
        if not h:
            continue
        if h not in handles:
            handles.append(h)
    return handles

def extract_json_block(s: str) -> Optional[dict]:
    if not s: return None
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`").strip()
    try:
        return json.loads(s)
    except:
        pass
    i, j = s.find("{"), s.rfind("}")
    if 0 <= i < j:
        try:
            return json.loads(s[i:j+1])
        except:
            return None
    return None

# ==================== Whitelist (NASDAQ + NYSE) ====================
def _download_ticker_table(url: str) -> pd.DataFrame:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    text = "\n".join([ln for ln in r.text.splitlines() if "|" in ln])
    return pd.read_csv(io.StringIO(text), sep="|")

@st.cache_data(show_spinner=True)
def load_valid_tickers() -> pd.DataFrame:
    # Try disk cache (age <= 7 days)
    if TICKER_CACHE_PATH.exists():
        try:
            mtime = datetime.fromtimestamp(TICKER_CACHE_PATH.stat().st_mtime, tz=timezone.utc)
            if now_utc() - mtime <= TICKER_CACHE_MAX_AGE:
                df = pd.read_csv(TICKER_CACHE_PATH, dtype=str)
                df["Symbol"] = df["Symbol"].str.upper()
                return df
        except Exception:
            pass

    frames = []
    for u in [
        "https://nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "https://nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
    ]:
        try:
            frames.append(_download_ticker_table(u))
        except Exception:
            pass

    if frames:
        df = pd.concat(frames, ignore_index=True)
        symcol = "Symbol" if "Symbol" in df.columns else ("NASDAQ Symbol" if "NASDAQ Symbol" in df.columns else None)
        if symcol:
            df = (df[[symcol]].rename(columns={symcol: "Symbol"}))
            df["Symbol"] = df["Symbol"].astype(str).str.upper()
            df = df[df["Symbol"].str.fullmatch(r"[A-Z]{1,5}")].drop_duplicates().reset_index(drop=True)
        else:
            df = pd.DataFrame({"Symbol": ["AAPL","MSFT","NVDA","AMZN","META","TSLA"]})
    else:
        df = pd.DataFrame({"Symbol": ["AAPL","MSFT","NVDA","AMZN","META","TSLA","GOOGL","GOOG","AMD","SPY","QQQ"]})

    try:
        df.to_csv(TICKER_CACHE_PATH, index=False)
    except Exception:
        pass
    return df


# ==================== Persistence helpers (seen-posts across runs) ====================
def _load_seen_raw_keys() -> Tuple[set, set]:
    """
    Returns (seen_keys, seen_urls) from existing RAW_POSTS_CSV.
    seen_keys uses the same 'platform:post_id' compound key used in-session.
    """
    seen_keys, seen_urls = set(), set()
    if not RAW_POSTS_CSV.exists():
        return seen_keys, seen_urls
    try:
        df = pd.read_csv(RAW_POSTS_CSV, dtype=str)
        for _, r in df.fillna("").iterrows():
            platform = r.get("platform", "").strip() or "reddit"
            pid      = r.get("post_id", "").strip()
            url      = r.get("url", "").strip()
            if pid:
                seen_keys.add(f"{platform}:{pid}")
            if url:
                seen_urls.add(url)
    except Exception:
        pass
    return seen_keys, seen_urls

def _is_seen_post(platform: str, post_id: str, url: str) -> bool:
    key = f"{platform}:{post_id}" if post_id else ""
    if key and key in st.session_state.raw_logged_ids:
        return True
    if url and url in st.session_state.raw_logged_urls:
        return True
    return False

# ==================== LLM (Ollama via HTTP) ====================
def ollama_extract(text: str, model: str) -> Optional[Dict[str, Any]]:
    prompt = f"""
You are a trading signal extractor.

Return ONLY compact JSON IF AND ONLY IF the text clearly contains a **directional trading prediction**
for a **real US-listed stock** (NYSE or NASDAQ symbol). If no explicit long/short call exists, return null.

Schema:
{{
  "tickers": ["..."],
  "direction": "long|short",
  "target": float or null,
  "confidence": 0.0-1.0
}}

Rules:
- Ignore spam, ads, promotions, unrelated content.
- Ignore random capitalised words that are not tickers.
- Do not output explanations. JSON only.

TEXT:
{text}
""".strip()

    try:
        r = requests.post(
            "http://localhost:11434/api/chat",
            json={"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False},
            timeout=60,
        )
        if r.status_code != 200:
            return None
        raw = r.json().get("message", {}).get("content", "")
    except Exception:
        return None

    js = extract_json_block(raw)
    if not isinstance(js, dict):
        return None

    tickers = normalize_tickers(js.get("tickers") or [])
    direction = (js.get("direction") or "").strip().lower()
    if direction not in ("long","short"):
        return None
    target = js.get("target")
    try:
        target = float(target) if target is not None else None
    except:
        target = None
    conf = normalize_conf(js.get("confidence"))
    return {"normalized": {"tickers": tickers, "direction": direction, "target": target, "confidence": conf},
            "raw": raw}

# ==================== Regex fallback ====================
TICKER_RE = r"\b([A-Z]{2,5})\b"
DIR_RE    = r"\b(LONG|SHORT|CALLS?|PUTS?|BULL(ISH)?|BEAR(ISH)?)\b"
TARGET_RE = r"target[:\s\$]*([0-9]+(?:\.[0-9]+)?)"

def regex_extract(text: str) -> Optional[Dict[str, Any]]:
    up = text.upper()
    tickers = normalize_tickers(re.findall(TICKER_RE, up))
    if not tickers: return None
    d = re.search(DIR_RE, up)
    direction = normalize_direction(d.group(1) if d else "neutral")
    if direction not in ("long","short"): return None
    targ = re.search(TARGET_RE, text, flags=re.IGNORECASE)
    target = float(targ.group(1)) if targ else None
    conf = 0.7 if "🚀" in text or "CONFIDENCE" in up else 0.6
    return {"normalized": {"tickers": tickers, "direction": direction, "target": target, "confidence": conf},
            "raw": '{"regex": true}'}

# ==================== Price fetch (robust, cached) ====================
@st.cache_data(ttl=120, show_spinner=False)
def fetch_mini_prices(ticker: str) -> Optional[pd.DataFrame]:
    """
    Robust price fetcher:
    - Tries multiple period/interval combos in order.
    - Falls back to Ticker.history if download() yields empty.
    - Coerces to numeric Close, builds Close from Adj Close if needed.
    - Returns UTC-naive index and 'Close' column.
    """
    tries = [
        ("1d",  "1m",  "download"),
        ("2d",  "1m",  "download"),
        ("5d",  "5m",  "download"),
        ("5d",  "5m",  "history"),
        ("1mo", "1h",  "history"),
    ]

    def _clean(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df = df.xs("Close", axis=1, level=0).to_frame(name="Close")
            except Exception:
                df.columns = df.columns.get_level_values(-1)
        if "Close" not in df.columns:
            if "Adj Close" in df.columns:
                df["Close"] = df["Adj Close"]
            else:
                for cand in ("close", "adjclose", "AdjClose", "CLOSE"):
                    if cand in df.columns:
                        df["Close"] = df[cand]; break
        if "Close" not in df.columns:
            return None
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df = df.dropna(subset=["Close"])
        if df.empty:
            return None
        idx = pd.to_datetime(df.index, utc=True).tz_localize(None)
        df = df.set_index(idx)[["Close"]].sort_index()
        return df

    for period, interval, method in tries:
        try:
            if method == "download":
                df = yf.download(ticker, period=period, interval=interval,
                                 progress=False, auto_adjust=True, prepost=True)
            else:
                df = yf.Ticker(ticker).history(period=period, interval=interval,
                                               auto_adjust=True, prepost=True, actions=False)
            df = _clean(df)
            if df is not None and not df.empty:
                return df
        except Exception:
            continue
    return None

# ==================== Trade simulation (with details) ====================
def _ensure_numeric_close_series(obj) -> Optional[pd.Series]:
    """Return a numeric Series of prices from either a Series or DataFrame; None if impossible."""
    if obj is None:
        return None
    s = obj
    if isinstance(s, pd.DataFrame):
        if "Close" in s.columns:
            s = s["Close"]
        else:
            num_cols = s.select_dtypes(include="number").columns
            s = s[num_cols[0]] if len(num_cols) else s.squeeze()
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return None
    s.index = pd.to_datetime(s.index, utc=True).tz_localize(None)
    return s

def evaluate_trade_details(
    series_like,
    ts_iso: str,
    direction: str,
    notional: float,
    tp_pct: float,
    sl_pct: float,
    horizon_min: int,
) -> Optional[Dict[str, Any]]:
    """
    Returns dict with:
      entry_time, entry_price, exit_time, exit_price, exit_reason ('TP','SL','HORIZON'),
      pnl (float), complete (bool)
    or None if cannot evaluate yet.
    """
    s = _ensure_numeric_close_series(series_like)
    if s is None or s.empty:
        return None

    ts = as_utc_naive(ts_iso)
    end_ts = ts + timedelta(minutes=horizon_min)
    now_naive = as_utc_naive(datetime.utcnow())

    # window in holding period
    try:
        window = s.loc[(s.index >= ts) & (s.index <= end_ts)]
    except Exception:
        return None

    # Entry = first bar >= ts
    if window.empty:
        idx = s.index.searchsorted(ts)
        if idx >= len(s):
            return None
        entry_time = s.index[idx]
        entry_price = float(s.iloc[idx])
        if (now_naive < end_ts) and (idx + 1 >= len(s)):
            return None
        window = s.iloc[idx : idx + max(2, math.ceil(horizon_min/5))]
    else:
        entry_time = window.index[0]
        entry_price = float(window.iloc[0])

    if not (math.isfinite(entry_price) and entry_price > 0):
        return None

    # Scan for TP/SL
    trail_vals = window.iloc[1:].astype(float).values
    trail_index = window.index[1:]
    exit_price = float(window.iloc[-1])
    exit_time = window.index[-1]
    exit_reason = None
    if direction == "long":
        tp, sl = entry_price * (1 + tp_pct), entry_price * (1 - sl_pct)
        for p, t in zip(trail_vals, trail_index):
            if p >= tp:
                exit_price, exit_time, exit_reason = float(p), t, "TP"; break
            if p <= sl:
                exit_price, exit_time, exit_reason = float(p), t, "SL"; break
        if exit_reason is None:
            if now_naive >= end_ts and len(window) >= 1:
                exit_reason = "HORIZON"
    else:  # short
        tp, sl = entry_price * (1 - tp_pct), entry_price * (1 + sl_pct)
        for p, t in zip(trail_vals, trail_index):
            if p <= tp:
                exit_price, exit_time, exit_reason = float(p), t, "TP"; break
            if p >= sl:
                exit_price, exit_time, exit_reason = float(p), t, "SL"; break
        if exit_reason is None:
            if now_naive >= end_ts and len(window) >= 1:
                exit_reason = "HORIZON"

    if exit_reason is None:
        return {
            "entry_time": entry_time, "entry_price": entry_price,
            "exit_time": None, "exit_price": None, "exit_reason": None,
            "pnl": None, "complete": False
        }

    # Compute P&L
    if direction == "long":
        pnl = notional * (exit_price / entry_price - 1.0)
    else:
        pnl = notional * (entry_price / exit_price - 1.0)

    return {
        "entry_time": entry_time, "entry_price": entry_price,
        "exit_time": exit_time, "exit_price": exit_price, "exit_reason": exit_reason,
        "pnl": float(pnl), "complete": True
    }

def compute_trade_details_for_predictions(
    pred_df: pd.DataFrame,
    notional: float,
    tp_pct: float,
    sl_pct: float,
    horizon_min: int
) -> pd.DataFrame:
    """Return details for each prediction row that passes trade threshold."""
    if pred_df.empty:
        return pd.DataFrame()
    out_rows = []
    cache: Dict[str, Optional[pd.DataFrame]] = {}
    for _, r in pred_df.iterrows():
        tkr = r["ticker"]
        if tkr not in cache:
            cache[tkr] = fetch_mini_prices(tkr)
        dfp = cache[tkr]
        if dfp is None or dfp.empty:
            out_rows.append({**r, "pnl": float("nan"), "entry_time": None, "entry_price": None,
                             "exit_time": None, "exit_price": None, "exit_reason": None, "complete": False})
            continue
        det = evaluate_trade_details(dfp["Close"], r["ts"], r["direction"],
                                     notional, tp_pct, sl_pct, horizon_min)
        if det is None:
            out_rows.append({**r, "pnl": float("nan"), "entry_time": None, "entry_price": None,
                             "exit_time": None, "exit_price": None, "exit_reason": None, "complete": False})
        else:
            row = {**r,
                   "pnl": det["pnl"] if det["pnl"] is not None else float("nan"),
                   "entry_time": det["entry_time"], "entry_price": det["entry_price"],
                   "exit_time": det["exit_time"], "exit_price": det["exit_price"],
                   "exit_reason": det["exit_reason"], "complete": det["complete"]}
            out_rows.append(row)
    return pd.DataFrame(out_rows)

# ==================== Trade log persistence (append-only) ====================
def _ensure_trades_csv():
    if not TRADES_CSV.exists():
        pd.DataFrame(columns=TRADE_LOG_COLUMNS).to_csv(TRADES_CSV, index=False)

def _load_trade_event_sets() -> Tuple[set, set]:
    """
    Returns (placed_ids, closed_ids) as sets of trade_id strings.
    """
    if not TRADES_CSV.exists():
        return set(), set()
    try:
        df = pd.read_csv(TRADES_CSV, usecols=["event_type","trade_id"], dtype=str)
        df = df.fillna("")
        placed = set(df.loc[df["event_type"]=="PLACED","trade_id"].tolist())
        closed = set(df.loc[df["event_type"]=="CLOSE","trade_id"].tolist())
        return placed, closed
    except Exception:
        return set(), set()

def _append_trade_events(rows: List[Dict[str, Any]]):
    if not rows:
        return
    _ensure_trades_csv()
    new_df = pd.DataFrame(rows)

    # Load existing (may have older schema)
    try:
        exist = pd.read_csv(TRADES_CSV)
    except Exception:
        exist = pd.DataFrame()

    # Union columns: existing + current schema (preserve order; TRADE_LOG_COLUMNS preferred)
    ordered = list(dict.fromkeys(TRADE_LOG_COLUMNS + exist.columns.tolist() + new_df.columns.tolist()))

    # Ensure all columns exist in both frames
    for c in ordered:
        if c not in exist.columns: exist[c] = ""
        if c not in new_df.columns: new_df[c] = ""

    # Align & write back
    exist = exist[ordered]
    new_df = new_df[ordered]
    out = pd.concat([exist, new_df], ignore_index=True)
    out.to_csv(TRADES_CSV, index=False)


def _ensure_raw_posts_csv():
    if not RAW_POSTS_CSV.exists():
        pd.DataFrame(columns=RAW_POST_COLUMNS).to_csv(RAW_POSTS_CSV, index=False)


def _append_raw_posts(rows: List[Dict[str, Any]]):
    if not rows:
        return
    _ensure_raw_posts_csv()
    new_df = pd.DataFrame(rows)
    try:
        exist = pd.read_csv(RAW_POSTS_CSV)
    except Exception:
        exist = pd.DataFrame(columns=RAW_POST_COLUMNS)

    ordered = list(dict.fromkeys(RAW_POST_COLUMNS + exist.columns.tolist() + new_df.columns.tolist()))
    for col in ordered:
        if col not in exist.columns:
            exist[col] = ""
        if col not in new_df.columns:
            new_df[col] = ""

    exist = exist[ordered]
    new_df = new_df[ordered]
    out = pd.concat([exist, new_df], ignore_index=True)
    out.to_csv(RAW_POSTS_CSV, index=False)

# ==================== Streamlit UI ====================
st.set_page_config(page_title="Crowd Signals (Reddit + LLM)", layout="wide")

# --- session state init ---
if "log" not in st.session_state: st.session_state.log = []
if "predictions" not in st.session_state:
    st.session_state.predictions = pd.DataFrame(columns=[
        "ts","platform","source","post_id","url","title","ticker","direction",
        "confidence","target","text_snippet","text_full","raw"
    ])
if "is_running" not in st.session_state: st.session_state.is_running = False
if "current_activity" not in st.session_state: st.session_state.current_activity = ""
if "scrape_state" not in st.session_state:
    st.session_state.scrape_state = {"source_idx": 0, "posts": [], "post_idx": 0, "sources": []}

_ensure_trades_csv()
_ensure_raw_posts_csv()

# trade caches
if "placed_ids" not in st.session_state or "closed_ids" not in st.session_state:
    st.session_state.placed_ids, st.session_state.closed_ids = _load_trade_event_sets()

# *** NEW: UI text state defaults (must exist before the text_area uses them) ***
if "rss_config_text" not in st.session_state:
    st.session_state.rss_config_text = RSS_DEFAULT_TEXT
if "x_handles_text" not in st.session_state:
    st.session_state.x_handles_text = X_DEFAULT_TEXT

# *** NEW: persistent seen sets for backfill/dup-protection ***
if "raw_logged_ids" not in st.session_state or "raw_logged_urls" not in st.session_state:
    seen_keys, seen_urls = _load_seen_raw_keys()
    st.session_state.raw_logged_ids  = set(seen_keys)
    st.session_state.raw_logged_urls = set(seen_urls)

# *** NEW: backfill controller state ***
if "backfill_active" not in st.session_state:
    st.session_state.backfill_active = True
if "backfill_seen_streak" not in st.session_state:
    st.session_state.backfill_seen_streak = {}
if "backfill_batches_for_source" not in st.session_state:
    st.session_state.backfill_batches_for_source = {}

# --- sidebar controls ---
with st.sidebar:
    st.title("Crowd Signals")
    model = st.text_input("Ollama model", value="mistral", help="Model must be available in Ollama")
    subs = st.text_input("Subreddits (comma)", value=",".join(DEFAULT_SUBS))
    max_posts = st.slider("Posts per source", 5, 30, 12, 1)
    interval = st.slider("Loop interval (seconds)", 30, 900, 180, 10)
    batch_size = st.slider("Posts per refresh (batch size)", 1, 20, 4, 1)
    pause_per_post = st.slider("Pacing per post (sec)", 0.0, 3.0, 1.0, 0.1)

    st.markdown("---")
    # Ticker re-cache
    reloaded = False
    if st.button("Re-cache ticker list (NASDAQ + NYSE)"):
        load_valid_tickers.clear()
        try: TICKER_CACHE_PATH.unlink(missing_ok=True)
        except Exception: pass
        reloaded = True

    VALID_TICKERS_DF = load_valid_tickers()
    if reloaded:
        st.success(f"Re-cached {len(VALID_TICKERS_DF)} tickers")
    VALID_TICKERS = set(VALID_TICKERS_DF["Symbol"].tolist())

    st.markdown("---")
    st.subheader("Confidence thresholds")
    min_conf_extract = st.slider("Min confidence to record", 0.0, 1.0, 0.6, 0.05)
    min_conf_trade   = st.slider("Min confidence to trade", 0.0, 1.0, 0.7, 0.05)

    st.markdown("---")
    st.subheader("Trade Simulator")
    notional = st.number_input("Notional per signal ($)", min_value=1.0, value=10.0, step=1.0)
    tp_pct = st.slider("Take profit (%)", 0.1, 10.0, 2.0, 0.1) / 100.0
    sl_pct = st.slider("Stop loss (%)", 0.1, 10.0, 1.5, 0.1) / 100.0
    horizon_min = st.slider("Holding window (minutes)", 5, 480, 60, 5)
    chart_minutes = st.slider("Chart window (minutes)", 30, 360, 120, 15)

    st.markdown("---")
    show_all = st.checkbox("Show all whitelist tickers (incl. no predictions)", value=True)
    max_rows = st.slider("Max rows to render", 20, 500, 120, 10)

    st.markdown("---")
    def _toggle_run():
        st.session_state.is_running = not st.session_state.is_running
    st.button("STOP" if st.session_state.is_running else "RUN", on_click=_toggle_run, type="primary")
    st.caption("RUN = continuous; the app scrapes in small batches on each refresh. Increase interval if Reddit returns 429.")

    st.markdown("---")
    st.subheader("Backfill / Crash Recovery")
    backfill_toggle = st.checkbox("Backfill on startup / resume", value=st.session_state.backfill_active,
                                  help="Keep fetching each source repeatedly until we hit a previously-seen item.")
    st.session_state.backfill_active = backfill_toggle
    max_backfill_batches = st.slider("Max backfill batches per source", 1, 40, 10, 1,
                                     help="Safety cap so one source doesn't loop forever if scraper can't page.")

subs_list_display = [s.strip() for s in subs.split(",") if s.strip()]
with st.container():
    st.subheader("Sources being monitored")
    st.markdown("**Reddit feeds**")
    if subs_list_display:
        st.markdown(", ".join(f"`r/{sub}`" for sub in subs_list_display))
    else:
        st.info("No Reddit feeds configured.")

    st.markdown("**RSS / News feeds**")
    st.caption("Format each line as Name|https://feed.url. Blank lines are ignored.")
    rss_text = st.text_area(
        "RSS feeds (Name|URL per line)",
        value=st.session_state.get("rss_config_text", RSS_DEFAULT_TEXT),
        key="rss_config_text_input",
    )
    if rss_text != st.session_state.rss_config_text:
        st.session_state.rss_config_text = rss_text

    st.markdown("**X / Twitter handles**")
    st.caption("Enter @handles separated by commas or new lines. Public data fetched via Nitter RSS.")
    x_text = st.text_area(
        "X handles",
        value=st.session_state.get("x_handles_text", X_DEFAULT_TEXT),
        key="x_handles_text_input",
    )
    if x_text != st.session_state.x_handles_text:
        st.session_state.x_handles_text = x_text

    st.markdown("**Stocktwits**")
    st.caption("Stocktwits News articles feed is enabled by default.")
    st.markdown("`Stocktwits News`")

# --- helpers ---
def log(msg: str):
    ts = now_iso()
    st.session_state.log.append(f"[{ts}] {msg}")
    st.session_state.log = st.session_state.log[-600:]

def filter_valid_signal(sig: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not sig or "normalized" not in sig:
        return None
    js = sig["normalized"]
    tickers = [t for t in (js.get("tickers") or []) if t in VALID_TICKERS]
    if not tickers:
        return None
    if js.get("direction") not in ("long","short"):
        return None
    conf = js.get("confidence")
    if conf is None or conf < min_conf_extract:
        return None
    js["tickers"] = tickers
    return {"normalized": js, "raw": sig.get("raw", "")}

# --- one step per refresh (non-blocking) ---
def run_step():
    subs_list = [s.strip() for s in subs.split(",") if s.strip()]
    rss_text = st.session_state.get("rss_config_text", RSS_DEFAULT_TEXT)
    x_text   = st.session_state.get("x_handles_text", X_DEFAULT_TEXT)
    rss_list_raw = parse_rss_config(rss_text)

    # Dedup RSS lines by URL
    seen_urls_local = set()
    rss_list: List[Dict[str, str]] = []
    for feed in rss_list_raw:
        url = feed.get("url")
        if url and url in seen_urls_local:
            continue
        if url:
            seen_urls_local.add(url)
        rss_list.append(feed)
    x_list = parse_x_handles(x_text)

    sources = []
    for sub in subs_list:
        sources.append({"platform": "reddit", "identifier": sub, "label": f"r/{sub}"})
    for feed in rss_list:
        sources.append({"platform": "rss", "identifier": feed["url"], "label": feed["name"]})
    for handle in x_list:
        sources.append({"platform": "x", "identifier": handle, "label": f"@{handle}"})
    sources.append({"platform": "stocktwits", "identifier": "news-articles", "label": "Stocktwits News"})

    if sources:
        log(
            "Sources: "
            f"reddit={sum(s['platform']=='reddit' for s in sources)}, "
            f"rss={sum(s['platform']=='rss' for s in sources)}, "
            f"x={sum(s['platform']=='x' for s in sources)}, "
            f"stocktwits={sum(s['platform']=='stocktwits' for s in sources)}"
        )
    if not sources:
        return

    ss = st.session_state.scrape_state
    ss["sources"] = sources
    if ss["source_idx"] >= len(sources):
        ss["source_idx"] = 0

    # If no posts cached for current source, fetch them once
    if not ss["posts"]:
        source = sources[ss["source_idx"]]
        st.session_state.current_activity = f"Scraping {source['label']} ({source['platform']}) …"
        log(st.session_state.current_activity)
        try:
            if source["platform"] == "reddit":
                ss["posts"] = scrape_sub_new(source["identifier"], max_posts=max_posts)
            elif source["platform"] == "rss":
                ss["posts"] = fetch_rss_feed(
                    source["identifier"], name=source["label"], max_items=max_posts, platform="rss"
                )
            elif source["platform"] == "x":
                ss["posts"] = scrape_x_feed(source["identifier"], max_posts=max_posts)
            elif source["platform"] == "stocktwits":
                ss["posts"] = scrape_stocktwits_news(max_items=max_posts)
            else:
                ss["posts"] = []
            ss["post_idx"] = 0

            log(f"Fetched {len(ss['posts'])} items from {source['label']} ({source['platform']})")

            for p in ss["posts"]:
                p.setdefault("platform", source["platform"])
                if not p.get("source"):
                    p["source"] = source["label"]
        except RuntimeError as e:
            if source["platform"] == "reddit":
                log(f"  {source['label']}: {e}. Backing off 60s.")
                time.sleep(60)
            else:
                log(f"  {source['label']}: {e}")
            ss["source_idx"] = (ss["source_idx"] + 1) % len(sources)
            ss["posts"] = []
            ss["post_idx"] = 0
            return
        except Exception as e:
            log(f"  {source['label']}: error {e}")
            ss["source_idx"] = (ss["source_idx"] + 1) % len(sources)
            ss["posts"] = []
            ss["post_idx"] = 0
            return
        if not ss["posts"]:
            ss["source_idx"] = (ss["source_idx"] + 1) % len(sources)
            ss["posts"] = []
            ss["post_idx"] = 0
            return

    # *** NEW: counters for this batch ***
    source = ss["sources"][ss["source_idx"]]
    new_items_this_batch = 0
    processed = 0

    while processed < batch_size and ss["post_idx"] < len(ss["posts"]):
        post = ss["posts"][ss["post_idx"]]
        ss["post_idx"] += 1
        processed += 1

        platform     = post.get("platform", "reddit")
        source_label = post.get("source") or post.get("subreddit") or ""
        text_full    = post.get("text") or ""
        snippet      = post.get("snippet") or ((text_full[:160] + "…") if len(text_full) > 160 else text_full)
        st.session_state.current_activity = (
            f"Analyzing {source_label or platform} ({platform}) ({ss['post_idx']}/{len(ss['posts'])})"
        )
        log(f"  Analyzing [{platform}] {snippet}")

        post_identifier = post.get("id") or post.get("url") or post.get("title")
        post_id = str(post_identifier) if post_identifier else f"{platform}-{int(time.time()*1000)}"
        url = str(post.get("url") or "")

        scraped_ts = post.get("scraped_at") or now_iso()

        # *** NEW: backfill stop condition uses persistent SEEN sets ***
        already_seen = _is_seen_post(platform, post_id, url)

        # Append to raw log IFF not seen before (persist+session)
        if not already_seen:
            _append_raw_posts(
                [{
                    "scraped_at": scraped_ts,
                    "platform": platform,
                    "source": source_label or platform,
                    "post_id": post_id,
                    "url": url,
                    "title": str(post.get("title") or ""),
                    "text": text_full,
                }]
            )
            # update in-memory seen sets
            st.session_state.raw_logged_ids.add(f"{platform}:{post_id}")
            if url: st.session_state.raw_logged_urls.add(url)
            new_items_this_batch += 1

        # Send FULL OP TEXT to the LLM (we can still analyze even if it was seen before)
        sig = ollama_extract(text_full, model=model) or regex_extract(text_full)
        sig = filter_valid_signal(sig)
        if not sig:
            if pause_per_post > 0: time.sleep(min(pause_per_post, 0.25))
            continue

        js, raw = sig["normalized"], sig.get("raw", "")
        for t in js["tickers"]:
            pred_row = {
                "ts": post.get("scraped_at") or scraped_ts,
                "platform": platform,
                "source": source_label or platform,
                "post_id": post_id,
                "url": url,
                "title": post.get("title"),
                "ticker": t,
                "direction": js.get("direction"),
                "confidence": js.get("confidence"),
                "target": js.get("target"),
                "text_snippet": snippet,
                "text_full": text_full,
                "raw": raw,
            }
            st.session_state.predictions = (
                pd.concat([st.session_state.predictions, pd.DataFrame([pred_row])], ignore_index=True)
                .drop_duplicates(subset=["platform","post_id","ticker","direction"], keep="last")
            )

            trade_id = f"{platform}-{post_id}-{t}-{js.get('direction')}"
            try:
                conf_val = float(js.get("confidence") or 0.0)
            except Exception:
                conf_val = 0.0
            if conf_val >= float(min_conf_trade) and trade_id not in st.session_state.placed_ids:
                placed_event = {
                    "event_type": "PLACED",
                    "trade_id": trade_id,
                    "post_id": post_id,
                    "ticker": t,
                    "direction": js.get("direction"),
                    "placed_at": now_iso(),
                    "signal_ts": post.get("scraped_at") or scraped_ts,
                    "entry_time": "", "entry_price": "",
                    "exit_time": "", "exit_price": "", "exit_reason": "", "pnl": "",
                    "confidence": conf_val,
                    "target": js.get("target"),
                    "subreddit": source_label or platform,
                    "post_url": url,
                    "title": post.get("title"),
                    "text_snippet": snippet,
                    "post_text": text_full,
                    "llm_json": raw if isinstance(raw, str) else json.dumps(raw),
                    "params_notional": float(notional),
                    "params_tp_pct": float(tp_pct),
                    "params_sl_pct": float(sl_pct),
                    "params_horizon_min": int(horizon_min),
                }
                _append_trade_events([placed_event])
                st.session_state.placed_ids.add(trade_id)
                log(f"[trade] PLACED {t} {js.get('direction')} conf={conf_val:.2f}")

        if pause_per_post > 0:
            time.sleep(min(pause_per_post, 0.25))

    # *** NEW: backfill loop control ***
    src_key = f"{source['platform']}:{source['identifier']}"
    if st.session_state.backfill_active:
        # init counters
        st.session_state.backfill_seen_streak.setdefault(src_key, 0)
        st.session_state.backfill_batches_for_source.setdefault(src_key, 0)
        st.session_state.backfill_batches_for_source[src_key] += 1

        if new_items_this_batch == 0:
            # we hit only seen items in this batch — advance source
            st.session_state.backfill_seen_streak[src_key] += 1
            log(f"[backfill] {source['label']}: no new items; streak={st.session_state.backfill_seen_streak[src_key]}")
            ss["source_idx"] = (ss["source_idx"] + 1) % len(sources)
            ss["posts"] = []
            ss["post_idx"] = 0
        else:
            # got new items — immediately fetch the same source again next refresh
            log(f"[backfill] {source['label']}: +{new_items_this_batch} new items; continuing…")
            ss["posts"] = []
            ss["post_idx"] = 0

        # safety cap per source
        if st.session_state.backfill_batches_for_source[src_key] >= max_backfill_batches:
            log(f"[backfill] {source['label']}: reached cap ({max_backfill_batches} batches); moving on.")
            ss["source_idx"] = (ss["source_idx"] + 1) % len(sources)
            ss["posts"] = []
            ss["post_idx"] = 0
    else:
        # normal round-robin progression
        if ss["post_idx"] >= len(ss["posts"]):
            ss["source_idx"] = (ss["source_idx"] + 1) % len(sources)
            ss["posts"] = []
            ss["post_idx"] = 0

# --- non-blocking scheduler ---
if st.session_state.is_running:
    st_autorefresh(interval=int(interval * 1000), key="auto")
    run_step()

# ==================== Compute trade details & log CLOSE events ====================
pred_df_all = st.session_state.predictions.copy()
if not pred_df_all.empty:
    ts_parsed = pd.to_datetime(pred_df_all["ts"], utc=True)
    cutoff = now_utc() - timedelta(days=5)  # limited by intraday data availability
    mask_recent = ts_parsed >= cutoff
    mask_trade = pred_df_all["confidence"].astype(float) >= float(min_conf_trade)
    eval_df = pred_df_all.loc[mask_recent & mask_trade].reset_index(drop=True)

    details_df = compute_trade_details_for_predictions(eval_df, notional, tp_pct, sl_pct, horizon_min)
    # Persist CLOSE events for completed trades (once)
    close_events = []
    for _, r in details_df.iterrows():
        trade_id = f"{r.get('platform', 'reddit')}-{r['post_id']}-{r['ticker']}-{r['direction']}"
        if bool(r.get("complete")) and trade_id not in st.session_state.closed_ids:
            close_events.append({
                "event_type": "CLOSE",
                "trade_id": trade_id,
                "post_id": r["post_id"],
                "ticker": r["ticker"],
                "direction": r["direction"],
                "placed_at": "",
                "signal_ts": r["ts"],
                "entry_time": r["entry_time"],
                "entry_price": r["entry_price"],
                "exit_time": r["exit_time"],
                "exit_price": r["exit_price"],
                "exit_reason": r["exit_reason"],
                "pnl": r["pnl"],
                "confidence": r["confidence"],
                "target": r["target"],
                "subreddit": r["source"],
                "post_url": r["url"],
                "title": r["title"],
                "text_snippet": r["text_snippet"],
                "post_text": (r["text_full"] if "text_full" in r and isinstance(r["text_full"], str)
                              else r["text_snippet"]),
                "llm_json": r["raw"] if isinstance(r["raw"], str) else json.dumps(r["raw"]),
                "params_notional": float(notional),
                "params_tp_pct": float(tp_pct),
                "params_sl_pct": float(sl_pct),
                "params_horizon_min": int(horizon_min),
            })
            st.session_state.closed_ids.add(trade_id)
            try:
                pnl_disp = f"{float(r['pnl']):+.2f}" if pd.notna(r["pnl"]) else "nan"
            except Exception:
                pnl_disp = "nan"
            log(f"[trade] CLOSE {r['ticker']} {r['direction']} pnl=${pnl_disp} ({r['exit_reason']})")

    if close_events:
        _append_trade_events(close_events)
else:
    details_df = pd.DataFrame()

# ==================== Header: status + Total P/L ====================
with st.container():
    cur = st.session_state.current_activity or "Idle"

    # make sure mode_str is defined before we print it
    mode_str = "Backfill" if st.session_state.get("backfill_active", False) else "Normal"

    st.markdown(f"**Status:** {cur} · **Mode:** {mode_str}")

    total_pnl = 0.0
    if not details_df.empty:
        total_pnl = float(pd.to_numeric(details_df["pnl"], errors="coerce").fillna(0).sum())
    pnl_color = "#16a34a" if total_pnl >= 0 else "#dc2626"
    st.markdown(
        f"<div style='padding:10px 14px;border-radius:10px;border:1px solid #e5e7eb;"
        f"background:#f8fafc; font-size:18px;'><b>Total P/L:</b> "
        f"<span style='color:{pnl_color};font-weight:700;'>${total_pnl:,.2f}</span>"
        f" &nbsp; (notional ${float(notional):.0f} each, TP={tp_pct*100:.1f}%, SL={sl_pct*100:.1f}%, "
        f"Horizon={int(horizon_min)}m, Trade conf ≥ {float(min_conf_trade):.2f})</div>",
        unsafe_allow_html=True
    )

# ==================== Layout ====================
left, right = st.columns([1,1])

with left:
    st.subheader("Scrape log & LLM output")
    st.text_area(
        "Activity log",
        value="\n".join(reversed(st.session_state.log)),
        height=280,
    )

    latest = st.session_state.predictions.sort_values("ts", ascending=False).head(20)
    if not latest.empty:
        st.markdown("**Latest extracted predictions**")
        for _, r in latest.iterrows():
            st.markdown(
                f"**[{r['ticker']}] {r['direction'].upper()}** · conf={float(r['confidence']):.2f}"
                f"{' · target='+str(r['target']) if pd.notna(r['target']) else ''}"
            )
            st.caption(f"{r['ts']} · r/{r['source']} · [link]({r['url']})")

            with st.expander("Full post text"):
                st.write(r.get("text_full") or r.get("text_snippet") or "")

            with st.expander("LLM raw JSON"):
                st.code(r["raw"] if isinstance(r["raw"], str) else json.dumps(r["raw"], indent=2), language="json")
    else:
        st.info("No predictions yet this session.")

with right:
    st.subheader("Prediction Board")

    counts = (st.session_state.predictions.groupby("ticker")
              .size()
              .rename("pred_count")
              .reset_index()) if not st.session_state.predictions.empty else pd.DataFrame(columns=["ticker","pred_count"])

    VALID_TICKERS_DF = load_valid_tickers()
    universe = VALID_TICKERS_DF["Symbol"].tolist()

    board = pd.DataFrame({"ticker": universe}).merge(counts, on="ticker", how="left")
    board["pred_count"] = pd.to_numeric(board["pred_count"], errors="coerce").fillna(0).astype("int64")
    board = board.sort_values(by=["pred_count","ticker"], ascending=[False, True]).reset_index(drop=True)
    if not st.sidebar.checkbox("Show zero-prediction rows", value=True, key="__show_all_dup__"):
        board = board[board["pred_count"] > 0]
    board = board.head(int(max_rows))

    if board.empty:
        st.info("Nothing to show yet — waiting for predictions.")
    else:
        latest_by_t = (st.session_state.predictions.sort_values("ts")
                       .groupby("ticker", as_index=False).tail(1).set_index("ticker")
                       if not st.session_state.predictions.empty else pd.DataFrame())

        # latest trade details per ticker (only for trades that pass trading conf)
        latest_details_by_t = (details_df.sort_values("ts")
                               .groupby("ticker", as_index=False).tail(1).set_index("ticker")
                               if not details_df.empty else pd.DataFrame())

        for _, row in board.iterrows():
            t = row["ticker"]
            pred_count = int(row["pred_count"])
            latest_row = latest_by_t.loc[t] if (not latest_by_t.empty and t in latest_by_t.index) else None

            c1, c2 = st.columns([1,1])

            if latest_row is not None:
                direction = latest_row["direction"]
                icon = "📈" if direction == "long" else ("📉" if direction == "short" else "➖")
                conf = float(latest_row["confidence"])
                src = latest_row["source"]; ts_str = latest_row["ts"]
                target_str = f" · target: {latest_row['target']}" if pd.notna(latest_row["target"]) else ""
                c1.markdown(
                    f"<div style='padding:8px 10px;border:1px solid #e5e7eb;border-radius:10px;'>"
                    f"<div style='font-weight:700;font-size:18px'>{t} {icon}"
                    f" <span style='font-size:12px;color:#6b7280'>({pred_count} preds)</span>"
                    f"</div>"
                    f"<div style='font-size:12px;color:#6b7280'>{ts_str} · r/{src}</div>"
                    f"<div style='font-size:12px'>direction: <b>{direction}</b> · confidence: <b>{conf:.2f}</b>{target_str}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            else:
                c1.markdown(
                    f"<div style='padding:8px 10px;border:1px solid #e5e7eb;border-radius:10px;'>"
                    f"<div style='font-weight:700;font-size:18px'>{t} "
                    f"<span style='font-size:12px;color:#6b7280'>({pred_count} preds)</span></div>"
                    f"<div style='font-size:12px;color:#6b7280'>No predictions yet</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            # RIGHT: mini chart + P/L badge
            pnl_val = None
            dfp = fetch_mini_prices(t) if pred_count > 0 else None
            if dfp is not None:
                df_2h = crop_last_minutes(dfp, minutes=int(chart_minutes))
                if df_2h is not None and not df_2h.empty:
                    c2.line_chart(df_2h["Close"], height=140, use_container_width=True)
                else:
                    c2.caption("No recent bars (market closed?)")
            else:
                c2.caption("No price data")

            if not latest_details_by_t.empty and t in latest_details_by_t.index:
                val = latest_details_by_t.loc[t]["pnl"]
                try:
                    pnl_val = float(val) if pd.notna(val) else None
                except Exception:
                    pnl_val = None

            if pnl_val is not None:
                col = "#16a34a" if pnl_val >= 0 else "#dc2626"
                sign = "+" if pnl_val >= 0 else ""
                c2.markdown(
                    f"<div style='margin-top:4px;font-weight:700;color:{col}'>P/L: {sign}${pnl_val:,.2f}</div>",
                    unsafe_allow_html=True
                )
            elif pred_count > 0 and latest_row is not None:
                c2.caption("P/L: waiting…")

st.caption(
    "Trades are persisted to `trades_log.csv` as append-only events (PLACED, CLOSE). "
    "We scrape each post's permalink to capture the full OP text (shown in 'Full post text' and passed to the LLM)."
)
