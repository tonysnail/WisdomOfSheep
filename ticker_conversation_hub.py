#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ticker_conversation_hub.py — Per-ticker rolling conversations with Ollama.

Features
- Importable API: ConversationHub (ingest_article, ask, ask_as_of)
- Storage backends: JsonlStore (many files) or SQLiteStore (single DB file)
- CLI: ingest (scan DB), ask, ask-as-of (time-travel), rollup
- Incremental: per-ticker last_ts gate
- Verbose logging: logs every per-ticker update (ADD/SKIP) and progress
- Time-travel safety: ask-as-of uses records ≤ cutoff; rollups carry covered range

Examples
    # CLI
    python ticker_conversation_hub.py ingest --db council/wisdom_of_sheep.sql --store sqlite --convos convos/conversations.sqlite --model mistral --verbose
    python ticker_conversation_hub.py ask --ticker HOOD --store sqlite --convos convos/conversations.sqlite --model mistral --q "Main risks now?"
    python ticker_conversation_hub.py ask-as-of --ticker HOOD --as-of 2025-09-30T23:59:59+00:00 --q "Tone around options revenue then?" --store sqlite --convos convos/conversations.sqlite --model mistral
    python ticker_conversation_hub.py rollup --ticker HOOD --keep-latest 300 --store sqlite --convos convos/conversations.sqlite --model mistral

    # Importable
    from ticker_conversation_hub import ConversationHub, SQLiteStore
    hub = ConversationHub(SQLiteStore("convos/conversations.sqlite"), model="mistral")
    hub.ingest_article(
        tickers=["HOOD","AAPL"],
        title="Robinhood adds multi-leg options",
        bullets=["Risk checks, staged rollout"],
        url="https://example.com",
        ts="2025-10-13T10:22:00+00:00",
        source="rss:nasdaq",
        verbose=True
    )
"""

from __future__ import annotations
import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
import threading
from math import exp, log1p
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

# -------------------- General utils --------------------
ROOT = Path(__file__).resolve().parent
DEFAULT_DB = ROOT / "council" / "wisdom_of_sheep.sql"
DEFAULT_CONVOS_DIR = ROOT / "convos"
DEFAULT_SQLITE_PATH = DEFAULT_CONVOS_DIR / "conversations.sqlite"
DEFAULT_MODEL = "mistral"
OLLAMA_API_URL = "http://localhost:11434/api/chat"
try:
    OLLAMA_THREADS = max(1, int(os.getenv("WOS_OLLAMA_THREADS", "4")))
except (TypeError, ValueError):
    OLLAMA_THREADS = 4

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def _clip(s: str, n: int) -> str:
    return " ".join((s or "").split())[:n]

# ISO8601 strings compare lexicographically when fully specified with timezone.
def _iso_newer(a: str, b: str) -> bool:
    return (a or "") > (b or "")

import math
from datetime import datetime, timezone, timedelta

def _dir_to_num(d: str) -> float:
    d = (d or "").lower()
    if d == "up": return 1.0
    if d == "down": return -1.0
    return 0.0  # neutral/uncertain fallback

def _impact_w(impact: str) -> float:
    m = (impact or "").lower()
    return 1.4 if m == "high" else (1.0 if m == "med" else 0.6)

def _iso_to_dt(s: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(s.replace("Z","+00:00"))
    except Exception:
        return None

def _classify_channel(source: str, url: str) -> str:
    s = (source or "").lower()
    u = (url or "").lower()
    social_domains = ["reddit.com", "x.com", "twitter.com", "stocktwits.com", "t.me", "discord.gg"]
    news_clues = ["nasdaq.com", "seekingalpha.com", "bloomberg.com", "reuters.com", "ft.com", "cnbc.com", "wsj.com"]
    if any(d in u for d in social_domains) or any(k in s for k in ["reddit", "stocktwits", "x", "twitter"]):
        return "social"
    if any(d in u for d in news_clues) or any(k in s for k in ["nasdaq", "reuters", "bloomberg", "news"]):
        return "news"
    return "other"

def _collect_ticker_deltas(store, ticker: str, as_of_iso: str, lookback_days: int, limit: int = 2000) -> list[dict]:
    """Pre-cutoff deltas for ticker within lookback window; requires ticker in 'who'."""
    recs = store.records_before(ticker, as_of_iso, limit=limit)
    asof = _iso_to_dt(as_of_iso) or datetime.now(timezone.utc)
    out = []
    for r in recs:
        if r.get("type") != "delta": 
            continue
        d = r["data"]
        ts = _iso_to_dt(d.get("t",""))
        if not ts: 
            continue
        age = (asof - ts).total_seconds() / 86400.0
        if age < 0 or age > float(lookback_days):
            continue
        who = [w.upper() for w in (d.get("who") or [])]
        if ticker.upper() not in who:
            continue
        out.append(d)
    # keep chronological, then cap to most recent
    out.sort(key=lambda x: x.get("t",""))
    return out[-limit:]

def _score_from_deltas(deltas: list[dict], as_of_iso: str, lambda_decay: float = 0.12) -> tuple[float, float, float, float]:
    """Return (des_raw, variance, sum_w, sum_w2)."""
    asof = _iso_to_dt(as_of_iso) or datetime.now(timezone.utc)
    sum_w = sum_ws = sum_ws2 = 0.0
    for d in deltas:
        ts = _iso_to_dt(d.get("t","")); 
        if not ts: 
            continue
        age_days = max(0.0, (asof - ts).total_seconds()/86400.0)
        w_rec = math.exp(-lambda_decay * age_days)
        w_imp = _impact_w(d.get("impact",""))
        w = w_rec * w_imp
        s = _dir_to_num(d.get("dir",""))
        sum_w += w
        sum_ws += w * s
        sum_ws2 += w * (s ** 2)
    if sum_w <= 0:
        return 0.0, 0.0, 0.0, 0.0
    mean = sum_ws / sum_w
    var = max(0.0, (sum_ws2 / sum_w) - (mean ** 2))
    return mean, var, sum_w, sum_ws  # return sum_ws as proxy; caller can compute sum_w2 if needed

def _confidence(sum_w: float, weights: list[float], var: float) -> float:
    if sum_w <= 0 or not weights:
        return 0.0
    sum_w2 = sum(w*w for w in weights) or 1e-9
    neff = (sum_w**2) / sum_w2
    stdev = math.sqrt(max(var, 0.0))
    c_cov = math.tanh(neff / 4.0)
    c_agree = 1.0 / (1.0 + 1.8 * stdev)
    return max(0.0, min(1.0, c_cov * c_agree))


# -------------------- Ollama chat plumbing --------------------
def _try_ollama_http(messages: List[Dict[str, str]], model: str, timeout_s: float = 30.0) -> Optional[str]:
    try:
        import requests  # lazy import; ok if missing
    except Exception:
        return None
    try:
        r = requests.post(
            OLLAMA_API_URL,
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {"num_thread": OLLAMA_THREADS},
            },
            timeout=timeout_s,
        )
        r.raise_for_status()
        js = r.json()
        msg = js.get("message") or {}
        return (msg.get("content") or "").strip()
    except Exception:
        return None

def _ollama_cli_singleturn(prompt: str, model: str, timeout_s: float = 30.0) -> str:
    cp = subprocess.run(["ollama", "run", model, prompt],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        timeout=timeout_s, check=False)
    return (cp.stdout.decode("utf-8", errors="ignore") or "").strip()

def _chat(messages: List[Dict[str, str]], model: str, timeout_s: float = 30.0) -> str:
    # Prefer HTTP multi-turn; fallback to merged single-turn CLI.
    out = _try_ollama_http(messages, model, timeout_s)
    if out is not None:
        return out
    merged = "\n\n".join(f"{m.get('role','user').upper()}:\n{m.get('content','')}" for m in messages)
    return _ollama_cli_singleturn(merged, model, timeout_s)

def _count_social_burst(store, ticker: str, as_of_iso: str, hours: int = 6) -> int:
    recs = store.records_before(ticker, as_of_iso, limit=4000)
    asof = _iso_to_dt(as_of_iso)
    if not asof: return 0
    cutoff = asof - timedelta(hours=hours)
    n = 0
    for r in recs[::-1]:  # newest first
        if r.get("type") != "delta": continue
        d = r["data"]
        if d.get("chan") != "social": continue
        ts = _iso_to_dt(d.get("t",""))
        if not ts or ts < cutoff: break
        if ticker.upper() in [w.upper() for w in (d.get("who") or [])]:
            n += 1
    return n

def _social_burst_multiplier(burst_count: int) -> float:
    # Smooth, capped boost: 0 -> 1.0 ; 3 -> ~1.25 ; 10 -> ~1.5
    return min(1.5, 1.0 + 0.2 * log1p(max(0, burst_count)))

# -------------------- Compression prompt --------------------
COMPRESS_SYS = (
  "You compress market news into a tiny, loss-aware delta for a stock's rolling memory.\n"
  "Return STRICT JSON only with fields: "
  '{"t":"ISO8601","src":"short source","who":["TICKERS"],'
  '"cat":["reg|product|earnings|macro|legal|ops|risk|m&a|comp"],'
  '"sum":"2–3 compact sentences: (1) what happened; (2) why it matters to the PRIMARY ticker; (3, optional) near-term setup/constraint.",'
  '"dir":"up|down|neutral|uncertain","impact":"low|med|high","why":"<=12 words"}\n'
  "Categorization: if content implies compliance/litigation/outage/regulatory scrutiny/liquidity or pricing pressure/churn/data-security/share-loss, "
  "INCLUDE 'risk' alongside other tags.\n"
  "Assume the first element of WHO is the PRIMARY ticker; write the summary from that ticker's perspective.\n"
)

def _compress_delta(model: str, when_iso: str, source: str, who: List[str], title: str, bullets: List[str], url: str) -> Optional[dict]:
    bullets = [b for b in bullets if isinstance(b, str)]
    B = bullets[:4] if bullets else ([title] if title else [])
    if not B:
        return None
    user = f"t={when_iso} src={_clip(source,32)} url={url}\nWHO={','.join(who[:6])}\nBULLETS:\n" + "\n".join(f"- {_clip(b,180)}" for b in B) + "\nJSON only."
    txt = _chat([{"role":"system","content":COMPRESS_SYS},{"role":"user","content":user}], model, timeout_s=25.0).strip()
    try:
        s = txt.find("{"); e = txt.rfind("}")
        if s != -1 and e != -1 and e > s:
            js = json.loads(txt[s:e+1])
            js["t"] = js.get("t") or when_iso
            js["src"] = js.get("src") or _clip(source, 24)
            js["who"] = js.get("who") or who[:6]
            js["sum"] = _clip(js.get("sum",""), 220)
            js["url"] = url
            return js
    except Exception:
        pass
    return None

def compute_ticker_signal(store, ticker: str, as_of_iso: str, lookback_days: int = 7,
                          peers: list[str] | None = None,
                          channel_filter: str = "all",
                          burst_hours: int = 6) -> dict:
    deltas = _collect_ticker_deltas(store, ticker, as_of_iso, lookback_days, limit=5000)

    # optional channel filtering
    if channel_filter in ("news", "social"):
        deltas = [d for d in deltas if d.get("chan") == channel_filter]

    # burst for social chatter
    burst_mult = 1.0
    if channel_filter in ("all", "social"):
        bc = _count_social_burst(store, ticker, as_of_iso, hours=burst_hours)
        burst_mult = _social_burst_multiplier(bc)

    # weights and aggregates
    asof = _iso_to_dt(as_of_iso) or datetime.now(timezone.utc)
    weights = []
    sum_w = sum_ws = sum_ws2 = 0.0
    for d in deltas:
        ts = _iso_to_dt(d.get("t",""))
        if not ts: 
            continue
        age_days = max(0.0, (asof - ts).total_seconds()/86400.0)
        w = exp(-0.12 * age_days) * _impact_w(d.get("impact",""))
        if d.get("chan") == "social":
            w *= burst_mult
        s = _dir_to_num(d.get("dir",""))
        weights.append(w)
        sum_w += w
        sum_ws += w * s
        sum_ws2 += w * (s**2)
    if sum_w > 0:
        des_raw = sum_ws / sum_w
        var = max(0.0, (sum_ws2 / sum_w) - (des_raw ** 2))
    else:
        des_raw = 0.0
        var = 0.0
    conf = _confidence(sum_w, weights, var)

    # sector baseline (peers), same channel filter to be apples-to-apples
    des_sector = 0.0
    if peers:
        peer_scores = []
        for p in peers:
            if p.upper() == ticker.upper(): continue
            p_deltas = _collect_ticker_deltas(store, p, as_of_iso, lookback_days, limit=3000)
            if channel_filter in ("news","social"):
                p_deltas = [d for d in p_deltas if d.get("chan") == channel_filter]
            if not p_deltas: continue
            # simple mean for peers
            pasof = _iso_to_dt(as_of_iso) or datetime.now(timezone.utc)
            pw_sum = pws = 0.0
            for d in p_deltas:
                ts = _iso_to_dt(d.get("t","")); 
                if not ts: continue
                age = max(0.0,(pasof-ts).total_seconds()/86400.0)
                w = exp(-0.12*age) * _impact_w(d.get("impact",""))
                if d.get("chan") == "social":
                    w *= burst_mult  # optional: same burst mult, or compute per-peer
                s = _dir_to_num(d.get("dir",""))
                pw_sum += w; pws += w*s
            if pw_sum > 0:
                peer_scores.append(pws/pw_sum)
        if peer_scores:
            des_sector = sum(peer_scores)/len(peer_scores)

    des_idio = des_raw - des_sector
    return {
        "ticker": ticker.upper(),
        "as_of": as_of_iso,
        "window_days": lookback_days,
        "channel": channel_filter,
        "burst_hours": burst_hours,
        "signal": {
            "des_raw": round(des_raw, 4),
            "des_sector": round(des_sector, 4),
            "des_idio": round(des_idio, 4),
            "confidence": round(conf, 3),
            "n_deltas": len(deltas)
        }
    }

# -------------------- Storage abstraction --------------------
class BaseStore:
    def get_last_timestamp(self, ticker: str) -> Optional[str]:
        raise NotImplementedError
    def append_delta(self, ticker: str, delta: dict) -> None:
        raise NotImplementedError
    def append_memory(self, ticker: str, note: str, range_meta: Optional[dict] = None) -> None:
        raise NotImplementedError
    def last_n_records(self, ticker: str, n: int = 500) -> List[dict]:
        raise NotImplementedError
    def records_before(self, ticker: str, cutoff_iso: str, limit: int = 500) -> List[dict]:
        raise NotImplementedError

class JsonlStore(BaseStore):
    """
    File-per-ticker store with a lightweight _index.json (last_ts per ticker).
    """
    def __init__(self, base_dir: Union[str, Path] = DEFAULT_CONVOS_DIR):
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        self.index_path = self.base / "_index.json"
        try:
            self.index = json.loads(self.index_path.read_text(encoding="utf-8"))
        except Exception:
            self.index = {}

    def _path(self, ticker: str) -> Path:
        return self.base / f"{ticker.upper()}.jsonl"

    def get_last_timestamp(self, ticker: str) -> Optional[str]:
        tk = ticker.upper()
        if tk in self.index:
            return self.index[tk]
        p = self._path(tk)
        if not p.exists():
            return None
        last = None
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("type") == "delta":
                        t = rec["data"].get("t")
                        if t:
                            last = t
                except Exception:
                    pass
        if last:
            self.index[tk] = last
            self.index_path.write_text(json.dumps(self.index, indent=2), encoding="utf-8")
        return last

    def _write_index(self, tk: str, ts: str) -> None:
        self.index[tk.upper()] = ts
        self.index_path.write_text(json.dumps(self.index, indent=2), encoding="utf-8")

    def append_delta(self, ticker: str, delta: dict) -> None:
        p = self._path(ticker)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"type": "delta", "data": delta}, ensure_ascii=False) + "\n")
        t = delta.get("t")
        if t:
            self._write_index(ticker, t)

    def append_memory(self, ticker: str, note: str, range_meta: Optional[dict] = None) -> None:
        p = self._path(ticker)
        data = {"t": _now_iso(), "note": note}
        if range_meta:
            data["range"] = range_meta
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"type":"memory","data":data}, ensure_ascii=False) + "\n")

    def last_n_records(self, ticker: str, n: int = 500) -> List[dict]:
        p = self._path(ticker)
        if not p.exists():
            return []
        lines = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
        return lines[-n:]

    def records_before(self, ticker: str, cutoff_iso: str, limit: int = 500) -> List[dict]:
        p = self._path(ticker)
        if not p.exists():
            return []
        recs: List[dict] = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("type") == "delta":
                    ts = (rec.get("data") or {}).get("t") or ""
                    if ts and ts <= cutoff_iso:
                        recs.append(rec)
                elif rec.get("type") == "memory":
                    rng = (rec.get("data") or {}).get("range") or {}
                    end = rng.get("end")
                    if end and end <= cutoff_iso:
                        recs.append(rec)
        # sort chronologically by ts or range.end and cap
        def _ts_of(r: dict) -> str:
            d = r.get("data") or {}
            return d.get("t") or (d.get("range") or {}).get("end") or ""
        recs.sort(key=_ts_of)
        return recs[-limit:]

class SQLiteStore(BaseStore):
    """
    Monolithic store in a single SQLite file.
    Tables:
      conversations(id INTEGER PK, ticker TEXT, ts TEXT, kind TEXT, data JSON)
      convo_index(ticker TEXT PK, last_ts TEXT)
    Index: idx_conv_ticker_ts (ticker, ts)
    """
    def __init__(self, path: Union[str, Path] = DEFAULT_SQLITE_PATH):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._lock = threading.RLock()
        # defer connection creation per-thread; just ensure schema exists once
        with self._conn() as conn:
            self._init(conn)

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(
                self.path,
                check_same_thread=False,   # allow use in this thread
                isolation_level=None,      # autocommit style
            )
            # sensible pragmas for concurrent reads/writes
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA temp_store=MEMORY;")
            self._local.conn = conn
        return conn

    def _init(self, conn: sqlite3.Connection) -> None:
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS conversations(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            ts TEXT NOT NULL,
            kind TEXT NOT NULL,
            data TEXT NOT NULL,
            post_id TEXT
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS convo_index(
            ticker TEXT PRIMARY KEY,
            last_ts TEXT
        )""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_conv_ticker_ts ON conversations(ticker, ts)")
        # add post_id if legacy
        cols = {row[1] for row in c.execute("PRAGMA table_info(conversations)")}
        if "post_id" not in cols:
            c.execute("ALTER TABLE conversations ADD COLUMN post_id TEXT")
        conn.commit()

    # Optional: explicit close for thread lifecycles (e.g., worker shutdown)
    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try: conn.close()
            finally: self._local.conn = None

    def get_last_timestamp(self, ticker: str) -> Optional[str]:
        conn = self._conn()
        row = conn.execute(
            "SELECT last_ts FROM convo_index WHERE ticker=?", (ticker.upper(),)
        ).fetchone()
        return row[0] if row and row[0] else None

    def _set_last_timestamp(self, ticker: str, ts: str) -> None:
        with self._lock:
            conn = self._conn()
            conn.execute(
                "INSERT INTO convo_index(ticker,last_ts) VALUES(?,?) "
                "ON CONFLICT(ticker) DO UPDATE SET last_ts=excluded.last_ts",
                (ticker.upper(), ts)
            )

    def append_delta(self, ticker: str, delta: dict) -> None:
        t = delta.get("t") or _now_iso()
        with self._lock:
            conn = self._conn()
            conn.execute(
                "INSERT INTO conversations(ticker, ts, kind, data, post_id) VALUES(?,?,?,?,?)",
                (ticker.upper(), t, "delta", json.dumps(delta, ensure_ascii=False), delta.get("post_id"))
            )
            self._set_last_timestamp(ticker, t)

    def append_memory(self, ticker: str, note: str, range_meta: Optional[dict] = None) -> None:
        t = _now_iso()
        data = {"t": t, "note": note}
        if range_meta: data["range"] = range_meta
        with self._lock:
            conn = self._conn()
            conn.execute(
                "INSERT INTO conversations(ticker, ts, kind, data, post_id) VALUES(?,?,?,?,?)",
                (ticker.upper(), t, "memory", json.dumps(data, ensure_ascii=False), None)
            )

    def last_n_records(self, ticker: str, n: int = 500) -> List[dict]:
        conn = self._conn()
        rows = conn.execute(
            "SELECT kind, data, post_id FROM conversations WHERE ticker=? ORDER BY ts DESC LIMIT ?",
            (ticker.upper(), int(n))
        ).fetchall()
        recs: List[dict] = []
        for kind, data_json, pid in rows[::-1]:
            data = json.loads(data_json)
            if pid and isinstance(data, dict) and not data.get("post_id"):
                data["post_id"] = pid
            record = {"type": kind, "data": data}
            if pid:
                record["post_id"] = pid
            recs.append(record)
        return recs

    # Efficient helpers for rollup
    def nth_latest_delta_ts(self, ticker: str, keep_latest: int) -> Optional[str]:
        conn = self._conn()
        row = conn.execute(
            "SELECT ts FROM conversations WHERE ticker=? AND kind='delta' ORDER BY ts DESC LIMIT 1 OFFSET ?",
            (ticker.upper(), max(keep_latest-1, 0))
        ).fetchone()
        return row[0] if row else None

    def fetch_deltas_older_than(self, ticker: str, ts_threshold: str, limit: int = 2000) -> List[dict]:
        conn = self._conn()
        rows = conn.execute(
            "SELECT data, post_id FROM conversations WHERE ticker=? AND kind='delta' AND ts < ? ORDER BY ts ASC LIMIT ?",
            (ticker.upper(), ts_threshold, int(limit))
        ).fetchall()
        out: List[dict] = []
        for data_json, pid in rows:
            data = json.loads(data_json)
            if pid and isinstance(data, dict) and not data.get("post_id"):
                data["post_id"] = pid
            out.append(data)
        return out

    def records_before(self, ticker: str, cutoff_iso: str, limit: int = 500) -> List[dict]:
        conn = self._conn()
        rows_d = conn.execute(
            "SELECT kind, data, post_id FROM conversations WHERE ticker=? AND kind='delta' AND ts<=? ORDER BY ts DESC LIMIT ?",
            (ticker.upper(), cutoff_iso, int(limit*2))
        ).fetchall()
        recs: List[dict] = []
        for kind, data_json, pid in rows_d:
            data = json.loads(data_json)
            if pid and isinstance(data, dict) and not data.get("post_id"):
                data["post_id"] = pid
            record = {"type": kind, "data": data}
            if pid:
                record["post_id"] = pid
            recs.append(record)

        # memory notes whose range.end ≤ cutoff
        rows_m = conn.execute(
            "SELECT kind, data, post_id FROM conversations WHERE ticker=? AND kind='memory' ORDER BY ts DESC LIMIT ?",
            (ticker.upper(), 5000)
        ).fetchall()
        for (k, d, pid) in rows_m:
            js = json.loads(d)
            rng = js.get("range") or {}
            if rng.get("end") and rng["end"] <= cutoff_iso:
                record = {"type": k, "data": js}
                if pid:
                    record["post_id"] = pid
                recs.append(record)

        # sort chronologically and cap
        def _ts_of(r: dict) -> str:
            d = r.get("data") or {}
            return d.get("t") or (d.get("range") or {}).get("end") or ""
        recs.sort(key=_ts_of)
        return recs[-limit:]

# -------------------- Records → chat messages --------------------
def _records_to_messages(items: List[dict]) -> List[dict]:
    sys = ("You are a concise analyst using rolling memory deltas. "
           "Be factual. Cite delta timestamps if used.")
    msgs = [{"role": "system", "content": sys}]
    def fmt_delta(d: dict) -> str:
        parts = [
            f"[{d.get('t','')}] {', '.join(d.get('who',[]))} {d.get('cat',[])}",
            f"{d.get('sum','')}",
            f"dir={d.get('dir','uncertain')} impact={d.get('impact','med')} why={d.get('why','')}",
            f"src={d.get('src','')} url={d.get('url','')}"
        ]
        return " | ".join(_clip(x, 280) for x in parts if x)
    for rec in items:
        if rec.get("type") == "memory":
            msgs.append({"role":"assistant", "content": f"(memory) {rec['data'].get('t','')}: {rec['data'].get('note','')}"})
        elif rec.get("type") == "delta":
            msgs.append({"role":"assistant", "content": f"(delta) {fmt_delta(rec['data'])}"})
    msgs.append({"role":"system","content":"When answering, cite recent deltas by timestamp if used."})
    return msgs


def build_as_of_messages(store, ticker: str, as_of_iso: str, limit: int = 500) -> List[dict]:
    """Return leak-safe chat messages for a ticker as of a cutoff timestamp."""
    recs = store.records_before(ticker, as_of_iso, limit)
    msgs = list(_records_to_messages(recs))
    cutoff_msg = {
        "role": "system",
        "content": f"AS_OF cutoff = {as_of_iso}. Ignore any knowledge after this timestamp.",
    }
    if msgs:
        msgs.insert(1, cutoff_msg)
    else:
        msgs.extend([
            {"role": "system", "content": "You are a concise analyst using rolling memory deltas."},
            cutoff_msg,
        ])
    return msgs


def chat(messages: List[Dict[str, str]], *, model: str = DEFAULT_MODEL, timeout_s: Optional[float] = None) -> str:
    """Public chat helper that mirrors ConversationHub.ask/ask_as_of behaviour."""
    return _chat(messages, model, timeout_s or 30.0)

# -------------------- Public API: ConversationHub --------------------
class ConversationHub:
    def __init__(self, store: BaseStore, model: str = DEFAULT_MODEL):
        self.store = store
        self.model = model

    def ingest_article(
        self,
        *,
        tickers: Sequence[str],
        title: str,
        bullets: Optional[Sequence[str]] = None,
        url: str = "",
        ts: Optional[str] = None,
        source: str = "news",
        verbose: bool = False,
        post_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Ingest a single article into multiple ticker conversations (idempotent via last_ts)."""
        ts = ts or _now_iso()
        tickers = [t.upper() for t in tickers if t]
        bullets_list = list(bullets or [])
        delta = _compress_delta(self.model, ts, source, list(tickers), title, bullets_list, url)
        chan = _classify_channel(source, url)
        delta["chan"] = chan
        
        if not delta:
            if verbose:
                print(f"[convo] compress_fail: title='{_clip(title,80)}' tickers={tickers}")
            return {
                "tickers": list(tickers),
                "appended": 0,
                "skipped_old": 0,
                "reason": "compress_fail",
                "delta": None,
            }

        if post_id:
            delta["post_id"] = post_id

        appended = 0
        skipped_old = 0
        for tk in tickers[:6]:
            last = self.store.get_last_timestamp(tk)
            if last and not _iso_newer(ts, last):
                skipped_old += 1
                if verbose:
                    print(f"[convo] SKIP {tk} ts={ts} (<= last_ts={last}) :: {_clip(delta.get('sum',''), 100)}")
                continue
            self.store.append_delta(tk, delta)
            appended += 1
            if verbose:
                print(f"[convo] ADD  {tk} ts={ts} :: {_clip(delta.get('sum',''), 100)}")
        return {
            "tickers": list(tickers),
            "appended": appended,
            "skipped_old": skipped_old,
            "reason": "ok",
            "delta": delta,
        }

    def ask(self, ticker: str, question: str, n: int = 500) -> str:
        recs = self.store.last_n_records(ticker, n=n)
        msgs = _records_to_messages(recs)
        msgs.append({"role":"user","content":question})
        return chat(msgs, model=self.model, timeout_s=40.0).strip()

    def ask_as_of(self, ticker: str, question: str, as_of_iso: str, limit: int = 500) -> str:
        """Leakage-free Q&A: only uses deltas with ts ≤ as_of_iso and rollups with range.end ≤ as_of_iso."""
        msgs = build_as_of_messages(self.store, ticker, as_of_iso, limit)
        msgs.append({"role":"user","content":question})
        return chat(msgs, model=self.model, timeout_s=40.0).strip()

# -------------------- DB helpers for CLI ingest --------------------
def _load_all_posts(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    cur = conn.cursor()
    rows = cur.execute("SELECT post_id, platform, source, url, title, author, scraped_at, posted_at, text FROM posts").fetchall()
    out: List[Dict[str, Any]] = []
    for pid, platform, source, url, title, author, scraped_at, posted_at, text in rows:
        out.append({
            "post_id": pid, "platform": platform or "", "source": source or "",
            "url": url or "", "title": title or "", "author": author or "",
            "scraped_at": scraped_at, "posted_at": posted_at, "text": text or ""
        })
    return out

def _load_stage(conn: sqlite3.Connection, post_id: str, stage: str) -> Optional[dict]:
    row = conn.execute("SELECT payload FROM stages WHERE post_id=? AND stage=?", (post_id, stage)).fetchone()
    if not row:
        return None
    try:
        return json.loads(row[0])
    except Exception:
        return None

def _summ_assets(summ: dict) -> List[str]:
    out: List[str] = []
    for it in (summ.get("assets_mentioned") or []):
        if isinstance(it, dict):
            t = (it.get("ticker") or "").strip().upper()
            if t:
                out.append(t)
    return out

_CASHTAG = re.compile(r"\$[A-Z]{1,5}\b")
def _cashtags(title: str, text: str) -> List[str]:
    s = f"{title} {text}".upper()
    return [m[1:] for m in _CASHTAG.findall(s)]

# -------------------- Rollup helpers --------------------
ROLLUP_SYS = (
    "You summarize many deltas into one durable memory note for this ticker.\n"
    "Keep to <= 5 bullet lines, each <= 18 words, capturing: key themes, competitor shifts, recurring risks, direction bias, catalysts."
)

def _rollup_notes(model: str, notes: List[str]) -> str:
    bullets = "\n".join(f"- {_clip(n, 240)}" for n in notes[:400])
    user = "Summarize these deltas into durable memory notes:\n" + bullets
    return _chat([{"role":"system","content":ROLLUP_SYS},{"role":"user","content":user}], model, timeout_s=25.0).strip()

# -------------------- CLI commands --------------------
def _make_store(kind: str, path: str) -> BaseStore:
    if kind == "sqlite":
        p = Path(path)
        if p.is_dir() or str(path).endswith(os.sep):
            p = p / "conversations.sqlite"
        return SQLiteStore(str(p))
    else:
        return JsonlStore(path)

def cmd_ingest(db_path: str, store_kind: str, convos_path: str, model: str, verbose: bool) -> None:
    store = _make_store(store_kind, convos_path)
    hub = ConversationHub(store=store, model=model)

    conn = sqlite3.connect(db_path)
    posts = _load_all_posts(conn)
    total = len(posts)
    added_total = 0
    skipped_old_total = 0

    print(f"[ingest] start | posts={total} model='{model}' store='{store_kind}' path='{convos_path}'")
    start = time.time()

    for i, p in enumerate(posts, start=1):
        pid = p["post_id"]
        if verbose:
            print(f"[{i:05d}/{total}] post_id={pid} title='{_clip(p.get('title',''), 100)}'")

        summ = _load_stage(conn, pid, "summariser")
        if not summ:
            if verbose:
                print("   ↳ skip: no summariser")
            continue

        ts = p.get("posted_at") or p.get("scraped_at") or _now_iso()

        who = _summ_assets(summ)
        if not who:
            who = list({*_cashtags(p["title"], p["text"])})
        if not who:
            if verbose:
                print("   ↳ skip: no tickers/cashtags")
            continue

        bullets = [b for b in (summ.get("summary_bullets") or []) if isinstance(b, str)]
        res = hub.ingest_article(
            tickers=who,
            title=p["title"],
            bullets=bullets,
            url=p.get("url") or "",
            ts=ts,
            source=p.get("source") or p.get("platform") or "news",
            verbose=True,  # always print per-ticker ADD/SKIP lines
            post_id=pid,
        )
        added_total += res["appended"]
        skipped_old_total += res["skipped_old"]

        if verbose and (i % 50 == 0):
            elapsed = time.time() - start
            rate = i / max(elapsed, 1)
            print(f"[progress] i={i}/{total} added={added_total} skipped_old={skipped_old_total} rate={rate:.2f}/s")

    conn.close()
    elapsed = time.time() - start
    print(f"[ingest] done | posts={total} added={added_total} skipped_old={skipped_old_total} elapsed={elapsed:.1f}s")

def cmd_ask(ticker: str, store_kind: str, convos_path: str, model: str, question: str) -> None:
    store = _make_store(store_kind, convos_path)
    hub = ConversationHub(store=store, model=model)
    ans = hub.ask(ticker.upper(), question, n=500)
    print("\n=== Answer ===\n" + ans + "\n")

def cmd_ask_as_of(ticker: str, store_kind: str, convos_path: str, model: str, question: str, as_of_iso: str) -> None:
    store = _make_store(store_kind, convos_path)
    hub = ConversationHub(store=store, model=model)
    ans = hub.ask_as_of(ticker.upper(), question, as_of_iso, limit=500)
    print("\n=== Answer (as-of) ===\n" + ans + "\n")

def cmd_rollup(ticker: str, store_kind: str, convos_path: str, model: str, keep_latest: int, verbose: bool) -> None:
    store = _make_store(store_kind, convos_path)

    # Build lines of older deltas → memory note; backend-specific for efficiency
    if isinstance(store, SQLiteStore):
        nth_ts = store.nth_latest_delta_ts(ticker, keep_latest)
        if not nth_ts:
            print(f"[rollup] no deltas or fewer than keep_latest={keep_latest} for {ticker}")
            return
        older = store.fetch_deltas_older_than(ticker, nth_ts, limit=5000)
        if not older:
            print(f"[rollup] nothing to roll up for {ticker}")
            return
        lines = [f"[{d.get('t','')}] {', '.join(d.get('who',[]))} :: {d.get('sum','')} (dir={d.get('dir','')}, imp={d.get('impact','')})" for d in older]
    else:
        recs = store.last_n_records(ticker, n=100000)  # effectively all
        deltas = [r["data"] for r in recs if r.get("type") == "delta"]
        if len(deltas) <= keep_latest:
            print(f"[rollup] nothing to roll up for {ticker}")
            return
        older = deltas[:-keep_latest]
        lines = [f"[{d.get('t','')}] {', '.join(d.get('who',[]))} :: {d.get('sum','')} (dir={d.get('dir','')}, imp={d.get('impact','')})" for d in older]

    # summarize and store with covered range to enable time-travel filtering
    note = _rollup_notes(model, lines)
    if older:
        start_ts = older[0].get("t","")
        end_ts   = older[-1].get("t","")
    else:
        start_ts = end_ts = _now_iso()
    store.append_memory(ticker, note, range_meta={"start": start_ts, "end": end_ts})
    if verbose:
        print(f"[rollup] memory added for {ticker} ({len(lines)} deltas summarized) range=[{start_ts} → {end_ts}]")
    else:
        print(f"[rollup] memory added for {ticker}")

# -------------------- CLI --------------------
def _cli():
    ap = argparse.ArgumentParser(description="Per-ticker conversation hub (Ollama). Importable + CLI.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # ingest
    p_ing = sub.add_parser("ingest", help="Scan DB and append compressed deltas per ticker.")
    p_ing.add_argument("--db", default=str(DEFAULT_DB))
    p_ing.add_argument("--store", choices=["jsonl","sqlite"], default="jsonl")
    p_ing.add_argument("--convos", default=str(DEFAULT_CONVOS_DIR),
                       help="For jsonl: directory. For sqlite: path to .sqlite file or a directory.")
    p_ing.add_argument("--model", default=DEFAULT_MODEL)
    p_ing.add_argument("--verbose", action="store_true")

    # ask
    p_ask = sub.add_parser("ask", help="Ask a question against a ticker's rolling conversation.")
    p_ask.add_argument("--ticker", required=True)
    p_ask.add_argument("--q", required=True)
    p_ask.add_argument("--store", choices=["jsonl","sqlite"], default="jsonl")
    p_ask.add_argument("--convos", default=str(DEFAULT_CONVOS_DIR))
    p_ask.add_argument("--model", default=DEFAULT_MODEL)

    # ask-as-of (time-travel)
    p_ask_asof = sub.add_parser("ask-as-of", help="Ask with an AS-OF cutoff to avoid future leakage.")
    p_ask_asof.add_argument("--ticker", required=True)
    p_ask_asof.add_argument("--q", required=True)
    p_ask_asof.add_argument("--as-of", required=True, help="ISO8601 cutoff, e.g. 2025-09-30T23:59:59+00:00")
    p_ask_asof.add_argument("--store", choices=["jsonl","sqlite"], default="jsonl")
    p_ask_asof.add_argument("--convos", default=str(DEFAULT_CONVOS_DIR))
    p_ask_asof.add_argument("--model", default=DEFAULT_MODEL)

    # rollup
    p_ru = sub.add_parser("rollup", help="Summarize older deltas into a memory note.")
    p_ru.add_argument("--ticker", required=True)
    p_ru.add_argument("--keep-latest", type=int, default=200)
    p_ru.add_argument("--store", choices=["jsonl","sqlite"], default="jsonl")
    p_ru.add_argument("--convos", default=str(DEFAULT_CONVOS_DIR))
    p_ru.add_argument("--model", default=DEFAULT_MODEL)
    p_ru.add_argument("--verbose", action="store_true")

    p_score = sub.add_parser("score", help="Compute DES sentiment metrics from stored deltas.")
    p_score.add_argument("--ticker", required=True)
    p_score.add_argument("--as-of", required=True)
    p_score.add_argument("--days", type=int, default=7)
    p_score.add_argument("--store", choices=["jsonl","sqlite"], default="jsonl")
    p_score.add_argument("--convos", default=str(DEFAULT_CONVOS_DIR))
    p_score.add_argument("--peers", default="", help="Comma-separated peer tickers for baseline; leave empty to skip.")

    p_score.add_argument("--channel", choices=["all","news","social"], default="all", help="Which channel to use for scoring.")
    p_score.add_argument("--burst-hours", type=int, default=6, help="Burst window for social amplification.")

    args = ap.parse_args()
    if args.cmd == "ingest":
        cmd_ingest(args.db, args.store, args.convos, args.model, args.verbose)
    elif args.cmd == "ask":
        cmd_ask(args.ticker, args.store, args.convos, args.model, args.q)
    elif args.cmd == "ask-as-of":
        cmd_ask_as_of(args.ticker, args.store, args.convos, args.model, args.q, args.as_of)
    elif args.cmd == "rollup":
        cmd_rollup(args.ticker, args.store, args.convos, args.model, args.keep_latest, args.verbose)
    elif args.cmd == "score":
        store = _make_store(args.store, args.convos)
        peers = [t.strip().upper() for t in args.peers.split(",") if t.strip()] or None
        out = compute_ticker_signal(
            store,
            args.ticker.upper(),
            args.as_of,
            lookback_days=args.days,
            peers=peers,
            channel_filter=args.channel,
            burst_hours=args.burst_hours,
        )
        print(json.dumps(out, indent=2))

if __name__ == "__main__":
    _cli()
