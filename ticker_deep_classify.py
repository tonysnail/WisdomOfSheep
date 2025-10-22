#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Finance-first ticker classifier (no Wikipedia, no LLM).

Process for building Ticker encyclopaedia:
1> build_ticker_universe.py
2> enrich_tickers.py
3> [ ticker_deep_classify.py ]

This script fills in remaining undetected tickers using Yahoo Finance
search/quote APIs.

It: 
  1) Queries Yahoo Finance search API (fast).
  2) Falls back to Yahoo v7 Quote (for longName/exchange) when needed.
  3) Detects hard cues: ETF, REIT, ADR, SPAC, Trust, Warrant, Unit, Rights, Forex, Futures, Crypto.
  4) Writes results to finance_company_types.csv
  5) Optionally patches tickers_enriched.csv in-place.


Usage
-----
python ticker_deep_classify.py \
  --input tickers_enriched.csv \
  --out finance_company_types.csv \
  --apply \
  --conf-thresh 0.65 \
  --limit 0 \
  --flush-cache \
  --progress-every 100 \
  --log-pass-through \
  --verbose \
  --qps 2.0 \
  --timeout 12

Single-ticker test:
python ticker_deep_classify.py --test BKUI --verbose --log-every-query --qps 1.5 --timeout 12

"""

from __future__ import annotations

import os
import re
import json
import time
import logging
import argparse
import random
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta

import requests
import pandas as pd


# ---------------- Config ----------------

HEADERS = {"User-Agent": "WisdomOfSheep/1.0 (Ticker classifier; contact: local)"}
CACHE_DIR = Path(".cache/finance_types"); CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_VERSION = "finance_only_v2"

YF_QUOTE_V7 = "https://query2.finance.yahoo.com/v7/finance/quote"
YF_SEARCH   = "https://query2.finance.yahoo.com/v1/finance/search"

# We only *apply* hard non-equity types from search:
ASSET_TYPES = {"ETF","ADR","SPAC?","REIT","TRUST","WARRANT","UNIT","RIGHTS","FOREX","FUTURES","CRYPTO"}

# Yahoo exchange display to our market/exchange code map
YF_EXCH_TO_META = {
    "NYSEARCA": ("NYSE Arca","PCX"), "ARCA": ("NYSE Arca","PCX"),
    "NYSE": ("NYSE","NYQ"),
    "NASDAQGS": ("NASDAQ","NMS"), "NASDAQ GM": ("NASDAQ","NGM"),
    "NASDAQCM": ("NASDAQ","NCM"), "NASDAQ": ("NASDAQ","NMS"),
    "BATS": ("Cboe BZX","BATS"), "CBOE": ("Cboe BZX","BATS"),
    "CBOE US": ("Cboe BZX","BATS"), "CBOE BZX": ("Cboe BZX","BATS"),
    "AMEX": ("NYSE American","ASE"), "NYSEAMERICAN": ("NYSE American","ASE"),
    "TSX": ("TSX","TSX"), "TSXV": ("TSX Venture","TSXV"),
    "LSE": ("LSE","LSE"),
}

# Suffix hints (W/WS/WT/U/UN/R)
SUFFIX_MAP = {"W":"WARRANT","WS":"WARRANT","WT":"WARRANT","U":"UNIT","UN":"UNIT","R":"RIGHTS"}


# ---------------- Logging ----------------

def setup_logging(verbose: bool):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("finance_types")

log = setup_logging(False)


# ---------------- Utils ----------------

def strip_security_suffix(sym: str) -> Tuple[str, Optional[str]]:
    s = sym.upper().strip()
    if s.endswith(("-WS","-WT","-UN")):
        base = s.rsplit("-", 1)[0]
        suf  = s.rsplit("-", 1)[1]
        return base, SUFFIX_MAP.get(suf)
    m = re.search(r"[-/](W|WS|WT|U|UN|R)$", s)
    if m:
        return s[:m.start()], SUFFIX_MAP.get(m.group(1))
    return s, None


class RateLimiter:
    def __init__(self, qps: float = 1.2, jitter: float = 0.25):
        self.min_interval = 1.0 / max(qps, 0.1)
        self.jitter = float(jitter)
        self._last = 0.0
    def wait(self):
        now = time.time()
        wait_for = self.min_interval - (now - self._last)
        if wait_for > 0:
            time.sleep(wait_for + random.uniform(0, self.jitter))
        self._last = time.time()


# ---------------- HTTP ----------------

def _get_json(url: str, params: Dict[str, Any], retries=2, pause=0.8, timeout=15.0, rl: RateLimiter=None):
    for a in range(retries + 1):
        try:
            if rl: rl.wait()
            r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 503):
                time.sleep(pause * (a + 1))
                continue
            # other status: just stop trying
            return None
        except Exception:
            time.sleep(pause * (a + 1))
    return None


# ---------------- Yahoo helpers ----------------

def yahoo_quote_single(symbol: str, timeout=15.0, rl: RateLimiter=None, log_queries: bool=False):
    params = {"symbols": symbol}
    if log_queries:
        log.info("  ↳ YF quote: %s", symbol)
    js = _get_json(YF_QUOTE_V7, params, timeout=timeout, rl=rl) or {}
    res = (js.get("quoteResponse", {}) or {}).get("result", [])
    if not res:
        return None
    q = res[0]
    return {
        "symbol": q.get("symbol"),
        "longName": q.get("longName") or q.get("shortName"),
        "shortName": q.get("shortName"),
        "exchange": q.get("exchange"),
        "fullExchangeName": q.get("fullExchangeName"),
        "quoteType": (q.get("quoteType") or "").upper(),
    }


def yahoo_finance_search(symbol: str, timeout=15.0, rl: RateLimiter=None, log_queries: bool=False):
    params = {"q": symbol, "lang": "en-US", "region": "US", "quotesCount": 6, "newsCount": 0}
    if log_queries:
        log.info("  ↳ YF search: %s", symbol)
    js = _get_json(YF_SEARCH, params, timeout=timeout, rl=rl) or {}
    quotes = js.get("quotes") or []
    if not quotes:
        return None

    preferred = None
    symU = symbol.upper()
    for q in quotes:
        if str(q.get("symbol", "")).upper() == symU:
            preferred = q
            break
    if preferred is None:
        for q in quotes:
            if (q.get("quoteType") or "").lower() in ("etf","equity","adr","mutualfund","currency","future","crypto"):
                preferred = q
                break
    if preferred is None:
        preferred = quotes[0]

    qt = (preferred.get("quoteType") or "").upper()
    name = preferred.get("shortname") or preferred.get("longname") or preferred.get("name") or ""
    exch = preferred.get("exchange") or preferred.get("exchDisp") or ""
    return {
        "name": name,
        "quoteType": qt,
        "exchange": exch,
        "symbol": preferred.get("symbol"),
        "source": "yahoo_search",
        "title": f"{name} ({preferred.get('symbol')})",
        "snippet": preferred.get("exchDisp") or preferred.get("exchange") or "",
        "link": f"https://finance.yahoo.com/quote/{preferred.get('symbol')}",
    }


def map_yahoo_quote_type(qt: str) -> Optional[str]:
    qt = (qt or "").upper()
    if qt == "ETF": return "ETF"
    if qt == "ADR": return "ADR"
    if qt in ("FUTURE", "FUTURES"): return "FUTURES"
    if qt in ("CURRENCY", "FOREX"):  return "FOREX"
    if qt in ("CRYPTO", "CRYPTOCURRENCY"): return "CRYPTO"
    # EQUITY or anything else → do not set
    return None


# ---------------- Finance guess ----------------

def finance_guess(symbol: str, log_queries=False, timeout=15.0, rl: RateLimiter=None) -> Optional[Dict[str, Any]]:
    # 1) Yahoo Finance search (fast)
    yf_hit = yahoo_finance_search(symbol, timeout=timeout, rl=rl, log_queries=log_queries)
    if yf_hit:
        at = map_yahoo_quote_type(yf_hit.get("quoteType"))
        if at:
            yf_hit["type"] = at
            return yf_hit
        # keep name even if no hard type
        return yf_hit

    # 2) Fallback: Yahoo v7 quote (longName/exchange)
    v7 = yahoo_quote_single(symbol, timeout=timeout, rl=rl, log_queries=log_queries)
    if v7 and v7.get("longName"):
        return {
            "name": v7["longName"],
            "type": map_yahoo_quote_type(v7.get("quoteType")),
            "exchange": v7.get("exchange"),
            "exchDisp": v7.get("fullExchangeName"),
            "source": "yahoo_quote",
            "title": f"{v7.get('longName','')} ({v7.get('symbol','')})",
            "snippet": v7.get("fullExchangeName") or "",
            "link": f"https://finance.yahoo.com/quote/{symbol}",
        }
    return None


# ---------------- Cache ----------------

def cache_path(symbol: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", symbol.upper())
    return CACHE_DIR / f"{safe}.{CACHE_VERSION}.json"

def load_cached(symbol: str, ttl_days: Optional[int]):
    p = cache_path(symbol)
    if not p.exists():
        return None
    if ttl_days:
        try:
            if datetime.fromtimestamp(p.stat().st_mtime) < datetime.now() - timedelta(days=ttl_days):
                return None
        except Exception:
            pass
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def save_cached(symbol: str, data: Dict[str, Any]):
    p = cache_path(symbol)
    try:
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


# ---------------- Sparse selection ----------------

def pick_sparse_rows(df: pd.DataFrame, limit: Optional[int]):
    miss_name   = df["longName"].isna()
    miss_si     = df["sector"].isna() & df["industry"].isna()
    unknown_mkt = df["market"].fillna("").str.upper().eq("UNKNOWN")
    candidates  = df[miss_name | miss_si | unknown_mkt].copy()
    if limit:
        candidates = candidates.head(limit)
    return candidates

def is_sparse_row(row) -> bool:
    """
    Decide whether a row needs probing (web calls) based on missing metadata.
    A row is 'sparse' if:
      - longName is empty/NaN, OR
      - BOTH sector and industry are empty/NaN, OR
      - market is UNKNOWN/empty
    """
    longname_empty = (pd.isna(row.longName) or not str(row.longName or "").strip())
    both_si_empty  = (pd.isna(row.sector) and pd.isna(row.industry))
    market_val = (str(row.market) if pd.notna(row.market) else "")
    market_unknown = (not market_val.strip()) or (market_val.strip().upper() == "UNKNOWN")
    return longname_empty or both_si_empty or market_unknown

# ---------------- Main ----------------

def main():
    parser = argparse.ArgumentParser(description="Finance-first ticker classifier (search-only).")
    parser.add_argument("--input", default="tickers/tickers_enriched.csv")
    parser.add_argument("--out",   default="tickers/finance_company_types.csv")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--conf-thresh", type=float, default=0.65)
    parser.add_argument("--limit", type=int, default=0, help="Max number of *sparse* rows to probe (0 = all).")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--flush-cache", action="store_true")
    parser.add_argument("--cache-ttl-days", type=int, default=30)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--qps", type=float, default=1.2)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--log-every-query", action="store_true")
    parser.add_argument("--test", dest="test_symbol", default=None,
                        help="Test a single ticker symbol (e.g., BKUI) and print detected name/type/exchange.")
    # Kept for compatibility; no longer needed since we always include all rows inline.
    parser.add_argument("--include-all", action="store_true", default=True)
    parser.add_argument("--log-pass-through", action="store_true",
                        help="Log a summary line for each non-sparse ticker included as pass-through (no web calls).")
    args = parser.parse_args()

    global log
    log = setup_logging(args.verbose)
    rl = RateLimiter(args.qps, jitter=0.25)

    # --- SINGLE-TICKER TEST MODE ---
    if args.test_symbol:
        sym = args.test_symbol.strip().upper()
        log.info("[test] %s — starting", sym)
        fin = finance_guess(sym, log_queries=True, timeout=args.timeout, rl=rl)
        v7  = yahoo_quote_single(sym, timeout=args.timeout, rl=rl, log_queries=True)

        def _fmt(o): return json.dumps(o, ensure_ascii=False, indent=2) if o else "None"
        log.info("[test] finance_guess result:\n%s", _fmt(fin))
        log.info("[test] yahoo_quote_single result:\n%s", _fmt(v7))

        detected_type = fin.get("type") if isinstance(fin, dict) else None
        detected_name = (fin.get("name") if isinstance(fin, dict) else None) or (v7.get("longName") if isinstance(v7, dict) else None)
        ex_disp = None
        if isinstance(fin, dict):
            ex_disp = fin.get("exchDisp") or fin.get("fullExchangeName") or fin.get("exchange")
        if not ex_disp and isinstance(v7, dict):
            ex_disp = v7.get("fullExchangeName") or v7.get("exchange")

        log.info("[test] summary → symbol=%s | name=%s | type=%s | exchange_display=%s | source=%s",
                 sym, (detected_name or "—"), (detected_type or "—"),
                 (ex_disp or "—"),
                 (fin.get("source") if isinstance(fin, dict) else "—"))
        return

    # --- Load input
    src = Path(args.input)
    if not src.exists():
        log.error("Input not found: %s", src)
        return

    df = pd.read_csv(src)
    for c in ["Symbol","longName","sector","industry","asset_type","market","themes","exchange","fullExchangeName"]:
        if c not in df.columns:
            df[c] = pd.NA

    # --- Unified pass over ALL rows; only probe sparse ones
    rows: List[Dict[str, Any]] = []
    t0 = time.time()

    total_rows = len(df)
    sparse_seen = 0
    non_sparse_seen = 0
    sparse_limit = args.limit if (args.limit and args.limit > 0) else None

    log.info("Total input rows: %d", total_rows)

    for idx, row in enumerate(df.itertuples(index=False), start=1):
        sym = str(row.Symbol).strip().upper()
        sparse = is_sparse_row(row)

        if sparse:
            if sparse_limit is not None and sparse_seen >= sparse_limit:
                # Treat as pass-through once we hit the sparse limit
                sparse = False

        if sparse:
            sparse_seen += 1
            log.info("[%d/%d] %s — sparse → probing", idx, total_rows, sym)

            cached = None if args.flush_cache else load_cached(sym, args.cache_ttl_days)
            if cached:
                log.info("summary → symbol=%s | name=%s | type=%s | exchange_display=%s | source=%s",
                         sym,
                         (cached.get("name_from_finance") or "—"),
                         (cached.get("asset_type_detected") or "—"),
                         (cached.get("finance_exchange") or "—"),
                         (cached.get("finance_source") or "cache"))
                rows.append(cached)
                continue

            fin = finance_guess(sym, log_queries=args.log_every_query, timeout=args.timeout, rl=rl)

            detected_type = fin.get("type") if isinstance(fin, dict) else None
            detected_name = fin.get("name") if isinstance(fin, dict) else None
            ex_disp = None
            if isinstance(fin, dict):
                ex_disp = fin.get("exchDisp") or fin.get("fullExchangeName") or fin.get("exchange")

            # v7 fallback for nicer name/exchange if missing
            if not detected_name or not ex_disp:
                v7 = yahoo_quote_single(sym, timeout=args.timeout, rl=rl, log_queries=False)
                if v7:
                    if not detected_name: detected_name = v7.get("longName")
                    if not ex_disp:       ex_disp       = v7.get("fullExchangeName") or v7.get("exchange")

            log.info("summary → symbol=%s | name=%s | type=%s | exchange_display=%s | source=%s",
                     sym, (detected_name or "—"), (detected_type or "—"),
                     (ex_disp or "—"),
                     (fin.get("source") if isinstance(fin, dict) else "—"))

            if fin and fin.get("type"):
                result = {
                    "Symbol": sym,
                    "name_from_finance": detected_name,
                    "finance_title": fin.get("title"),
                    "finance_source": fin.get("source"),
                    "finance_link": fin.get("link"),
                    "finance_exchange": ex_disp,
                    "asset_type_detected": fin.get("type"),
                    "confidence": 0.95,
                    "why": f"finance_cue:{fin.get('source')}",
                }
            else:
                result = {
                    "Symbol": sym,
                    "name_from_finance": detected_name,  # may be from v7 quote
                    "finance_title": fin.get("title") if isinstance(fin, dict) else None,
                    "finance_source": fin.get("source") if isinstance(fin, dict) else None,
                    "finance_link": fin.get("link") if isinstance(fin, dict) else None,
                    "finance_exchange": ex_disp,
                    "asset_type_detected": None,
                    "confidence": 0.0,
                    "why": "no_hard_cue",
                }

            save_cached(sym, result)
            rows.append(result)

            # polite pause
            time.sleep(0.08 if not fin or not fin.get("type") else 0.12)

        else:
            # Non-sparse: pass-through row (so it's visible in the log/output *now*)
            non_sparse_seen += 1
            existing_name = row.longName if (isinstance(row.longName, str) and row.longName.strip()) else None
            if isinstance(row.fullExchangeName, str) and row.fullExchangeName.strip():
                existing_exchange = row.fullExchangeName
            elif isinstance(row.exchange, str) and row.exchange.strip():
                existing_exchange = row.exchange
            elif isinstance(row.market, str) and row.market.strip():
                existing_exchange = row.market
            else:
                existing_exchange = None

            rows.append({
                "Symbol": sym,
                "name_from_finance": existing_name,
                "finance_title": None,
                "finance_source": "original_enriched",
                "finance_link": None,
                "finance_exchange": existing_exchange,
                "asset_type_detected": None,
                "confidence": 0.0,
                "why": "already_complete_or_not_sparse"
            })

            if args.log_pass_through:
                existing_type = (str(row.asset_type).upper()
                                 if (isinstance(row.asset_type, str) and row.asset_type.strip())
                                 else "—")
                log.info("summary → symbol=%s | name=%s | type=%s | exchange_display=%s | source=%s",
                         sym,
                         (existing_name or "—"),
                         existing_type,
                         (existing_exchange or "—"),
                         "original_enriched")

        # optional periodic progress ping
        if idx <= 5 or idx % args.progress_every == 0 or idx == total_rows:
            log.info("[%d/%d] processed", idx, total_rows)

    # Final counters
    log.info("Finished scanning rows: total=%d, sparse_probed=%d, pass_through=%d",
             total_rows, sparse_seen, non_sparse_seen)

    # --- Write results
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    log.info("Wrote %s (%d rows) in %.1fs", args.out, len(out), time.time() - t0)

    # --- Apply to enriched CSV (only if a hard cue was detected)
    if args.apply and len(out):
        patched = df.copy()
        by_sym  = out.set_index("Symbol")
        applied = 0

        for i, r in patched.iterrows():
            s = str(r["Symbol"]).strip().upper()
            if s not in by_sym.index:
                continue

            row_out = by_sym.loc[s]
            at = (row_out.get("asset_type_detected") or "").upper()
            conf = float(row_out.get("confidence") or 0.0)

            # Fill longName if empty and we have a finance-derived name
            current_name = r.get("longName")
            if (pd.isna(current_name) or not str(current_name or "").strip()):
                nm = row_out.get("name_from_finance")
                if isinstance(nm, str) and nm.strip():
                    patched.at[i, "longName"] = nm.strip()

            # Backfill market/exchange if UNKNOWN/blank using finance_exchange
            market_now = (str(r.get("market")).upper() if pd.notna(r.get("market")) else "UNKNOWN")
            ex_disp = row_out.get("finance_exchange")
            if market_now == "UNKNOWN" and isinstance(ex_disp, str) and ex_disp.strip():
                key = ex_disp.strip().upper()
                if key in YF_EXCH_TO_META:
                    market, ex_code = YF_EXCH_TO_META[key]
                    patched.at[i, "market"] = market
                    if (pd.isna(r.get("exchange")) or not str(r.get("exchange") or "").strip()):
                        patched.at[i, "exchange"] = ex_code
                    if (pd.isna(r.get("fullExchangeName")) or not str(r.get("fullExchangeName") or "").strip()):
                        patched.at[i, "fullExchangeName"] = market

            # Apply asset_type only for hard non-equity types, if confidence allows
            if at in ASSET_TYPES and conf >= args.conf_thresh:
                raw_current = r.get("asset_type")
                current = (str(raw_current).upper() if pd.notna(raw_current) else "")
                if current in ("", "EQUITY", "UNKNOWN") or pd.isna(raw_current):
                    patched.at[i, "asset_type"] = at
                    if at == "ETF" and (not isinstance(patched.at[i, "themes"], str) or not patched.at[i, "themes"]):
                        patched.at[i, "themes"] = "asset_management"
                    applied += 1

        # Backup original and write patched file in place
        backup = src.with_suffix(".bak.csv")
        src.replace(backup)
        patched.to_csv(src, index=False)
        log.info("Applied %d asset_type updates (confidence ≥ %.2f). Backup: %s → Patched: %s",
                 applied, args.conf_thresh, backup.name, src.name)


if __name__ == "__main__":
    main()
