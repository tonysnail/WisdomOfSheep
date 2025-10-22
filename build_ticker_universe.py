#!/usr/bin/env python3
"""
Build a master symbol universe for WisdomOfSheep.

Usage (for all tickers including futures / commodities / forex / crypto):
python build_ticker_universe.py --verbose --include-futures --include-forex --include-crypto

Process for building Ticker encyclopaedia:
1> [ build_ticker_universe.py ]
2> enrich_tickers.py
3> ticker_deep_classify.py

Outputs:
  tickers_all.csv  (single column: Symbol)

Includes:
- US equities from Nasdaq Trader Symbol Directory:
    * nasdaqlisted.txt     (NASDAQ listed)
    * otherlisted.txt      (NYSE, NYSE American, NYSE Arca, etc.)
- Optional: Futures (=F), Forex (=X), Crypto (-USD) in Yahoo formats
- Optional: universe_overrides.csv merged last to persist ad-hoc symbols
- --ensure ATCH,NVDA,ES=F,... to force-include on the fly

Usage examples:
  python build_ticker_universe.py --verbose
  python build_ticker_universe.py --verbose --include-futures --include-forex --include-crypto
  python build_ticker_universe.py --verbose --ensure ATCH --include-futures

Notes:
- Futures (=F) cover commodities (CL=F, GC=F, ZC=F, etc.), equity index, and rates.
- Forex uses PAIR=X (EURUSD=X). Default: majors autogen; or provide a seed file.
- Crypto defaults to a small top set; or provide a seed file.

"""

from __future__ import annotations
import argparse, csv, io, sys, re, logging, requests
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

# ------------------------------------------------------------
# Remote sources (authoritative, free)
# ------------------------------------------------------------
NASDAQLISTED_URL = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
OTHERLISTED_URL  = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"

# ------------------------------------------------------------
# Defaults / Files
# ------------------------------------------------------------
OUT_CSV_DEFAULT = "tickers/tickers_all.csv"
OVERRIDES_CSV   = "tickers/universe_overrides.csv"  # optional, single column: Symbol

# Skip Yahoo test-style symbols (e.g., ZAZZT)
SKIP_SYMBOL_PATTERNS = [re.compile(r"^Z[A-Z]{3}T$")]

# Keep SPAC/rights/units/warrants in equities by default (filter later in enricher if desired)
ALLOW_UNITS_RIGHTS_WARRANTS_DEFAULT = True

# ------------------------------------------------------------
# Built-in seeds (used if seed files are missing)
# ------------------------------------------------------------
BUILTIN_FUTURES = [
    # Energy
    "CL=F","BZ=F","NG=F","HO=F","RB=F",
    # Metals
    "GC=F","SI=F","HG=F","PL=F","PA=F","ALI=F",
    # Grains / Oilseeds
    "ZC=F","ZW=F","ZS=F","ZM=F","ZL=F","ZO=F","KE=F","ZR=F",
    # Softs
    "KC=F","SB=F","CC=F","CT=F","OJ=F",
    # Livestock
    "LE=F","GF=F","HE=F",
    # Equity index / Rates (handy macro refs)
    "ES=F","NQ=F","YM=F","RTY=F","ZN=F","ZB=F","ZF=F","ZT=F",
]

BUILTIN_CRYPTO = [
    "BTC-USD","ETH-USD","SOL-USD","BNB-USD","ADA-USD",
    "XRP-USD","DOGE-USD","AVAX-USD","DOT-USD","ATOM-USD"
]

FOREX_MAJORS = ["EUR","USD","JPY","GBP","AUD","NZD","CAD","CHF"]

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def setup_logging(verbose: bool) -> logging.Logger:
    lvl = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=lvl, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger("universe")

def normalize_symbol(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.strip()
    if s.startswith("$"): s = s[1:]
    return s.upper()

def should_skip_symbol(sym: str) -> bool:
    return any(pat.match(sym) for pat in SKIP_SYMBOL_PATTERNS)

def fetch_text(url: str, log: logging.Logger) -> str:
    log.info("Downloading: %s", url)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.text

def parse_pipe_table(text: str) -> Tuple[List[str], List[List[str]]]:
    """
    Parses NASDAQ's pipe-delimited symbol files.
    Skips footer lines like: "File Creation Time: ..."
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    header = lines[0].split("|")
    rows: List[List[str]] = []
    for ln in lines[1:]:
        if ln.startswith("File Creation Time"):
            continue
        parts = ln.split("|")
        # pad if trailing pipes missing
        if len(parts) < len(header):
            parts += [""] * (len(header) - len(parts))
        rows.append(parts[:len(header)])
    return header, rows

def as_bool(x: str) -> bool:
    return (x or "").strip().upper() in ("Y","YES","TRUE","T","1")

def is_warrant_unit_rights(sym: str) -> bool:
    return bool(re.search(r"(?:-?W$|/W$|-?U$|/U$|-?R$|/R$)", sym))

def load_csv_single_col(path: Path, col: str = "Symbol") -> List[str]:
    if not path.exists(): return []
    out: List[str] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.lower() == col.lower():  # skip header
            continue
        out.append(normalize_symbol(ln.split(",")[0]))
    return [s for s in out if s]

def load_seed_list(path: Path) -> List[str]:
    if not path.exists(): return []
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")]

def add_seed_rows(rows: Dict[str, Dict[str, Any]], symbols: List[str], listing: str, source: str):
    for s in symbols:
        s = normalize_symbol(s)
        if not s or should_skip_symbol(s): continue
        if s not in rows:
            rows[s] = {"Symbol": s, "Name": "", "Listing": listing,
                       "RawMarketCategory": "", "ETF": False, "Source": source}

# ------------------------------------------------------------
# Build US equities (NASDAQ + NYSE family)
# ------------------------------------------------------------
def load_us_equities(log: logging.Logger,
                     allow_units_rights_warrants: bool = ALLOW_UNITS_RIGHTS_WARRANTS_DEFAULT
                     ) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    # NASDAQ listed
    nas_txt = fetch_text(NASDAQLISTED_URL, log)
    nas_header, nas_rows = parse_pipe_table(nas_txt)
    h = {h:i for i,h in enumerate(nas_header)}
    for r in nas_rows:
        sym = normalize_symbol(r[h.get("Symbol", 0)])
        if not sym or should_skip_symbol(sym): continue
        test_issue = as_bool(r[h.get("Test Issue", 0)])
        if test_issue:  # ignore NASDAQ test issues
            continue
        if not allow_units_rights_warrants and is_warrant_unit_rights(sym):
            continue
        out.append({
            "Symbol": sym,
            "Name": r[h.get("Security Name", 1)],
            "Listing": "NASDAQ",
            "RawMarketCategory": r[h.get("Market Category", 2)],
            "ETF": as_bool(r[h.get("ETF", 6)]),
            "Source": "nasdaqlisted.txt",
        })

    # NYSE / NYSE American / Arca / etc.
    oth_txt = fetch_text(OTHERLISTED_URL, log)
    oth_header, oth_rows = parse_pipe_table(oth_txt)
    h = {h:i for i,h in enumerate(oth_header)}
    for r in oth_rows:
        sym = normalize_symbol(r[h.get("ACT Symbol", 0)])
        if not sym or should_skip_symbol(sym): continue
        test_issue = as_bool(r[h.get("Test Issue", 6)])
        if test_issue:
            continue
        if not allow_units_rights_warrants and is_warrant_unit_rights(sym):
            continue
        exch = (r[h.get("Exchange", 2)] or "").upper()  # NYSE/A/P etc.
        out.append({
            "Symbol": sym,
            "Name": r[h.get("Security Name", 1)],
            "Listing": exch,
            "RawMarketCategory": "",
            "ETF": as_bool(r[h.get("ETF", 4)]),
            "Source": "otherlisted.txt",
        })

    # De-dupe (prefer rows with Name/Listing present)
    dedup: Dict[str, Dict[str, Any]] = {}
    for row in out:
        s = row["Symbol"]
        if s in dedup:
            cur = dedup[s]
            if not cur.get("Name") and row.get("Name"): cur["Name"] = row["Name"]
            if not cur.get("Listing") and row.get("Listing"): cur["Listing"] = row["Listing"]
            cur["ETF"] = cur.get("ETF", False) or row.get("ETF", False)
        else:
            dedup[s] = row
    return list(dedup.values())

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Build tickers_all.csv (US equities + optional futures/forex/crypto).")
    ap.add_argument("--out", default=OUT_CSV_DEFAULT)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--no-units-rights-warrants", action="store_true",
                    help="Drop UNIT/RIGHTS/WARRANT tickers from the equity universe.")
    ap.add_argument("--include-futures", action="store_true")
    ap.add_argument("--include-forex", action="store_true")
    ap.add_argument("--include-crypto", action="store_true")
    ap.add_argument("--futures-seed", default="futures_seed.txt",
                    help="Optional text file with one Yahoo futures symbol per line (e.g., CL=F).")
    ap.add_argument("--forex-seed", default="",
                    help="Optional text file with one Yahoo forex symbol per line (e.g., EURUSD=X). If omitted, majors are auto-generated.")
    ap.add_argument("--crypto-seed", default="crypto_seed.txt",
                    help="Optional text file with one Yahoo crypto symbol per line (e.g., BTC-USD).")
    ap.add_argument("--ensure", type=str, default="",
                    help="Comma-separated symbols to force-include (e.g., ATCH,NVDA,ES=F,EURUSD=X)")
    ap.add_argument("--use-overrides", action="store_true",
                    help=f"Merge {OVERRIDES_CSV} (single column 'Symbol') after all sources.")
    args = ap.parse_args()

    log = setup_logging(args.verbose)

    # 1) US-listed equities (NASDAQ + NYSE family)
    equities = load_us_equities(log, allow_units_rights_warrants=(not args.no_units_rights_warrants))

    # 2) Start rows with equities
    rows: Dict[str, Dict[str, Any]] = {e["Symbol"]: e for e in equities}

    # 3) Optional add-ons (Futures / FX / Crypto)
    if args.include_futures:
        fut_seed = load_seed_list(Path(args.futures_seed))
        futs = fut_seed if fut_seed else BUILTIN_FUTURES
        add_seed_rows(rows, futs, "FUT", "futures_seed" if fut_seed else "futures_builtin")

    if args.include_forex:
        if args.forex_seed:
            fx = load_seed_list(Path(args.forex_seed))
        else:
            # Autogenerate conventional majors/crosses; prefer base!=USD to avoid USDEUR=X etc.
            pairs: List[str] = []
            for a in FOREX_MAJORS:
                for b in FOREX_MAJORS:
                    if a == b: continue
                    if a == "USD":  # prefer EURUSD=X not USDEUR=X
                        continue
                    pairs.append(f"{a}{b}=X")
            fx = sorted(set(pairs))
        add_seed_rows(rows, fx, "FX", "fx_seed" if args.forex_seed else "fx_autogen")

    if args.include_crypto:
        cr_seed = load_seed_list(Path(args.crypto_seed))
        cr = cr_seed if cr_seed else BUILTIN_CRYPTO
        add_seed_rows(rows, cr, "CRYPTO", "crypto_seed" if cr_seed else "crypto_builtin")

    # 4) Force-include ad-hoc tickers
    ensured = [normalize_symbol(s) for s in args.ensure.split(",") if s.strip()]
    if ensured:
        add_seed_rows(rows, ensured, "ENSURE", "ensure")
        log.info("Force-included: %s", ", ".join(sorted(ensured)))

    # 5) Optional overrides file merged last
    if args.use_overrides:
        extra = load_csv_single_col(Path(OVERRIDES_CSV))
        if extra:
            add_seed_rows(rows, extra, "OVERRIDE", "universe_overrides.csv")
            log.info("Merged overrides: %d symbols from %s", len(extra), OVERRIDES_CSV)
        else:
            log.info("No overrides loaded (missing or empty %s).", OVERRIDES_CSV)

    # Final skip for known test patterns
    for s in list(rows.keys()):
        if should_skip_symbol(s):
            rows.pop(s, None)

    # 6) Write CSV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Symbol"])
        for sym in sorted(rows.keys()):
            w.writerow([sym])

    eq_cnt = sum(1 for v in rows.values() if v.get("Source") in ("nasdaqlisted.txt","otherlisted.txt"))
    log.info("Wrote %s with %d symbols (US equities=%d, extras=%d).",
             out_path, len(rows), eq_cnt, len(rows)-eq_cnt)

if __name__ == "__main__":
    main()
