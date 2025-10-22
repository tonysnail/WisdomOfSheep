#!/usr/bin/env python3
"""
Enrich a ticker list with sector/industry/themes + leaders, ETF mapping, universe filters,
and exchange metadata with regular trading hours (local + UTC/GMT).

Process for building Ticker encyclopaedia:
1> build_ticker_universe.py
2> [ enrich_tickers.py ]
3> ticker_deep_classify.py

Usage:

Just extract company names for tickers (no trading history to determine market leaders)
python enrich_tickers.py --verbose --no-history --allow-types EQUITY,ADR,ETF,FUTURES,FOREX,CRYPTO

python enrich_tickers.py \
  --verbose \
  --allow-types EQUITY,ADR,ETF,FUTURES,FOREX,CRYPTO \
  --chunk-hist 20 \
  --pause 1.0 \
  --retries 3 \
  --backoff 2.0


Reads file:   tickers_all.csv       ( created by 'build_ticker_universe.py' )
Outputs file: tickers_enriched.csv

Now with:
- Equities + ETFs + Futures (=F) + Forex (=X) + Crypto (-USD) support
- Better asset-type detection (quoteType + symbol pattern)
- 24/5 FX and 24/7 Crypto session handling; generic Futures session
- Verbose logging / CLI flags
- Skip known Yahoo test symbols (e.g., ZAZZT)
- OTC exchange mapping (PNK/OTC*)
- Tighter theme regexes + guardrails (e.g., 'silver' no longer matches Swiss 'AG')
- History fetch retries with exponential backoff and single-symbol info retry
"""

from __future__ import annotations
import argparse, logging, re, sys, json, time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, date, timezone
import numpy as np
import pandas as pd
import yfinance as yf

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    raise SystemExit("Python 3.9+ required for zoneinfo")

# ===================== Defaults =====================

INPUT_CSV_DEFAULT  = "tickers/tickers_all.csv"
OUT_ENRICH_DEFAULT = "tickers/tickers_enriched.csv"
OUT_SECTOR_DEFAULT = "tickers/sector_leaders.csv"
OUT_THEME_DEFAULT  = "tickers/theme_leaders.csv"
PATTERNS_JSON      = "tickers/patterns.json"     # optional: extend theme regexes
INFO_CACHE = Path(".cache/info_cache.parquet")

# ===================== ETF maps =====================

SECTOR_TO_ETF = {
    "Information Technology": "XLK", "Health Care": "XLV", "Financials": "XLF",
    "Consumer Discretionary": "XLY", "Consumer Staples": "XLP", "Energy": "XLE",
    "Industrials": "XLI", "Materials": "XLB", "Real Estate": "XLRE",
    "Communication Services": "XLC", "Utilities": "XLU",
}

THEME_TO_ETFS = {
    # Commodities / Mining
    "gold":["GLD","GDX","IAU","GDXJ"], "silver":["SLV","SIL"], "copper":["CPER","COPX"],
    "lithium":["LIT"], "uranium":["URA","URNM"], "steel":["SLX"], "aluminum":["JJU"], "coal":["KOL"],
    # Energy / Power
    "oil_gas":["XLE","XOP","IEO","VDE"], "renewables":["ICLN","TAN"], "solar":["TAN"], "nuclear":["NLR","URA"],
    "hydrogen":["HDRO"], "battery_storage":["LIT","BATT"],
    # IT stack
    "semiconductors":["SMH","SOXX","SOXQ","XSD"], "ai":["BOTZ","IRBO","QQQ"], "cloud":["WCLD","SKYY"],
    "software":["IGV","VGT"], "cybersecurity":["HACK","CIBR"], "hardware":["VGT"], "networking":["IGN"],
    # Financials
    "fintech":["FINX","IPAY"], "banks":["KBE","KRE","XLF"], "insurance":["KIE"], "asset_management":["XLF"],
    # Health Care
    "biotech":["XBI","IBB"], "pharma":["IHE","PJP"], "medtech":["IHI"],
    # Industrials / Transport
    "aerospace_defense":["ITA","DFEN"], "autos":["CARZ"], "ev":["DRIV","IDRV","KARS"], "airlines":["JETS"],
    "shipping":["SEA"], "rail":["XTN"],
    # Consumer / Media
    "retail":["XRT"], "ecommerce":["EBIZ","ONLN"], "media_streaming":["XLC"], "gaming":["HERO","ESPO"],
    # Real Estate / Utilities
    "reit":["VNQ","XLRE"], "utilities":["XLU"], "grid_infra":["PAVE"],
    # Materials / Ag
    "chemicals":["XLB"], "fertilizers":["MOO","SOIL"], "agriculture":["MOO"],
    # Misc
    "cannabis":["MSOS","MJ"], "crypto_blockchain":["BITQ","BLOK","WGMI"],
}

# ---------- Alias building for news text matching ----------

_CCY_NAME = {
    "USD":"US dollar","EUR":"Euro","JPY":"Japanese yen","GBP":"British pound",
    "AUD":"Australian dollar","NZD":"New Zealand dollar","CAD":"Canadian dollar","CHF":"Swiss franc",
    "CNY":"Chinese yuan","CNH":"Offshore yuan","MXN":"Mexican peso","BRL":"Brazilian real"
}

_FUTS_HUMAN = {
    "CL":"WTI crude oil", "BZ":"Brent crude oil", "NG":"natural gas", "HO":"heating oil", "RB":"RBOB gasoline",
    "GC":"gold", "SI":"silver", "HG":"copper", "PL":"platinum", "PA":"palladium", "ALI":"aluminum",
    "ZC":"corn", "ZW":"wheat", "ZS":"soybeans", "ZM":"soybean meal", "ZL":"soybean oil", "ZO":"oats",
    "KE":"KC wheat", "ZR":"rough rice",
    "KC":"coffee", "SB":"sugar", "CC":"cocoa", "CT":"cotton", "OJ":"orange juice",
    "ES":"S&P 500", "NQ":"Nasdaq 100", "YM":"Dow Jones", "RTY":"Russell 2000",
    "ZN":"10-year T-Note", "ZB":"30-year T-Bond", "ZF":"5-year T-Note", "ZT":"2-year T-Note"
}

# Map futures root → (sector, primary_theme)
_FUTS_CLASS = {
    # Energy
    "CL": ("Energy", "oil_gas"), "BZ": ("Energy", "oil_gas"), "NG": ("Energy", "oil_gas"),
    "HO": ("Energy", "oil_gas"), "RB": ("Energy", "oil_gas"),
    # Metals / Materials
    "GC": ("Materials", "gold"), "SI": ("Materials", "silver"), "HG": ("Materials", "copper"),
    "PL": ("Materials", "chemicals"), "PA": ("Materials", "chemicals"), "ALI": ("Materials", "aluminum"),
    # Ags
    "ZC": ("Materials", "agriculture"), "ZW": ("Materials", "agriculture"),
    "ZS": ("Materials", "agriculture"), "ZM": ("Materials", "agriculture"),
    "ZL": ("Materials", "agriculture"), "ZO": ("Materials", "agriculture"),
    "KE": ("Materials", "agriculture"), "ZR": ("Materials", "agriculture"),
    "KC": ("Materials", "agriculture"), "SB": ("Materials", "agriculture"),
    "CC": ("Materials", "agriculture"), "CT": ("Materials", "agriculture"),
    "OJ": ("Materials", "agriculture"),
    # Equity index
    "ES": (None, "index"), "NQ": (None, "index"), "YM": (None, "index"), "RTY": (None, "index"),
    # Rates
    "ZN": (None, "rates"), "ZB": (None, "rates"), "ZF": (None, "rates"), "ZT": (None, "rates"),
}

_CRYPTO_NAME = {
    "BTC":"Bitcoin","ETH":"Ethereum","SOL":"Solana","BNB":"BNB","ADA":"Cardano",
    "XRP":"XRP","DOGE":"Dogecoin","AVAX":"Avalanche","DOT":"Polkadot","ATOM":"Cosmos",
}

# ===================== Theme regex (tightened) =====================

THEME_PATTERNS = {
    # Commodities / Mining
    "gold":[r"\bgold\b", r"\bau\b", r"precious metal"],
    # IMPORTANT: removed r"\bag\b" to avoid Swiss 'AG' false-positives; added XAG code.
    "silver":[r"\bsilver\b", r"\bXAG\b"],
    "copper":[r"\bcopper\b", r"porphyry", r"\bcu\b"],
    "lithium":[r"\blithium\b", r"spodumene", r"sal\s*de\s*vida", r"brine"],
    "uranium":[r"\buranium\b", r"\bU3O8\b"],
    "steel":[r"\bsteel\b"],
    # IMPORTANT: removed overly broad "\bal\b"
    "aluminum":[r"\balumin(i)?um\b"],
    "coal":[r"\bcoal\b"],

    # Energy / Power
    "oil_gas":[r"\boil\b", r"brent", r"wti", r"natural gas", r"\bngl\b"],
    "renewables":[r"renewable", r"wind", r"solar", r"hydro", r"geothermal"],
    "solar":[r"\bsolar\b", r"photovoltaic", r"\bpv\b"],
    "nuclear":[r"\bnuclear\b", r"\breactor\b", r"\bSMR\b"],
    "hydrogen":[r"\bhydrogen\b", r"\bH2\b", r"electrolyzer"],
    "battery_storage":[r"battery", r"anode", r"cathode", r"lithium-ion", r"solid-state"],

    # IT stack
    "semiconductors":[r"\bsemi", r"wafer", r"foundry", r"fab", r"\bEDA\b", r"nvidia", r"\bamd\b", r"intel"],
    "ai":[r"\bAI\b", r"machine learning", r"deep learning", r"\bLLM\b", r"accelerator", r"\bGPU\b"],
    "cloud":[r"\bcloud\b", r"\bSaaS\b", r"hyperscaler", r"aws", r"azure", r"\bgcp\b"],
    "software":[r"\bsoftware\b", r"\bdevops\b", r"subscription", r"platform"],
    "cybersecurity":[r"cyber", r"endpoint", r"firewall", r"zero trust", r"\bSIEM\b", r"\bXDR\b"],
    "hardware":[r"\bhardware\b", r"peripheral", r"server", r"storage", r"\bPC\b", r"laptop"],
    "networking":[r"network", r"router", r"switch", r"5g", r"optical", r"datacenter"],

    # Financials
    "fintech":[r"fintech", r"payments", r"wallet", r"\bBNPL\b"],
    "banks":[r"\bbank\b", r"lender", r"credit union"],
    "insurance":[r"insurance", r"reinsurance"],
    "asset_management":[r"asset management", r"\bETF\b", r"brokerage"],

    # Health Care
    "biotech":[r"biotech", r"therapeutic", r"oncology", r"biopharma", r"gene therapy"],
    "pharma":[r"pharma", r"\bdrug\b", r"\bFDA\b", r"pipeline"],
    "medtech":[r"medical device", r"diagnostic", r"imaging", r"implant"],

    # Industrials / Transport
    "aerospace_defense":[r"aerospace", r"defense", r"missile", r"satellite"],
    "autos":[r"automotive", r"\bOEM\b", r"auto parts"],
    "ev":[r"\bEV\b", r"electric vehicle", r"charging"],
    # IMPORTANT: avoid generic 'carrier' to stop maritime names matching
    "airlines":[r"\bairline(s)?\b", r"\bair\s*carrier(s)?\b"],
    # IMPORTANT: narrow shipping to maritime terms; avoid packaging 'containers'
    "shipping":[
        r"\bshipping\b",
        r"\bcontainerships?\b", r"\bcontainer\s*ship(s)?\b",
        r"\btanker(s)?\b", r"\bdry\s*bulk\b", r"\bbaltic\s*dry\b",
        r"\bcharter(er|ing)?\b", r"\bfreight\s*rate(s)?\b"
    ],
    "rail":[r"\brail\b", r"railroad", r"railway"],

    # Consumer / Media
    "retail":[r"retail", r"department store", r"apparel"],
    "ecommerce":[r"e-?commerce", r"marketplace", r"online retail"],
    "media_streaming":[r"streaming", r"\bOTT\b", r"content platform"],
    "gaming":[r"videogame", r"publisher", r"esports"],

    # Real Estate / Utilities
    "reit":[r"\bREIT\b", r"real estate investment trust"],
    "utilities":[r"utility", r"electric power", r"water utility", r"gas utility"],
    "grid_infra":[r"transmission", r"distribution", r"substation", r"grid", r"inverter"],

    # Materials / Ag
    "chemicals":[r"chemical", r"specialty chemicals"],
    "fertilizers":[r"fertilizer", r"potash", r"phosphate", r"ammonia", r"\bNPK\b"],
    "agriculture":[r"agricultur", r"farm", r"seed", r"crop", r"tractor", r"agtech"],

    # Misc
    "cannabis":[r"cannabis", r"marijuana", r"\bTHC\b", r"\bCBD\b"],
    "crypto_blockchain":[r"crypto", r"bitcoin", r"ethereum", r"blockchain", r"\bweb3\b"],
}

# ===================== Exchange map (incl. OTC) =====================

EXCHANGE_META: Dict[str, Dict[str, Any]] = {
    # US
    "NMS":{"market":"NASDAQ","tz":"America/New_York","days":"Mon-Fri","open":"09:30","close":"16:00"},
    "NASDAQ":{"market":"NASDAQ","tz":"America/New_York","days":"Mon-Fri","open":"09:30","close":"16:00"},
    "NYQ":{"market":"NYSE","tz":"America/New_York","days":"Mon-Fri","open":"09:30","close":"16:00"},
    "NYSE":{"market":"NYSE","tz":"America/New_York","days":"Mon-Fri","open":"09:30","close":"16:00"},
    "ASE":{"market":"NYSE American","tz":"America/New_York","days":"Mon-Fri","open":"09:30","close":"16:00"},
    "AMEX":{"market":"NYSE American","tz":"America/New_York","days":"Mon-Fri","open":"09:30","close":"16:00"},
    "PCX":{"market":"NYSE Arca","tz":"America/New_York","days":"Mon-Fri","open":"09:30","close":"16:00"},
    "BATS":{"market":"Cboe BZX","tz":"America/New_York","days":"Mon-Fri","open":"09:30","close":"16:00"},
    # OTC (added)
    "PNK":{"market":"OTC Pink","tz":"America/New_York","days":"Mon-Fri","open":"09:30","close":"16:00"},
    "OTC":{"market":"OTC Markets","tz":"America/New_York","days":"Mon-Fri","open":"09:30","close":"16:00"},
    "OTCQB":{"market":"OTCQB","tz":"America/New_York","days":"Mon-Fri","open":"09:30","close":"16:00"},
    "OTCQX":{"market":"OTCQX","tz":"America/New_York","days":"Mon-Fri","open":"09:30","close":"16:00"},
    # UK
    "LSE":{"market":"LSE","tz":"Europe/London","days":"Mon-Fri","open":"08:00","close":"16:30"},
    # Eurozone
    "XETRA":{"market":"XETRA","tz":"Europe/Berlin","days":"Mon-Fri","open":"09:00","close":"17:30"},
    "FRA":{"market":"Frankfurt","tz":"Europe/Berlin","days":"Mon-Fri","open":"08:00","close":"20:00"},
    "PAR":{"market":"Euronext Paris","tz":"Europe/Paris","days":"Mon-Fri","open":"09:00","close":"17:30"},
    "AMS":{"market":"Euronext Amsterdam","tz":"Europe/Amsterdam","days":"Mon-Fri","open":"09:00","close":"17:40"},
    "BRU":{"market":"Euronext Brussels","tz":"Europe/Brussels","days":"Mon-Fri","open":"09:00","close":"17:30"},
    "MIL":{"market":"Borsa Italiana","tz":"Europe/Rome","days":"Mon-Fri","open":"09:00","close":"17:30"},
    "MAD":{"market":"Bolsa de Madrid","tz":"Europe/Madrid","days":"Mon-Fri","open":"09:00","close":"17:30"},
    "SIX":{"market":"SIX Swiss","tz":"Europe/Zurich","days":"Mon-Fri","open":"09:00","close":"17:30"},
    # Canada
    "TOR":{"market":"TSX","tz":"America/Toronto","days":"Mon-Fri","open":"09:30","close":"16:00"},
    "TSX":{"market":"TSX","tz":"America/Toronto","days":"Mon-Fri","open":"09:30","close":"16:00"},
    "VAN":{"market":"TSX Venture","tz":"America/Toronto","days":"Mon-Fri","open":"09:30","close":"16:00"},
    "TSXV":{"market":"TSX Venture","tz":"America/Toronto","days":"Mon-Fri","open":"09:30","close":"16:00"},
    "NEO":{"market":"NEO","tz":"America/Toronto","days":"Mon-Fri","open":"09:30","close":"16:00"},
    # APAC
    "TSE":{"market":"Tokyo (TSE)","tz":"Asia/Tokyo","days":"Mon-Fri","open":"09:00","close":"15:00","split":["11:30","12:30"]},
    "JPX":{"market":"Tokyo (TSE)","tz":"Asia/Tokyo","days":"Mon-Fri","open":"09:00","close":"15:00","split":["11:30","12:30"]},
    "HKG":{"market":"Hong Kong (HKEX)","tz":"Asia/Hong_Kong","days":"Mon-Fri","open":"09:30","close":"16:00","split":["12:00","13:00"]},
    "ASX":{"market":"ASX","tz":"Australia/Sydney","days":"Mon-Fri","open":"10:00","close":"16:00"},
    "SHG":{"market":"Shanghai (SSE)","tz":"Asia/Shanghai","days":"Mon-Fri","open":"09:30","close":"15:00","split":["11:30","13:00"]},
    "SHE":{"market":"Shenzhen (SZSE)","tz":"Asia/Shanghai","days":"Mon-Fri","open":"09:30","close":"15:00","split":["11:30","13:00"]},
    "SES":{"market":"Singapore (SGX)","tz":"Asia/Singapore","days":"Mon-Fri","open":"09:00","close":"17:00"},
    "KSC":{"market":"Korea (KRX)","tz":"Asia/Seoul","days":"Mon-Fri","open":"09:00","close":"15:30"},
    "KOSDAQ":{"market":"KOSDAQ","tz":"Asia/Seoul","days":"Mon-Fri","open":"09:00","close":"15:30"},
    "TPE":{"market":"Taiwan (TWSE)","tz":"Asia/Taipei","days":"Mon-Fri","open":"09:00","close":"13:30"},
    # Others
    "SAO":{"market":"B3 (Brazil)","tz":"America/Sao_Paulo","days":"Mon-Fri","open":"10:00","close":"17:30"},
    "MEX":{"market":"BMV (Mexico)","tz":"America/Mexico_City","days":"Mon-Fri","open":"08:30","close":"15:00"},
    "JSE":{"market":"JSE (South Africa)","tz":"Africa/Johannesburg","days":"Mon-Fri","open":"09:00","close":"17:00"},
}

# Known Yahoo/Nasdaq test symbols (skip)
SKIP_SYMBOL_PATTERNS = [re.compile(r"^Z[A-Z]{3}T$")]  # e.g., ZAZZT, ZBZZT, ZCZZT

# Warrants/units/rights & SPAC detection
TICKER_SUFFIX_FLAGS = [(r"-?W$|/W$", "WARRANT"), (r"-?U$|/U$", "UNIT"), (r"-?R$|/R$", "RIGHTS")]
SPAC_HINTS = [r"\bacquisition\b", r"\bholdings\b", r"\bcapital\b", r"\bblank check\b"]

# ===================== Logging =====================

def setup_logging(verbose: bool, debug: bool):
    level = logging.DEBUG if debug else (logging.INFO if verbose else logging.WARNING)
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("yfinance").setLevel(logging.WARNING if not debug else logging.INFO)
    logging.getLogger("urllib3").setLevel(logging.WARNING if not debug else logging.INFO)
    return logging.getLogger("enrich")

log = logging.getLogger("enrich")

# ===================== Helpers =====================
def load_info_cache():
    if INFO_CACHE.exists():
        try: return pd.read_parquet(INFO_CACHE)
        except Exception: return pd.DataFrame(columns=["Symbol"])
    return pd.DataFrame(columns=["Symbol"])

def save_info_cache(df):
    INFO_CACHE.parent.mkdir(parents=True, exist_ok=True)
    try: df.to_parquet(INFO_CACHE, index=False)
    except Exception: pass

def normalize_symbol(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.strip()
    if s.startswith("$"): s = s[1:]
    return s.upper()

def should_skip_symbol(sym: str) -> bool:
    return any(p.match(sym) for p in SKIP_SYMBOL_PATTERNS)

def detect_special_symbol(sym: str) -> Optional[str]:
    for pat, label in TICKER_SUFFIX_FLAGS:
        if re.search(pat, sym): return label
    return None

def detect_spac(name: str) -> bool:
    if not name: return False
    return any(re.search(p, name.lower()) for p in SPAC_HINTS)

def asset_type_from_yahoo(quoteType: str, sym: str, longName: str) -> str:
    qt = (quoteType or "").lower()
    s = (sym or "").upper()

    # Direct types from Yahoo
    if qt in ("etf","mutualfund","closedendfund","index"):
        return "ETF"
    if qt in ("futures","future"):
        return "FUTURES"
    if qt in ("currency","forex"):
        return "FOREX"
    if qt in ("crypto","cryptocurrency"):
        return "CRYPTO"

    # Pattern fallbacks (Yahoo sometimes sets qt="EQUITY" for non-equities)
    if s.endswith("=F"):
        return "FUTURES"
    if s.endswith("=X"):
        return "FOREX"
    if "-" in s and s.endswith("-USD"):
        return "CRYPTO"

    sp = detect_special_symbol(s)
    if   sp == "WARRANT": return "WARRANT"
    elif sp == "UNIT":    return "UNIT"
    elif sp == "RIGHTS":  return "RIGHTS"
    if (longName or "").lower().find("adr") >= 0: return "ADR"
    if detect_spac(longName): return "SPAC?"
    return "EQUITY"

def load_theme_overrides() -> Dict[str, List[str]]:
    p = Path(PATTERNS_JSON)
    if not p.exists(): return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("Failed to parse patterns.json: %s", e)
        return {}

def compile_theme_lexicon() -> Dict[str, List[re.Pattern]]:
    lex = dict(THEME_PATTERNS)
    lex.update(load_theme_overrides())
    compiled = {k: [re.compile(p, re.IGNORECASE) for p in v] for k, v in lex.items()}
    log.debug("Compiled theme lexicon with %d themes", len(compiled))
    return compiled

def tag_themes(name: str, industry: str, lex: Dict[str, List[re.Pattern]]) -> List[str]:
    hay = f"{name or ''} | {industry or ''}".lower()
    tags = []
    for theme, pats in lex.items():
        if any(p.search(hay) for p in pats):
            tags.append(theme)
    # Guardrail: don't tag 'shipping' if it's clearly packaging/containers industry context
    if "shipping" in tags and ("packaging" in hay or "container" in (industry or "").lower()):
        tags = [t for t in tags if t != "shipping"]
    return sorted(set(tags))

def rankify(s: pd.Series, ascending=False) -> pd.Series:
    return s.rank(ascending=ascending, method="min", na_option="keep")

def _parse_hhmm(s: str) -> Tuple[int,int]:
    h, m = s.split(":"); return int(h), int(m)

# Special handling for non-exchange tickers
def special_exchange_for_symbol(sym: str) -> Optional[Dict[str, Any]]:
    s = (sym or "").upper()
    if s.endswith("=X"):
        # 24/5 FX (approx)
        return {"market":"FOREX","tz":"Etc/UTC","days":"Mon-Fri","open":"00:00","close":"23:59"}
    if "-" in s and s.endswith("-USD"):
        # 24/7 Crypto (approx)
        return {"market":"CRYPTO","tz":"Etc/UTC","days":"Mon-Sun","open":"00:00","close":"23:59"}
    if s.endswith("=F"):
        # Generic futures "session" (varies per contract; use US futures desk rhythm)
        return {"market":"FUTURES","tz":"America/New_York","days":"Sun-Fri","open":"18:00","close":"17:00"}
    return None

def resolve_exchange(info_exchange: str, full_exchange: str, sym: str = "") -> Dict[str, Any]:
    # First, handle special symbol classes
    sp = special_exchange_for_symbol(sym)
    if sp is not None:
        sp = dict(sp); sp["source_key"] = "SPECIAL"; return sp

    key = (info_exchange or "").upper()
    if key in EXCHANGE_META:
        meta = dict(EXCHANGE_META[key]); meta["source_key"] = key; return meta
    fx = (full_exchange or "").lower()
    for k, meta in EXCHANGE_META.items():
        label = (meta.get("market","") or "").lower()
        if label and label in fx:
            m = dict(meta); m["source_key"] = k; return m
    return {"market":"UNKNOWN","tz":"America/New_York","days":"Mon-Fri","open":"09:30","close":"16:00","source_key":"FALLBACK"}

def session_utc_for_date(tz_name: str, open_hhmm: str, close_hhmm: str, d: date) -> Tuple[str,str]:
    tz = ZoneInfo(tz_name)
    oh, om = _parse_hhmm(open_hhmm)
    ch, cm = _parse_hhmm(close_hhmm)
    local_open = datetime(d.year, d.month, d.day, oh, om, tzinfo=tz)
    local_close= datetime(d.year, d.month, d.day, ch, cm, tzinfo=tz)
    return local_open.astimezone(timezone.utc).strftime("%H:%M"), local_close.astimezone(timezone.utc).strftime("%H:%M")

def today_utc_session(meta: Dict[str,Any], ref_utc: Optional[datetime]=None) -> Tuple[str,str]:
    ref = ref_utc or datetime.now(timezone.utc)
    return session_utc_for_date(meta["tz"], meta["open"], meta["close"], ref.date())

# ---------- Yahoo symbol normalization & retries ----------

def infer_missing_fields(row: dict) -> dict:
    """Fill longName/shortName/sector/themes for non-equities when info is empty."""
    sym = (row.get("Symbol") or "").upper()
    at  = (row.get("asset_type") or "").upper()
    ln  = row.get("longName")
    sn  = row.get("shortName")
    sector = row.get("sector")
    industry = row.get("industry")
    themes = row.get("themes") or ""

    # FUTURES
    if at == "FUTURES" or sym.endswith("=F"):
        root = sym.split("=F")[0]
        human = _FUTS_HUMAN.get(root)
        if not ln and human: ln = f"{human} Futures"
        if not sn and human: sn = human
        sec, th = _FUTS_CLASS.get(root, (None, None))
        sector = sector or sec
        industry = industry or ""
        if th and th not in (themes or ""):
            themes = ", ".join([t for t in [themes, th] if t])

    # FOREX
    if at == "FOREX" or sym.endswith("=X"):
        pair = sym[:-2] if sym.endswith("=X") else sym
        if len(pair) >= 6:
            a, b = pair[:3], pair[3:6]
            if not ln:
                an = _CCY_NAME.get(a, a); bn = _CCY_NAME.get(b, b)
                ln = f"{an} / {bn}"
            if not sn: sn = f"{a}/{b}"
        sector = sector or None
        industry = industry or ""

    # CRYPTO
    if at == "CRYPTO" or "-USD" in sym:
        coin = sym.split("-USD")[0]
        if not ln:
            ln = _CRYPTO_NAME.get(coin, coin)
        if not sn:
            sn = coin
        # theme for crypto to help NLP linking
        if "crypto_blockchain" not in (themes or ""):
            themes = ", ".join([t for t in [themes, "crypto_blockchain"] if t])

    row["longName"] = ln
    row["shortName"] = sn
    row["sector"] = sector
    row["industry"] = industry
    row["themes"] = themes
    return row

_YAHOO_ALT_GENERATORS = []

def _init_yahoo_alt_generators():
    global _YAHOO_ALT_GENERATORS
    if _YAHOO_ALT_GENERATORS:
        return
    def gen_bloomberg_to_yahoo(sym):
        # e.g., BRK.B -> BRK-B ; GOOG.L style (we don't add .L here because we don't know venue)
        if "." in sym:
            base, suf = sym.split(".", 1)
            # Preferreds from ABR$D / ABR^D / ABR.PR.D -> ABR-PD
            if suf.upper() in list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
                # class shares like BRK.B
                return [f"{base}-{suf.upper()}"]
            # common vendor preferred formats
            if suf.upper() in ("PR", "PRA", "PRB", "PRC"):
                return [sym.replace(".PR", "-P")]
        return []
    def gen_preferreds(sym):
        # ABR$D -> ABR-PD ; ABR^D -> ABR-PD ; ABR.PR.D -> ABR-PD
        s = sym
        s = re.sub(r"\$", "-P", s)  # $D -> -PD
        s = re.sub(r"\^", "-P", s)  # ^D -> -PD
        s = re.sub(r"\.PR\.", "-P", s, flags=re.IGNORECASE)  # .PR.D -> -PD
        # If ended with a single letter after -P, keep it (ABR-PD)
        return [s] if s != sym else []
    def gen_units_warrants(sym):
        # AAM.U -> AAM-U ; AAM.W -> AAM-W ; ALT: sometimes -UN / -WS used on Yahoo
        out = []
        if sym.endswith(".U"):
            out += [sym[:-2] + "-U", sym[:-2] + "-UN"]
        if sym.endswith(".W"):
            out += [sym[:-2] + "-W", sym[:-2] + "-WS", sym[:-2] + "-WT"]
        return out
    _YAHOO_ALT_GENERATORS = [gen_preferreds, gen_units_warrants, gen_bloomberg_to_yahoo]

_init_yahoo_alt_generators()

def yahoo_symbol_alternatives(sym: str, limit: int = 5) -> List[str]:
    alts = []
    for gen in _YAHOO_ALT_GENERATORS:
        try:
            alts += gen(sym)
        except Exception:
            pass
    # Dedup & trim
    seen = set([sym])
    out = []
    for s in alts:
        if s and s not in seen:
            out.append(s); seen.add(s)
        if len(out) >= limit:
            break
    return out

def safe_yf_info_for_symbol(sym: str) -> Dict[str, Any]:
    """
    Try .info for the symbol; if 404, attempt a few Yahoo-style alternatives.
    """
    try:
        info = yf.Ticker(sym).info or {}
        if info: return info
    except Exception:
        pass
    for alt in yahoo_symbol_alternatives(sym):
        try:
            info = yf.Ticker(alt).info or {}
            if info:
                info["__resolvedSymbol"] = alt  # remember what worked
                return info
        except Exception:
            continue
    return {}

def _strip_company_suffixes(name: str) -> str:
    return re.sub(r"\b(inc(orporated)?|corp(oration)?|co(mpany)?|plc|sa|nv|ag|ltd|limited|s\.?a\.?)\b\.?,?", "", name, flags=re.I).strip()

def build_aliases(symbol: str, asset_type: str, long_name: Optional[str], short_name: Optional[str]) -> str:
    s = (symbol or "").upper()
    at = (asset_type or "").upper()
    aliases = set()

    # Always include raw symbol
    aliases.add(s)

    # Equities: names + cleaned variants
    if at in {"EQUITY","ADR","ETF","WARRANT","UNIT","RIGHTS","SPAC?"} or at == "":
        for nm in [long_name, short_name]:
            if nm and isinstance(nm, str):
                aliases.add(nm)
                aliases.add(_strip_company_suffixes(nm))
        # class shares
        if re.search(r"[-\.][ABCD]$", s):
            cl = s[-1]
            for nm in [long_name, short_name]:
                if nm:
                    aliases.add(f"{_strip_company_suffixes(nm)} Class {cl}")
        # preferred/units/warrants hints
        if at == "WARRANT" or s.endswith("-W") or s.endswith("-WS") or s.endswith(".W"):
            aliases.update({"warrant", "warrants"})
        if at == "UNIT" or s.endswith("-U") or s.endswith("-UN") or s.endswith(".U"):
            aliases.update({"unit", "units"})

    # Futures
    if at == "FUTURES" or s.endswith("=F"):
        root = s.split("=F")[0]
        human = _FUTS_HUMAN.get(root)
        if human:
            aliases.update({human, f"{human} futures", f"{human} price"})
        # generic forms
        aliases.update({f"{root} futures", f"{root} contract", f"{root} front-month"})
        # commodity words helpful in headlines (best-effort)
        if human in {"gold","silver","copper","platinum","palladium","aluminum"}:
            aliases.add(f"{human} spot")

    # Forex
    if at == "FOREX" or s.endswith("=X"):
        pair = s[:-2] if s.endswith("=X") else s
        if len(pair) >= 6:
            a, b = pair[:3], pair[3:6]
            aliases.update({pair, f"{a}/{b}", f"{a}{b}", f"{a}-{b}", f"{a} {b}"})
            # currency names
            a_name = _CCY_NAME.get(a, a); b_name = _CCY_NAME.get(b, b)
            aliases.update({f"{a_name} / {b_name}", f"{a_name}-{b_name}", f"{a_name} vs {b_name}"})

    # Crypto
    if at == "CRYPTO" or ("-USD" in s):
        coin = s.split("-USD")[0]
        # Best-effort name from long/short; else coin ticker
        if long_name: aliases.add(long_name)
        aliases.update({coin, f"{coin} price", f"{coin}/USD", f"{coin} USD", f"{coin}-USD"})
        # common names
        common = {"BTC":"Bitcoin","ETH":"Ethereum","SOL":"Solana","BNB":"BNB","ADA":"Cardano","XRP":"XRP","DOGE":"Dogecoin","AVAX":"Avalanche","DOT":"Polkadot","ATOM":"Cosmos"}
        if coin in common:
            nm = common[coin]; aliases.update({nm, f"{nm} price", f"{nm} to USD"})

    # Final clean-up
    cleaned = sorted({a.strip() for a in aliases if a and isinstance(a, str)})
    return ", ".join(cleaned)

# ===================== Data fetch =====================

def fetch_yf_info(tickers: List[str], chunk=40, pause=0.4) -> pd.DataFrame:
    rows = []
    n = len(tickers)
    t0 = time.time()
    for i in range(0, n, chunk):
        batch = tickers[i:i+chunk]
        log.info("Info fetch [%d/%d] symbols %d..%d", min(i+chunk, n), n, i+1, min(i+len(batch), n))
        b0 = time.time()
        tk = yf.Tickers(" ".join(batch))
        for sym, t in tk.tickers.items():
            info = {}
            try:
                info = t.info or {}
            except Exception:
                info = {}
            if not info:
                # our stronger per-symbol retry with transliteration
                info = safe_yf_info_for_symbol(sym)

            rows.append({
                "Symbol": sym,
                "resolvedSymbol": info.get("__resolvedSymbol"),  # NEW (may be None)
                "longName": info.get("longName"),
                "shortName": info.get("shortName"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "marketCap": info.get("marketCap"),
                "beta": info.get("beta"),
                "averageVolume": info.get("averageVolume"),
                "currency": info.get("currency"),
                "country": info.get("country"),
                "quoteType": info.get("quoteType"),
                "exchange": info.get("exchange"),
                "fullExchangeName": info.get("fullExchangeName"),
            })
        b1 = time.time()
        log.debug("   batch time: %.2fs", b1-b0)
        if pause>0 and i+chunk<n:
            time.sleep(pause)
    t1 = time.time()
    df = pd.DataFrame(rows).drop_duplicates(subset=["Symbol"])
    log.info("Info fetched: %d rows (%.2fs total)", len(df), t1-t0)
    return df


def fetch_history_metrics(tickers: List[str], period="1y", interval="1d",
                          chunk=20, pause=1.0, retries=3, backoff=2.0, jitter=0.4) -> pd.DataFrame:
    import random
    rows = []; n = len(tickers); t0 = time.time()
    i = 0
    while i < n:
        batch = tickers[i:i+chunk]
        log.info("Hist fetch  [%d/%d] symbols %d..%d", min(i+chunk, n), n, i+1, min(i+len(batch), n))
        attempt = 0
        while attempt <= retries:
            try:
                h = yf.download(batch, period=period, interval=interval,
                                auto_adjust=True, threads=True, group_by="ticker", progress=False)
                break
            except Exception as e:
                msg = str(e)
                if "Rate limited" in msg or "Too Many Requests" in msg:
                    sleep_s = (backoff ** attempt) + random.uniform(0, jitter)
                    attempt += 1
                    log.warning("Rate limited on batch starting %s: retry %d in %.1fs", batch[0], attempt, sleep_s)
                    time.sleep(sleep_s)
                    continue
                else:
                    log.error("Download error on batch starting %s: %s", batch[0], e)
                    h = pd.DataFrame(); break

        # add rows helper (same as before, minor None guards)
        def add_rows(sym_list, h):
            if h is None or h.empty:
                for s in sym_list: rows.append({"Symbol": s})
                return
            if isinstance(h.columns, pd.MultiIndex):
                roots = set(h.columns.levels[0])
                for s in sym_list:
                    if s not in roots:
                        rows.append({"Symbol": s}); continue
                    c = h[s].get("Close", pd.Series()).dropna()
                    v = h[s].get("Volume", pd.Series()).reindex_like(c).fillna(0)
                    if c.empty: rows.append({"Symbol": s}); continue
                    def tot_ret(days):
                        if len(c) < days+1: return None
                        return float(c.iloc[-1]/c.iloc[-days-1] - 1.0)
                    r3, r6, r12 = tot_ret(63), tot_ret(126), tot_ret(252)
                    last_px = float(c.iloc[-1]); adv = float(v.rolling(20).mean().iloc[-1] or 0.0)
                    rows.append({"Symbol": s, "ret_3m": r3, "ret_6m": r6, "ret_12m": r12,
                                 "last_close": last_px, "ADV_USD_20": (adv*last_px) if adv else None})
            else:
                s = sym_list[0]
                if h.empty or "Close" not in h:
                    rows.append({"Symbol": s}); return
                c = h["Close"].dropna(); v = h.get("Volume", pd.Series()).reindex_like(c).fillna(0)
                if c.empty: rows.append({"Symbol": s}); return
                def tot_ret(days):
                    if len(c) < days+1: return None
                    return float(c.iloc[-1]/c.iloc[-days-1] - 1.0)
                r3, r6, r12 = tot_ret(63), tot_ret(126), tot_ret(252)
                last_px = float(c.iloc[-1]); adv = float(v.rolling(20).mean().iloc[-1] or 0.0)
                rows.append({"Symbol": s, "ret_3m": r3, "ret_6m": r6, "ret_12m": r12,
                             "last_close": last_px, "ADV_USD_20": (adv*last_px) if adv else None})

        add_rows(batch, h)
        # polite pause with jitter
        time.sleep(pause + random.uniform(0, jitter))
        i += chunk

    df = pd.DataFrame(rows); log.info("Hist fetched: %d rows (%.2fs total)", len(df), time.time()-t0)
    return df

# ===================== Leaders & filtering =====================

def apply_universe_filters(df: pd.DataFrame, min_cap: int, min_adv: float, allow_types: set) -> pd.DataFrame:
    before = len(df)
    df = df.copy()
    df["asset_type"] = df["asset_type"].astype(str).str.upper()

    mask_type = df["asset_type"].isin(allow_types)

    equity_like = df["asset_type"].isin({"EQUITY","ADR","ETF"})
    non_equity  = ~equity_like

    # Equities: enforce cap + ADV
    cap_ok_equity = (df["marketCap"].fillna(0) >= min_cap)
    adv_ok_equity = (df["ADV_USD_20"].fillna(0) >= min_adv)

    # Non-equities (Futures/FX/Crypto): no marketCap; ensure we at least have a price
    price_ok = df["last_close"].notna()
    # Optional ADV floor for non-equities: keep if ADV missing (many FX/crypto return 0 volume on Yahoo)
    adv_ok_non_eq = (df["ADV_USD_20"].isna()) | (df["ADV_USD_20"] >= 0)

    mask_equity    = equity_like & cap_ok_equity & adv_ok_equity
    mask_non_equity= non_equity & price_ok & adv_ok_non_eq

    final_mask = mask_type & (mask_equity | mask_non_equity)

    dropped = int((~final_mask).sum())
    log.info("Universe filter: start=%d | kept=%d | dropped=%d", before, int(final_mask.sum()), dropped)

    d = df[final_mask].copy()
    log.info("Universe after filters: %d symbols", len(d))
    return d

def compute_sector_leaders(df: pd.DataFrame, n=12) -> pd.DataFrame:
    d = df.copy()
    for col in ("marketCap","ADV_USD_20","ret_3m","ret_6m","ret_12m"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d["r_mcap"] = rankify(d["marketCap"].fillna(0), ascending=False)
    d["r_liq"]  = rankify(d["ADV_USD_20"].fillna(0), ascending=False)
    for col in ("ret_3m","ret_6m","ret_12m"):
        d[f"r_{col}"] = rankify(d[col].fillna(-1), ascending=False)
    d["r_momo"] = (d["r_ret_3m"] + d["r_ret_6m"] + d["r_ret_12m"]) / 3.0
    d["leader_score"] = 0.4*d["r_mcap"] + 0.3*d["r_liq"] + 0.3*d["r_momo"]
    d["sector_proxy_etf"] = d["sector"].map(SECTOR_TO_ETF)
    leaders = (d.sort_values(["sector","leader_score"], ascending=[True, False])
                 .groupby("sector").head(n).reset_index(drop=True))
    for sector, g in leaders.groupby("sector"):
        top = g.head(3)[["Symbol","leader_score"]].to_dict("records")
        log.info("Sector leader preview %-24s → %s", sector or "Unknown", top)
    cols = ["sector","sector_proxy_etf","Symbol","longName","industry","asset_type",
            "ADV_USD_20","marketCap","ret_3m","ret_6m","ret_12m","leader_score",
            "themes","themes_primary","market","exchange_tz","open_local","close_local","open_utc","close_utc","trading_days","split_session"]
    return leaders[cols].sort_values(["sector","leader_score"], ascending=[True,False])

def compute_theme_leaders(df: pd.DataFrame, n=20) -> pd.DataFrame:
    d = df.copy()
    d["themes_list"] = d["themes"].apply(lambda s: [t.strip() for t in s.split(",") if t.strip()] if isinstance(s,str) and s else [])
    d = d.explode("themes_list").rename(columns={"themes_list":"theme"})
    d = d[d["theme"].notna() & (d["theme"]!="")]

    for col in ("marketCap","ADV_USD_20","ret_3m","ret_6m","ret_12m"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d["r_mcap"] = d.groupby("theme")["marketCap"].transform(lambda s: rankify(s.fillna(0), ascending=False))
    d["r_liq"]  = d.groupby("theme")["ADV_USD_20"].transform(lambda s: rankify(s.fillna(0), ascending=False))
    for col in ("ret_3m","ret_6m","ret_12m"):
        d[f"r_{col}"] = d.groupby("theme")[col].transform(lambda s: rankify(s.fillna(-1), ascending=False))
    d["r_momo"] = (d["r_ret_3m"] + d["r_ret_6m"] + d["r_ret_12m"]) / 3.0
    d["leader_score"] = 0.35*d["r_mcap"] + 0.35*d["r_liq"] + 0.30*d["r_momo"]
    d["theme_proxy_etfs"] = d["theme"].map(lambda t: ", ".join(THEME_TO_ETFS.get(t, [])))
    leaders = (d.sort_values(["theme","leader_score"], ascending=[True, False])
                 .groupby("theme").head(n).reset_index(drop=True))
    for theme, g in leaders.groupby("theme"):
        top = g.head(3)[["Symbol","leader_score"]].to_dict("records")
        log.info("Theme leader preview  %-24s → %s", theme, top)
    cols = ["theme","theme_proxy_etfs","Symbol","longName","sector","industry","asset_type",
            "ADV_USD_20","marketCap","ret_3m","ret_6m","ret_12m","leader_score",
            "market","exchange_tz","open_local","close_local","open_utc","close_utc","trading_days","split_session"]
    return leaders[cols].sort_values(["theme","leader_score"], ascending=[True,False])

# ===================== Main =====================

def main():
    ap = argparse.ArgumentParser(description="Verbose ticker enricher")
    ap.add_argument("--input", default=INPUT_CSV_DEFAULT)
    ap.add_argument("--out-enrich", default=OUT_ENRICH_DEFAULT)
    ap.add_argument("--out-sector", default=OUT_SECTOR_DEFAULT)
    ap.add_argument("--out-theme", default=OUT_THEME_DEFAULT)
    ap.add_argument("--min-cap", type=int, default=300_000_000)
    ap.add_argument("--min-adv", type=float, default=10_000_000)
    ap.add_argument("--allow-types", type=str, default="EQUITY,ADR,ETF,FUTURES,FOREX,CRYPTO")
    ap.add_argument("--chunk-info", type=int, default=40)
    ap.add_argument("--chunk-hist", type=int, default=60)
    ap.add_argument("--pause", type=float, default=0.4)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--backoff", type=float, default=2.0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--no-history", action="store_true", help="Skip history downloads (faster; aliases/info only).")
    ap.add_argument("--aliases-only", action="store_true", help="Only write Symbol+aliases (for NLP), skipping info/history.")
    ap.add_argument("--force-info", action="store_true", help="Ignore cache and refetch .info for all symbols.")
    args = ap.parse_args()

    global log
    log = setup_logging(args.verbose, args.debug)

    src = Path(args.input)
    if not src.exists():
        log.error("Input file not found: %s", src); sys.exit(1)

    raw = pd.read_csv(src).dropna(subset=["Symbol"])
    tickers_all = [normalize_symbol(s) for s in raw["Symbol"].tolist() if isinstance(s, str)]
    tickers = [t for t in sorted(set(tickers_all)) if not should_skip_symbol(t)]
    log.info("Loaded %d symbols from %s (dedup → %d). After skip-pattern filter: %d",
             len(raw), src.name, len(set(tickers_all)), len(tickers))

    if args.aliases_only:
        tmp = pd.DataFrame({
            "Symbol": tickers,
            "asset_type": ["" for _ in tickers],
            "aliases": [build_aliases(t, "", "", "") for t in tickers],
        })
        tmp.to_csv(args.out_enrich, index=False)
        log.info("Wrote %s (aliases-only, %d rows)", args.out_enrich, len(tmp))
        sys.exit(0)

    cache = load_info_cache()

    if args.force_info or cache.empty:
        log.info("Force info fetch (or empty cache): fetching info for all %d symbols", len(tickers))
        info = fetch_yf_info(tickers, chunk=args.chunk_info, pause=args.pause)
    else:
        cache_syms = set(cache["Symbol"].astype(str).str.upper().tolist())
        # Identify symbols fully missing OR present but with poor coverage (no names/sector/industry)
        need_cols = ["longName", "shortName", "sector", "industry", "quoteType", "exchange", "fullExchangeName"]
        cache_ok = cache.copy()
        for c in need_cols:
            if c not in cache_ok.columns:
                cache_ok[c] = np.nan
        mask_poor = cache_ok["Symbol"].isin(tickers) & cache_ok[need_cols].isna().all(axis=1)
        to_fetch = sorted(set(tickers) - cache_syms | set(cache_ok.loc[mask_poor, "Symbol"]))
        log.info("Cache has %d symbols; to_fetch=%d (missing or poor coverage)", len(cache_syms), len(to_fetch))

        new_info = fetch_yf_info(to_fetch, chunk=args.chunk_info, pause=args.pause) if to_fetch else pd.DataFrame(columns=["Symbol"])
        info = (pd.concat([cache, new_info], ignore_index=True)
                  .drop_duplicates(subset=["Symbol"], keep="last")
                  .reset_index(drop=True))

    save_info_cache(info)

    # Asset types & coverage
    info["asset_type"] = [asset_type_from_yahoo(qt, sym, nm)
                          for qt, sym, nm in zip(info["quoteType"], info["Symbol"], info["longName"])]
    info = info.to_dict("records")
    info = [infer_missing_fields(r) for r in info]
    info = pd.DataFrame(info)
    log.info("Asset types: %s", info["asset_type"].value_counts(dropna=False).to_dict())
    sec_cov = info["sector"].notna().mean(); ind_cov = info["industry"].notna().mean()
    log.info("Coverage: sector=%.1f%%, industry=%.1f%%", 100*sec_cov, 100*ind_cov)
    info["aliases"] = [
        build_aliases(sym, atype, ln, sn)
        for sym, atype, ln, sn in zip(info["Symbol"], info["asset_type"], info["longName"], info["shortName"])
    ]

    # Themes
    lex = compile_theme_lexicon()
    info["themes"] = [", ".join(tag_themes(nm or sn, ind or "", lex))
                      for nm, sn, ind in zip(info["longName"], info["shortName"], info["industry"])]
    info = info.to_dict("records")
    info = [infer_missing_fields(r) for r in info]
    info = pd.DataFrame(info)
    log.info("Theme tagging done. Non-empty theme rows: %d / %d", (info["themes"].astype(str)!="").sum(), len(info))

    # Exchange meta + UTC session
    markets, tzs, days, splits, opens_local, closes_local, opens_utc, closes_utc = ([] for _ in range(8))
    unknown_keys = set()
    for sym, ex, fx in zip(info["Symbol"], info.get("exchange", pd.Series([None]*len(info))),
                           info.get("fullExchangeName", pd.Series([None]*len(info)))):
        meta = resolve_exchange(ex, fx, sym)
        if meta["market"] == "UNKNOWN":
            unknown_keys.add((ex or "None", fx or "None"))
        markets.append(meta["market"]); tzs.append(meta["tz"]); days.append(meta["days"])
        splits.append(",".join(meta.get("split", [])) if meta.get("split") else "")
        opens_local.append(meta["open"]); closes_local.append(meta["close"])
        o,c = today_utc_session(meta); opens_utc.append(o); closes_utc.append(c)
    log.info("Exchange mapping done; unknown keys: %d", len(unknown_keys))
    if unknown_keys:
        for pair in sorted(list(unknown_keys))[:10]:
            log.warning("Unknown exchange mapping sample: exchange=%s fullExchange=%s", pair[0], pair[1])
        if len(unknown_keys) > 10:
            log.warning("... and %d more unmatched exchanges", len(unknown_keys)-10)

    info["market"] = markets; info["exchange_tz"] = tzs; info["trading_days"] = days
    info["split_session"] = splits; info["open_local"] = opens_local; info["close_local"] = closes_local
    info["open_utc"] = opens_utc; info["close_utc"] = closes_utc

    # History/performance
    perf = pd.DataFrame({"Symbol": info["Symbol"]})
    if not args.no_history:
        perf = fetch_history_metrics(info["Symbol"].tolist(), period="1y", interval="1d",
                                     chunk=args.chunk_hist, pause=args.pause,
                                     retries=args.retries, backoff=args.backoff)
    else:
        log.info("Skipping history due to --no-history")

    df = info.merge(perf, on="Symbol", how="left")
    # Ensure history columns exist even when --no-history
    for col in ["last_close", "ADV_USD_20", "ret_3m", "ret_6m", "ret_12m"]:
        if col not in df.columns:
            df[col] = np.nan

    # ADV fallback if needed (equities only — volume often absent for FX/CRYPTO)
    m = (df["ADV_USD_20"].isna() | (df["ADV_USD_20"] <= 0)) & df["asset_type"].isin(["EQUITY","ADR","ETF"])
    fallback_n = int(m.sum())
    if fallback_n:
        df.loc[m, "ADV_USD_20"] = (df.loc[m, "averageVolume"].fillna(0) * df.loc[m, "last_close"].fillna(0)).astype(float)
    log.info("ADV fallback applied to %d equity-like rows", fallback_n)

    # Primary themes (sector-biased sort; keep top 3)
    def primary_themes(row):
        themes = [t.strip() for t in str(row.get("themes","")).split(",") if t.strip()]
        if not themes: return ""
        sector = (row.get("sector") or "").lower()
        bias = {
            "information technology":{"ai","cloud","software","cybersecurity","semiconductors","hardware","networking"},
            "health care":{"biotech","pharma","medtech"},
            "energy":{"oil_gas","renewables","solar","nuclear","hydrogen","battery_storage"},
            "materials":{"steel","aluminum","chemicals","fertilizers","copper","lithium","gold","silver","uranium"},
        }
        w = []
        for t in themes:
            wt = 2 if any(k in sector and t in v for k,v in bias.items()) else 1
            w.append((t, wt))
        w.sort(key=lambda x:(-x[1], x[0]))
        return ", ".join([t for t,_ in w[:3]])
    df["themes_primary"] = df.apply(primary_themes, axis=1)

    # Save enriched BEFORE filters
    cols = ["Symbol","longName","sector","industry","asset_type","themes","themes_primary", "aliases",
            "market","exchange","fullExchangeName","exchange_tz","trading_days","split_session",
            "open_local","close_local","open_utc","close_utc",
            "marketCap","beta","country","currency","averageVolume","last_close","ADV_USD_20",
            "ret_3m","ret_6m","ret_12m"]
    enriched = df[cols].sort_values("Symbol").reset_index(drop=True)
    Path(args.out_enrich).parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(args.out_enrich, index=False)
    log.info("Wrote %s (%d rows)", args.out_enrich, len(enriched))

    # Universe filters (controls what leaders are built from)
    allow_types = set(t.strip().upper() for t in args.allow_types.split(",") if t.strip())
    uni = apply_universe_filters(enriched, args.min_cap, args.min_adv, allow_types)

    # Leaders
    sector_leaders = compute_sector_leaders(uni[uni["asset_type"].isin(["EQUITY","ADR","ETF"])], n=12)
    theme_leaders  = compute_theme_leaders(uni, n=20)
    sector_leaders.to_csv(args.out_sector, index=False)
    theme_leaders.to_csv(args.out_theme, index=False)
    log.info("Wrote %s (%d rows)", args.out_sector, len(sector_leaders))
    log.info("Wrote %s (%d rows)", args.out_theme, len(theme_leaders))

    # Final summaries
    log.info("Top asset types (universe): %s", uni["asset_type"].value_counts(dropna=False).to_dict())
    log.info("Top sectors by count (universe): %s", uni["sector"].value_counts(dropna=False).head(10).to_dict())
    log.info("Top markets by count (universe): %s", uni["market"].value_counts(dropna=False).head(10).to_dict())
    log.info("Done.")

if __name__ == "__main__":
    main()
