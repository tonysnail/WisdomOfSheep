"""Shared data models and helper utilities for council LLM stages."""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union, Literal

import pandas as pd
from pydantic import BaseModel, Field, field_validator, conint, confloat

# ---------------------------------------------------------------------------
# Shared enums and schemas
# ---------------------------------------------------------------------------

TimeHint = Literal["intraday", "swing_days", "swing_weeks", "multi_months", "long_term", "uncertain"]
Stance = Literal["bullish", "bearish", "neutral", "uncertain"]
ClaimType = Literal["valuation", "liquidity", "project_status", "macro_theme", "performance", "other"]
VerdictStatus = Literal["supported", "refuted", "mixed", "insufficient"]
Direction = Literal["up", "down", "none", "uncertain"]
SuitableFor = Literal["scalp", "swing", "position", "avoid"]
Action = Literal["no_trade", "monitor", "paper_trade", "consider_small_size"]

GLOBAL_RULE = (
    'Return STRICT JSON ONLY. Add a final string field named "why" at the end of your JSON object '
    'and at the end of any scored sub-objects, with 1–3 sentences explaining your choices.'
    'Hard rule: Every object in arrays must include a non-empty "why" field with 1–3 sentences. If any "why" is empty or missing, your output is invalid.'
)

_TIMEFRAME_ALLOWED = ["intraday", "swing_days", "swing_weeks", "multi_months", "long_term", "uncertain"]
_STANCE_ALLOWED = ["bullish", "bearish", "neutral", "uncertain"]
_VERDICT_ALLOWED = ["supported", "refuted", "mixed", "insufficient"]
_DIRECTION_ALLOWED = ["up", "down", "none", "uncertain"]

_TIMEFRAME_MAP = {
    "intraday": "intraday",
    "daytrade": "intraday",
    "daytrading": "intraday",
    "swing": "swing_days",
    "swingdays": "swing_days",
    "swingweeks": "swing_weeks",
    "weekswing": "swing_weeks",
    "multimonth": "multi_months",
    "multimonths": "multi_months",
    "longterm": "long_term",
    "longrun": "long_term",
}

_DIRECTION_MAP = {
    "up": "up",
    "bull": "up",
    "bullish": "up",
    "green": "up",
    "down": "down",
    "bear": "down",
    "bearish": "down",
    "red": "down",
    "flat": "none",
    "sideways": "none",
    "rangebound": "none",
    "uncertain": "uncertain",
}

# ---------------------------------------------------------------------------
# Data models shared across stages
# ---------------------------------------------------------------------------


class AssetRef(BaseModel):
    ticker: Optional[str] = None
    market: Optional[str] = None


class EntityTimeframeOut(BaseModel):
    assets: List[AssetRef] = Field(default_factory=list)
    time_hint: TimeHint
    uncertainty: conint(ge=0, le=3) = 1
    why: str

    @field_validator("assets", mode="before")
    @classmethod
    def _norm_assets(cls, v):
        return v or []


class AssetMention(BaseModel):
    ticker: Optional[str] = None
    name_or_description: str
    exchange_or_market: Optional[str] = None


class NumberMention(BaseModel):
    label: str
    value: str
    unit: Optional[str] = None


class SummariserSpamOut(BaseModel):
    spam_likelihood_pct: conint(ge=0, le=100)
    # Only required/meaningful if spam_likelihood_pct >= 40
    why: str = ""


class QualityFlags(BaseModel):
    repetition_or_template: bool = False
    vague_claims: bool = False
    mentions_spread_or_liquidity: bool = False


class SummariserOut(BaseModel):
    summary_bullets: List[str]
    assets_mentioned: List[AssetMention]
    claimed_catalysts: List[str] = Field(default_factory=list)
    claimed_risks: List[str] = Field(default_factory=list)
    numbers_mentioned: List[NumberMention] = Field(default_factory=list)
    author_stance: Stance
    quality_flags: QualityFlags

    # NEW: populated by the follow-up call (not the first prompt)
    spam_likelihood_pct: conint(ge=0, le=100) = 0
    spam_why: str = ""

    why: str


class Claim(BaseModel):
    id: str
    text: str
    type: ClaimType
    entity: Optional[str] = None
    why: str


class ClaimsOut(BaseModel):
    claims: List[Claim]
    why: str


class Citation(BaseModel):
    source: Literal["reddit", "rss", "stocktwits", "x"]
    title: str
    url: str
    published_at: Optional[str] = None
    snippet: str


class Verdict(BaseModel):
    id: str
    status: VerdictStatus
    confidence: confloat(ge=0.0, le=1.0) = 0.0
    reason: str
    citations: List[Citation] = Field(default_factory=list)
    why: str


class VerifierOut(BaseModel):
    verdicts: List[Verdict]
    overall_notes: List[str] = Field(default_factory=list)
    why: str


class ContextOut(BaseModel):
    context_bullets: List[str] = Field(default_factory=list)
    relevant_history: List[str] = Field(default_factory=list)
    comparables_or_benchmarks: List[str] = Field(default_factory=list)
    stale_risk_level: conint(ge=0, le=3) = 2
    watchouts: List[str] = Field(default_factory=list)
    why: str


class SetupQuality(BaseModel):
    evidence_specificity: conint(ge=0, le=3)
    timeliness: conint(ge=0, le=3)
    edge_vs_consensus: conint(ge=0, le=3)
    why: str


class ForOut(BaseModel):
    bull_points: List[str] = Field(default_factory=list)
    implied_catalysts: List[str] = Field(default_factory=list)
    setup_quality: SetupQuality
    what_would_improve: List[str] = Field(default_factory=list)
    why: str


class LiquidityConcerns(BaseModel):
    mentioned: bool = False
    details: Optional[str] = None
    why: str


class AgainstOut(BaseModel):
    bear_points: List[str] = Field(default_factory=list)
    red_flags: List[str] = Field(default_factory=list)
    data_gaps: List[str] = Field(default_factory=list)
    liquidity_concerns: LiquidityConcerns
    why: str


class Tradability(BaseModel):
    suitable_for: List[SuitableFor]
    blocking_issues: List[str] = Field(default_factory=list)
    why: str


class DirectionOut(BaseModel):
    implied_direction: Direction
    timeframe: TimeHint
    strength: conint(ge=0, le=3)
    tradability: Tradability
    why: str


# ---------------------------------------------------------------------------
# Normalisation helpers shared across stages
# ---------------------------------------------------------------------------

_TICKER_DB: Dict[str, Dict[str, Any]] = {}
_TICKER_DB_READY = False
TICKER_CSV_PATH = os.environ.get("WOS_TICKERS_CSV", "tickers/tickers_enriched.csv")

_CASHTAG_RE = re.compile(r"\$([A-Za-z]{1,5})(?![A-Za-z0-9])")
_ISIN_RE = re.compile(r"\b[A-Z]{2}[A-Z0-9]{9}\d\b")
_LEVERAGE_RE = re.compile(r"(?i)\b(long|short)\b.*?\b(?:factor|x)\s*([1-9]\d?)")
_OPT_RE = re.compile(r"\b(\d{1,2}/\d{1,2}/\d{2,4})\s*\$?\s*(\d+(?:\.\d+)?)\s*(call|put)s?\b", re.I)
_NUM_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
_MAG = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}


def _lazy_load_ticker_db() -> None:
    global _TICKER_DB_READY, _TICKER_DB
    if _TICKER_DB_READY:
        return
    try:
        if os.path.exists(TICKER_CSV_PATH):
            df = pd.read_csv(TICKER_CSV_PATH, dtype=str).fillna("")
            for _, row in df.iterrows():
                sym = str(row.get("Symbol", "")).strip().upper()
                if not sym:
                    continue
                _TICKER_DB[sym] = {
                    "longName": row.get("longName", ""),
                    "exchange": row.get("exchange", "")
                    or row.get("fullExchangeName", "")
                    or row.get("market", ""),
                    "market": row.get("market", ""),
                    "sector": row.get("sector", ""),
                    "industry": row.get("industry", ""),
                    "asset_type": row.get("asset_type", ""),
                    "themes": row.get("themes", ""),
                    "themes_primary": row.get("themes_primary", ""),
                }
        _TICKER_DB_READY = True
    except Exception:
        _TICKER_DB_READY = True


def lookup_ticker_meta(tkr: str) -> Optional[Dict[str, Any]]:
    if not tkr:
        return None
    _lazy_load_ticker_db()
    return _TICKER_DB.get(tkr.strip().upper())


def extract_cashtags(text: str) -> List[str]:
    return [m.group(1).upper() for m in _CASHTAG_RE.finditer(text or "")]


def looks_like_isin(s: str) -> bool:
    return bool(_ISIN_RE.fullmatch((s or "").strip().upper()))


def find_isins(text: str) -> List[str]:
    return [m.group(0).upper() for m in _ISIN_RE.finditer(text or "")]


def extract_leverage(text: str) -> Tuple[Optional[str], Optional[int]]:
    match = _LEVERAGE_RE.search(text or "")
    if not match:
        return (None, None)
    side = match.group(1).lower()
    try:
        factor = int(match.group(2))
    except Exception:
        factor = None
    return (side, factor)


def lower_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k).lower(): lower_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [lower_keys(x) for x in obj]
    return obj


def nz(val: Any) -> str:
    return (val or "").strip() if isinstance(val, str) else ("" if val is None else str(val).strip())


def coerce_reason_list(value: Any) -> List[str]:
    """Coerce the LLM produced bullet list into a list of strings.

    Newer prompts require every array item to include a ``why`` field which means
    the model now returns objects like ``{"claim": ..., "why": ...}``.  The
    historical data model, however, still expects plain strings.  This helper
    squeezes the richer structure back into strings so downstream code keeps
    working while we capture the justification inline.
    """

    if not isinstance(value, list):
        return []

    coerced: List[str] = []
    for item in value:
        if isinstance(item, str):
            token = item.strip()
            if token:
                coerced.append(token)
            continue

        if isinstance(item, dict):
            claim_keys = [
                "claim",
                "point",
                "text",
                "statement",
                "summary",
                "detail",
                "reason",
            ]
            claim = ""
            for key in claim_keys:
                claim = nz(item.get(key))
                if claim:
                    break
            why = nz(item.get("why"))

            # Fall back to the why text if we did not get a claim like field.
            if not claim and why:
                claim = why
                why = ""

            if claim:
                text = claim
                if why and why.lower() != claim.lower():
                    text = f"{text} — WHY: {why}"
                coerced.append(text)
            continue

        # Unknown item type – coerce its repr just so we do not lose signal.
        coerced.append(str(item))

    return coerced


def norm_ticker(t: Optional[str]) -> Optional[str]:
    token = nz(t)
    if not token:
        return None
    token = re.sub(r"[^A-Za-z0-9\.\-]", "", token)
    return token.upper() if token else None


def norm_exchange(mkt: Optional[str]) -> Optional[str]:
    market = nz(mkt).upper()
    if not market:
        return None
    if market in {"N/A", "NA", "NONE", "UNKNOWN", "UNK", "NULL"}:
        return None
    if market in {"NASDAQ", "NASDQ", "NAS"}:
        return "NASDAQ"
    if market in {"NYSE", "NEW YORK STOCK EXCHANGE"}:
        return "NYSE"
    if market in {"OTC", "OTCBB", "PINK"}:
        return "OTC"
    return market


def looks_like_us_equity(t: Optional[str]) -> bool:
    return bool(re.fullmatch(r"[A-Z]{1,5}(?:\.[A-Z]{1,3})?", (t or "").strip()))


def ensure_why(block: Dict[str, Any], default_note: str) -> None:
    if "why" not in block or not isinstance(block["why"], str) or not block["why"].strip():
        block["why"] = default_note


def ensure_top_level_why(block: Dict[str, Any], fallback: str = "auto-fixed: missing why") -> None:
    if not isinstance(block.get("why"), str) or not block["why"].strip():
        block["why"] = fallback


def first_str(value: Any) -> Optional[str]:
    if isinstance(value, list):
        if not value:
            return None
        value = value[0]
    return str(value) if value is not None else None


def coerce_enum(val: Any, allowed: Union[List[str], set], default: str, mapping: Optional[Dict[str, str]] = None) -> str:
    token = first_str(val)
    token = re.sub(r"[\s_\-]+", "", (token or "").strip().lower())
    if mapping and token in mapping:
        token = mapping[token]
    return token if token in allowed else default


def coerce_assets(val: Any) -> List[Dict[str, Any]]:
    def _parse_symbol(raw: str) -> Tuple[Optional[str], Optional[str]]:
        raw = (raw or "").strip().lstrip("$")
        exchange = None
        symbol = raw

        if symbol and looks_like_isin(symbol):
            return (None, None)

        if ":" in raw:
            left, right = raw.split(":", 1)
            exchange = left.strip().upper()
            symbol = right.strip()
        if "." in symbol and len(symbol.rsplit(".", 1)[-1]) <= 3:
            base, suff = symbol.rsplit(".", 1)
            symbol = base
            exchange = exchange or suff.upper()
        symbol = symbol.replace(" ", "").upper()
        return (symbol if symbol else None, exchange)

    output: List[Dict[str, Any]] = []
    if isinstance(val, list):
        for item in val:
            if isinstance(item, dict):
                ticker = item.get("ticker") or item.get("symbol") or item.get("tkr") or item.get("name") or ""
                market = item.get("market")
                symbol, exchange = _parse_symbol(str(ticker))
                output.append(
                    {
                        "ticker": norm_ticker(symbol),
                        "market": norm_exchange((market or exchange) if isinstance(market, str) or exchange else None),
                    }
                )
            elif isinstance(item, str):
                symbol, exchange = _parse_symbol(item)
                output.append({"ticker": norm_ticker(symbol), "market": norm_exchange(exchange)})
        return output
    if isinstance(val, str):
        symbol, exchange = _parse_symbol(val)
        return [{"ticker": norm_ticker(symbol), "market": norm_exchange(exchange)}]
    return []


def coerce_number_like(value: Any) -> str:
    if not isinstance(value, str):
        value = str(value or "")
    match = _NUM_RE.search(value.replace("’", "'"))
    if not match:
        return value.strip()
    return match.group(0).replace(",", "")


def expand_magnitude(token: Any) -> str:
    if not isinstance(token, str):
        token = str(token or "")
    exact = re.fullmatch(r"(?i)\s*(\d+(?:\.\d+)?)\s*([KMB])\s*", token)
    if exact:
        val = float(exact.group(1))
        mul = _MAG[exact.group(2).upper()]
        result = val * mul
        return str(int(result)) if result.is_integer() else str(result)

    in_phrase = re.search(r"(?i)(\d+(?:\.\d+)?)\s*([KMB])", token)
    if not in_phrase:
        return token.strip()
    val = float(in_phrase.group(1))
    mul = _MAG[in_phrase.group(2).upper()]
    expanded = val * mul
    replacement = str(int(expanded)) if expanded.is_integer() else str(expanded)
    return token[: in_phrase.start()] + replacement + token[in_phrase.end():]


def split_percent_value(value: str, unit: Optional[str]) -> Tuple[str, Optional[str]]:
    token = nz(value)
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*%", token)
    if match:
        return (match.group(1), "%")
    if token.endswith("%"):
        return (token[:-1].strip(), "%")
    return (token, unit if isinstance(unit, str) else None)


def normalize_summariser_spam(js: dict) -> dict:
    data = lower_keys(js if isinstance(js, dict) else {})
    try:
        pct = int(float(data.get("spam_likelihood_pct", 0)))
    except Exception:
        pct = 0
    pct = max(0, min(100, pct))
    reason = data.get("why")
    if not isinstance(reason, str):
        reason = ""
    return {"spam_likelihood_pct": pct, "why": (reason or "").strip()}


def preclean_post_text(text: str) -> str:
    token = re.sub(r"[ \t]+", " ", (text or "")).strip()
    token = re.sub(r"\n{3,}", "\n\n", token)

    paragraphs = [p.strip() for p in token.split("\n\n") if p.strip()]
    seen: set[str] = set()
    output: List[str] = []
    for paragraph in paragraphs:
        h = hashlib.md5(paragraph.encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            output.append(paragraph)
    return "\n\n".join(output)


__all__ = [
    "GLOBAL_RULE",
    "TimeHint",
    "Stance",
    "ClaimType",
    "VerdictStatus",
    "Direction",
    "SuitableFor",
    "Action",
    "AssetRef",
    "EntityTimeframeOut",
    "AssetMention",
    "NumberMention",
    "SummariserSpamOut",
    "SummariserOut",
    "QualityFlags",
    "Claim",
    "ClaimsOut",
    "Citation",
    "Verdict",
    "VerifierOut",
    "ContextOut",
    "SetupQuality",
    "ForOut",
    "LiquidityConcerns",
    "AgainstOut",
    "Tradability",
    "DirectionOut",
    "TICKER_CSV_PATH",
    "lookup_ticker_meta",
    "extract_cashtags",
    "find_isins",
    "extract_leverage",
    "lower_keys",
    "nz",
    "norm_ticker",
    "norm_exchange",
    "looks_like_us_equity",
    "ensure_why",
    "ensure_top_level_why",
    "first_str",
    "coerce_enum",
    "coerce_assets",
    "coerce_number_like",
    "expand_magnitude",
    "split_percent_value",
    "normalize_summariser_spam",
    "preclean_post_text",
    "_TIMEFRAME_ALLOWED",
    "_TIMEFRAME_MAP",
    "_STANCE_ALLOWED",
    "_VERDICT_ALLOWED",
    "_DIRECTION_ALLOWED",
    "_DIRECTION_MAP",
]
