# round_table.py
# Wisdom of Sheep — Council + Pydantic v2 + Ollama + CSV test mode + verbose traces + JSON repair

""" 

Usage:

Run one stage, pulling text automatically by post_id:

python round_table.py stage --post-id t3_1nq4yev --stage summariser


Run legacy CLI dummy test: Random article:

python round_table.py --dummytest raw_posts_log.csv --random --model mistral --pretty --verbose --no-timeout



Run legacy CLI dummy test: Specific Article (post-id):

python round_table.py --dummytest raw_posts_log.csv --post-id t3_1nq4yev --model mistral --pretty --verbose --no-timeout

"""

from __future__ import annotations

import json
from json import JSONDecodeError
import re
import argparse
import random
import hashlib
import os
import csv
import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple, Callable, Union, Literal
from datetime import datetime, timezone, timedelta
import sqlite3, json, time
import pandas as pd
import requests
from pathlib import Path
from pydantic import BaseModel, Field, conint, confloat

from council.common import (
    Action,
    AgainstOut,
    AssetRef,
    Claim,
    ClaimType,
    ClaimsOut,
    ContextOut,
    Direction,
    DirectionOut,
    EntityTimeframeOut,
    ForOut,
    LiquidityConcerns,
    Tradability,
    NumberMention,
    SummariserOut,
    TimeHint,
    VerdictStatus,
    VerifierOut,
    lookup_ticker_meta,
    looks_like_us_equity,
    preclean_post_text,
)
from council.entity_stage import run_entity as run_entity_stage
from council.summariser_stage import run_summariser as run_summariser_stage
from council.claims_stage import run_claims as run_claims_stage
from council.verifier_stage import run_verifier as run_verifier_stage
from council.context_stage import run_context as run_context_stage
from council.bull_case_stage import run_bull_case as run_bull_stage
from council.bear_case_stage import run_bear_case as run_bear_stage
from council.direction_stage import run_direction as run_direction_stage
from council.chairman_stage import run_chairman_stage


# Path to Master Council Database - we write our output here, and read bullet points from previous news articles when needed.

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "council" / "wisdom_of_sheep.sql"

# ===== Trade Signal (Moderator) =====

class CouncilScores(BaseModel):
    information_quality: conint(ge=0, le=100)
    trade_clarity: conint(ge=0, le=100)
    evidence_strength: conint(ge=0, le=100)
    why: str


class LiquidityRisk(BaseModel):
    spread_pct: Optional[confloat(ge=0.0, le=1.0)] = None
    note: Optional[str] = None


class SourceRef(BaseModel):
    type: Literal["reddit", "rss", "stocktwits", "x"]
    post_id: Optional[str] = None
    url: Optional[str] = None


class AuditClaim(BaseModel):
    id: str
    text: str
    verdict: VerdictStatus
    why: str


class AuditBlock(BaseModel):
    post_bullets: List[str]
    claims: List[AuditClaim]
    context_used: List[str] = Field(default_factory=list)
    timestamps: Dict[str, str]
    why: str
    extras: Dict[str, Any] = Field(default_factory=dict)  # ← add this


class TradeSignal(BaseModel):
    signal_id: str
    source: SourceRef
    asset: AssetRef
    headline_summary: str
    direction: Direction
    timeframe: TimeHint
    confidence: confloat(ge=0.0, le=1.0)
    liquidity_risk: LiquidityRisk
    council_scores: CouncilScores
    rationale: List[str]
    blocking_issues: List[str]
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    audit: AuditBlock
    action: Action
    next_checks: List[str]
    why: str


# --- Date extraction from text (for sanity checks) ---
_MONTHS = ("january","february","march","april","may","june",
           "july","august","september","october","november","december")
_MON_ABBR = tuple(m[:3] for m in _MONTHS)

_DATE_SLASH_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b")
_DATE_SLASH_SHORT_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})\b")
_DATE_MONTHNAME_RE = re.compile(
    r"\b(?:(?:"
    + "|".join([re.escape(m) for m in _MONTHS + _MON_ABBR])
    + r"))\s+(\d{1,2})(?:,\s*(\d{4}))?\b",
    re.I
)
_YEAR_RE = re.compile(r"\b(20\d{2})\b")  # conservative: only 2000–2099

_DIRECTION_MAP = {
    "up": "up", "bull": "up", "bullish": "up", "green": "up",
    "down": "down", "bear": "down", "bearish": "down", "red": "down",
    "flat": "none", "sideways": "none", "rangebound": "none",
    "uncertain": "uncertain",
}

_HELP_TOKENS = (
    "what happened", "can someone explain", "i don't understand",
    "why did", "how did", "what did i do wrong", "help", "confused"
)


# ---- SQL helpers ----

def get_post_text(post_id: str) -> str:
    """Fetch post text for a given post_id from SQL (preferred) or fallback CSV."""
    import sqlite3, os, pandas as pd

    if DB_PATH.exists():
        with sqlite3.connect(str(DB_PATH)) as conn:
            cur = conn.cursor()
            row = cur.execute("SELECT text FROM posts WHERE post_id = ?", (post_id,)).fetchone()
            if row:
                return row[0]

    # fallback to raw_posts_log.csv
    if os.path.exists("raw_posts_log.csv"):
        df = pd.read_csv("raw_posts_log.csv")
        row = df.loc[df["post_id"] == post_id]
        if not row.empty:
            return str(row.iloc[0]["text"])

    return None

def _now_iso():
    import time
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())

def write_stage(post_id, stage, payload):
    con = sqlite3.connect(str(DB_PATH))
    con.execute("""
        INSERT INTO stages(post_id, stage, created_at, payload)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(post_id, stage) DO UPDATE SET
          created_at=excluded.created_at,
          payload=excluded.payload
    """, (post_id, stage, _now_iso(), json.dumps(payload, ensure_ascii=False)))
    con.commit()
    con.close()

def upsert_post(row):
    con = sqlite3.connect(str(DB_PATH))
    con.execute("""
        INSERT INTO posts(post_id, platform, source, url, title, author,
                          scraped_at, posted_at, score, text)
        VALUES (:post_id, :platform, :source, :url, :title, :author,
                :scraped_at, :posted_at, :score, :text)
        ON CONFLICT(post_id) DO UPDATE SET
          platform=excluded.platform,
          source=excluded.source,
          url=excluded.url,
          title=excluded.title,
          author=excluded.author,
          scraped_at=excluded.scraped_at,
          posted_at=excluded.posted_at,
          score=excluded.score,
          text=excluded.text;
    """, row)
    con.commit()
    con.close()

# ---- redex helpers ----
_CASHTAG_RE = re.compile(r"\$([A-Za-z]{1,5})(?![A-Za-z0-9])")
_ISIN_RE = re.compile(r"\b[A-Z]{2}[A-Z0-9]{9}\d\b")
_LEVERAGE_RE = re.compile(r"(?i)\b(long|short)\b.*?\b(?:factor|x)\s*([1-9]\d?)")
_OPT_RE = re.compile(
    r"\b(\d{1,2}/\d{1,2}/\d{2,4})\s*\$?\s*(\d+(?:\.\d+)?)\s*(call|put)s?\b",
    re.I
)

# ---- E: numeric helpers ----
_NUM_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
_MAG = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}

def _extract_cashtags(text: str) -> list[str]:
    return [m.group(1).upper() for m in _CASHTAG_RE.finditer(text or "")]

def _looks_like_isin(s: str) -> bool:
    return bool(_ISIN_RE.fullmatch((s or "").strip().upper()))

def _parse_option_contracts(text: str) -> List[Dict[str, str]]:
    """
    Capture contracts like '10/24/2025 $6 calls' or '10/24 $6 put'.
    Returns: [{"expiry":"10/24/2025","strike":"6","type":"call"}, ...]
    """
    out: List[Dict[str, str]] = []
    for m in _OPT_RE.finditer(text or ""):
        out.append({
            "expiry": m.group(1),
            "strike": m.group(2),
            "type": m.group(3).lower()
        })
    return out

def _extract_date_tokens_from_text(text: str) -> Dict[str, set]:
    """
    Returns {'raw': set_of_raw_substrings, 'years': set_of_years}
    We store raw lexemes so we can require an exact mention match.
    """
    t = text or ""
    raw = set()
    years = set()

    for m in _DATE_SLASH_RE.finditer(t):
        raw.add(m.group(0))
    for m in _DATE_SLASH_SHORT_RE.finditer(t):
        raw.add(m.group(0))
    for m in _DATE_MONTHNAME_RE.finditer(t):
        # capture exact substring as seen
        raw.add(m.group(0))
    for m in _YEAR_RE.finditer(t):
        years.add(m.group(1))
    return {"raw": raw, "years": years}


# =========================
# LLM Client (Ollama)
# =========================

def _strip_json_comments(text: str) -> str:
    """Remove ``//`` and ``/* */`` comments from JSON-ish text."""

    if not text or ("//" not in text and "/*" not in text):
        return text

    out: List[str] = []
    in_string = False
    escape = False
    single_line = False
    multi_line = False
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""

        if single_line:
            if ch == "\n":
                single_line = False
                out.append(ch)
            i += 1
            continue

        if multi_line:
            if ch == "*" and nxt == "/":
                multi_line = False
                i += 2
            else:
                i += 1
            continue

        if in_string:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue

        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue

        if ch == "/" and nxt == "/":
            single_line = True
            i += 2
            continue

        if ch == "/" and nxt == "*":
            multi_line = True
            i += 2
            continue

        out.append(ch)
        i += 1

    return "".join(out)


def _patch_malformed_json_tokens(text: str) -> str:
    """Attempt to repair common JSON mistakes emitted by weaker models.

    In particular, some models occasionally omit the colon between a key and
    value when expressing comparison style hints (e.g. ``"value">=0.5``).
    Such payloads are invalid JSON and would otherwise cause the round-table
    run to abort.  This helper rewrites those patterns into valid JSON so that
    downstream normalisation can continue handling the semantic meaning.
    """

    if not text or '"' not in text:
        return text

    comparator_pattern = re.compile(
        r'"([^"\\]+)"\s*(>=|<=|==|!=|>|<)\s*([^,}\]]+)',  # missing colon between key/comparator/value
        re.MULTILINE,
    )

    def _normalise_literal(raw: str) -> str:
        candidate = raw.strip()

        # Some models leave a dangling quote after the literal; trim it when it
        # is clearly unmatched so that ``json.loads`` does not error.
        if candidate.endswith('"') and candidate.count('"') == 1:
            candidate = candidate[:-1].rstrip()

        try:
            parsed = json.loads(candidate)
        except Exception:
            return candidate

        if isinstance(parsed, str):
            return parsed

        # Re-serialise primitives (numbers, booleans, null) to preserve their
        # JSON representation while avoiding Python-specific casing (e.g.
        # ``True`` -> ``true``).
        return json.dumps(parsed)

    def repl(match: re.Match) -> str:
        key = match.group(1)
        comparator = match.group(2)
        value = _normalise_literal(match.group(3))
        coerced = comparator + value
        return f'"{key}": {json.dumps(coerced)}'

    return comparator_pattern.sub(repl, text)


def _try_json_with_comments(text: str) -> Optional[dict]:
    candidates = [text]
    cleaned = _strip_json_comments(text)
    if cleaned != text:
        candidates.append(cleaned)

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except JSONDecodeError:
            patched = _patch_malformed_json_tokens(candidate)
            if patched != candidate:
                try:
                    return json.loads(patched)
                except JSONDecodeError:
                    continue
        except Exception:
            continue

    return None


def _extract_json_block(s: str) -> Optional[dict]:
    """Robustly pull a JSON object from a model response."""
    if not s:
        return None
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()
    parsed = _try_json_with_comments(s)
    if parsed is not None:
        return parsed
    i, j = s.find("{"), s.rfind("}")
    if 0 <= i < j:
        inner = s[i:j + 1]
        return _try_json_with_comments(inner)
    return None


class LLMClient(ABC):
    @abstractmethod
    def generate_json(self, role: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        raise NotImplementedError


class OllamaChatClient(LLMClient):
    def __init__(
        self,
        model: str = "mistral",
        host: str = "http://localhost:11434",
        timeout: Optional[float] = None,
    ):
        """Initialise client for Ollama chat completions.

        Args:
            model: Ollama model name.
            host: Base URL for the Ollama server.
            timeout: Request timeout in seconds. ``None`` waits indefinitely.
        """
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    # --- inside OllamaChatClient.generate_json(...) ---
    def generate_json(self, role: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        def _call(sys, usr, temperature=0.2):
            body = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": sys},
                    {"role": "user", "content": usr},
                ],
                "stream": False,
                "format": "json",
                "options": {"temperature": temperature},
            }
            r = requests.post(f"{self.host}/api/chat", json=body, timeout=self.timeout)
            r.raise_for_status()
            return (r.json() or {}).get("message", {}).get("content", "")

        # ↓ try stricter (lower temp) when the role is verifier
        content = _call(system_prompt, user_prompt, temperature=(0.0 if role == "verifier" else 0.2))
        js = _extract_json_block(content)

        # NEW: if the model returned a bare array for verifier, wrap it
        if role == "verifier" and isinstance(js, list):
            js = {"verdicts": js, "overall_notes": [], "why": "auto-wrapped array→object"}

        if isinstance(js, dict):
            # if it only has 'verdicts', still okay; we'll normalize later
            return js

        # One strict retry
        correction_sys = system_prompt + "\nRespond with ONLY a single JSON object with keys: verdicts, overall_notes, why."
        correction_usr = user_prompt + "\nIf you returned an array last time, wrap it under {\"verdicts\": [...] }."
        content2 = _call(correction_sys, correction_usr, temperature=(0.0 if role == "verifier" else 0.2))
        js2 = _extract_json_block(content2)
        if role == "verifier" and isinstance(js2, list):
            js2 = {"verdicts": js2, "overall_notes": [], "why": "auto-wrapped array→object (retry)"}
        if isinstance(js2, dict):
            return js2

        raise RuntimeError(f"Ollama response for role {role} did not contain valid JSON.")




# =========================
# Utilities
# =========================

def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def _compute_spread_pct(numbers: List[NumberMention]) -> Optional[float]:
    bid = ask = None
    for n in numbers:
        label = (n.label or "").lower()
        try:
            val = float(n.value)
        except Exception:
            continue
        if "bid" in label:
            bid = val
        if "ask" in label or "offer" in label:
            ask = val
    if bid is not None and ask is not None and ask > 0:
        mid = (bid + ask) / 2.0
        if mid > 0:
            return round((ask - bid) / mid, 4)
    return None

def _make_signal_id(source_type: str, post_text: str) -> str:
    h = hashlib.sha1(post_text.encode("utf-8")).hexdigest()[:10]
    return f"{source_type}_{h}"

def _first_str(x: Any) -> Optional[str]:
    if isinstance(x, list):
        if not x:
            return None
        x = x[0]
    return str(x) if x is not None else None

def _ensure_top_level_why(d: dict, fallback: str = "auto-fixed: missing why") -> None:
    # Top-level only; do not touch nested objects.
    if not isinstance(d.get("why"), str) or not d["why"].strip():
        d["why"] = fallback

def _snippet(text: str, query_terms: List[str], window: int = 320) -> str:
    t = re.sub(r"\s+", " ", (text or "")).strip()
    if not t:
        return ""
    idx = None
    for term in query_terms:
        m = re.search(re.escape(term), t, flags=re.I)
        if m:
            idx = m.start(); break
    if idx is None:
        return t[:window] + ("…" if len(t) > window else "")
    start = max(0, idx - window // 2)
    end   = min(len(t), start + window)
    return (("…" if start > 0 else "") + t[start:end] + ("…" if end < len(t) else ""))

def _compose_council_scores_why(info_quality: int, trade_clarity: int, evidence_strength: int,
                                spread_pct: Optional[float], supported: int, refuted: int, insufficient: int) -> str:
    parts = [
        f"evidence_strength={evidence_strength}% (supported={supported}, refuted={refuted}, insufficient={insufficient})",
        f"information_quality={info_quality}",
        f"trade_clarity={trade_clarity}",
    ]
    if spread_pct is not None:
        parts.append(f"spread≈{spread_pct*100:.1f}%")
    return "; ".join(parts)

def _compose_moderator_why(direction: str, timeframe: str, ev_strength_pct: int, spread_pct: Optional[float],
                           bull_points: List[str], bear_points: List[str]) -> str:
    cues = []
    if direction and direction != "uncertain":
        cues.append(f"direction={direction}")
    if timeframe and timeframe != "uncertain":
        cues.append(f"timeframe={timeframe}")
    cues.append(f"evidence_strength={ev_strength_pct}%")
    if spread_pct is not None:
        cues.append(f"spread≈{spread_pct*100:.1f}%")
    if bull_points:
        cues.append(f"bull='{bull_points[0]}'")
    if bear_points:
        cues.append(f"bear='{bear_points[0]}'")
    return "; ".join(cues) or "n/a"

def _stale_bucket(age_days: int) -> int:
    """Map age in days to 0..3 bucket."""
    if age_days <= 3:
        return 0
    if age_days <= 7:
        return 1
    if age_days <= 30:
        return 2
    return 3

def _is_help_post(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in _HELP_TOKENS)

def _apply_quality_penalties(council_scores: dict, metrics: dict, verifier_obj) -> dict:
    # Base scores present? ensure ints
    for k in ("information_quality","trade_clarity","evidence_strength"):
        council_scores[k] = int(council_scores.get(k, 50))

    # Penalty: autofixed top-level why
    autofix_top = int(metrics.get("autofixed_top_level_why_count", 0))
    council_scores["information_quality"] -= 10 * min(autofix_top, 3)

    # Penalty: inner missing whys we had to tolerate after retries
    missing_inner = int(metrics.get("missing_whys_inner_after_retry", 0))
    council_scores["information_quality"] -= 5 * min(missing_inner, 4)
    council_scores["trade_clarity"]        -= 5 * min(missing_inner, 4)

    # Penalty: any verifier verdict lacks citations
    no_cites = 0
    fallback_all = False
    try:
        verdicts = list(getattr(verifier_obj, "verdicts", []))
        # detect controlled fallback (all verdict whys mention Auto-fallback)
        if verdicts and all(isinstance(getattr(v, "why", ""), str) and "auto-fallback" in getattr(v, "why", "").lower() for v in verdicts):
            fallback_all = True

        if not fallback_all:
            for v in verdicts:
                if not getattr(v, "citations", []):
                    no_cites += 1
    except Exception:
        pass
    if no_cites:
        council_scores["evidence_strength"] -= 10 * min(no_cites, 3)

    # Clamp 0..100 (only numeric fields)
    for k in ("information_quality","trade_clarity","evidence_strength"):
        v = int(council_scores.get(k, 0))
        council_scores[k] = max(0, min(100, v))

    # Rebuild 'why' with FINAL numbers (post-penalty)
    supported = sum(1 for v in getattr(verifier_obj, "verdicts", []) if getattr(v, "status", "") == "supported")
    refuted = sum(1 for v in getattr(verifier_obj, "verdicts", []) if getattr(v, "status", "") == "refuted")
    insufficient = sum(1 for v in getattr(verifier_obj, "verdicts", []) if getattr(v, "status", "") == "insufficient")
    council_scores["why"] = (
        f"evidence_strength={council_scores['evidence_strength']}% "
        f"(supported={supported}, refuted={refuted}, insufficient={insufficient}); "
        f"information_quality={council_scores['information_quality']}; "
        f"trade_clarity={council_scores['trade_clarity']}; "
        f"penalties(top_auto_fix={autofix_top}, missing_inner={missing_inner}, no_cites={no_cites})"
    )
    return council_scores

# =========================
# Metrics helpers (for penalties)
# =========================

def _count_missing_inner_whys(obj: Any, is_root: bool = True, *, _in_list: bool = False) -> int:
    """
    Count nested dicts (excluding root) that *should* have a non-empty 'why'.
    We intentionally count only dicts that appear inside lists (array items),
    since those represent scored sub-objects in our schemas (claims, verdicts, bullets, etc.).

    We skip known non-scored dicts (e.g., citations, quality_flags, liquidity_risk).
    """
    NON_SCORED_KEYS = {"source", "title", "url", "published_at", "snippet"}  # citation-like
    SKIP_DICT_NAMES = {"quality_flags", "liquidity_risk"}  # benign singletons

    cnt = 0
    if isinstance(obj, dict):
        # Skip citation-like dicts
        if NON_SCORED_KEYS.issubset(set(obj.keys())):
            return 0

        # If this dict lives inside a list, require a 'why' unless it's clearly a benign block
        if not is_root and _in_list:
            kset = set(obj.keys())
            # If it looks like a benign singleton, skip
            if not any(k in SKIP_DICT_NAMES for k in kset):
                w = obj.get("why")
                if not isinstance(w, str) or not w.strip():
                    cnt += 1

        for v in obj.values():
            cnt += _count_missing_inner_whys(v, is_root=False, _in_list=False)
    elif isinstance(obj, list):
        for it in obj:
            cnt += _count_missing_inner_whys(it, is_root=False, _in_list=True)
    return cnt


def _count_missing_whys_in_list(items: Any) -> int:
    """Count how many dict items in a list lack a non-empty 'why'."""
    if not isinstance(items, list):
        return 0
    cnt = 0
    for it in items:
        if isinstance(it, dict):
            w = it.get("why")
            if not isinstance(w, str) or not w.strip():
                cnt += 1
    return cnt

# =========================
# Pipeline Runner (with retries & verbose)
# =========================

class RoundTable:
    def __init__(self, llm: LLMClient, verbose: bool = False, dump_dir: Optional[str] = None, retry_on_invalid: int = 2):
        self.llm = llm
        self.verbose = verbose
        self.dump_dir = dump_dir
        self.retry_on_invalid = max(0, int(retry_on_invalid))
        self.metrics: Dict[str, Dict[str, int]] = {}  # collect per-role metrics for penalties
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)

    def _trace(self, title: str, content: Any):
        if self.verbose:
            print(f"\n=== {title} ===")
            if isinstance(content, (dict, list)):
                print(json.dumps(content, indent=2, ensure_ascii=False))
            else:
                print(str(content))

    def _dump(self, name: str, obj: Any):
        if not self.dump_dir:
            return
        path = os.path.join(self.dump_dir, f"{name}.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def _record_metrics(self, role: str, normalized_obj: Dict[str, Any]):
        """
        Penalize only real schema-quality issues that matter:
        - top-level 'why' auto-fixed (minor)
        - missing 'why' inside claims[] or verdicts[] (the only arrays where inner whys are mandatory)
        """
        penalized_roles = {"entity", "summariser", "claims", "verifier", "context"}
        if role not in penalized_roles:
            return

        role_metrics = self.metrics.setdefault(role, {})

        # track top-level autofixes
        top_why = normalized_obj.get("why")
        if isinstance(top_why, str) and top_why.lower().startswith("auto-fixed"):
            role_metrics["autofixed_top_level_why_count"] = role_metrics.get("autofixed_top_level_why_count", 0) + 1

        # ONLY count inner missing whys for arrays that require them
        missing_inner = 0
        if role == "claims":
            missing_inner += _count_missing_whys_in_list(normalized_obj.get("claims"))
        elif role == "verifier":
            missing_inner += _count_missing_whys_in_list(normalized_obj.get("verdicts"))

        role_metrics["missing_whys_inner_after_retry"] = role_metrics.get("missing_whys_inner_after_retry", 0) + missing_inner

    def _call_role(self,
                   role: str,
                   system_prompt: str,
                   user_prompt: str,
                   normalizer: Callable[[dict], dict],
                   Schema: Any) -> Any:
        # 1st attempt
        raw = self.llm.generate_json(role, system_prompt, user_prompt)
        self._trace(f"{role.upper()} RAW", raw)
        self._dump(f"{role}_raw", raw)

        # Normalize and try to validate
        fixed = normalizer(raw if isinstance(raw, dict) else {})
        self._trace(f"{role.upper()} NORMALIZED", fixed)
        self._dump(f"{role}_normalized", fixed)
        try:
            obj = Schema.model_validate(fixed)
            self._trace(f"{role.upper()} VALIDATION", "OK")
            # record metrics from normalized product
            self._record_metrics(role, fixed)
            return obj
        except Exception as e1:
            self._trace(f"{role.upper()} VALIDATION ERROR", str(e1))

        # Retry once (or configured times) with strict correction
        attempts = 0
        last_err = None
        while attempts < self.retry_on_invalid:
            attempts += 1
            correction_sys = system_prompt + "\n\nYour previous JSON did not match the schema. " \
                                              "Output EXACTLY the required JSON keys and types. No explanations."
            prev = json.dumps(raw, ensure_ascii=False) if isinstance(raw, dict) else str(raw)
            correction_user = user_prompt + f"\n\nPREVIOUS_JSON:\n{prev}"
            raw2 = self.llm.generate_json(role, correction_sys, correction_user)
            self._trace(f"{role.upper()} RETRY RAW", raw2)
            self._dump(f"{role}_retry_raw_{attempts}", raw2)

            fixed2 = normalizer(raw2 if isinstance(raw2, dict) else {})
            self._trace(f"{role.upper()} RETRY NORMALIZED", fixed2)
            self._dump(f"{role}_retry_normalized_{attempts}", fixed2)
            try:
                obj = Schema.model_validate(fixed2)
                self._trace(f"{role.upper()} VALIDATION", f"OK after retry {attempts}")
                # record metrics from final normalized product
                self._record_metrics(role, fixed2)
                return obj
            except Exception as e2:
                last_err = e2
                self._trace(f"{role.upper()} VALIDATION ERROR (retry {attempts})", str(e2))

        # If still invalid, raise with context
        raise RuntimeError(f"{role} JSON could not be validated after {self.retry_on_invalid+1} attempts: {last_err}")

    # ---- Role runners ----
    def run_entity(self, post_text: str) -> EntityTimeframeOut:
        return run_entity_stage(self, post_text)

    def run_summariser(self, post_text: str) -> SummariserOut:
        return run_summariser_stage(self, post_text)

    def run_claims(self, post_text: str) -> ClaimsOut:
        return run_claims_stage(self, post_text)

    def run_verifier(self, claims: ClaimsOut, evidence_map: Dict[str, List[Dict[str, Any]]]) -> VerifierOut:
        return run_verifier_stage(self, claims, evidence_map)


    def run_context(self, asset: Optional[AssetRef], post_text: str) -> ContextOut:
        return run_context_stage(self, asset, post_text)

    def run_for(self, bundle: Dict[str, Any]) -> ForOut:
        return run_bull_stage(self, json.dumps(bundle, ensure_ascii=False))

    def run_against(self, bundle: Dict[str, Any]) -> AgainstOut:
        return run_bear_stage(self, json.dumps(bundle, ensure_ascii=False))

    def run_direction(self, bundle: Dict[str, Any]) -> DirectionOut:
        return run_direction_stage(self, json.dumps(bundle, ensure_ascii=False))

    def fuse_moderator(
        self,
        source: SourceRef,
        post_text: str,
        entity: EntityTimeframeOut,
        summ: SummariserOut,
        claims: ClaimsOut,
        verifier: VerifierOut,
        context: ContextOut,
        for_out: ForOut,
        against_out: AgainstOut,
        dir_out: DirectionOut,
    ) -> TradeSignal:

        def _looks_like_symbol(t: Optional[str]) -> bool:
            return bool(re.match(r"^[A-Z][A-Z0-9\.\-]{0,9}$", (t or "").strip()))

        # --- local helpers (keep function self-contained) ---
        _ISIN_RE = re.compile(r"\b[A-Z]{2}[A-Z0-9]{9}\d\b")
        def _find_isins(text: str) -> list[str]:
            return [m.group(0).upper() for m in _ISIN_RE.finditer(text or "")]

        _LEV_RE = re.compile(r"(?i)\b(long|short)\b.*?\b(?:factor|x)\s*([1-9]\d?)")
        def _extract_leverage(text: str) -> tuple[Optional[str], Optional[int]]:
            m = _LEV_RE.search(text or "")
            if not m:
                return (None, None)
            side = m.group(1).lower()
            try:
                x = int(m.group(2))
            except Exception:
                x = None
            return (side, x)

        # ---- collect distinct tickers mentioned ----
        tickers: list[str] = []
        for a in entity.assets:
            # only admit plausible US-tradable equities; ignore obvious ISINs/foreign codes
            if a.ticker and looks_like_us_equity(a.ticker) and (a.market in (None, "US", "NASDAQ", "NYSE", "OTC")):
                tickers.append(a.ticker)

        for am in getattr(summ, "assets_mentioned", []):
            t = getattr(am, "ticker", None)
            if t and looks_like_us_equity(t):
                tickers.append(t)

        uniq_tickers = sorted({t for t in tickers if t})

        # --- pick a primary ticker by salience (prefer a real US equity)
        def _score_ticker(t: str) -> float:
            t_u = t.upper()
            t_l = t.lower()
            txt = (post_text or "")
            # base: US equity gets a small preference
            score = 1.0 if looks_like_us_equity(t_u) else 0.0
            # hard counts
            score += 2.0 * len(re.findall(rf"\${re.escape(t_u)}(?![A-Za-z0-9])", txt))
            score += 1.0 * len(re.findall(rf"\b{re.escape(t_u)}\b", txt))
            # claims/entities mention
            for c in getattr(claims, "claims", []):
                if (getattr(c, "entity", "") or "").lower().strip() in {t_l}:
                    score += 2.0
                if re.search(rf"\b{re.escape(t_u)}\b", getattr(c, "text", "")):
                    score += 1.0
            # summariser assets_mentioned bonus
            for am in getattr(summ, "assets_mentioned", []):
                if (getattr(am, "ticker", "") or "").upper() == t_u:
                    score += 1.0
            return score

        single_ticker = None
        if len(uniq_tickers) == 1:
            single_ticker = uniq_tickers[0]
        elif len(uniq_tickers) > 1:
            scores = {t: _score_ticker(t) for t in uniq_tickers}
            # pick the max if it clearly leads (>=1.5x runner-up OR ahead by ≥3 points)
            top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            if top and (len(top) == 1 or (top[0][1] >= 1.5 * top[1][1] or top[0][1] - top[1][1] >= 3.0)):
                single_ticker = top[0][0]

            def _is_ambiguous_short(t: str) -> bool:
                # Very short symbols (<=2) are often ambiguous across venues/products
                return bool(re.fullmatch(r"[A-Z]{1,2}", t or ""))

            if single_ticker and _is_ambiguous_short(single_ticker):
                # Require stronger salience: at least one explicit support (assets_mentioned or >=2 cashtag hits)
                cashtags = _extract_cashtags(post_text)
                explicit_hits = sum(1 for am in getattr(summ, "assets_mentioned", []) if (am.ticker or "").upper() == single_ticker)
                if cashtags.count(single_ticker) < 2 and explicit_hits == 0:
                    # Demote to basket/watchlist instead of primary
                    observe_pool = [t for t in uniq_tickers if t != single_ticker] + [single_ticker]
                    single_ticker = None

        # --- prefer a single cashtag if present in the post text
        cashtags = _extract_cashtags(post_text)
        if len(cashtags) == 1:
            single_ticker = cashtags[0]
        elif cashtags:
            # If multiple cashtags, but exactly one also appears in uniq_tickers, pick it
            overlap = [t for t in cashtags if t in uniq_tickers]
            if len(overlap) == 1:
                single_ticker = overlap[0]

        # choose asset only if exactly one / clear leader
        if single_ticker:
            # Prefer the market from entity.assets; if missing, fall back to CSV meta
            ent_market = None
            for a in entity.assets:
                if (a.ticker or "").upper() == single_ticker:
                    ent_market = a.market
                    break

            if not ent_market:
                meta = lookup_ticker_meta(single_ticker)
                ent_market = (meta.get("exchange") or meta.get("market")) if meta else None
                ent_market = _norm_exchange(ent_market)

            asset = AssetRef(ticker=single_ticker, market=ent_market)
        else:
            asset = AssetRef()

        # --- basket monitor: track "the rest" as a watchlist (sidecar) ---
        OBSERVE_BASKET_MIN = 2
        OBSERVE_BASKET_MAX = 10
        observe_pool = [t for t in uniq_tickers if t != single_ticker]
        observe_basket: list[str] = (
            observe_pool[:OBSERVE_BASKET_MAX] if len(observe_pool) >= OBSERVE_BASKET_MIN else []
        )


        # ---- initialize gates/controls early (avoid UnboundLocalError) ----
        forced_no_trade: bool = False
        reasons: List[str] = []

        # ---- lightweight poll detector (non-blocking by itself) ----
        text_l = (post_text or "").lower()
        q_tokens = ("who", "which", "what", "favorite", "favourite", "anyone", "tips", "thoughts", "dd", "10x")
        looks_like_poll = (
            any(q in text_l for q in q_tokens)
            and (len(uniq_tickers) >= 2)            # multiple tickers listed
            and not summ.claimed_catalysts          # no catalysts
            and (len(verifier.verdicts) == 0 or all(v.status == "insufficient" for v in verifier.verdicts))
        )

        # ---- detect ISINs / leveraged ETP semantics (EU turbos, minis, etc.) ----
        isins_seen = _find_isins(post_text)
        lev_side, lev_x = _extract_leverage(post_text)

        spread_pct = _compute_spread_pct(summ.numbers_mentioned)

        # ---- verdict maps & counts ----
        verdict_map = {v.id: v for v in verifier.verdicts}
        supported_ct = sum(1 for v in verifier.verdicts if v.status == "supported")
        refuted_ct   = sum(1 for v in verifier.verdicts if v.status == "refuted")
        insufficient_ct = sum(1 for v in verifier.verdicts if v.status == "insufficient")
        mixed_ct     = sum(1 for v in verifier.verdicts if v.status == "mixed")
        total_claims = max(1, len(verifier.verdicts))

        # ---- evidence strength: external vs self weighting ----
        def _is_self_cite(cite_url: str, src_url: str) -> bool:
            try:
                from urllib.parse import urlparse
                a, b = urlparse(cite_url).netloc, urlparse(src_url).netloc
                return bool(a) and bool(b) and (a == b)
            except Exception:
                return False

        src_url = (source.url or "").strip()

        ext_supported = ext_mixed = ext_refuted = 0
        self_supported = self_mixed = self_refuted = 0

        for v in verifier.verdicts:
            # treat verdict as external if it has ANY external citation
            has_external = any(not _is_self_cite(getattr(c, "url", "") or "", src_url) for c in getattr(v, "citations", []))
            if has_external:
                if v.status == "supported": ext_supported += 1
                elif v.status == "mixed":   ext_mixed += 1
                elif v.status == "refuted": ext_refuted += 1
            else:
                if v.status == "supported": self_supported += 1
                elif v.status == "mixed":   self_mixed += 1
                elif v.status == "refuted": self_refuted += 1

        ext_rate  = (ext_supported + ext_mixed + ext_refuted) / total_claims
        # self_rate: do NOT let self-refutes boost strength
        self_rate = (self_supported + self_mixed) / total_claims

        ev_strength = int(round(100 * (0.8 * ext_rate + 0.2 * self_rate))) if verifier.verdicts else 0

        # Clamp FOR.setup_quality if no external evidence
        if ev_strength == 0:
            sq = for_out.setup_quality
            sq_dict = sq.model_dump()
            for k in ("evidence_specificity", "timeliness", "edge_vs_consensus"):
                sq_dict[k] = min(int(sq_dict.get(k, 0)), 1)
            _ensure_top_level_why(sq_dict, "auto-fixed: no external evidence; capped setup quality")
            for_out = ForOut.model_validate({**for_out.model_dump(), "setup_quality": sq_dict})

        # Cap direction strength with zero evidence
        if ev_strength == 0:
            dir_dict = dir_out.model_dump()
            dir_dict["strength"] = min(int(dir_dict.get("strength", 0)), 1)
            _ensure_top_level_why(dir_dict, "auto-fixed: zero-evidence cap")
            dir_out = DirectionOut.model_validate(dir_dict)

        # ---- Hard gate A: no single tradable asset (but don't penalize if we have a watchlist)
        if not single_ticker and not observe_basket:
            reasons.append("Multiple/unspecified instruments; mapping may be non-equity (fx/futures/krx)")

        # ---- Hard gate B: poll + zero evidence
        if ev_strength == 0 and looks_like_poll:
            forced_no_trade = True
            reasons.append("Speculative poll / no thesis and no verified evidence")

        # ---- Help/education post? (disable trading suggestions)
        is_help = _is_help_post(post_text)
        if is_help:
            forced_no_trade = True
            reasons.append("Educational/help post (no actionable thesis)")


        # Defaults from direction role
        direction = dir_out.implied_direction
        timeframe = dir_out.timeframe

        # ---- Build rationale (skip ticker-only points)
        def _non_trivial_point(s: str) -> bool:
            s = (s or "").strip()
            if not s:
                return False
            return not bool(re.fullmatch(r"^\$?[A-Z][A-Z0-9\.\-]{0,9}$", s))

        rationale: List[str] = []
        b0 = next((bp for bp in for_out.bull_points if _non_trivial_point(bp)), None)
        if b0:
            rationale.append("Bull: " + b0)
        a0 = next(iter(against_out.bear_points), None)
        if a0:
            rationale.append("Bear: " + a0)
        if spread_pct is not None:
            rationale.append(f"Liquidity: spread ≈ {spread_pct*100:.1f}%")
        if lev_x:
            rationale.append(f"Product mechanics: {lev_side or 'leveraged'} ×{lev_x} ETP — path-dependent, issuer-priced")
        if isins_seen:
            rationale.append("ISIN detected → likely structured product; quotes may decouple intraday")
        if not single_ticker and observe_basket:
            rationale.append(f"Basket monitor: observing {', '.join(observe_basket)}")
 

        # ---- Clarity floor (allows paper_trade with no external evidence)
        clarity_floor = (
            (summ.author_stance in ("bullish", "bearish")) and
            bool(b0) and
            (context.stale_risk_level <= 1) and
            (spread_pct is None or spread_pct < 0.05)
        )

        # ---- Confidence rubric (before action selection)
        conf = 0.05
        conf += 0.20 * (ev_strength / 100.0)
        conf += 0.10 if summ.author_stance in ("bullish", "bearish") else 0.0
        conf += 0.10 if for_out.setup_quality.evidence_specificity >= 2 else 0.0
        conf += 0.05 if for_out.setup_quality.timeliness >= 2 else 0.0
        if spread_pct is not None and spread_pct >= 0.05:
            conf -= 0.10
        confidence = float(max(0.05, min(0.70, conf)))
        # Additional cap when ISIN/leveraged ETP detected (issuer mechanics, resets, barriers)
        if isins_seen or lev_x:
            confidence = min(confidence, 0.20)

        is_help = _is_help_post(post_text)

        # If this is an educational/help post, don't generate a trade suggestion
        if is_help:
            forced_no_trade = True
            reasons.append("Educational/help post (no actionable thesis)")

        # ---- Soft demotion instead of hard 'no evidence' gate
        action: Action
        if forced_no_trade:
            direction = "uncertain"
            timeframe = "uncertain"
            confidence = 0.0
            ev_strength = 0
            # ensure rationale reflects the reason
            for r in reversed(reasons):
                rationale.insert(0, f"Bear: {r}")
            action = "no_trade"
        else:
            if ev_strength == 0:
                confidence = min(confidence, 0.15)
                # keep model's direction if decisive, else uncertain
                if direction == "uncertain":
                    action = "monitor"
                else:
                    action = "paper_trade" if clarity_floor else "monitor"
            else:
                # ev_strength > 0
                action = "consider_small_size" if confidence >= 0.50 else "paper_trade"
                if direction == "uncertain":
                    action = "monitor"

        # ---- Blocking issues
        blocks: List[str] = []
        if forced_no_trade:
            blocks.extend(reasons)
        if (spread_pct is not None and spread_pct >= 0.05):
            blocks.append("Illiquidity / wide spread")
        if ev_strength == 0 and not forced_no_trade:
            blocks.append("No verified catalysts/evidence")

        if forced_no_trade and "No clear tradable US ticker; treat as avoid." not in blocks:
            blocks.append("No clear tradable US ticker; treat as avoid.")
        if lev_x:
            blocks.append(f"Leveraged ETP ({lev_side or 'unknown'} ×{lev_x}) — path dependency / resets / issuer spread")
        # If we're in basket-monitor mode, remove the 'no clear ticker' block so we don't over-warn
        if not single_ticker and observe_basket:
            blocks = [b for b in blocks if b != "No clear tradable US ticker; treat as avoid."]
  
 

        # ---- Build audit claims
        audit_claims: List[AuditClaim] = []
        for c in claims.claims:
            v = verdict_map.get(c.id)
            audit_claims.append(AuditClaim(
                id=c.id,
                text=c.text,
                verdict=v.status if v else "insufficient",
                why=(v.why if v else "No verifier output for this claim.")
            ))

        # ---- Base council scores (pre-penalties)
        info_quality = 35
        trade_clarity = 25
        if forced_no_trade:
            info_quality = 5
            trade_clarity = 5
        council_scores = CouncilScores(
            information_quality=info_quality,
            trade_clarity=trade_clarity,
            evidence_strength=ev_strength,
            why=_compose_council_scores_why(info_quality, trade_clarity, ev_strength, spread_pct, supported_ct, refuted_ct, insufficient_ct)
        )

        # ---- Apply schema-quality penalties
        all_metrics = {
            "autofixed_top_level_why_count": sum(self.metrics.get(r, {}).get("autofixed_top_level_why_count", 0) for r in self.metrics),
            "missing_whys_inner_after_retry": sum(self.metrics.get(r, {}).get("missing_whys_inner_after_retry", 0) for r in self.metrics),
        }
        if self.verbose:
            self._trace("METRICS (penalties input)", {
                "by_role": self.metrics,
                "council_penalty_totals": all_metrics,
            })
        council_scores = CouncilScores.model_validate(
            _apply_quality_penalties(council_scores.model_dump(), all_metrics, verifier)
        )

        headline = "Council assessment based on post text, local evidence, and context."

        # Build the audit block (and include assets_seen via extras)
        assets_seen = [{"ticker": a.ticker, "market": a.market} for a in entity.assets] if entity.assets else []

        extras = {"assets_seen": assets_seen}
        options_seen = _parse_option_contracts(post_text)
        if options_seen:
            seen_keys = set()
            dedup = []              # deduplicate
            for o in options_seen:
                key = (o.get("expiry"), o.get("strike"), o.get("type"))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                dedup.append(o)
            extras["options_seen"] = dedup
        if isins_seen:
            extras["isins_seen"] = isins_seen
        if not single_ticker and observe_basket:
            extras["watchlist"] = observe_basket

        audit_block = AuditBlock(
            post_bullets=summ.summary_bullets,
            claims=audit_claims,
            context_used=context.context_bullets,
            timestamps={"ingested_utc": _iso_now(), "researched_utc": _iso_now()},
            why="Weighted Local Research > Post > Context; liquidity (if any) penalized.",
            extras=extras  # requires AuditBlock to include: extras: Dict[str, Any] = Field(default_factory=dict)
        )

        signal = TradeSignal(
            signal_id=_make_signal_id(source.type, post_text),
            source=source,
            asset=asset or AssetRef(),
            headline_summary=headline,
            direction=direction,
            timeframe=timeframe,
            confidence=round(confidence, 2),
            liquidity_risk=LiquidityRisk(
                spread_pct=spread_pct,
                note="Bid/Ask mentioned in post" if spread_pct is not None else None
            ),
            council_scores=council_scores,   # ← don’t zero this due to asset class
            rationale=rationale or ["No clear edge identified."],
            blocking_issues=blocks,          # ← filtered hard-fails only
            citations=[],
            audit=audit_block,               # ← includes assets_seen (or extras)
            action=action,
            next_checks=[
                "Locate dated catalysts (filings/news).",
                "Check volume and spread trend.",
                "Validate key claims via reliable sources.",
                *(
                    [
                        "Read the ETP’s KID/prospectus: reset rules, daily factor mechanics, and financing costs.",
                        "Check if product has knock-out/barrier and how intraday calculation works.",
                        "Compare intraday issuer quotes vs. underlying chart for that window."
                    ] if (isins_seen or lev_x) else []
                ),
                *(
                    [f"Observe basket intraday: {', '.join(observe_basket)} (breadth, yields, futures/ETF alignment)"]
                    if (not single_ticker and observe_basket) else []
                )                
            ],
            why=_compose_moderator_why(
                direction, timeframe, council_scores.evidence_strength,
                spread_pct, for_out.bull_points, against_out.bear_points
            )
        )
        self._trace("MODERATOR SIGNAL", signal.model_dump())
        self._dump("moderator_signal", signal.model_dump())
        return signal






# =========================
# CSV evidence helpers
# =========================

def load_raw_posts_csv(path: str) -> pd.DataFrame:
    """Expect columns: scraped_at, platform, source, post_id, url, title, text."""
    df = pd.read_csv(path, dtype=str).fillna("")
    if "scraped_at" in df.columns:
        try:
            df["scraped_at"] = pd.to_datetime(df["scraped_at"], errors="coerce", utc=True).dt.tz_convert(None)
        except Exception:
            pass
    return df

def pick_post(df: pd.DataFrame, latest: bool = False, select_post_id: Optional[str] = None) -> Dict[str, Any]:
    if df.empty:
        raise RuntimeError("CSV is empty.")

    if select_post_id:
        if "post_id" not in df.columns:
            raise RuntimeError("CSV has no 'post_id' column to select by.")
        mask = df["post_id"].astype(str) == str(select_post_id)
        if not mask.any():
            raise RuntimeError(f"Post id '{select_post_id}' not found in CSV.")
        row = df.loc[mask].iloc[0]
    else:
        row = (
            df.sort_values("scraped_at", ascending=False).iloc[0]
            if (latest and "scraped_at" in df.columns)
            else df.sample(1).iloc[0]
        )

    text = str(row.get("text") or "")
    title = str(row.get("title") or "")
    post_text = (title + "\n\n" + text).strip()
    return {
        "platform": str(row.get("platform") or "reddit"),
        "source": str(row.get("source") or ""),
        "post_id": str(row.get("post_id") or ""),
        "url": str(row.get("url") or ""),
        "scraped_at": str(row.get("scraped_at") or ""),
        "post_text": post_text
    }

def build_evidence_map_from_csv(
    df: pd.DataFrame,
    claims: List[Claim],
    lookback_days: int = 120,
    per_claim_k: int = 3,
    source_post_id: Optional[str] = None,   # <— NEW
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Build per-claim evidence from a local CSV of posts/news.
    Expected df columns: platform, source, post_id, url, title, text, scraped_at.

    - Time-filter to last `lookback_days`.
    - Require an alias (ticker/entity/commodity/futures/FX) hit in the body unless we fail to build any aliases at all.
    - Score by naive term hits; dedupe by URL; cap to `per_claim_k`.
    """
    out: Dict[str, List[Dict[str, Any]]] = {}

    if df.empty:
        return {c.id: [] for c in claims}

    # Time filter
    since = datetime.now(timezone.utc) - timedelta(days=int(lookback_days))
    if "scraped_at" in df.columns:
        try:
            mask = pd.to_datetime(df["scraped_at"], errors="coerce", utc=True) >= since
            df = df.loc[mask].copy()
        except Exception:
            pass

    if df.empty:
        return {c.id: [] for c in claims}

    # Pre-join body for speed
    titles = df.get("title", pd.Series("", index=df.index)).astype(str)
    texts  = df.get("text",  pd.Series("", index=df.index)).astype(str)
    joined = (titles + " " + texts).tolist()

    # Helper: make snippet centered on matched terms
    def _mk_item(i: int, terms: List[str], score: float, claim_id: Optional[str] = None) -> Dict[str, Any]:
        row = df.iloc[i]
        body = f"{row.get('title','')} {row.get('text','')}"
        return {
            "source": str(row.get("platform","reddit")),
            "title": str(row.get("title",""))[:160],
            "url": str(row.get("url","")),
            "published_at": str(row.get("scraped_at","")),
            "snippet": _snippet(body, terms, window=320),
            "claim_id": claim_id,   # 🔧 always include
        }

    # Identify the source row(s) for the picked post so we can force-include it as low-weight evidence
    source_indices: list[int] = []
    if source_post_id and "post_id" in df.columns:
        try:
            source_indices = df.index[(df["post_id"].astype(str) == str(source_post_id))].tolist()
        except Exception:
            source_indices = []

    # ---------- Alias construction (futures/FX/commodities aware) ----------
    alias_set: set[str] = set()

    # Equities: 1–5 letters (e.g., NIO, NVDA)
    eq_regex  = re.compile(r"\b([A-Z]{1,5})(?:\b|[^A-Za-z0-9])")
    # Futures: root(1–4)+month code+yy (e.g., CCZ25, CLF26, ESZ5)
    fut_regex = re.compile(r"\b([A-Z]{1,4}[FGHJKMNQUVXZ]\d{1,2})\b")
    # FX pairs with caret (e.g., ^GBPUSD, ^EURUSD)
    fx_regex  = re.compile(r"\^([A-Z]{3}[A-Z]{3})\b")

    COMMODITY_WORDS = {
        # Softs / agricultural
        "cocoa", "coffee", "sugar", "cotton",
        "wheat", "corn", "soy", "soybeans", "soymeal", "soyoil",
        "oats", "rice", "canola", "rapeseed", "barley",

        # Livestock
        "cattle", "feeder cattle", "live cattle",
        "hogs", "lean hogs", "pork", "beef",

        # Energy
        "oil", "crude", "brent", "wti", "gasoline", "diesel",
        "natgas", "natural gas", "propane", "lng",
        "coal", "uranium",

        # Metals
        "gold", "silver", "platinum", "palladium",
        "copper", "aluminum", "nickel", "zinc", "lead", "tin",
        "iron ore", "steel", "lithium", "cobalt", "manganese", "graphite",

        # Other emerging/critical materials
        "rare earth", "rare earths", "vanadium", "tungsten", "molybdenum",
        "potash", "phosphate",

        # Index-like / catch-all
        "commodity", "futures", "spot", "etf"
    }

    for c in claims:
        text = (c.text or "")

        # Entity name
        if c.entity:
            ent = c.entity.strip().lower()
            if ent:
                alias_set.add(ent)

        # Equities
        for m in eq_regex.finditer(text):
            t = m.group(1).upper()
            if 1 <= len(t) <= 5:
                alias_set.add(t.lower())
                alias_set.add(("$" + t).lower())
                alias_set.add(("nasdaq:" + t).lower())
                alias_set.add(("nyse:" + t).lower())
                alias_set.add(t.lower())

        # Futures (e.g., CCZ25) and root (e.g., CC)
        for m in fut_regex.finditer(text):
            t = m.group(1).upper()
            alias_set.add(t.lower())
            root_m = re.match(r"([A-Z]{1,4})[FGHJKMNQUVXZ]\d{1,2}", t)
            if root_m:
                alias_set.add(root_m.group(1).lower())

        # FX caret pairs (e.g., ^GBPUSD) + legs
        for m in fx_regex.finditer(text):
            t = m.group(1).upper()  # GBPUSD
            alias_set.add(t.lower())
            alias_set.add(t[:3].lower())
            alias_set.add(t[3:].lower())

        # Commodity words present in claim text
        t_lower = text.lower()
        for w in COMMODITY_WORDS:
            if w in t_lower:
                alias_set.add(w)

    relax_alias_requirement = False
    if not alias_set:
        # If we failed to build any aliases, don't block evidence entirely.
        relax_alias_requirement = True

    # ---------- Per-claim evidence selection ----------
    # ---------- Per-claim evidence selection ----------
    for c in claims:
        terms = [t.lower() for t in re.findall(r"[A-Za-z0-9\.\-]{2,}", (c.text or ""))[:8]]
        if c.entity:
            terms.append(c.entity.strip().lower())
        terms = list({t for t in terms if t})

        scored: List[Tuple[int, float]] = []
        for i, body in enumerate(joined):
            b = body.lower()

            if not relax_alias_requirement and alias_set and not any(alias in b for alias in alias_set):
                continue

            score = 0.0
            for t in terms:
                if t and t in b:
                    score += 1.0
            if score > 0:
                scored.append((i, score))

        # 🔁 build the list, starting with forced source row(s), then top scored
        items: List[Dict[str, Any]] = []
        seen_urls: set[str] = set()

        # (A) Force-include the source post row(s) as ultra-low-weight evidence
        for i in source_indices:
            item = _mk_item(i, terms, 0.0, claim_id=c.id)
            u = item["url"]
            if u and u not in seen_urls:
                items.append(item)
                seen_urls.add(u)
                if len(items) >= per_claim_k:
                    break

        # (B) Add scored evidence, sorted by score desc, dedup by URL, then cap to k
        if len(items) < per_claim_k:
            scored.sort(key=lambda x: x[1], reverse=True)
            for i, score in scored:
                if len(items) >= per_claim_k:
                    break
                item = _mk_item(i, terms, score, claim_id=c.id)
                u = item["url"]
                if u in seen_urls:
                    continue
                seen_urls.add(u)
                items.append(item)

        out[c.id] = items

    return out



# =========================
# High-level API (import in wisdom_of_sheep.py)
# =========================

_STAGE_CHOICES = {
    "entity",
    "summariser",
    "claims",
    "context",
    "for",
    "against",
    "direction",
    "moderator",
    "verifier",
    "chairman",
}


def run_stage(
    post_id: str,
    stage: str,
    *,
    model: str = "mistral",
    host: str = "http://localhost:11434",
    timeout: Optional[float] = None,
    verbose: bool = False,
    dump_dir: Optional[str] = None,
    log_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Run a single council stage in-process and return the payload."""

    stage_norm = (stage or "").strip().lower()
    if stage_norm not in _STAGE_CHOICES:
        raise ValueError(f"Unsupported stage '{stage}'. Valid options: {sorted(_STAGE_CHOICES)}")

    text = get_post_text(post_id)
    if not text:
        raise RuntimeError(f"No text available for post {post_id} in DB or raw_posts_log.csv")

    results = run_stages_for_post(
        post_id=post_id,
        title="",
        text="",
        model=model,
        host=host,
        verbose=verbose,
        dump_dir=dump_dir,
        timeout=timeout,
        stages=[stage_norm],
        autofill_deps=True,
        pretty_print_stage_output=False,
        log_callback=log_callback,
    )

    payload = results.get(stage_norm)
    if payload is None:
        raise RuntimeError(f"Stage '{stage_norm}' did not produce a payload")

    return payload


def analyze_post(
    post_text: str,
    model: str = "mistral",
    host: str = "http://localhost:11434",
    evidence_df: Optional[pd.DataFrame] = None,
    evidence_lookback_days: int = 120,
    max_evidence_per_claim: int = 3,
    source: Optional[SourceRef] = None,
    verbose: bool = False,
    dump_dir: Optional[str] = None,
    post_scraped_at: Optional[datetime] = None,   # <— NEW
    timeout: Optional[float] = None,              # <— allow unlimited by default
) -> Tuple[TradeSignal, Dict[str, Any]]:
    """Run full council on a single post_text using your Ollama server."""
    
    post_text = preclean_post_text(post_text)

    llm = OllamaChatClient(model=model, host=host, timeout=timeout)
    pipeline = RoundTable(llm, verbose=verbose, dump_dir=dump_dir, retry_on_invalid=3)

    # Show the original post once at the start when verbose
    if verbose:
        pipeline._trace("SOURCE POST", post_text)

    ent = pipeline.run_entity(post_text)
    summ = pipeline.run_summariser(post_text)
    claims = pipeline.run_claims(post_text)

    ev_map = build_evidence_map_from_csv(
        evidence_df if evidence_df is not None else pd.DataFrame(columns=["platform","source","post_id","url","title","text","scraped_at"]),
        claims.claims,
        lookback_days=evidence_lookback_days,
        per_claim_k=max_evidence_per_claim,
        source_post_id=(source.post_id if source else None)
    )

    verifier = pipeline.run_verifier(claims, ev_map)
    primary_asset = ent.assets[0] if ent.assets else None
    context = pipeline.run_context(primary_asset, post_text)

    # NEW: override staleness by age
    if post_scraped_at is not None:
        age_days = max(0, (datetime.now(timezone.utc) - post_scraped_at.replace(tzinfo=timezone.utc)).days)
        bucket = _stale_bucket(age_days)
        # mutate via model_dump / re-validate to keep type safety
        ctx_dict = context.model_dump()
        ctx_dict["stale_risk_level"] = bucket
        context = ContextOut.model_validate(ctx_dict)

    bundle = {
        "post_id": post_id,
        "post": post_text,
        "entity": ent.model_dump(),
        "summary": summ.model_dump(),
        "claims": claims.model_dump(),
        "verifier": verifier.model_dump(),
        "context": context.model_dump()
    }
    for_out = pipeline.run_for(bundle)
    against_out = pipeline.run_against(bundle)
    dir_out = pipeline.run_direction(bundle)

    src = source or SourceRef(type="reddit", post_id=None, url=None)
    signal = pipeline.fuse_moderator(
        source=src,
        post_text=post_text,
        entity=ent, summ=summ, claims=claims, verifier=verifier,
        context=context, for_out=for_out, against_out=against_out, dir_out=dir_out
    )
    return signal, signal.model_dump()


def run_from_csv_random(
    csv_path: str = "raw_posts_log.csv",
    latest: bool = False,
    model: str = "mistral",
    host: str = "http://localhost:11434",
    evidence_lookback_days: int = 120,
    max_evidence_per_claim: int = 3,
    verbose: bool = False,
    dump_dir: Optional[str] = None,
    timeout: Optional[float] = None,
    post_id: Optional[str] = None,       # <— NEW
) -> Tuple[TradeSignal, Dict[str, Any]]:
    """Pick a random (or latest) post from CSV and analyze it."""
    df = load_raw_posts_csv(csv_path)
    picked = pick_post(df, latest=latest, select_post_id=post_id)
    post_text = picked["post_text"]
    source = SourceRef(type=picked["platform"] or "reddit", post_id=picked["post_id"], url=picked["url"])

    scraped_dt = None
    try:
        raw_ts = picked.get("scraped_at")
        if raw_ts:
            # Be forgiving about formats; coerce and attach UTC
            ts = pd.to_datetime(raw_ts, errors="coerce", utc=True)
            if pd.notna(ts):
                scraped_dt = ts.to_pydatetime()
    except Exception:
        scraped_dt = None

    # Optional safety net: if still None, default to "now" to avoid stale=3 from model heuristics
    if scraped_dt is None:
        scraped_dt = datetime.now(timezone.utc)

    return analyze_post(
        post_text=post_text,
        model=model,
        host=host,
        evidence_df=df,
        evidence_lookback_days=evidence_lookback_days,
        max_evidence_per_claim=max_evidence_per_claim,
        source=source,
        verbose=verbose,
        dump_dir=dump_dir,
        post_scraped_at=scraped_dt,           # <— pass it
        timeout=timeout,                      # <— NEW
    )


# =========================
# Stage runner orchestration
# =========================

ALLOWED_SOURCE_TYPES = {"reddit", "rss", "stocktwits", "x"}


def _coerce_source_type(platform: str | None, post_id: str | None, url: str | None) -> str:
    """Map arbitrary platform labels to SourceRef literal set."""

    p = (platform or "").strip().lower()
    if p in ALLOWED_SOURCE_TYPES:
        return p

    u = (url or "").strip().lower()
    if post_id and post_id.startswith(("t3_", "t1_", "t5_")):
        return "reddit"
    if "reddit.com" in u:
        return "reddit"
    if "stocktwits.com" in u:
        return "stocktwits"
    if "x.com" in u or "twitter.com" in u:
        return "x"
    return "rss"


def _ensure_tables() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(DB_PATH)) as con:
        cur = con.cursor()
        cur.execute(
            """
        CREATE TABLE IF NOT EXISTS posts(
            post_id     TEXT PRIMARY KEY,
            platform    TEXT,
            source      TEXT,
            url         TEXT,
            title       TEXT,
            author      TEXT,
            scraped_at  TEXT,
            posted_at   TEXT,
            score       REAL,
            text        TEXT
        );
        """
        )
        cur.execute(
            """
        CREATE TABLE IF NOT EXISTS stages(
            post_id    TEXT,
            stage      TEXT,
            created_at TEXT,
            payload    TEXT,
            PRIMARY KEY(post_id, stage)
        );
        """
        )
        cur.execute(
            """
        CREATE TABLE IF NOT EXISTS post_extras(
            post_id      TEXT PRIMARY KEY,
            payload_json TEXT
        );
        """
        )
        con.commit()


def _fetch_post_row(post_id: str) -> Optional[Dict[str, Any]]:
    with sqlite3.connect(str(DB_PATH)) as con:
        con.row_factory = sqlite3.Row
        row = con.execute("SELECT * FROM posts WHERE post_id = ?", (post_id,)).fetchone()
        if not row:
            return None
        return dict(row)


def _fetch_csv_row(csv_path: str, post_id: str) -> Optional[Dict[str, Any]]:
    """Return a dict matching CSV headings for a single post_id, or None."""

    df = load_raw_posts_csv(csv_path)
    if "post_id" not in df.columns:
        return None
    mask = df["post_id"].astype(str) == str(post_id)
    if not mask.any():
        return None
    row = df.loc[mask].iloc[0].fillna("")
    return {
        "scraped_at": str(row.get("scraped_at", "")),
        "platform": str(row.get("platform", "")),
        "source": str(row.get("source", "")),
        "post_id": str(row.get("post_id", "")),
        "url": str(row.get("url", "")),
        "title": str(row.get("title", "")),
        "text": str(row.get("text", "")),
        "final_url": str(row.get("final_url", "")),
        "fetch_status": str(row.get("fetch_status", "")),
        "domain": str(row.get("domain", "")),
    }


def _write_post_extras(post_id: str, extras: Dict[str, Any]) -> None:
    try:
        with sqlite3.connect(str(DB_PATH)) as con:
            con.execute(
                """
            INSERT INTO post_extras(post_id, payload_json)
            VALUES (?, ?)
            ON CONFLICT(post_id) DO UPDATE SET payload_json=excluded.payload_json
            """,
                (post_id, json.dumps(extras, ensure_ascii=False)),
            )
            con.commit()
    except Exception:
        # extras are optional; swallow errors to keep pipeline running
        pass


def _fetch_stage_json(post_id: str, stage: str) -> Optional[Dict[str, Any]]:
    with sqlite3.connect(str(DB_PATH)) as con:
        con.row_factory = sqlite3.Row
        row = con.execute(
            "SELECT payload FROM stages WHERE post_id = ? AND stage = ?",
            (post_id, stage),
        ).fetchone()
        if not row:
            return None
        try:
            return json.loads(row["payload"])
        except Exception:
            return None


def _load_or_run_entity(
    pipeline: RoundTable,
    post_id: str,
    post_text: str,
    autofill: bool,
    *,
    force_rerun: bool = False,
) -> EntityTimeframeOut:
    js = None if force_rerun else _fetch_stage_json(post_id, "entity")
    if js:
        return EntityTimeframeOut.model_validate(js)
    if not autofill:
        raise RuntimeError("entity stage missing and auto-fill disabled")
    ent = pipeline.run_entity(post_text)
    write_stage(post_id, "entity", ent.model_dump())
    return ent


def _load_or_run_summariser(
    pipeline: RoundTable,
    post_id: str,
    post_text: str,
    autofill: bool,
    *,
    force_rerun: bool = False,
) -> SummariserOut:
    js = None if force_rerun else _fetch_stage_json(post_id, "summariser")
    if js:
        return SummariserOut.model_validate(js)
    if not autofill:
        raise RuntimeError("summariser stage missing and auto-fill disabled")
    sm = pipeline.run_summariser(post_text)
    write_stage(post_id, "summariser", sm.model_dump())
    return sm


def _load_or_run_claims(
    pipeline: RoundTable,
    post_id: str,
    post_text: str,
    autofill: bool,
    *,
    force_rerun: bool = False,
) -> ClaimsOut:
    js = None if force_rerun else _fetch_stage_json(post_id, "claims")
    if js:
        return ClaimsOut.model_validate(js)
    if not autofill:
        raise RuntimeError("claims stage missing and auto-fill disabled")
    cl = pipeline.run_claims(post_text)
    write_stage(post_id, "claims", cl.model_dump())
    return cl


def _load_or_build_context(
    pipeline: RoundTable,
    post_id: str,
    post_text: str,
    ent: Optional[EntityTimeframeOut],
    autofill: bool,
    *,
    force_rerun: bool = False,
) -> ContextOut:
    js = None if force_rerun else _fetch_stage_json(post_id, "context")
    if js:
        return ContextOut.model_validate(js)
    if not autofill:
        raise RuntimeError("context stage missing and auto-fill disabled")
    primary = ent.assets[0] if ent and ent.assets else None
    ctx = pipeline.run_context(primary, post_text)
    write_stage(post_id, "context", ctx.model_dump())
    return ctx


def _load_or_build_verifier(
    pipeline: RoundTable,
    post_id: str,
    claims: ClaimsOut,
    evidence_df: pd.DataFrame,
    lookback_days: int,
    per_claim_k: int,
    source_ref: SourceRef,
    autofill: bool,
    *,
    force_rerun: bool = False,
) -> VerifierOut:
    js = None if force_rerun else _fetch_stage_json(post_id, "verifier")
    if js:
        return VerifierOut.model_validate(js)
    if not autofill:
        raise RuntimeError("verifier stage missing and auto-fill disabled")
    ev_map = build_evidence_map_from_csv(
        evidence_df,
        claims.claims,
        lookback_days,
        per_claim_k,
        source_post_id=source_ref.post_id,
    )
    ver = pipeline.run_verifier(claims, ev_map)
    write_stage(post_id, "verifier", ver.model_dump())
    return ver


def run_stages_for_post(
    *,
    post_id: str,
    title: str,
    text: str,
    platform: str = "manual",
    source: str = "manual",
    url: str = "",
    model: str = "mistral",
    host: str = "http://localhost:11434",
    verbose: bool = False,
    dump_dir: Optional[str] = None,
    timeout: Optional[float] = None,
    stages: List[str],
    autofill_deps: bool = True,
    evidence_csv: Optional[str] = None,
    refresh_from_csv: bool = False,
    evidence_lookback_days: int = 120,
    max_evidence_per_claim: int = 3,
    pretty_print_stage_output: bool = False,
    echo_post: bool = False,
    log_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Run selected stages, returning a mapping of stage name to payload."""

    logger = logging.getLogger("round_table.runner")
    callback_failed = False

    def _emit(message: str, level: int = logging.INFO) -> None:
        nonlocal callback_failed
        logger.log(level, message)
        if log_callback:
            try:
                log_callback(message)
            except Exception:
                if not callback_failed:
                    logger.exception("log callback failed")
                    callback_failed = True

    _ensure_tables()

    row_existing = _fetch_post_row(post_id)
    csv_row: Optional[Dict[str, Any]] = None

    if refresh_from_csv and evidence_csv:
        csv_row = _fetch_csv_row(evidence_csv, post_id)
        if csv_row:
            _emit("Found row in CSV for this post-id:")
            _emit(json.dumps(csv_row, indent=2, ensure_ascii=False))
            _emit("→ Upserting into DB with original scraped_at/platform/source/url.")
            upsert_post(
                {
                    "post_id": post_id,
                    "platform": csv_row.get("platform") or platform,
                    "source": csv_row.get("source") or source,
                    "url": csv_row.get("url") or url,
                    "title": csv_row.get("title") or title,
                    "author": "",
                    "scraped_at": csv_row.get("scraped_at") or _now_iso(),
                    "posted_at": csv_row.get("scraped_at") or _now_iso(),
                    "score": 0.0,
                    "text": csv_row.get("text") or text,
                }
            )
            _write_post_extras(
                post_id,
                {
                    "final_url": csv_row.get("final_url"),
                    "fetch_status": csv_row.get("fetch_status"),
                    "domain": csv_row.get("domain"),
                },
            )
        else:
            _emit(
                f"--refresh-from-csv requested, but post-id {post_id} was not found in {evidence_csv}",
                level=logging.WARNING,
            )
    elif row_existing is None and evidence_csv:
        csv_row = _fetch_csv_row(evidence_csv, post_id)
        if csv_row:
            _emit("Found row in CSV:")
            _emit(json.dumps(csv_row, indent=2, ensure_ascii=False))
            _emit("→ Upserting into DB with original scraped_at/platform/source/url.")
            upsert_post(
                {
                    "post_id": post_id,
                    "platform": csv_row.get("platform") or platform,
                    "source": csv_row.get("source") or source,
                    "url": csv_row.get("url") or url,
                    "title": csv_row.get("title") or title,
                    "author": "",
                    "scraped_at": csv_row.get("scraped_at") or _now_iso(),
                    "posted_at": csv_row.get("scraped_at") or _now_iso(),
                    "score": 0.0,
                    "text": csv_row.get("text") or text,
                }
            )
            _write_post_extras(
                post_id,
                {
                    "final_url": csv_row.get("final_url"),
                    "fetch_status": csv_row.get("fetch_status"),
                    "domain": csv_row.get("domain"),
                },
            )
        else:
            _emit("Row not found in CSV; will upsert using provided arguments.")

    row_existing = _fetch_post_row(post_id)
    if (
        row_existing is None
        or (text and text != (row_existing.get("text") or ""))
        or (title and title != (row_existing.get("title") or ""))
    ):
        _emit("Upserting post into DB (from provided data)")
        upsert_post(
            {
                "post_id": post_id,
                "platform": platform,
                "source": source,
                "url": url,
                "title": title or (row_existing.get("title") if row_existing else ""),
                "author": (row_existing.get("author") if row_existing else ""),
                "scraped_at": (
                    row_existing.get("scraped_at")
                    if row_existing and row_existing.get("scraped_at")
                    else _now_iso()
                ),
                "posted_at": (
                    row_existing.get("posted_at")
                    if row_existing and row_existing.get("posted_at")
                    else _now_iso()
                ),
                "score": (row_existing.get("score") if row_existing else 0.0),
                "text": text or (row_existing.get("text") if row_existing else ""),
            }
        )

    row = _fetch_post_row(post_id)
    if not row:
        raise RuntimeError(f"Post '{post_id}' not found or failed to upsert.")

    title_final = row.get("title") or (csv_row.get("title") if csv_row else "") or ""
    text_final = row.get("text") or (csv_row.get("text") if csv_row else "") or ""
    post_text = (
        title_final + ("\n\n" + text_final if text_final else "")
    ).strip()

    if echo_post:
        pretty = preclean_post_text(post_text)
        _emit("===== POST TEXT (title + body) BEGIN =====")
        for line in pretty.splitlines():
            _emit(line)
        _emit("===== POST TEXT (title + body) END =====")
        _emit(f"(chars={len(pretty)})")

    client = OllamaChatClient(model=model, host=host, timeout=timeout)
    pipeline = RoundTable(client, verbose=verbose, dump_dir=dump_dir, retry_on_invalid=3)

    if evidence_csv:
        evidence_df = load_raw_posts_csv(evidence_csv)
    else:
        with sqlite3.connect(str(DB_PATH)) as con:
            evidence_df = pd.read_sql_query(
                "SELECT scraped_at, platform, source, post_id, url, title, text FROM posts",
                con,
            )

    src = SourceRef(
        type=_coerce_source_type(row.get("platform"), post_id, row.get("url")),
        post_id=post_id,
        url=row.get("url") or "",
    )

    results: Dict[str, Any] = {}
    ent: Optional[EntityTimeframeOut] = None
    summ: Optional[SummariserOut] = None
    claims: Optional[ClaimsOut] = None
    context: Optional[ContextOut] = None
    verifier: Optional[VerifierOut] = None
    for_out: Optional[ForOut] = None
    against_out: Optional[AgainstOut] = None
    dir_out: Optional[DirectionOut] = None

    def _finish(stage_name: str, obj: Any) -> None:
        payload = obj.model_dump() if hasattr(obj, "model_dump") else obj
        write_stage(post_id, stage_name, payload)
        results[stage_name] = payload

        if stage_name == "summariser":
            try:
                spam_pct = int(payload.get("spam_likelihood_pct", 0))
            except Exception:
                spam_pct = 0
            why_spam = payload.get("spam_why") or payload.get("spam_reasons") or ""
            if isinstance(why_spam, list):
                why_spam = "; ".join([str(x).strip() for x in why_spam if str(x).strip()])[:180]
            why_spam = str(why_spam).strip()
            if spam_pct >= 40:
                _emit(
                    f"Summariser spam_likelihood_pct={spam_pct}%  why={why_spam or '(model provided no reason)'}"
                )
            else:
                _emit(f"Summariser spam_likelihood_pct={spam_pct}%")

        if pretty_print_stage_output:
            _emit(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            _emit(f"✓ Stage {stage_name} complete.")

    for stage in stages:
        _emit(f"→ Running stage: {stage}")

        if stage == "entity":
            ent = _load_or_run_entity(pipeline, post_id, post_text, autofill_deps, force_rerun=True)
            _finish("entity", ent)
            continue

        if stage == "summariser":
            summ = _load_or_run_summariser(pipeline, post_id, post_text, autofill_deps, force_rerun=True)
            _finish("summariser", summ)
            continue

        if stage == "claims":
            claims = _load_or_run_claims(pipeline, post_id, post_text, autofill_deps, force_rerun=True)
            _finish("claims", claims)
            continue

        if stage == "context":
            if ent is None:
                try:
                    ent_json = _fetch_stage_json(post_id, "entity")
                    ent = EntityTimeframeOut.model_validate(ent_json) if ent_json else None
                except Exception:
                    ent = None
                if ent is None and autofill_deps:
                    ent = _load_or_run_entity(pipeline, post_id, post_text, True, force_rerun=False)
            context = _load_or_build_context(
                pipeline, post_id, post_text, ent, autofill_deps, force_rerun=True
            )
            _finish("context", context)
            continue

        if stage == "verifier":
            if claims is None:
                js = _fetch_stage_json(post_id, "claims")
                if js:
                    claims = ClaimsOut.model_validate(js)
                elif autofill_deps:
                    claims = _load_or_run_claims(pipeline, post_id, post_text, True, force_rerun=False)
                else:
                    raise RuntimeError("verifier requires claims but they are missing")
            verifier = _load_or_build_verifier(
                pipeline,
                post_id,
                claims,
                evidence_df,
                evidence_lookback_days,
                max_evidence_per_claim,
                src,
                autofill_deps,
                force_rerun=True,
            )
            _finish("verifier", verifier)
            continue

        if stage in {"for", "against", "direction", "moderator"}:
            if ent is None:
                js = _fetch_stage_json(post_id, "entity")
                if js:
                    ent = EntityTimeframeOut.model_validate(js)
                elif autofill_deps:
                    ent = _load_or_run_entity(pipeline, post_id, post_text, True, force_rerun=False)

            if summ is None:
                js = _fetch_stage_json(post_id, "summariser")
                if js:
                    summ = SummariserOut.model_validate(js)
                elif autofill_deps:
                    summ = _load_or_run_summariser(pipeline, post_id, post_text, True, force_rerun=False)

            if claims is None:
                js = _fetch_stage_json(post_id, "claims")
                if js:
                    claims = ClaimsOut.model_validate(js)
                elif autofill_deps:
                    claims = _load_or_run_claims(pipeline, post_id, post_text, True, force_rerun=False)

            if verifier is None:
                js = _fetch_stage_json(post_id, "verifier")
                if js:
                    verifier = VerifierOut.model_validate(js)
                elif autofill_deps and claims is not None:
                    verifier = _load_or_build_verifier(
                        pipeline,
                        post_id,
                        claims,
                        evidence_df,
                        evidence_lookback_days,
                        max_evidence_per_claim,
                        src,
                        True,
                        force_rerun=False,
                    )

            if context is None:
                js = _fetch_stage_json(post_id, "context")
                if js:
                    context = ContextOut.model_validate(js)
                elif autofill_deps:
                    context = _load_or_build_context(
                        pipeline, post_id, post_text, ent, True, force_rerun=False
                    )

            bundle = {
                "post_id": post_id,
                "post": post_text,
                "entity": (ent.model_dump() if ent else {}),
                "summary": (summ.model_dump() if summ else {}),
                "claims": (claims.model_dump() if claims else {}),
                "verifier": (
                    verifier.model_dump()
                    if verifier
                    else {"verdicts": [], "overall_notes": [], "why": "auto-empty"}
                ),
                "context": (
                    context.model_dump()
                    if context
                    else {
                        "context_bullets": [],
                        "relevant_history": [],
                        "comparables_or_benchmarks": [],
                        "stale_risk_level": 2,
                        "watchouts": [],
                        "why": "auto-empty",
                    }
                ),
            }

            if stage == "for":
                for_out = pipeline.run_for(bundle)
                _finish("for", for_out)
                continue

            if stage == "against":
                against_out = pipeline.run_against(bundle)
                _finish("against", against_out)
                continue

            if stage == "direction":
                dir_out = pipeline.run_direction(bundle)
                _finish("direction", dir_out)
                continue

            if stage == "moderator":
                _emit(
                    "Moderator stage requested, but the moderator pipeline is not available. Skipping."
                )
                continue

            raise RuntimeError(f"Unknown stage: {stage}")

        elif stage == "chairman":
            chairman_payload = run_chairman_stage(post_id, emit=_emit, store=False)
            _finish("chairman", chairman_payload)
            continue

        else:
            raise RuntimeError(f"Unknown stage: {stage}")

    return results


# =========================
# CLI
# =========================

def main():
    ap = argparse.ArgumentParser(description="Wisdom of Sheep — Round Table council runner (Ollama)")
    ap.add_argument("--dummytest", nargs="?", const="raw_posts_log.csv", default=None,
                    help="Run a one-off test on a CSV (defaults to ./raw_posts_log.csv if path omitted).")
    ap.add_argument("--latest", action="store_true", help="With --dummytest, pick the latest row instead of random.")
    ap.add_argument("--random", action="store_true", help="Explicitly pick random row (default if neither flag).")
    ap.add_argument("--model", default="mistral", help="Ollama model name.")
    ap.add_argument("--host", default="http://localhost:11434", help="Ollama host base URL.")
    ap.add_argument("--evidence-lookback-days", type=int, default=120, help="Lookback for CSV evidence.")
    ap.add_argument("--max-evidence-per-claim", type=int, default=3, help="Max evidence items per claim.")
    ap.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    ap.add_argument("--verbose", action="store_true", help="Print prompts and role I/O.")
    ap.add_argument("--dump-dir", default=None, help="Directory to dump raw/normalized JSON per role.")
    ap.add_argument("--no-timeout", action="store_true", help="Disable all model/HTTP timeouts. Use with care.")
    ap.add_argument(
        "--timeout-secs",
        type=float,
        default=None,
        help="Override per-call timeout in seconds (default: wait forever). Ignored if --no-timeout is set.",
    )
    ap.add_argument("--post-id", default=None,help="Pick a specific post_id from the CSV (e.g., t3_1nqd7vg). Overrides --latest/--random.")
    args = ap.parse_args()

    # Decide the effective timeout for all LLM/HTTP calls
    if args.no_timeout:
        effective_timeout = None
    else:
        effective_timeout = args.timeout_secs

    latest = bool(args.latest and not args.random)

    if args.dummytest:
        signal, signal_dict = run_from_csv_random(
            csv_path=args.dummytest,
            latest=latest,
            model=args.model,
            host=args.host,
            evidence_lookback_days=args.evidence_lookback_days,
            max_evidence_per_claim=args.max_evidence_per_claim,
            verbose=args.verbose,
            dump_dir=args.dump_dir,
            timeout=effective_timeout,   # <— NEW
            post_id=args.post_id,   # <— NEW
        )
    else:
        print(json.dumps({"error": "Provide --dummytest to run a standalone check against raw_posts_log.csv"}, ensure_ascii=False))
        return

    print(json.dumps(signal_dict, indent=2 if args.pretty else None, ensure_ascii=False))


if __name__ == "__main__":
    import argparse, json, sys

    parser = argparse.ArgumentParser(description="Wisdom of Sheep round_table")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # === Mode 1: stage runner (legacy CLI hook) ===
    p_stage = subparsers.add_parser("stage", help="Run a single pipeline stage")
    p_stage.add_argument("--post-id", required=True)
    p_stage.add_argument("--stage", required=True,
                         choices=[
                             "entity",
                             "summariser",
                             "claims",
                             "context",
                             "for",
                             "against",
                             "direction",
                             "moderator",
                             "verifier",
                             "chairman",
                         ])

    # === Mode 2: legacy test CLI ===
    p_test = subparsers.add_parser("test", help="Run legacy test harness")
    p_test.add_argument("--dummytest", action="store_true")
    p_test.add_argument("--random", action="store_true")
    p_test.add_argument("--pretty", action="store_true")
    p_test.add_argument("--verbose", action="store_true")
    p_test.add_argument("--model", default="mistral")

    args = parser.parse_args()

    if args.mode == "stage":
        try:
            out = run_stage(args.post_id, args.stage)
        except Exception as exc:
            raise SystemExit(str(exc)) from exc

        print(json.dumps(out, indent=2))

    elif args.mode == "test":
        # Your old test harness logic
        main_test_cli(args)
