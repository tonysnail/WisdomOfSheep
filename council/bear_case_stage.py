"""Bear case stage: capture red flags and liquidity concerns, using Researcher context."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional, List

from .common import (
    GLOBAL_RULE,
    AgainstOut,
    coerce_reason_list,
    ensure_top_level_why,
    lower_keys,
)

# ───────────────────────────── Repo / DB helpers ──────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DB_PATH = _REPO_ROOT / "council" / "wisdom_of_sheep.sql"


def _latest_stage(conn: sqlite3.Connection, post_id: str, stage: str) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        """
        SELECT payload
        FROM stages
        WHERE post_id = ? AND stage = ?
        ORDER BY datetime(created_at) DESC
        LIMIT 1
        """,
        (post_id, stage),
    ).fetchone()
    if not row:
        return None
    try:
        return json.loads(row[0])
    except Exception:
        return None


def _pick_primary_ticker(bundle_obj: Dict[str, Any]) -> Optional[str]:
    primary = bundle_obj.get("primary_ticker") or bundle_obj.get("primaryTicker")
    if isinstance(primary, str) and primary.strip():
        return primary.strip().upper()
    assets = bundle_obj.get("assets_mentioned") or bundle_obj.get("assets") or []
    if isinstance(assets, list):
        for a in assets:
            if isinstance(a, dict):
                t = (a.get("ticker") or "").strip().upper()
                if t:
                    return t
    return None


# ───────────────────────────── Prompt + schema ────────────────────────────────

AGAINST_SCHEMA = {
    "bear_points": ["string"],
    "red_flags": ["string"],
    "data_gaps": ["string"],
    "liquidity_concerns": {"mentioned": True, "details": "string|null", "why": "string"},
    "why": "string",
}

AGAINST_SYSTEM_PROMPT = (
    "AGAINST analyst: reasons it may NOT be a good trade.\n"
    "Use the post bundle AND the RESEARCHER_CONTEXT (technical summary lines and hub sentiment). "
    "Respect verifier/moderator outcomes. Do not invent new facts beyond provided inputs. "
    f"{GLOBAL_RULE}\nSchema: {json.dumps(AGAINST_SCHEMA)}"
)

AGAINST_USER_PROMPT = "INPUTS:\n{bundle}\n\nRESEARCHER_CONTEXT:\n{researcher_brief}"


def _normalize_against(raw: dict) -> dict:
    data = lower_keys(raw)

    for key in ["bear_points", "red_flags", "data_gaps"]:
        data[key] = coerce_reason_list(data.get(key))

    keywords = {
        "spread",
        "illiquid",
        "liquidity",
        "volume",
        "thinly",
        "wide spread",
        "no volume",
        "delta",
        "gamma",
        "hedge",
        "hedging",
        "mms",
        "market maker",
        "market makers",
        "dealer",
        "dealers",
        "vanna",
        "charm",
        "delta hedging",
        "gamma squeeze",
        "pin risk",
    }

    if "liquidity_concerns" not in data or not isinstance(data["liquidity_concerns"], dict):
        data["liquidity_concerns"] = {"mentioned": False, "details": None, "why": "auto-fixed"}
    else:
        lc = data["liquidity_concerns"]
        mentioned = bool(lc.get("mentioned", False))
        details = lc.get("details")
        if details is not None and not isinstance(details, str):
            details = str(details)
        has_keyword = bool(details) and any(token in details.lower() for token in keywords)
        if mentioned and not has_keyword:
            mentioned = False
            details = None
        data["liquidity_concerns"] = {
            "mentioned": mentioned,
            "details": details if mentioned else None,
            "why": lc.get("why") if isinstance(lc.get("why"), str) and lc.get("why").strip() else "auto-fixed: coerced liquidity signal",
        }

    ensure_top_level_why(data, "auto-fixed: missing why")
    return data


# ─────────────────────────── Researcher context glue ──────────────────────────

def _compose_researcher_brief(post_id: Optional[str], bundle_obj: Dict[str, Any]) -> str:
    """
    Build an LLM-ready context from Researcher:
      - technical_research (summary_lines)
      - sentiment_research (DES fields, n_deltas, narrative)
    STRICT: if unavailable or unresolved → raise RuntimeError.
    """
    if not post_id:
        raise RuntimeError("researcher-context-missing: post_id not provided in bundle")
    if not _DB_PATH.exists():
        raise RuntimeError(f"researcher-context-missing: db not found at {_DB_PATH}")

    try:
        conn = sqlite3.connect(str(_DB_PATH))
        conn.row_factory = sqlite3.Row
    except Exception:
        raise RuntimeError("researcher-context-missing: unable to open db connection")

    try:
        tech = _latest_stage(conn, post_id, "technical_research")
        sent = _latest_stage(conn, post_id, "sentiment_research")
    finally:
        conn.close()

    ticker = _pick_primary_ticker(bundle_obj)

    # Technical (must exist for at least one ticker)
    tech_lines: List[str] = []
    ordered = (tech.get("ordered") or []) if isinstance(tech, dict) else []
    tickers_map = tech.get("tickers") if isinstance(tech, dict) else None
    if not isinstance(tickers_map, dict) or (not ordered and not tickers_map):
        raise RuntimeError("researcher-context-missing: technical_research stage not present for post")

    if ticker and ticker in tickers_map:
        tech_entry = tickers_map.get(ticker) or {}
        tech_lines = tech_entry.get("summary_lines") or []
    elif ordered:
        first = ordered[0]
        tech_entry = tickers_map.get(first) or {}
        tech_lines = tech_entry.get("summary_lines") or []

    if not tech_lines:
        raise RuntimeError("researcher-context-missing: technical summary_lines not found")

    tech_txt = "; ".join(str(x).strip() for x in tech_lines if isinstance(x, str) and x.strip())

    # Sentiment (must exist for at least one ticker)
    if not isinstance(sent, dict):
        raise RuntimeError("researcher-context-missing: sentiment_research stage not present for post")

    s_map = sent.get("tickers") if isinstance(sent, dict) else None
    if not isinstance(s_map, dict) or not s_map:
        raise RuntimeError("researcher-context-missing: sentiment_research has no ticker entries")

    s_entry = s_map.get(ticker) if ticker else None
    if not s_entry:
        first_key = next(iter(s_map))
        s_entry = s_map.get(first_key)

    result = (s_entry or {}).get("result") or {}
    if not result:
        raise RuntimeError("researcher-context-missing: sentiment result missing for ticker")

    try:
        des_raw = result.get("des_raw")
        des_idio = result.get("des_idio")
        conf = result.get("confidence")
        n = result.get("n_deltas")
        nar = result.get("narrative")
        parts = []
        if isinstance(des_raw, (int, float)):
            parts.append(f"DES(raw)={des_raw:+.2f}")
        if isinstance(des_idio, (int, float)):
            parts.append(f"idio={des_idio:+.2f}")
        if isinstance(conf, (int, float)):
            parts.append(f"conf={conf:.2f}")
        if isinstance(n, (int, float)):
            parts.append(f"deltas={int(n)}")
        sent_txt = ("Sentiment: " + ", ".join(parts)) if parts else ""
        if isinstance(nar, str) and nar.strip():
            sent_txt = (sent_txt + " | " if sent_txt else "") + f"Narrative: {nar.strip()[:500]}"
    except Exception:
        raise RuntimeError("researcher-context-missing: sentiment fields not parseable")

    if not tech_txt and not sent_txt:
        raise RuntimeError("researcher-context-missing: no technical or sentiment summary available")

    out = [f"Technical: {tech_txt}"]
    if sent_txt:
        out.append(sent_txt)
    return "\n".join(out)


# ───────────────────────────────── Runner ─────────────────────────────────────

def run_bear_case(table, bundle_json: str) -> AgainstOut:
    """Execute the bear-case analyst stage; requires Researcher context to be present in SQL for this post."""
    try:
        bundle_obj = json.loads(bundle_json) if isinstance(bundle_json, str) else {}
    except Exception:
        bundle_obj = {}
    post_id = (bundle_obj.get("source_post_id") or bundle_obj.get("post_id") or "").strip() or None

    # Will raise RuntimeError if missing -> pipeline fails fast
    researcher_brief = _compose_researcher_brief(post_id, bundle_obj)

    return table._call_role(  # pylint: disable=protected-access
        "against",
        AGAINST_SYSTEM_PROMPT,
        AGAINST_USER_PROMPT.format(bundle=bundle_json, researcher_brief=researcher_brief),
        _normalize_against,
        AgainstOut,
    )
