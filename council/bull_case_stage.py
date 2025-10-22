"""Bull case stage: articulate reasons a trade could work well, using Researcher context."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional, List

from .common import (
    GLOBAL_RULE,
    ForOut,
    coerce_reason_list,
    ensure_top_level_why,
    ensure_why,
    lower_keys,
)

# ───────────────────────────── Repo / DB helpers ──────────────────────────────

# council/…/this_file.py  → repo_root = parents[1]
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
    # Try explicit fields commonly present in the bundle
    primary = bundle_obj.get("primary_ticker") or bundle_obj.get("primaryTicker")
    if isinstance(primary, str) and primary.strip():
        return primary.strip().upper()

    # Try summariser-derived assets list
    assets = bundle_obj.get("assets_mentioned") or bundle_obj.get("assets") or []
    if isinstance(assets, list):
        for a in assets:
            if isinstance(a, dict):
                t = (a.get("ticker") or "").strip().upper()
                if t:
                    return t

    # Nothing resolvable
    return None


# ───────────────────────────── Prompt + schema ────────────────────────────────

FOR_SCHEMA = {
    "bull_points": ["string"],
    "implied_catalysts": ["string"],
    "setup_quality": {
        "evidence_specificity": "0-3",
        "timeliness": "0-3",
        "edge_vs_consensus": "0-3",
        "why": "string",
    },
    "what_would_improve": ["string"],
    "why": "string",
}

FOR_SYSTEM_PROMPT = (
    "FOR analyst: reasons it could move favorably.\n"
    "Use the post bundle AND the RESEARCHER_CONTEXT (technical summary lines and hub sentiment). "
    "Respect any verifier/moderator outcomes. Do not invent new facts beyond provided inputs. "
    f"{GLOBAL_RULE}\nSchema: {json.dumps(FOR_SCHEMA)}"
)

FOR_USER_PROMPT = "INPUTS:\n{bundle}\n\nRESEARCHER_CONTEXT:\n{researcher_brief}"


def _normalize_for(raw: dict) -> dict:
    data = lower_keys(raw)
    data["bull_points"] = coerce_reason_list(data.get("bull_points"))
    data["implied_catalysts"] = coerce_reason_list(data.get("implied_catalysts"))
    if "setup_quality" not in data or not isinstance(data["setup_quality"], dict):
        data["setup_quality"] = {
            "evidence_specificity": 0,
            "timeliness": 0,
            "edge_vs_consensus": 0,
            "why": "auto-fixed",
        }
    else:
        quality = data["setup_quality"]
        for key in ["evidence_specificity", "timeliness", "edge_vs_consensus"]:
            try:
                quality[key] = max(0, min(3, int(quality.get(key, 0))))
            except Exception:
                quality[key] = 0
        ensure_why(quality, "auto-fixed: missing why")
    data["what_would_improve"] = coerce_reason_list(data.get("what_would_improve"))
    ensure_top_level_why(data, "auto-fixed: missing why")
    return data


# ─────────────────────────── Researcher context glue ──────────────────────────

def _compose_researcher_brief(post_id: Optional[str], bundle_obj: Dict[str, Any]) -> str:
    """
    Build an LLM-ready context from:
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
        # fallback to any entry
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

def run_bull_case(table, bundle_json: str) -> ForOut:
    """Run the bull-case stage; requires Researcher context to be present in SQL for this post."""
    try:
        bundle_obj = json.loads(bundle_json) if isinstance(bundle_json, str) else {}
    except Exception:
        bundle_obj = {}
    post_id = (bundle_obj.get("source_post_id") or bundle_obj.get("post_id") or "").strip() or None

    # Will raise RuntimeError if missing -> pipeline fails fast
    researcher_brief = _compose_researcher_brief(post_id, bundle_obj)

    return table._call_role(  # pylint: disable=protected-access
        "for",
        FOR_SYSTEM_PROMPT,
        FOR_USER_PROMPT.format(bundle=bundle_json, researcher_brief=researcher_brief),
        _normalize_for,
        ForOut,
    )
