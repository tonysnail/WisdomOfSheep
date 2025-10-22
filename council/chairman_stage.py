#!/usr/bin/env python3
# council/chairman_stage.py
"""
Chairman stage — summarises all council analysis into:
1) A plain-English verdict.
2) Final metrics JSON for plotting / bot inputs.

- Reads prior stages from council/wisdom_of_sheep.sql (hard-coded).
- Pulls authoritative TECHNICALS from technical_research (tool-aware).
- Pulls DES sentiment from sentiment_research (and corpus_llm_sent as fallback).
- Sends ALL council payloads (entity, researcher, claims, context, for, against, direction, verifier,
  plus optional technical_research/sentiment_research/corpus_llm_sent/summariser) to the LLM
  alongside a compact INPUT.
- Calls Ollama (hard-coded model "mistral") with a strict JSON schema.
- If the plain-English verdict is missing, performs a short follow-up LLM call.
- Reconciles 'close' to most recent source (compute_indicators > price_window).
- Rounds numerics; clamps ranges; normalises next_checks.
- Stores the final result back into `stages` as stage='chairman'.

CLI:
  
  python council/chairman_stage.py run --post-id t3_1nurhhz

  python council/chairman_stage.py show-input --post-id t3_XXXX
  python council/chairman_stage.py show-stages --post-id t3_XXXX
"""

from __future__ import annotations

import argparse
import json
from json import JSONDecodeError
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

# -----------------------------
# Hard-coded configuration
# -----------------------------
DB_PATH = Path(__file__).resolve().parent / "wisdom_of_sheep.sql"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_MODEL = "mistral"  # required
OLLAMA_TIMEOUT = 120
try:
    OLLAMA_THREADS = max(1, int(os.getenv("WOS_OLLAMA_THREADS", "4")))
except (TypeError, ValueError):
    OLLAMA_THREADS = 4


# -----------------------------
# Utilities
# -----------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _emit(emit: Optional[Callable[[str], None]], message: str, *, error: bool = False) -> None:
    if emit:
        emit(message)
    else:
        if error:
            print(message, file=sys.stderr)
        else:
            print(message)


def _jprint(obj: Any, emit: Optional[Callable[[str], None]] = None) -> None:
    text = json.dumps(obj, indent=2, ensure_ascii=False)
    _emit(emit, text)


def _strip_json_comments(text: str) -> str:
    """Remove ``//`` and ``/* */`` style comments from a JSON blob.

    Ollama occasionally returns otherwise-valid JSON that includes developer-style
    comments. Python's :func:`json.loads` is strict and will refuse to parse these
    payloads, causing the entire Chairman stage to abort. We strip the comments
    while preserving string contents so the downstream parser can accept the
    payload.
    """

    if not text or ("//" not in text and "/*" not in text):
        return text

    out_chars: List[str] = []
    in_string = False
    escape = False
    i = 0
    length = len(text)
    single_line = False
    multi_line = False

    while i < length:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < length else ""

        if single_line:
            if ch == "\n":
                single_line = False
                out_chars.append(ch)
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
            out_chars.append(ch)
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
            out_chars.append(ch)
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

        out_chars.append(ch)
        i += 1

    return "".join(out_chars)


def _patch_malformed_json_tokens(text: str) -> str:
    """Repair common JSON mistakes emitted by some local models."""

    if not text or '"' not in text:
        return text

    comparator_pattern = re.compile(
        r'"([^"\\]+)"\s*(>=|<=|==|!=|>|<)\s*([^,}\]]+)',
        re.MULTILINE,
    )

    def _normalise_literal(raw: str) -> str:
        candidate = raw.strip()

        if candidate.endswith('"') and candidate.count('"') == 1:
            candidate = candidate[:-1].rstrip()

        try:
            parsed = json.loads(candidate)
        except Exception:
            return candidate

        if isinstance(parsed, str):
            return parsed

        return json.dumps(parsed)

    def repl(match: re.Match) -> str:
        key = match.group(1)
        comparator = match.group(2)
        value = _normalise_literal(match.group(3))
        coerced = comparator + value
        return f'"{key}": {json.dumps(coerced)}'

    return comparator_pattern.sub(repl, text)


def _json_loads_allowing_comments(text: str) -> Dict[str, Any]:
    """Attempt to parse JSON, tolerating ``//`` comments and fixer rewrites."""

    last_error: Optional[Exception] = None

    def _attempt(candidate: str) -> Optional[Dict[str, Any]]:
        nonlocal last_error
        try:
            return json.loads(candidate)
        except JSONDecodeError as exc:
            last_error = exc
            patched = _patch_malformed_json_tokens(candidate)
            if patched != candidate:
                try:
                    return json.loads(patched)
                except JSONDecodeError as exc2:
                    last_error = exc2
                    return None
        except Exception as exc:
            last_error = exc
        return None

    first_pass = _attempt(text)
    if first_pass is not None:
        return first_pass

    cleaned = _strip_json_comments(text)
    if cleaned != text:
        second_pass = _attempt(cleaned)
        if second_pass is not None:
            return second_pass

    if isinstance(last_error, JSONDecodeError):
        raise last_error
    if last_error is not None:
        raise last_error
    raise JSONDecodeError("Invalid JSON", text, 0)


def _parse_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s.endswith("%"):
            s = s[:-1]
        return float(s)
    except Exception:
        return None


def _ensure_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return default


def _ensure_num_or_none(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _round_or_none(x: Any, n: int = 2) -> Optional[float]:
    try:
        if x is None:
            return None
        return round(float(x), n)
    except Exception:
        return None


def _get_ticker(prior: Dict[str, Any]) -> str:
    # Preferred: from entity
    ent = prior.get("entity") or {}
    assets = ent.get("assets")
    if isinstance(assets, list) and assets:
        a0 = assets[0] or {}
        t = a0.get("ticker")
        if t:
            return str(t)
    # Fallbacks: scan other blocks text
    for key in ("claims", "context", "for", "against", "researcher", "direction"):
        block = prior.get(key) or {}
        if isinstance(block, dict):
            text_candidates = []
            for _, v in block.items():
                if isinstance(v, str):
                    text_candidates.append(v)
            text = " ".join(text_candidates)
            m = re.search(r'\b[A-Z]{1,5}\b', text)
            if m:
                return m.group(0)
    return ""

def _synth_plain_english(compact: Dict[str, Any], fm: Dict[str, Any]) -> str:
    t = (fm.get("technical") or {}) if isinstance(fm.get("technical"), dict) else {}
    s = (fm.get("sentiment") or {}) if isinstance(fm.get("sentiment"), dict) else {}

    ticker = fm.get("ticker") or compact.get("ticker") or "the stock"
    direction = (fm.get("implied_direction") or "neutral").lower()
    risk = (fm.get("risk_level") or "medium").lower()

    close = t.get("close")
    rsi = t.get("rsi14")
    macd = t.get("macd_hist")
    below20 = (t.get("price_above_sma20") is False)
    below50 = (t.get("price_above_sma50") is False)
    below200 = (t.get("price_above_sma200") is False)
    trend_dir = (t.get("trend_direction") or "").lower()
    trend_str = t.get("trend_strength")

    bits = []

    # Sentence 1: direction + key technicals
    dir_phrase = {
        "down": "likely to continue lower",
        "up": "likely to push higher",
        "neutral": "likely to chop sideways",
    }.get(direction, "likely to chop sideways")

    tech_reasons = []
    if isinstance(macd, (int, float)) and macd < 0:
        tech_reasons.append("negative MACD")
    if isinstance(rsi, (int, float)):
        if rsi < 45:
            tech_reasons.append(f"RSI {round(float(rsi),1)} (weak)")
        elif rsi > 55:
            tech_reasons.append(f"RSI {round(float(rsi),1)} (firm)")
    below_ma = []
    if below20: below_ma.append("20")
    if below50: below_ma.append("50")
    if below200: below_ma.append("200")
    if below_ma:
        tech_reasons.append("below " + "/".join(below_ma) + "-DMA")

    if trend_dir in ("up","down","flat") and trend_str is not None:
        tech_reasons.append(f"trend {trend_dir} (strength {int(trend_str)})")

    prefix = f"{ticker} is {dir_phrase}"
    if isinstance(close, (int, float)):
        prefix += f" (last ≈ ${round(float(close),2)})"
    if tech_reasons:
        prefix += " on " + ", ".join(tech_reasons)

    bits.append(prefix + ".")

    # Sentence 2: risk + sentiment, if any
    sent_bits = []
    des = s.get("des_raw")
    conf = s.get("conf")
    if isinstance(des, (int, float)) and isinstance(conf, (int, float)):
        # only mention if confidence is non-trivial
        if conf >= 0.25:
            tilt = "positive" if des >= 0.55 else ("negative" if des <= 0.45 else "mixed")
            sent_bits.append(f"social DES {round(float(des),2)} ({tilt}), conf {round(float(conf),2)}")

    tail = f"Risk {risk}"
    if sent_bits:
        tail += f"; sentiment {', '.join(sent_bits)}"

    bits.append(tail + ".")

    # Join and compress to two sentences
    out = " ".join(bits).strip()
    # Hard cap: keep to ~240 chars
    if len(out) > 240:
        out = out[:237].rstrip() + "…"
    return out
    
# -----------------------------
# DB access
# -----------------------------
def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def load_latest_stage_payload(conn: sqlite3.Connection, post_id: str, stage: str) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        """
        SELECT payload
        FROM stages
        WHERE post_id=? AND stage=?
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (post_id, stage),
    ).fetchone()
    if not row:
        return None
    payload = row["payload"]
    try:
        if isinstance(payload, (bytes, bytearray)):
            payload = payload.decode("utf-8", "ignore")
        obj = json.loads(payload)
        return obj
    except Exception:
        return None


def gather_prior(
    conn: sqlite3.Connection,
    post_id: str,
    *,
    emit: Optional[Callable[[str], None]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Collect prior stages into a dict, and optional payloads. Also return a full council payload bundle."""
    stages_needed = ["entity", "researcher", "claims", "context", "for", "against", "direction", "verifier"]
    prior: Dict[str, Any] = {}
    missing: List[str] = []
    for st in stages_needed:
        obj = load_latest_stage_payload(conn, post_id, st)
        if obj is None:
            missing.append(st)
        else:
            prior[st] = obj

    # Optional enrichments (not fatal if missing)
    optional: Dict[str, Any] = {}
    for st in ["technical_research", "sentiment_research", "corpus_llm_sent", "summariser"]:
        obj = load_latest_stage_payload(conn, post_id, st)
        if obj is not None:
            optional[st] = obj

    if missing:
        _emit(
            emit,
            f"[WARN] Missing prerequisite stages for {post_id}: {', '.join(missing)}",
            error=True,
        )

    council_bundle = {}
    council_bundle.update(prior)
    council_bundle.update(optional)

    return prior, {"optional": optional, "council_bundle": council_bundle}


# -----------------------------
# TECHNICAL harvesting (tool-aware)
# -----------------------------
_SUMMARY_RX = {
    "price_window": re.compile(r"Price window close\s+[\d\.]+\s+\(([-+]?[\d\.]+)%\)", re.I),
    "rsi_line": re.compile(r"\bRSI\s*14\s+([-\d\.]+)", re.I),
    "macd_hist": re.compile(r"\bMACD\s*hist\s+([-\d\.]+)", re.I),
    "close": re.compile(r"\bClose\s+([-\d\.]+)", re.I),
    "trend": re.compile(r"\bTrend\s+(up|down|flat)\s*;\s*Strength\s+(\d+)", re.I),
}


def parse_summary_lines(lines: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for line in lines or []:
        m = _SUMMARY_RX["price_window"].search(line)
        if m:
            out["price_window_close_pct"] = _parse_float(m.group(1))
        m = _SUMMARY_RX["rsi_line"].search(line)
        if m:
            out["rsi14"] = _parse_float(m.group(1))
        m = _SUMMARY_RX["macd_hist"].search(line)
        if m:
            out["macd_hist"] = _parse_float(m.group(1))
        m = _SUMMARY_RX["close"].search(line)
        if m:
            out["close"] = _parse_float(m.group(1))
        m = _SUMMARY_RX["trend"].search(line)
        if m:
            out["trend_direction"] = (m.group(1) or "").lower()
            out["trend_strength"] = _parse_float(m.group(2))
    return out


def _pct_from_price_window_rows(rows: List[Dict[str, Any]]) -> Optional[float]:
    if not rows or len(rows) < 2:
        return None
    try:
        first = float(rows[0]["Close"])
        last = float(rows[-1]["Close"])
        if first == 0:
            return None
        return round(100.0 * (last - first) / first, 2)
    except Exception:
        return None


def harvest_technical_from_research(tech_research_payload: Dict[str, Any], ticker: str) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "close": None,
        "price_window_close_pct": None,
        "rsi14": None,
        "macd_line": None,
        "macd_signal": None,
        "macd_hist": None,
        "sma20": None,
        "sma50": None,
        "sma200": None,
        "golden_cross": None,
        "price_above_sma20": None,
        "price_above_sma50": None,
        "price_above_sma200": None,
        "trend_direction": None,
        "trend_strength": None,
        "trend_slope_pct_per_day": None,
        "trend_r2": None,
        "realized_vol_annual_pct": None,
        "baseline_vol_annual_pct": None,
        "vol_ratio": None,
        "vol_state": None,
        "last_close": None,
        "nearest_support": None,
        "nearest_resistance": None,
        "distance_to_support_pct": None,
        "distance_to_resistance_pct": None,
        "bollinger_last_event": None,
        "bollinger_last_event_date": None,
        "bollinger_bandwidth": None,
        "bollinger_pct_b": None,
        "obv_trend": None,
        "obv_slope": None,
        "obv_r2": None,
        "mfi": None,
        "mfi_state": None,
        "_asof": {},
    }
    if not isinstance(tech_research_payload, dict) or not ticker:
        return base

    tblock = ((tech_research_payload.get("tickers") or {}).get(ticker)) or {}
    if not isinstance(tblock, dict):
        return base

    # From human summary lines
    summary_lines = tblock.get("summary_lines") or []
    if isinstance(summary_lines, list):
        parsed = parse_summary_lines(summary_lines)
        for k, v in parsed.items():
            if v is not None:
                base[k] = v

    results = tblock.get("results") or []
    last_ci_close = None
    last_ci_asof = None
    last_pw_asof = None
    if isinstance(results, list):
        for item in results:
            res = (item or {}).get("result") or {}
            tool = res.get("tool") or (item or {}).get("tool")
            if not tool:
                continue

            if tool == "price_window":
                rows = res.get("data") or []
                if rows:
                    try:
                        base["close"] = float(rows[-1]["Close"])
                    except Exception:
                        pass
                if base.get("price_window_close_pct") is None:
                    pct = _pct_from_price_window_rows(rows)
                    if pct is not None:
                        base["price_window_close_pct"] = pct
                last_pw_asof = res.get("to") or None
                base["_asof"]["price_window"] = last_pw_asof

            elif tool == "compute_indicators":
                base["_asof"]["compute_indicators"] = res.get("asof")
                ci_close = _parse_float(res.get("close"))
                if ci_close is not None:
                    last_ci_close = ci_close
                    last_ci_asof = res.get("asof")
                rsi = _parse_float(res.get("rsi14"))
                if rsi is not None:
                    base["rsi14"] = rsi
                macd = res.get("macd") or {}
                if isinstance(macd, dict):
                    base["macd_line"] = _parse_float(macd.get("line")) or base.get("macd_line")
                    base["macd_signal"] = _parse_float(macd.get("signal")) or base.get("macd_signal")
                    base["macd_hist"] = _parse_float(macd.get("hist")) or base.get("macd_hist")
                sma = res.get("sma") or {}
                if isinstance(sma, dict):
                    base["sma20"] = _parse_float(sma.get("sma20")) or base.get("sma20")
                    base["sma50"] = _parse_float(sma.get("sma50")) or base.get("sma50")
                    base["sma200"] = _parse_float(sma.get("sma200")) or base.get("sma200")
                crosses = res.get("crosses") or {}
                if isinstance(crosses, dict):
                    if base.get("golden_cross") is None:
                        base["golden_cross"] = bool(crosses.get("golden_cross"))
                    if base.get("price_above_sma20") is None:
                        base["price_above_sma20"] = bool(crosses.get("price_above_sma20"))
                    if base.get("price_above_sma50") is None:
                        base["price_above_sma50"] = bool(crosses.get("price_above_sma50"))
                    if base.get("price_above_sma200") is None:
                        base["price_above_sma200"] = bool(crosses.get("price_above_sma200"))

            elif tool == "trend_strength":
                base["_asof"]["trend_strength"] = res.get("asof")
                base["trend_direction"] = res.get("direction") or base.get("trend_direction")
                base["trend_strength"] = _parse_float(res.get("strength")) or base.get("trend_strength")
                base["trend_slope_pct_per_day"] = _parse_float(res.get("slope_pct_per_day")) or base.get(
                    "trend_slope_pct_per_day"
                )
                base["trend_r2"] = _parse_float(res.get("r2")) or base.get("trend_r2")

            elif tool == "volatility_state":
                base["_asof"]["volatility_state"] = res.get("asof")
                base["realized_vol_annual_pct"] = _parse_float(res.get("realized_vol_annual_pct")) or base.get(
                    "realized_vol_annual_pct"
                )
                base["baseline_vol_annual_pct"] = _parse_float(res.get("baseline_vol_annual_pct")) or base.get(
                    "baseline_vol_annual_pct"
                )
                base["vol_ratio"] = _parse_float(res.get("ratio")) or base.get("vol_ratio")
                base["vol_state"] = res.get("state") or base.get("vol_state")

            elif tool == "support_resistance_check":
                base["_asof"]["support_resistance"] = res.get("asof")
                base["last_close"] = _parse_float(res.get("last_close")) or base.get("last_close")
                base["nearest_support"] = _parse_float(res.get("nearest_support")) or base.get("nearest_support")
                base["nearest_resistance"] = _parse_float(res.get("nearest_resistance")) or base.get("nearest_resistance")
                base["distance_to_support_pct"] = _parse_float(res.get("distance_to_support_pct")) or base.get(
                    "distance_to_support_pct"
                )
                base["distance_to_resistance_pct"] = _parse_float(res.get("distance_to_resistance_pct")) or base.get(
                    "distance_to_resistance_pct"
                )

            elif tool == "bollinger_breakout_scan":
                base["_asof"]["bollinger"] = res.get("asof")
                base["bollinger_last_event"] = res.get("last_event") or base.get("bollinger_last_event")
                base["bollinger_last_event_date"] = res.get("last_event_date") or base.get("bollinger_last_event_date")
                base["bollinger_bandwidth"] = _parse_float(res.get("bandwidth")) or base.get("bollinger_bandwidth")
                base["bollinger_pct_b"] = _parse_float(res.get("%b")) or base.get("bollinger_pct_b")

            elif tool == "obv_trend":
                base["_asof"]["obv_trend"] = res.get("asof")
                base["obv_trend"] = res.get("trend") or base.get("obv_trend")
                base["obv_slope"] = _parse_float(res.get("slope")) or base.get("obv_slope")
                base["obv_r2"] = _parse_float(res.get("r2")) or base.get("obv_r2")

            elif tool == "mfi_flow":
                base["_asof"]["mfi_flow"] = res.get("asof")
                base["mfi"] = _parse_float(res.get("mfi")) or base.get("mfi")
                base["mfi_state"] = res.get("state") or base.get("mfi_state")

    # Reconcile latest close preference
    def _norm_key(s: Optional[str]) -> str:
        if not s:
            return ""
        return s.replace("T", " ").replace("Z", "")

    if last_ci_close is not None:
        if _norm_key(last_ci_asof) >= _norm_key(last_pw_asof or ""):
            base["close"] = last_ci_close

    if base.get("last_close") and not base.get("close"):
        base["close"] = base["last_close"]

    return base


# -----------------------------
# SENTIMENT harvesting (authoritative; multi-source)
# -----------------------------
def _walk_sent_candidates(obj: Any, path: str = "$") -> List[Dict[str, Any]]:
    """Recursively walk an object to find dicts that look like sentiment blocks."""
    cands: List[Dict[str, Any]] = []
    if isinstance(obj, dict):
        # Normalise key names for comparison
        keys = set(k.lower() for k in obj.keys())
        # direct match
        if ("des_raw" in keys or "des" in keys or "idio" in keys) and ("conf" in keys or "confidence" in keys):
            cands.append({"node": obj, "path": path})
        # keep walking
        for k, v in obj.items():
            cands.extend(_walk_sent_candidates(v, f'{path}.{k}'))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            cands.extend(_walk_sent_candidates(v, f"{path}[{i}]"))
    return cands


def _pick_best_sentiment(cands: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """Pick the best candidate by simple heuristics: prefer ones with channel and non-zero/confident values."""
    if not cands:
        return {"des_raw": None, "idio": None, "conf": None, "deltas": None}
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for c in cands:
        n = c["node"]
        # unify keys
        des_raw = n.get("des_raw", n.get("des"))
        idio = n.get("idio", n.get("des_idio"))
        conf = n.get("conf", n.get("confidence"))
        deltas = n.get("deltas", n.get("n"))
        channel = (n.get("channel") or "").lower() if isinstance(n.get("channel"), str) else ""
        score = 0.0
        if des_raw is not None:
            score += 1.0
        if idio is not None:
            score += 0.5
        if conf is not None:
            try:
                cf = float(conf)
                score += 0.5 + min(max(cf, 0.0), 1.0)  # extra for non-zero conf
            except Exception:
                score += 0.3
        if channel in ("social", "news"):
            score += 0.25  # small nudge for explicit channel
        scored.append((score, c))
    # pick max
    scored.sort(key=lambda x: x[0], reverse=True)
    best = scored[0][1]["node"]
    out = {
        "des_raw": _ensure_num_or_none(best.get("des_raw", best.get("des"))),
        "idio": _ensure_num_or_none(best.get("idio", best.get("des_idio"))),
        "conf": _ensure_num_or_none(best.get("conf", best.get("confidence"))),
        "deltas": _ensure_num_or_none(best.get("deltas", best.get("n"))),
    }
    # cast deltas to int when appropriate
    if out["deltas"] is not None:
        try:
            out["deltas"] = int(out["deltas"])
        except Exception:
            pass
    return out


def harvest_sentiment_all_sources(
    sentiment_research_payload: Dict[str, Any],
    corpus_llm_sent_payload: Dict[str, Any],
    prior: Dict[str, Any],
    ticker: str,
) -> Dict[str, Optional[float]]:
    """
    Pull DES-derived sentiment. Order of preference:
      1) sentiment_research (any path, including tickers[T].plan.results.sentiment)
      2) corpus_llm_sent (any path)
      3) any *.sentiment object from other stages

    Returns dict {des_raw, idio, conf, deltas} with None when unknown.
    """
    # 1) sentiment_research: try direct known path
    out = {"des_raw": None, "idio": None, "conf": None, "deltas": None}
    traw = sentiment_research_payload or {}
    if isinstance(traw, dict):
        tblock = ((traw.get("tickers") or {}).get(ticker)) or {}
        # common flat
        if isinstance(tblock, dict):
            if tblock.get("des_raw") is not None:
                out["des_raw"] = _ensure_num_or_none(tblock.get("des_raw"))
            if tblock.get("idio") is not None:
                out["idio"] = _ensure_num_or_none(tblock.get("idio"))
            if tblock.get("conf") is not None:
                out["conf"] = _ensure_num_or_none(tblock.get("conf"))
            if tblock.get("deltas") is not None:
                try:
                    out["deltas"] = int(tblock.get("deltas"))
                except Exception:
                    out["deltas"] = _ensure_num_or_none(tblock.get("deltas"))
        # check nested plan/results/sentiment
        if all(v is None for v in out.values()):
            try:
                plan = (tblock.get("plan") or {}).get("results") or {}
                s = plan.get("sentiment") or {}
                if s:
                    out["des_raw"] = _ensure_num_or_none(s.get("des_raw", s.get("des")))
                    out["idio"] = _ensure_num_or_none(s.get("idio", s.get("des_idio")))
                    out["conf"] = _ensure_num_or_none(s.get("conf", s.get("confidence")))
                    d = s.get("deltas", s.get("n"))
                    out["deltas"] = int(d) if d is not None and str(d).isdigit() else _ensure_num_or_none(d)
            except Exception:
                pass
        # if still blank, walk everything in sentiment_research
        if all(v is None for v in out.values()):
            cands = _walk_sent_candidates(traw)
            if cands:
                out = _pick_best_sentiment(cands)

    # 2) corpus_llm_sent as fallback if needed
    if any(v is None for v in out.values()):
        craw = corpus_llm_sent_payload or {}
        if isinstance(craw, dict):
            cands = _walk_sent_candidates(craw)
            if cands:
                best = _pick_best_sentiment(cands)
                for k in out.keys():
                    if out[k] is None and best.get(k) is not None:
                        out[k] = best[k]

    # 3) final fallback: any .sentiment in prior['for'|'against'|'researcher']
    if any(v is None for v in out.values()):
        for key in ("for", "against", "researcher"):
            s = (prior.get(key) or {}).get("sentiment")
            if isinstance(s, dict):
                out["des_raw"] = out["des_raw"] if out["des_raw"] is not None else _ensure_num_or_none(s.get("des_raw", s.get("des")))
                out["idio"] = out["idio"] if out["idio"] is not None else _ensure_num_or_none(s.get("idio", s.get("des_idio")))
                out["conf"] = out["conf"] if out["conf"] is not None else _ensure_num_or_none(s.get("conf", s.get("confidence")))
                d = s.get("deltas", s.get("n"))
                if out["deltas"] is None and d is not None:
                    try:
                        out["deltas"] = int(d)
                    except Exception:
                        out["deltas"] = _ensure_num_or_none(d)
                if not any(v is None for v in out.values()):
                    break

    return out


# -----------------------------
# Reduce prior → compact INPUT
# -----------------------------
def reduce_input(prior: Dict[str, Any], optional_bundle: Dict[str, Any]) -> Dict[str, Any]:
    d = prior.get("direction") or {}
    timeframe = d.get("timeframe") or "long_term"
    strength = _ensure_int(d.get("strength", 0), 0)

    ticker = _get_ticker(prior)

    optional = optional_bundle.get("optional") or {}
    council_bundle = optional_bundle.get("council_bundle") or {}

    # Build technical snapshot from technical_research (tool-aware)
    tre_payload = optional.get("technical_research") or {}
    technical = harvest_technical_from_research(tre_payload, ticker)

    # Sentiment: try sentiment_research → corpus_llm_sent → stage fallbacks
    sent = harvest_sentiment_all_sources(
        optional.get("sentiment_research") or {},
        optional.get("corpus_llm_sent") or {},
        prior,
        ticker,
    )

    # Catalysts
    catalysts: List[str] = []
    f = prior.get("for") or {}
    if isinstance(f.get("implied_catalysts"), list):
        catalysts = [str(x) for x in f["implied_catalysts"]]

    # Watchouts: union of context.watchouts and against.red_flags
    watchouts = list((prior.get("context") or {}).get("watchouts") or [])
    red_flags = (prior.get("against") or {}).get("red_flags")
    if isinstance(red_flags, list):
        for w in red_flags:
            sw = str(w)
            if sw not in watchouts:
                watchouts.append(sw)

    compact = {
        "ticker": ticker,
        "timeframe": timeframe,
        "direction_strength": strength,
        "technical": technical,
        "sentiment": sent,
        "catalysts": catalysts,
        "watchouts": watchouts,
        "uncertainty_0to3": _ensure_int((prior.get("entity") or {}).get("uncertainty", 1), 1),
        "stale_risk_0to3": _ensure_int((prior.get("context") or {}).get("stale_risk_level", 1), 1),
        # Include the raw council payloads so the LLM can reference all outputs.
        "council": council_bundle,
    }
    return compact


# -----------------------------
# Ollama
# -----------------------------
def ollama_chat(
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    *,
    emit: Optional[Callable[[str], None]] = None,
) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_thread": OLLAMA_THREADS},
    }
    if emit:
        _emit(emit, f"[Chairman] Requesting Ollama model '{OLLAMA_MODEL}' …")
    resp = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    content = data.get("message", {}).get("content") or data.get("content") or ""
    return content


# -----------------------------
# Prompting
# -----------------------------
CHAIRMAN_SYSTEM = """You are the Chairman of the Council. You receive a compact JSON INPUT with:
- ticker, timeframe, direction_strength
- technical (rich snapshot gathered from technical_research tools)
- sentiment (from sentiment_research / corpus_llm_sent)
- catalysts, watchouts
- uncertainty_0to3, stale_risk_0to3
- council (all raw prior stage payloads: entity, researcher, claims, context, for, against, direction, verifier, technical_research, sentiment_research, corpus_llm_sent, summariser)

Your tasks:
1) Infer a directional view and risk from the EVIDENCE (especially TECHNICALS and SENTIMENT).
2) Output STRICT JSON with keys:
{
  "final_metrics": {
    "ticker": "T",
    "timeframe": "swing_days|swing_weeks|long_term",
    "implied_direction": "up|down|neutral",
    "direction_strength": 0|1|2|3,
    "conviction_0to100": int,
    "risk_level": "low|medium|high",
    "tradability_score_0to100": int,
    "uncertainty_0to3": 0|1|2|3,
    "stale_risk_0to3": 0|1|2|3,
    "technical": {
      "close": number|null,
      "price_window_close_pct": number|null,
      "rsi14": number|null,
      "macd_hist": number|null,
      "trend_direction": "up|down|flat"|null,
      "trend_strength": number|null,
      "sma20": number|null,
      "sma50": number|null,
      "sma200": number|null,
      "golden_cross": true|false|null,
      "price_above_sma20": true|false|null,
      "price_above_sma50": true|false|null,
      "price_above_sma200": true|false|null
    },
    "sentiment": {"des_raw": number|null,"idio": number|null,"conf": number|null,"deltas": number|null},
    "catalysts": [string, ...],
    "watchouts": [string, ...],
    "verifier": {"used": false, "notes": []},
    "data_gaps": [],
    "next_checks": [{"metric":"string","op":"string","value":(number|string),"action":"string"}],
    "timestamps": {"analysis_started": string|null, "research_updated": string|null},
    "why": "short justification",
    "blocking_issues": []
  }
}
Rules:
- Be numerically CONSISTENT with INPUT. Do NOT invent numbers.
- If a numeric is unknown, set null. Use integers for *_0to100.
- Prefer risk_level = high when drawdown is large and momentum negative (e.g. macd_hist<0, trend down).
- next_checks MUST be machine-parseable: include metric, op, value, action.
- STRICT JSON ONLY. NO commentary outside JSON.
"""

# Smaller prompt (fallback) without the bulky council bundle:
CHAIRMAN_SYSTEM_COMPACT = """You are the Chairman. Given a compact INPUT (ticker, timeframe, direction_strength, technical, sentiment, catalysts, watchouts, uncertainty, stale_risk), return STRICT JSON ONLY:
{
  "final_metrics": {
    "ticker": "T",
    "timeframe": "swing_days|swing_weeks|long_term",
    "implied_direction": "up|down|neutral",
    "direction_strength": 0|1|2|3,
    "conviction_0to100": int,
    "risk_level": "low|medium|high",
    "tradability_score_0to100": int,
    "uncertainty_0to3": 0|1|2|3,
    "stale_risk_0to3": 0|1|2|3,
    "technical": {
      "close": number|null,
      "price_window_close_pct": number|null,
      "rsi14": number|null,
      "macd_hist": number|null,
      "trend_direction": "up|down|flat"|null,
      "trend_strength": number|null,
      "sma20": number|null,
      "sma50": number|null,
      "sma200": number|null,
      "golden_cross": true|false|null,
      "price_above_sma20": true|false|null,
      "price_above_sma50": true|false|null,
      "price_above_sma200": true|false|null
    },
    "sentiment": {"des_raw": number|null,"idio": number|null,"conf": number|null,"deltas": number|null},
    "catalysts": [string, ...],
    "watchouts": [string, ...],
    "verifier": {"used": false, "notes": []},
    "data_gaps": [],
    "next_checks": [{"metric":"string","op":"string","value":(number|string),"action":"string"}],
    "timestamps": {"analysis_started": string|null, "research_updated": string|null},
    "why": "short justification",
    "blocking_issues": []
  }
}
Rules: Use only INPUT numbers; unknown→null; STRICT JSON ONLY.
"""

FOLLOWUP_SYSTEM = """You are the Chairman. Return STRICT JSON ONLY. NO prose outside JSON.
Input will contain:
- INPUT: compact input (ticker, timeframe, technical, sentiment, etc.)
- FINAL_METRICS: the last JSON you produced

Task: Write 1–2 sentences, plain English, on what the stock is most likely to do and why,
referencing the technical picture. Return EXACTLY this JSON shape and nothing else:
{"plain_english_result":"<one or two sentences>"}"""


# -----------------------------
# Validation, normalisation & repair
# -----------------------------
def _round_technical_block(t: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(t, dict):
        return {}
    keys_2dp = [
        "close",
        "price_window_close_pct",
        "rsi14",
        "macd_line",
        "macd_signal",
        "macd_hist",
        "sma20",
        "sma50",
        "sma200",
        "trend_strength",
        "trend_slope_pct_per_day",
        "trend_r2",
        "realized_vol_annual_pct",
        "baseline_vol_annual_pct",
        "vol_ratio",
        "last_close",
        "nearest_support",
        "nearest_resistance",
        "distance_to_support_pct",
        "distance_to_resistance_pct",
        "bollinger_bandwidth",
        "bollinger_pct_b",
        "obv_slope",
        "obv_r2",
        "mfi",
    ]
    out = dict(t)
    for k in keys_2dp:
        if k in out:
            out[k] = _round_or_none(out.get(k), 2)
    return out


def repair_result(result: Dict[str, Any], compact: Dict[str, Any]) -> Dict[str, Any]:
    fm = result.get("final_metrics", {})
    if not isinstance(fm, dict):
        fm = {}

    fm.setdefault("ticker", compact.get("ticker") or "")
    fm.setdefault("timeframe", compact.get("timeframe") or "long_term")
    fm.setdefault("implied_direction", "neutral")
    fm["direction_strength"] = _ensure_int(fm.get("direction_strength", compact.get("direction_strength", 0)), 0)
    fm["conviction_0to100"] = _ensure_int(fm.get("conviction_0to100", 50), 50)
    fm["tradability_score_0to100"] = _ensure_int(fm.get("tradability_score_0to100", 50), 50)
    fm["uncertainty_0to3"] = _ensure_int(fm.get("uncertainty_0to3", compact.get("uncertainty_0to3", 1)), 1)
    fm["stale_risk_0to3"] = _ensure_int(fm.get("stale_risk_0to3", compact.get("stale_risk_0to3", 1)), 1)

    # Technical
    t = fm.get("technical") or {}
    if not isinstance(t, dict):
        t = {}
    src_t = compact.get("technical") or {}
    for key in [
        "close",
        "price_window_close_pct",
        "rsi14",
        "macd_line",
        "macd_signal",
        "macd_hist",
        "trend_direction",
        "trend_strength",
        "sma20",
        "sma50",
        "sma200",
        "golden_cross",
        "price_above_sma20",
        "price_above_sma50",
        "price_above_sma200",
    ]:
        if src_t.get(key) is not None:
            if key in {"golden_cross", "price_above_sma20", "price_above_sma50", "price_above_sma200"}:
                t[key] = bool(src_t.get(key))
            else:
                t[key] = src_t.get(key)
        else:
            t.setdefault(key, None)
    fm["technical"] = _round_technical_block(t)

    # Risk heuristic
    risk = fm.get("risk_level")
    drawdown = _ensure_num_or_none(t.get("price_window_close_pct"))
    macd_h = _ensure_num_or_none(t.get("macd_hist"))
    trend_dir = (t.get("trend_direction") or "").lower() if isinstance(t.get("trend_direction"), str) else None

    risk_score = 0
    if drawdown is not None and drawdown <= -20:
        risk_score += 2
    if macd_h is not None and macd_h < 0:
        risk_score += 1
    if trend_dir == "down":
        risk_score += 1

    if risk is None:
        if risk_score >= 3:
            risk = "high"
        elif risk_score == 2:
            risk = "medium"
        else:
            risk = "low"
    fm["risk_level"] = risk

    # Sentiment: overwrite with authoritative compact
    src_sent = compact.get("sentiment") or {}
    sent = fm.get("sentiment") or {}
    for k in ("des_raw", "idio", "conf", "deltas"):
        if src_sent.get(k) is not None:
            sent[k] = src_sent.get(k)
        else:
            sent.setdefault(k, None)
    fm["sentiment"] = sent

    # Timestamps
    ts = fm.get("timestamps") or {}
    if not isinstance(ts, dict):
        ts = {}
    ts.setdefault("analysis_started", utc_now_iso())
    prov = (compact.get("technical") or {}).get("_asof") or {}
    asofs = [v for v in prov.values() if isinstance(v, str) and v]
    ts.setdefault("research_updated", max(asofs) if asofs else None)
    fm["timestamps"] = ts

    # Ensure arrays
    fm.setdefault("catalysts", compact.get("catalysts") or [])
    fm.setdefault("watchouts", compact.get("watchouts") or [])
    fm.setdefault("verifier", {"used": False, "notes": []})
    fm.setdefault("data_gaps", [])
    fm.setdefault("next_checks", [])
    fm.setdefault("why", fm.get("why") or "Chairman synthesis from council + technicals + sentiment.")
    fm.setdefault("blocking_issues", [])

    # Range clamps
    fm["conviction_0to100"] = max(0, min(100, fm["conviction_0to100"]))
    fm["tradability_score_0to100"] = max(0, min(100, fm["tradability_score_0to100"]))
    fm["uncertainty_0to3"] = max(0, min(3, fm["uncertainty_0to3"]))
    fm["stale_risk_0to3"] = max(0, min(3, fm["stale_risk_0to3"]))

    # Normalise next_checks
    normalized_checks: List[Dict[str, Any]] = []
    for chk in fm.get("next_checks") or []:
        if isinstance(chk, dict) and all(k in chk for k in ("metric", "op", "value")):
            normalized_checks.append(chk)
        elif isinstance(chk, dict) and "metric" in chk and "condition" in chk:
            cond = str(chk.get("condition"))
            metric = str(chk.get("metric"))
            m = re.search(r"(crosses above|crosses below|below|above|reclaims)", cond, re.I)
            val = None
            op = None
            if m:
                op = m.group(1).lower().replace(" ", "_")
                n = re.search(r"([-+]?\d+(\.\d+)?)", cond)
                val = float(n.group(1)) if n else cond
            normalized_checks.append({"metric": metric, "op": op or cond, "value": val, "action": "monitor"})
        else:
            normalized_checks.append(chk)
    fm["next_checks"] = normalized_checks

    return {"final_metrics": fm}


# -----------------------------
# Persistence
# -----------------------------
def store_result(
    conn: sqlite3.Connection,
    post_id: str,
    result: Dict[str, Any],
    *,
    emit: Optional[Callable[[str], None]] = None,
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO stages (post_id, stage, created_at, payload)
        VALUES (?, 'chairman', ?, ?)
        """,
        (post_id, utc_now_iso(), json.dumps(result, ensure_ascii=False)),
    )
    conn.commit()
    if emit:
        _emit(emit, f"[Chairman] ✅ Stored output for {post_id}")
        _jprint(result, emit)


# -----------------------------
# Runner
# -----------------------------
def build_messages_for_chairman(compact_input: Dict[str, Any], compact_mode: bool = False) -> List[Dict[str, str]]:
    system_prompt = CHAIRMAN_SYSTEM_COMPACT if compact_mode else CHAIRMAN_SYSTEM
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps({"INPUT": compact_input}, ensure_ascii=False)},
    ]


def build_messages_for_followup(compact_input: Dict[str, Any], final_metrics: Dict[str, Any]) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": FOLLOWUP_SYSTEM},
        {"role": "user", "content": json.dumps({"INPUT": compact_input, "FINAL_METRICS": final_metrics}, ensure_ascii=False)},
    ]


def run_chairman_stage(
    post_id: str,
    *,
    emit: Optional[Callable[[str], None]] = None,
    store: bool = True,
) -> Dict[str, Any]:
    conn = db_connect()
    try:
        prior, bundles = gather_prior(conn, post_id, emit=emit)
        compact = reduce_input(prior, bundles)

        if emit:
            _emit(emit, "[Chairman] TECH SNAPSHOT (merged):")
            _jprint(compact.get("technical"), emit)
            _emit(emit, "[Chairman] SENTIMENT SNAPSHOT (merged):")
            _jprint(compact.get("sentiment"), emit)

        messages = build_messages_for_chairman(compact, compact_mode=False)
        content = ollama_chat(messages, emit=emit)
        if emit:
            _emit(emit, "\n[Chairman] ----------------- LLM response (raw) -----------------")
            _emit(emit, content)
            _emit(emit, "[Chairman] ------------------------------------------------------")

        try:
            parsed = _json_loads_allowing_comments(content)
        except Exception:
            _emit(emit, "[WARN] LLM did not return valid JSON. Retrying with compact prompt…", error=True)
            compact_small = dict(compact)
            compact_small.pop("council", None)
            messages_small = build_messages_for_chairman(compact_small, compact_mode=True)
            content2 = ollama_chat(messages_small, emit=emit)
            if emit:
                _emit(emit, "\n[Chairman] ----------------- LLM response (raw, compact) --------")
                _emit(emit, content2)
                _emit(emit, "[Chairman] ------------------------------------------------------")
            try:
                parsed = _json_loads_allowing_comments(content2)
            except Exception:
                m = re.search(r"\{[\s\S]*\}", content2)
                if not m:
                    raise RuntimeError("[Chairman] Unable to parse LLM JSON.")
                parsed = _json_loads_allowing_comments(m.group(0))

        repaired = repair_result(parsed, compact)

        plain = parsed.get("plain_english_result")
        if not plain:
            try:
                messages2 = build_messages_for_followup(compact, repaired.get("final_metrics", {}))
                pe = ollama_chat(messages2, temperature=0.1, emit=emit)
                if emit:
                    _emit(emit, "[Chairman] --------- LLM follow-up (plain english verdict) ---------")
                    _emit(emit, pe)
                    _emit(emit, "[Chairman] ---------------------------------------------------------")
                try:
                    pe_json = _json_loads_allowing_comments(pe)
                except Exception:
                    m = re.search(r"\{[\s\S]*\}", pe)
                    if not m:
                        raise ValueError("no-json")
                    pe_json = _json_loads_allowing_comments(m.group(0))
                plain = pe_json.get("plain_english_result")
            except Exception:
                plain = _synth_plain_english(compact, repaired.get("final_metrics", {}))

        output = repaired
        output["plain_english_result"] = (plain or "Summary unavailable.").strip()
        if store:
            store_result(conn, post_id, output, emit=emit)
        return output
    finally:
        conn.close()


def run_one(post_id: str) -> None:
    run_chairman_stage(post_id, emit=print, store=True)


def cmd_show_input(post_id: str) -> None:
    conn = db_connect()
    prior, bundles = gather_prior(conn, post_id)
    compact = reduce_input(prior, bundles)
    _jprint(compact)


def cmd_show_stages(post_id: str) -> None:
    conn = db_connect()
    rows = conn.execute(
        """
        SELECT stage, created_at, length(payload) AS bytes
        FROM stages
        WHERE post_id=?
        ORDER BY created_at DESC, stage
        """,
        (post_id,),
    ).fetchall()
    for r in rows:
        print(f"{r['stage']:>20} | {r['created_at']} | {r['bytes']}")


# -----------------------------
# CLI
# -----------------------------
def main():
    p = argparse.ArgumentParser(description="Wisdom of Sheep — Chairman stage")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="Run Chairman on a post-id and store result")
    r.add_argument("--post-id", required=True)

    si = sub.add_parser("show-input", help="Show compact INPUT fed to the Chairman")
    si.add_argument("--post-id", required=True)

    ss = sub.add_parser("show-stages", help="List stages present for a post")
    ss.add_argument("--post-id", required=True)

    args = p.parse_args()
    if args.cmd == "run":
        run_one(args.post_id)
    elif args.cmd == "show-input":
        cmd_show_input(args.post_id)
    elif args.cmd == "show-stages":
        cmd_show_stages(args.post_id)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
