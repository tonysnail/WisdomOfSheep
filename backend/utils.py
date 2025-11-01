from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_SPAM_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def now_iso() -> str:
    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_article_time(raw: Optional[str]) -> str:
    if not raw:
        return now_iso()
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")
    except Exception:
        return raw


def clean_strings(value: Any) -> List[str]:
    items: List[str] = []
    if isinstance(value, list):
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                items.append(text)
    elif isinstance(value, str):
        for part in value.splitlines():
            text = part.strip()
            if text:
                items.append(text)
    return items


def extract_claim_texts(payload: Dict[str, Any]) -> List[str]:
    claims = payload.get("claims")
    texts: List[str] = []
    if isinstance(claims, list):
        for claim in claims:
            if isinstance(claim, dict):
                text = claim.get("text")
                if text is None:
                    continue
                clean = str(text).strip()
                if clean:
                    texts.append(clean)
    return texts


def direction_estimate(direction_payload: Optional[Dict[str, Any]], summariser_payload: Optional[Dict[str, Any]]) -> str:
    if isinstance(direction_payload, dict):
        raw = direction_payload.get("implied_direction")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    if isinstance(summariser_payload, dict):
        stance = summariser_payload.get("author_stance")
        if isinstance(stance, str) and stance.strip():
            s = stance.strip().lower()
            mapping = {
                "bullish": "up",
                "bearish": "down",
                "neutral": "neutral",
                "uncertain": "uncertain",
                "up": "up",
                "down": "down",
            }
            return mapping.get(s, stance.strip())
    return "uncertain"


def primary_ticker(
    summariser_payload: Optional[Dict[str, Any]],
    entity_payload: Optional[Dict[str, Any]],
) -> Optional[str]:
    candidates: List[str] = []
    if isinstance(summariser_payload, dict):
        assets = summariser_payload.get("assets_mentioned") or []
        if isinstance(assets, list):
            for asset in assets:
                if isinstance(asset, dict):
                    ticker = asset.get("ticker")
                    if isinstance(ticker, str) and ticker.strip():
                        candidates.append(ticker.strip().upper())
    if not candidates and isinstance(entity_payload, dict):
        assets = entity_payload.get("assets") or []
        if isinstance(assets, list):
            for asset in assets:
                if isinstance(asset, dict):
                    ticker = asset.get("ticker")
                    if isinstance(ticker, str) and ticker.strip():
                        candidates.append(ticker.strip().upper())
    return candidates[0] if candidates else None


def parse_spam_likelihood(value: Any) -> int:
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return 0
        pct = int(float(value))
    elif isinstance(value, str):
        match = _SPAM_NUMBER_RE.search(value)
        if not match:
            return 0
        try:
            pct = int(float(match.group(0)))
        except (TypeError, ValueError):
            return 0
    else:
        return 0

    return max(0, min(100, pct))


def parse_spam_reason(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                parts.append(text)
        return "; ".join(parts)
    return ""


def build_markets_from_entity(payload: Any) -> List[str]:
    if not payload:
        return []
    try:
        assets = payload.get("assets") or []
        vals = []
        for a in assets:
            t = (a.get("ticker") or "").strip()
            if t:
                vals.append(t)
        seen = set()
        uniq = []
        for x in vals:
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        return uniq[:8]
    except Exception:  # noqa: BLE001
        return []


def build_signal_from_dir_or_mod(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    out = {}
    for k in ("bias", "direction", "stance", "verdict", "confidence"):
        if k in payload:
            out[k] = payload[k]
    if "bias" not in out and "direction" in out:
        out["bias"] = out["direction"]
    return out


def trim(text: str, limit: int) -> str:
    if text and len(text) > limit:
        return text[:limit] + "\n…[truncated]…"
    return text or ""


def safe_bool_env(flag: bool) -> str:
    return "1" if flag else "0"


__all__ = [
    "build_markets_from_entity",
    "build_signal_from_dir_or_mod",
    "clean_strings",
    "direction_estimate",
    "extract_claim_texts",
    "normalize_article_time",
    "now_iso",
    "parse_spam_likelihood",
    "parse_spam_reason",
    "primary_ticker",
    "safe_bool_env",
    "trim",
]
