"""Entity stage: identifies tradable assets and time intent from raw post text."""

from __future__ import annotations

import json
from typing import Any

from .common import (
    GLOBAL_RULE,
    EntityTimeframeOut,
    coerce_assets,
    coerce_enum,
    ensure_why,
    lookup_ticker_meta,
    lower_keys,
    norm_exchange,
    norm_ticker,
    looks_like_isin,
    looks_like_us_equity,
    _TIMEFRAME_ALLOWED,
    _TIMEFRAME_MAP,
)

# Lightweight schema hint used in the system prompt.
ENTITY_SCHEMA = {
    "assets": [{"ticker": "string|null", "market": "string|null"}],
    "time_hint": ["intraday", "swing_days", "swing_weeks", "multi_months", "long_term", "uncertain"],
    "uncertainty": "0-3 integer",
    "why": "string",
}

ENTITY_SYSTEM_PROMPT = (
    "Identify assets and time intent in the post text. Use ONLY the text. "
    f"{GLOBAL_RULE}\nSchema (informal): {json.dumps(ENTITY_SCHEMA)}"
)
ENTITY_USER_PROMPT = "POST_TEXT:\n<<<\n{post}\n>>>"


def _normalize_entity(raw: dict) -> dict:
    """Repair and coerce the entity stage output into schema-compatible form."""
    data = lower_keys(raw)

    # Ingest assets from various sloppy keys.
    data["assets"] = coerce_assets(
        data.get("assets")
        or data.get("tickers")
        or data.get("symbols")
        or data.get("asset")
        or []
    )

    cleaned: list[dict[str, Any]] = []
    for asset in data["assets"]:
        ticker = norm_ticker(asset.get("ticker"))
        market = asset.get("market")

        if not ticker or ticker.isdigit():
            continue
        if not looks_like_us_equity(ticker) and not looks_like_isin(ticker):
            continue

        meta = lookup_ticker_meta(ticker)
        enriched_market = (
            meta.get("exchange")
            if meta and meta.get("exchange")
            else (meta.get("market") if meta else market)
        )
        enriched_market = norm_exchange(enriched_market)
        cleaned.append({"ticker": ticker, "market": enriched_market})

    data["assets"] = cleaned

    raw_time = data.get("time_hint") or data.get("timeframe")
    if isinstance(raw_time, list):
        raw_time = raw_time[0] if raw_time else None
    data["time_hint"] = coerce_enum(raw_time, _TIMEFRAME_ALLOWED, "uncertain", mapping=_TIMEFRAME_MAP)

    try:
        data["uncertainty"] = max(0, min(3, int(data.get("uncertainty", 1))))
    except Exception:
        data["uncertainty"] = 1

    ensure_why(data, "auto-fixed: missing why")
    allowed = {"assets", "time_hint", "uncertainty", "why"}
    return {k: data[k] for k in allowed if k in data}


def run_entity(table, post_text: str) -> EntityTimeframeOut:
    """Execute the entity stage through the shared RoundTable LLM helper."""
    return table._call_role(  # pylint: disable=protected-access
        "entity",
        ENTITY_SYSTEM_PROMPT,
        ENTITY_USER_PROMPT.format(post=post_text),
        _normalize_entity,
        EntityTimeframeOut,
    )
