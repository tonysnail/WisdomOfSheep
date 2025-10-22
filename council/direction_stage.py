"""Direction stage: infer likely direction, timeframe, and tradability guidance."""

from __future__ import annotations

import json

from .common import (
    GLOBAL_RULE,
    DirectionOut,
    ensure_top_level_why,
    coerce_enum,
    lower_keys,
    _DIRECTION_ALLOWED,
    _DIRECTION_MAP,
    _TIMEFRAME_ALLOWED,
    _TIMEFRAME_MAP,
)

DIR_SCHEMA = {
    "implied_direction": ["up", "down", "none", "uncertain"],
    "timeframe": ["intraday", "swing_days", "swing_weeks", "multi_months", "long_term", "uncertain"],
    "strength": "0-3",
    "tradability": {
        "suitable_for": ["scalp", "swing", "position", "avoid"],
        "blocking_issues": ["string"],
        "why": "string",
    },
    "why": "string",
}

DIRECTION_SYSTEM_PROMPT = (
    "Direction & timeframe given the inputs; account for verifier outcomes. "
    f"{GLOBAL_RULE}\nSchema: {json.dumps(DIR_SCHEMA)}"
)
DIRECTION_USER_PROMPT = "INPUTS:\n{bundle}"


def _normalize_direction(raw: dict) -> dict:
    data = lower_keys(raw)
    data["implied_direction"] = coerce_enum(
        data.get("implied_direction"),
        _DIRECTION_ALLOWED,
        "uncertain",
        mapping=_DIRECTION_MAP,
    )
    data["timeframe"] = coerce_enum(
        data.get("timeframe"),
        _TIMEFRAME_ALLOWED,
        "uncertain",
        mapping=_TIMEFRAME_MAP,
    )
    try:
        data["strength"] = max(0, min(3, int(data.get("strength", 0))))
    except Exception:
        data["strength"] = 0

    tradability = data.get("tradability")
    if not isinstance(tradability, dict):
        tradability = {"suitable_for": ["avoid"], "blocking_issues": [], "why": "auto-fixed"}
    else:
        suitable = tradability.get("suitable_for")
        if not isinstance(suitable, list) or not suitable:
            tradability["suitable_for"] = ["avoid"]
        else:
            allowed = {"scalp", "swing", "position", "avoid"}
            tradability["suitable_for"] = [
                item for item in suitable if isinstance(item, str) and item in allowed
            ] or ["avoid"]
        blocking = tradability.get("blocking_issues")
        tradability["blocking_issues"] = blocking if isinstance(blocking, list) else []
        ensure_top_level_why(tradability, "auto-fixed: missing why")
    data["tradability"] = tradability

    ensure_top_level_why(data, "auto-fixed: missing why")
    allowed = {"implied_direction", "timeframe", "strength", "tradability", "why"}
    return {k: data[k] for k in allowed if k in data}


def run_direction(table, bundle_json: str) -> DirectionOut:
    """Run the direction stage and return the structured guidance."""
    return table._call_role(  # pylint: disable=protected-access
        "direction",
        DIRECTION_SYSTEM_PROMPT,
        DIRECTION_USER_PROMPT.format(bundle=bundle_json),
        _normalize_direction,
        DirectionOut,
    )
