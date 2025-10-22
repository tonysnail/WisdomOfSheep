"""Context stage: provide neutral background and staleness assessment."""

from __future__ import annotations

import json
from typing import Optional

from .common import (
    GLOBAL_RULE,
    AssetRef,
    ContextOut,
    ensure_why,
    lower_keys,
)

CONTEXT_SCHEMA = {
    "context_bullets": ["string"],
    "relevant_history": ["string"],
    "comparables_or_benchmarks": ["string"],
    "stale_risk_level": "0-3 integer",
    "watchouts": ["string"],
    "why": "string",
}

CONTEXT_SYSTEM_PROMPT = (
    "Provide neutral background context from model memory only; do not assert current facts; tag staleness. "
    f"{GLOBAL_RULE}\nSchema: {json.dumps(CONTEXT_SCHEMA)}"
)
CONTEXT_USER_PROMPT = "ASSET:\n{asset}\n\nPOST_TEXT:\n<<<\n{post}\n>>>"


def _normalize_context(raw: dict) -> dict:
    data = lower_keys(raw)

    def as_list(value):
        if isinstance(value, list):
            return value
        if value is None:
            return []
        if isinstance(value, (str, int, float)):
            return [value]
        return []

    data["context_bullets"] = as_list(data.get("context_bullets"))
    data["relevant_history"] = as_list(data.get("relevant_history"))
    data["comparables_or_benchmarks"] = as_list(data.get("comparables_or_benchmarks"))
    data["watchouts"] = as_list(data.get("watchouts"))

    try:
        data["stale_risk_level"] = max(0, min(3, int(data.get("stale_risk_level", 2))))
    except Exception:
        data["stale_risk_level"] = 2

    ensure_why(data, "auto-fixed: missing why")
    return data


def run_context(table, asset: Optional[AssetRef], post_text: str) -> ContextOut:
    """Execute the context stage with optional asset focus."""
    return table._call_role(  # pylint: disable=protected-access
        "context",
        CONTEXT_SYSTEM_PROMPT,
        CONTEXT_USER_PROMPT.format(
            asset=json.dumps(asset.model_dump() if asset else {}),
            post=post_text,
        ),
        _normalize_context,
        ContextOut,
    )
