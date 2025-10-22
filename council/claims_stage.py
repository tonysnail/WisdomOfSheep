"""Claims stage: extracts concrete, checkable claims from the post text."""

from __future__ import annotations

import json

from .common import (
    GLOBAL_RULE,
    ClaimsOut,
    coerce_enum,
    ensure_top_level_why,
    ensure_why,
    first_str,
    lower_keys,
)

CLAIMS_SCHEMA = {
    "claims": [
        {
            "id": "string",
            "text": "string",
            "type": ["valuation", "liquidity", "project_status", "macro_theme", "performance", "other"],
            "entity": "string|null",
            "why": "string",
        }
    ],
    "why": "string",
}

CLAIMS_SYSTEM_PROMPT = (
    "Extract concrete, checkable claims from the post. No outside facts. "
    f"{GLOBAL_RULE}\nSchema: {json.dumps(CLAIMS_SCHEMA)}"
)
CLAIMS_USER_PROMPT = "POST_TEXT:\n<<<\n{post}\n>>>"


def _normalize_claims(raw: dict) -> dict:
    data = lower_keys(raw)
    claims = data.get("claims") or []
    norm = []
    allowed = ["valuation", "liquidity", "project_status", "macro_theme", "performance", "other"]

    for idx, item in enumerate(claims, start=1):
        if not isinstance(item, dict):
            continue
        claim = lower_keys(item)
        claim.setdefault("id", f"c{idx}")
        claim.setdefault("text", "")
        raw_type = claim.get("type")
        if isinstance(raw_type, list):
            raw_type = first_str(raw_type)
        claim["type"] = coerce_enum(raw_type, allowed, "other")
        entity = first_str(claim.get("entity"))
        if isinstance(entity, str):
            entity = entity.strip() or None
        claim["entity"] = entity
        ensure_why(claim, "auto-fixed: missing why")
        norm.append(claim)

    data["claims"] = norm
    ensure_top_level_why(data, "auto-fixed: missing why")
    return data


def run_claims(table, post_text: str) -> ClaimsOut:
    """Run the claims stage via the shared RoundTable helper."""
    return table._call_role(  # pylint: disable=protected-access
        "claims",
        CLAIMS_SYSTEM_PROMPT,
        CLAIMS_USER_PROMPT.format(post=post_text),
        _normalize_claims,
        ClaimsOut,
    )
