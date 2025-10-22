"""Verifier stage: adjudicate claims using provided evidence bundles."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from .common import (
    GLOBAL_RULE,
    ClaimsOut,
    VerifierOut,
    coerce_enum,
    ensure_top_level_why,
    lower_keys,
    nz,
    _VERDICT_ALLOWED,
)

VERIFIER_SCHEMA = {
    "verdicts": [
        {
            "id": "string",
            "status": ["supported", "refuted", "mixed", "insufficient"],
            "confidence": "0.0-1.0",
            "reason": "string",
            "citations": [
                {
                    "source": ["reddit", "rss", "stocktwits", "x"],
                    "title": "string",
                    "url": "string",
                    "published_at": "ISO8601",
                    "snippet": "string",
                }
            ],
            "why": "string",
        }
    ],
    "overall_notes": ["string"],
    "why": "string",
}

VERIFIER_SYSTEM_PROMPT = (
    "Decide per-claim verdicts using ONLY the evidence mapped to each claim's id. "
    "Evidence is provided as a dict keyed by claim_id; ignore any evidence that does not match the current claim's id. "
    "If a claim has no matching evidence items, return status='insufficient' with confidence=0.0 and empty citations. "
    "For each verdict you MUST include citations drawn only from the matching evidence items you used. "
    "Be conservative — do not infer beyond the evidence. Return a JSON object with exactly these keys: verdicts, overall_notes, why. "
    f"{GLOBAL_RULE}\nSchema: {json.dumps(VERIFIER_SCHEMA)}"
)
VERIFIER_USER_PROMPT = (
    "CLAIMS (array of objects with .id):\n{claims}\n\n"
    "EVIDENCE (dict: claim_id -> list of items with source/title/url/published_at/snippet/claim_id):\n{evidence}\n\n"
    "Instructions:\n"
    "- For each claim in CLAIMS, look up EVIDENCE[claim.id] and use ONLY those items.\n"
    "- If EVIDENCE[claim.id] is empty or weak, set status='insufficient', confidence=0.0, and provide no citations.\n"
    "- If you support/refute/mix, include citations drawn strictly from EVIDENCE[claim.id].\n\n"
    "Example output (JSON only):\n"
    "{\n"
    "  \"verdicts\": [\n"
    "    {\n"
    "      \"id\": \"1\",\n"
    "      \"status\": \"supported\",\n"
    "      \"confidence\": 0.7,\n"
    "      \"reason\": \"Contract amounts and dates match sources.\",\n"
    "      \"citations\": [\n"
    "        {\"source\":\"rss\",\"title\":\"Company signs 5M lb deal\",\"url\":\"https://news.example/abc\",\"published_at\":\"2024-05-01T00:00:00Z\",\"snippet\":\"...\"}\n"
    "      ]\n"
    "    },\n"
    "    {\n"
    "      \"id\": \"2\",\n"
    "      \"status\": \"insufficient\",\n"
    "      \"confidence\": 0.0,\n"
    "      \"reason\": \"No matching evidence in EVIDENCE[2].\",\n"
    "      \"citations\": []\n"
    "    }\n"
    "  ],\n"
    "  \"overall_notes\": [],\n"
    "  \"why\": \"Per-claim decisions based solely on provided EVIDENCE.*\"\n"
    "}\n"
)

_STATUS_ALIASES = {
    "true": "supported",
    "false": "refuted",
    "support": "supported",
    "refute": "refuted",
    "supported": "supported",
    "refuted": "refuted",
    "mixed/partial": "mixed",
    "partial": "mixed",
    "unknown": "insufficient",
    "n/a": "insufficient",
}


def _normalize_verifier(raw: dict) -> dict:
    data = lower_keys(raw)
    verdicts = data.get("verdicts")
    if not isinstance(verdicts, list):
        verdicts = []

    output = []
    for item in verdicts:
        if not isinstance(item, dict):
            continue
        claim_id = nz(item.get("id") or item.get("claim_id") or item.get("claimid"))
        status = coerce_enum(item.get("status"), _VERDICT_ALLOWED, "insufficient", mapping=_STATUS_ALIASES)
        confidence = item.get("confidence")
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.0
        reason = nz(item.get("reason"))

        citations = []
        evidence_items = item.get("evidence") or item.get("citations") or []
        if isinstance(evidence_items, list):
            for evidence in evidence_items:
                if not isinstance(evidence, dict):
                    continue
                src = (evidence.get("source") or "").lower()
                if src not in {"reddit", "rss", "stocktwits", "x"}:
                    url = evidence.get("url") or ""
                    if "reddit.com" in url:
                        src = "reddit"
                    elif "x.com" in url or "twitter.com" in url:
                        src = "x"
                    else:
                        src = "rss"
                citations.append(
                    {
                        "source": src,
                        "title": evidence.get("title") or "",
                        "url": evidence.get("url") or "",
                        "published_at": evidence.get("published_at") or nz(evidence.get("date")) or None,
                        "snippet": evidence.get("snippet") or nz(evidence.get("text")) or "",
                    }
                )

        why = item.get("why")
        if not isinstance(why, str) or not why.strip():
            why = "auto-fixed: missing why"

        output.append(
            {
                "id": claim_id,
                "status": status,
                "confidence": max(0.0, min(1.0, confidence)),
                "reason": reason or "",
                "citations": citations,
                "why": why,
            }
        )

    data["verdicts"] = output
    if not isinstance(data.get("overall_notes"), list):
        data["overall_notes"] = []
    ensure_top_level_why(data)
    allowed = {"verdicts", "overall_notes", "why"}
    return {k: data[k] for k in allowed if k in data}


def run_verifier(
    table,
    claims: ClaimsOut,
    evidence_map: Dict[str, List[Dict[str, Any]]],
) -> VerifierOut:
    """Run the verifier stage with the usual retry/correction logic."""
    try:
        verdicts = table._call_role(  # pylint: disable=protected-access
            "verifier",
            VERIFIER_SYSTEM_PROMPT,
            VERIFIER_USER_PROMPT.format(
                claims=json.dumps(claims.model_dump(), ensure_ascii=False, indent=2),
                evidence=json.dumps(evidence_map, ensure_ascii=False, indent=2),
            ),
            _normalize_verifier,
            VerifierOut,
        )
        if not verdicts.verdicts:
            fallback = _build_insufficient_fallback(claims)
            table._record_metrics("verifier", fallback)  # pylint: disable=protected-access
            return VerifierOut.model_validate(fallback)
        return verdicts
    except Exception as exc:  # pragma: no cover - defensive fallback
        table._trace("VERIFIER FALLBACK", str(exc))  # pylint: disable=protected-access
        fallback = _build_insufficient_fallback(claims, reason="Verifier failed to produce valid JSON")
        table._record_metrics("verifier", fallback)  # pylint: disable=protected-access
        return VerifierOut.model_validate(fallback)


def _build_insufficient_fallback(claims: ClaimsOut, reason: str = "Verifier produced no verdicts") -> dict:
    entries = []
    for claim in claims.claims:
        entries.append(
            {
                "id": claim.id,
                "status": "insufficient",
                "confidence": 0.0,
                "reason": f"{reason}; defaulting to insufficient.",
                "citations": [],
                "why": "Auto-fallback.",
            }
        )
    return _normalize_verifier({"verdicts": entries, "overall_notes": [reason], "why": "Auto-fallback"})
