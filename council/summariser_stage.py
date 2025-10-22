"""Summariser stage: produce factual bullets, metadata and spam likelihood."""

from __future__ import annotations

import json
from typing import Any

from .common import (
    GLOBAL_RULE,
    SummariserOut,
    SummariserSpamOut,
    coerce_enum,
    ensure_top_level_why,
    expand_magnitude,
    lookup_ticker_meta,
    lower_keys,
    norm_exchange,
    norm_ticker,
    nz,
    coerce_number_like,
    split_percent_value,
    _STANCE_ALLOWED,
    normalize_summariser_spam,
)

SUMM_SCHEMA = {
    "summary_bullets": ["string"],
    "assets_mentioned": [
        {"ticker": "string|null", "name_or_description": "string", "exchange_or_market": "string|null"}
    ],
    "claimed_catalysts": ["string"],
    "claimed_risks": ["string"],
    "numbers_mentioned": [
        {"label": "string", "value": "string", "unit": "string|null"}
    ],
    "author_stance": ["bullish", "bearish", "neutral", "uncertain"],
    "quality_flags": {
        "repetition_or_template": True,
        "vague_claims": True,
        "mentions_spread_or_liquidity": True,
    },
    "why": "string",
}

SPAM_SCHEMA = {
    "spam_likelihood_pct": "0-100 integer",
    "why": "string",
}

SUMMARISER_SYSTEM_PROMPT = (
    "Summarise into factual bullets; extract assets, numbers, catalysts/risks, stance and quality flags. "
    f"No outside facts. {GLOBAL_RULE}\nSchema: {json.dumps(SUMM_SCHEMA)}"
)
SUMMARISER_USER_PROMPT = "POST_TEXT:\n<<<\n{post}\n>>>"

SPAM_SYSTEM_PROMPT = (
    "JSON ONLY: {'spam_likelihood_pct':int,'why':string}. "
    "Spam = (a) off-topic for markets, (b) promo/CTA/brand marketing, "
    "(c) low-info advice/venting with no tradable thesis. Use POST_TEXT + SUMMARISER_JSON + HEURISTICS. "
    "Market catalysts = events that move *securities*: earnings/guidance/filings/M&A/FDA/macro prints/rate decisions/"
    "contracts/orders/IPO/splits/dividends. Product features/benefits are NOT market catalysts. First match wins:\n"
    "1) Promo/CTA or consumer product ad → 95–100; why='commercial promotion/CTA'.\n"
    "2) Off-topic: no tradable asset and no market catalyst/timeframe/plan → 85–100; why='off-topic (non-market)'.\n"
    "3) Advice/venting/meta AND no market catalyst → 50–70; why='low-information solicitation/meta-discussion'.\n"
    "4) Clear trade thesis → 0–15; why='clear thesis'.\n"
    "5) Else: if tickers present but no market catalyst → 30–50; else → 60–80. Always include a brief why (≤120 chars)."
)
SPAM_USER_PROMPT = (
    "POST_TEXT:\n<<<\n{post}\n>>>\n\n"
    "SUMMARISER_JSON:\n{summary_json}\n\n"
    "HEURISTICS: has_ticker={has_ticker} has_numbers={has_numbers} has_catalysts={has_catalysts} is_megathread={is_megathread}\n"
    "Return ONLY the JSON object."
)


def _normalize_summariser(raw: dict) -> dict:
    data = lower_keys(raw)

    for key in [
        "summary_bullets",
        "assets_mentioned",
        "claimed_catalysts",
        "claimed_risks",
        "numbers_mentioned",
    ]:
        if not isinstance(data.get(key), list):
            data[key] = []

    stance = data.get("author_stance")
    if isinstance(stance, list):
        stance = stance[0] if stance else None
    data["author_stance"] = coerce_enum(stance, _STANCE_ALLOWED, "uncertain")

    qf = data.get("quality_flags")
    if not isinstance(qf, dict):
        qf = {}
    data["quality_flags"] = {
        "repetition_or_template": bool(qf.get("repetition_or_template", False)),
        "vague_claims": bool(qf.get("vague_claims", False)),
        "mentions_spread_or_liquidity": bool(qf.get("mentions_spread_or_liquidity", False)),
    }

    assets_norm = []
    for item in data["assets_mentioned"]:
        if isinstance(item, dict):
            ticker = norm_ticker(item.get("ticker"))
            name_guess = nz(item.get("name_or_description"))
            ex_guess = norm_exchange(item.get("exchange_or_market"))

            meta = lookup_ticker_meta(ticker) if ticker else None
            if meta:
                name_final = meta.get("longName") or name_guess or (ticker or "")
                ex_final = norm_exchange(meta.get("exchange") or meta.get("market") or ex_guess)
            else:
                name_final = ticker or name_guess
                ex_final = ex_guess

            assets_norm.append(
                {
                    "ticker": ticker,
                    "name_or_description": str(name_final) if name_final is not None else "",
                    "exchange_or_market": ex_final,
                }
            )
        elif isinstance(item, str):
            assets_norm.append(
                {"ticker": None, "name_or_description": item.strip(), "exchange_or_market": None}
            )
    data["assets_mentioned"] = assets_norm

    numbers_norm = []
    for item in data["numbers_mentioned"]:
        if isinstance(item, dict):
            label = nz(item.get("label"))
            value_raw = expand_magnitude(nz(item.get("value")))
            value_raw = coerce_number_like(value_raw)
            value, unit = split_percent_value(value_raw, item.get("unit"))
            label_lower = label.lower()
            if "contract" in label_lower:
                unit = "contracts"
            numbers_norm.append({"label": label, "value": value, "unit": unit})
        elif isinstance(item, str):
            numbers_norm.append({"label": item.strip(), "value": "", "unit": None})
    for entry in numbers_norm:
        try:
            numeric = float(entry.get("value", ""))
        except Exception:
            continue
        label_lower = (entry.get("label") or "").lower()
        if entry.get("unit") == "%" and abs(numeric) > 50 and (
            "price" in label_lower
            or "points" in label_lower
            or "down" in label_lower
            or "up" in label_lower
        ):
            entry["unit"] = None
    data["numbers_mentioned"] = numbers_norm

    ensure_top_level_why(data)
    return data


def run_summariser(table, post_text: str) -> SummariserOut:
    """Execute the summariser stage and append spam heuristics."""
    summariser = table._call_role(  # pylint: disable=protected-access
        "summariser",
        SUMMARISER_SYSTEM_PROMPT,
        SUMMARISER_USER_PROMPT.format(post=post_text),
        _normalize_summariser,
        SummariserOut if "spam_likelihood_pct" in SummariserOut.model_fields else SummariserOut,
    )

    result = summariser.model_dump()
    has_ticker = any((a.get("ticker") or "").strip() for a in result.get("assets_mentioned", []))
    has_numbers = bool(result.get("numbers_mentioned"))
    has_catalysts = bool(result.get("claimed_catalysts"))

    text_lower = (post_text or "").lower()
    is_megathread = any(
        kw in text_lower
        for kw in (
            "daily discussion",
            "general discussion",
            "advice thread",
            "the lounge",
            "open thread",
            "daily thread",
            "megathread",
        )
    )

    try:
        spam_raw = table.llm.generate_json(  # pylint: disable=protected-access
            role="summariser_spam",
            system_prompt=SPAM_SYSTEM_PROMPT,
            user_prompt=SPAM_USER_PROMPT.format(
                post=post_text,
                summary_json=json.dumps(result, ensure_ascii=False),
                has_ticker=str(has_ticker).lower(),
                has_numbers=str(has_numbers).lower(),
                has_catalysts=str(has_catalysts).lower(),
                is_megathread=str(is_megathread).lower(),
            ),
        )
        spam_fixed = normalize_summariser_spam(spam_raw if isinstance(spam_raw, dict) else {})
        spam_obj = SummariserSpamOut.model_validate(spam_fixed)

        if "spam_likelihood_pct" in SummariserOut.model_fields:
            merged = result
            merged["spam_likelihood_pct"] = int(spam_obj.spam_likelihood_pct)
            merged["spam_why"] = spam_obj.why or ""
            summariser = SummariserOut.model_validate(merged)

        table._trace(  # pylint: disable=protected-access
            "Summariser spam",
            f"spam_likelihood_pct={spam_obj.spam_likelihood_pct}  why={spam_obj.why}",
        )
    except Exception as exc:  # pragma: no cover - defensive
        table._trace("SUMMARISER SPAM FOLLOW-UP ERROR", str(exc))  # pylint: disable=protected-access

    return summariser
