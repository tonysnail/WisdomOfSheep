from __future__ import annotations

import json
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


def _norm_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        text = str(value).strip()
    except Exception:  # noqa: BLE001
        return None
    return text or None


def _parse_interest_debug(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="ignore")
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            data = json.loads(text)
        except Exception:  # noqa: BLE001
            return {}
        return data if isinstance(data, dict) else {}
    if isinstance(raw, dict):
        return raw
    return {}


class InterestRecord(BaseModel):
    status: str = "pending"
    ticker: Optional[str] = None
    interest_score: Optional[float] = None
    interest_label: Optional[str] = None
    interest_why: Optional[str] = None
    council_recommended: Optional[bool] = None
    council_priority: Optional[str] = None
    calculated_at: Optional[str] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)


def build_interest_record(
    *,
    status: Any,
    ticker: Any,
    score: Any,
    label: Any,
    why: Any,
    recommended: Any,
    priority: Any,
    calculated_at: Any,
    error_code: Any,
    error_message: Any,
    debug_json: Any,
) -> Optional[InterestRecord]:
    status_raw = _norm_optional_str(status)
    score_val: Optional[float]
    try:
        score_val = float(score) if score is not None else None
    except Exception:  # noqa: BLE001
        score_val = None

    ticker_str = _norm_optional_str(ticker)
    label_str = _norm_optional_str(label)
    why_str = _norm_optional_str(why)
    priority_str = _norm_optional_str(priority)
    created_str = _norm_optional_str(calculated_at)
    error_code_str = _norm_optional_str(error_code)
    error_message_str = _norm_optional_str(error_message)

    recommended_bool: Optional[bool]
    if recommended is None:
        recommended_bool = None
    else:
        raw = str(recommended).strip()
        if raw == "" or raw.lower() == "none":
            recommended_bool = None
        else:
            try:
                recommended_bool = bool(int(float(raw)))
            except Exception:  # noqa: BLE001
                recommended_bool = bool(recommended)

    if (
        status_raw is None
        and score_val is None
        and error_code_str is None
        and error_message_str is None
    ):
        return None

    debug_dict = _parse_interest_debug(debug_json)
    metrics: Dict[str, Any] = {}
    if isinstance(debug_dict, dict):
        metrics_payload = debug_dict.get("metrics")
        if isinstance(metrics_payload, dict):
            metrics = dict(metrics_payload)
        else:
            metrics = {}
        for key in ("spam_pct", "platform", "source", "posted_at"):
            if key in debug_dict and key not in metrics:
                metrics[key] = debug_dict[key]

    status_clean = status_raw or ("ok" if score_val is not None else "pending")

    return InterestRecord(
        status=status_clean,
        ticker=ticker_str,
        interest_score=score_val,
        interest_label=label_str,
        interest_why=why_str,
        council_recommended=recommended_bool,
        council_priority=priority_str,
        calculated_at=created_str,
        error_code=error_code_str,
        error_message=error_message_str,
        metrics=metrics,
    )


__all__ = ["InterestRecord", "build_interest_record"]
