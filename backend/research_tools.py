from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .conversation import compute_ticker_signal, fetch_recent_deltas, get_conversation_hub

TECH_TOOLS: set[str] = {
    "price_window",
    "compute_indicators",
    "trend_strength",
    "volatility_state",
    "support_resistance_check",
    "bollinger_breakout_scan",
    "obv_trend",
    "mfi_flow",
}
SENTIMENT_TOOLS: set[str] = {"news_hub_score", "news_hub_ask_as_of"}


def _try_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_tool_label(tool: str) -> str:
    return tool.replace("_", " ").replace("-", " ").title()


def _format_tool_result(tool: str, result: Dict[str, Any]) -> str:
    if not isinstance(result, dict):
        return f"{_format_tool_label(tool)}: no data"

    if tool == "price_window":
        rows = result.get("data")
        if isinstance(rows, list) and rows:
            first = rows[0]
            last = rows[-1]
            first_close = _try_float(first.get("Close"))
            last_close = _try_float(last.get("Close"))
            if first_close is not None and last_close is not None:
                change = last_close - first_close
                pct = (change / first_close * 100.0) if first_close else None
                pct_str = f" ({pct:+.1f}%)" if pct is not None else ""
                return f"Price window close {last_close:.2f}{pct_str}".strip()
        note = result.get("note") or "no data"
        return f"Price window: {note}"

    if tool == "compute_indicators":
        rsi = _try_float(result.get("rsi14"))
        macd = result.get("macd") if isinstance(result.get("macd"), dict) else {}
        hist = _try_float(macd.get("hist")) if isinstance(macd, dict) else None
        close = _try_float(result.get("close"))
        parts = []
        if rsi is not None:
            parts.append(f"RSI14 {rsi:.1f}")
        if hist is not None:
            parts.append(f"MACD hist {hist:+.2f}")
        if close is not None:
            parts.append(f"Close {close:.2f}")
        if not parts:
            note = result.get("note") or "no indicators"
            parts.append(str(note))
        return "; ".join(parts)

    if tool == "trend_strength":
        direction = result.get("direction")
        strength = result.get("strength")
        slope = _try_float(result.get("slope_pct_per_day"))
        r2 = _try_float(result.get("r2"))
        parts = []
        if direction:
            parts.append(f"Trend {direction}")
        if strength is not None:
            parts.append(f"Strength {strength}")
        if slope is not None:
            parts.append(f"Slope {slope:+.2f}%/day")
        if r2 is not None:
            parts.append(f"R² {r2:.2f}")
        return "; ".join(parts) or "Trend strength: no data"

    if tool == "volatility_state":
        state = result.get("state")
        ratio = _try_float(result.get("ratio"))
        rv = _try_float(result.get("realized_vol_annual_pct"))
        if state or ratio is not None or rv is not None:
            parts = []
            if state:
                parts.append(f"Vol {state}")
            if ratio is not None:
                parts.append(f"Ratio {ratio:.2f}")
            if rv is not None:
                parts.append(f"RV {rv:.2f}%")
            return "; ".join(parts)
        note = result.get("note") or "volatility unavailable"
        return str(note)

    if tool == "support_resistance_check":
        sup = _try_float(result.get("nearest_support"))
        res = _try_float(result.get("nearest_resistance"))
        pct_sup = _try_float(result.get("distance_to_support_pct"))
        pct_res = _try_float(result.get("distance_to_resistance_pct"))
        parts = []
        if sup is not None:
            if pct_sup is not None:
                parts.append(f"Support {sup:.2f} ({pct_sup:+.1f}%)")
            else:
                parts.append(f"Support {sup:.2f}")
        if res is not None:
            if pct_res is not None:
                parts.append(f"Resistance {res:.2f} ({pct_res:+.1f}%)")
            else:
                parts.append(f"Resistance {res:.2f}")
        if parts:
            return "; ".join(parts)
        note = result.get("note") or "no levels"
        return str(note)

    if tool == "bollinger_breakout_scan":
        event = result.get("last_event")
        date = result.get("last_event_date")
        pct_b = _try_float(result.get("%b"))
        bw = _try_float(result.get("bandwidth"))
        parts = []
        if event:
            parts.append(f"Last {event.replace('_', ' ')}")
        if date:
            parts.append(f"on {date}")
        if pct_b is not None:
            parts.append(f"%B {pct_b:.2f}")
        if bw is not None:
            parts.append(f"Bandwidth {bw:.3f}")
        return " ".join(parts).strip() or "Bollinger scan: no signal"

    if tool == "obv_trend":
        trend = result.get("trend")
        slope = _try_float(result.get("slope"))
        r2 = _try_float(result.get("r2"))
        parts = []
        if trend:
            parts.append(f"OBV {trend}")
        if slope is not None:
            parts.append(f"Slope {slope:+.0f}")
        if r2 is not None:
            parts.append(f"R² {r2:.2f}")
        return "; ".join(parts) or "OBV trend: no data"

    if tool == "mfi_flow":
        mfi = _try_float(result.get("mfi"))
        state = result.get("state")
        parts = []
        if mfi is not None:
            parts.append(f"MFI {mfi:.0f}")
        if state:
            parts.append(state)
        return " ".join(parts).strip() or "MFI flow: no data"

    note = result.get("note")
    return f"{_format_tool_label(tool)}: {note or 'no data'}"


def summarize_technical_results(results: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    insights: List[Dict[str, Any]] = []
    lines: List[str] = []
    for record in results:
        tool = record.get("tool") or "unknown"
        status = record.get("status") or ("ok" if record.get("result") is not None else "error")
        if status != "ok":
            text = f"{_format_tool_label(tool)}: {record.get('error', 'failed')}"
        else:
            text = _format_tool_result(tool, record.get("result") or {})
        insights.append({"tool": tool, "text": text, "status": status})
        lines.append(text)
    return insights, lines


def summarize_sentiment(data: Dict[str, Any]) -> str:
    if "error" in data:
        return f"Sentiment unavailable: {data['error']}"
    ticker = data.get("ticker")
    lookback = data.get("lookback_days")
    ticker_series = data.get("series_ticker")
    sector_series = data.get("series_sector")
    parts: List[str] = []
    if isinstance(ticker_series, list) and ticker_series:
        last = ticker_series[-1]
        avg = _try_float(last.get("avg_combined"))
        posts = last.get("posts")
        if avg is not None:
            part = f"Ticker avg {avg:+.2f}"
            if posts is not None:
                part += f" across {posts} post(s)"
            parts.append(part)
    if isinstance(sector_series, list) and sector_series:
        last = sector_series[-1]
        avg = _try_float(last.get("avg_combined"))
        posts = last.get("posts")
        if avg is not None:
            part = f"Sector avg {avg:+.2f}"
            if posts is not None:
                part += f" across {posts} post(s)"
            parts.append(part)
    counts = data.get("counts")
    if isinstance(counts, dict):
        considered = counts.get("considered")
        if considered:
            parts.append(f"Considered {considered} post(s)")
    prefix = f"{ticker} sentiment ({lookback}d)" if ticker else "Sentiment"
    return f"{prefix}: " + ("; ".join(parts) if parts else "no signal")


def build_research_summary_text(
    ticker: str,
    article_time: str,
    hypotheses: List[Dict[str, Any]],
    rationale: str,
    technical_lines: Sequence[str],
    sentiment_summary: Optional[str],
) -> str:
    lines: List[str] = [f"Research focus: {ticker}", f"Article time: {article_time}"]
    if hypotheses:
        lines.append("")
        lines.append("Hypotheses:")
        for idx, hyp in enumerate(hypotheses, start=1):
            if not isinstance(hyp, dict):
                continue
            text = str(hyp.get("text") or "").strip()
            hyp_type = str(hyp.get("type") or "").strip()
            if text:
                if hyp_type:
                    lines.append(f"{idx}. ({hyp_type}) {text}")
                else:
                    lines.append(f"{idx}. {text}")
    if rationale:
        lines.append("")
        lines.append(f"Rationale: {rationale}")
    if technical_lines:
        lines.append("")
        lines.append("Technical checks:")
        for entry in technical_lines:
            lines.append(f"- {entry}")
    if sentiment_summary:
        lines.append("")
        lines.append(sentiment_summary)
    return "\n".join(lines).strip()


def execute_technical_plan(plan: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    from technical_analyser import (
        tool_bollinger_breakout_scan,
        tool_compute_indicators,
        tool_mfi_flow,
        tool_obv_trend,
        tool_price_window,
        tool_support_resistance_check,
        tool_trend_strength,
        tool_volatility_state,
    )

    steps = plan.get("steps") if isinstance(plan, dict) else None
    if not isinstance(steps, list):
        steps = []

    results: List[Dict[str, Any]] = []
    for idx, raw_step in enumerate(steps):
        record: Dict[str, Any] = {
            "index": idx,
            "tool": None,
            "status": "skipped",
        }
        if not isinstance(raw_step, dict):
            record["error"] = "invalid_step"
            results.append(record)
            continue

        tool = raw_step.get("tool")
        record.update(
            {
                "tool": tool,
                "covers": raw_step.get("covers") or [],
                "tests": raw_step.get("tests") or [],
                "pass_if": raw_step.get("pass_if"),
                "fail_if": raw_step.get("fail_if"),
                "why": raw_step.get("why"),
            }
        )

        args = raw_step.get("args")
        if not isinstance(args, dict):
            record["status"] = "error"
            record["error"] = "invalid_args"
            results.append(record)
            continue

        try:
            if tool == "price_window":
                res = tool_price_window(
                    str(args["ticker"]).strip(),
                    str(args["from"]).strip(),
                    str(args["to"]).strip(),
                    str(args.get("interval", "1d")).strip() or "1d",
                )
            elif tool == "compute_indicators":
                res = tool_compute_indicators(
                    str(args["ticker"]).strip(),
                    int(args.get("window_days", 60)),
                )
            elif tool == "trend_strength":
                res = tool_trend_strength(
                    str(args["ticker"]).strip(),
                    int(args.get("lookback_days", 30)),
                )
            elif tool == "volatility_state":
                res = tool_volatility_state(
                    str(args["ticker"]).strip(),
                    int(args.get("days", 20)),
                    int(args.get("baseline_days", 10)),
                )
            elif tool == "support_resistance_check":
                res = tool_support_resistance_check(
                    str(args["ticker"]).strip(),
                    int(args.get("days", 30)),
                )
            elif tool == "bollinger_breakout_scan":
                res = tool_bollinger_breakout_scan(
                    str(args["ticker"]).strip(),
                    int(args.get("days", 20)),
                )
            elif tool == "obv_trend":
                res = tool_obv_trend(
                    str(args["ticker"]).strip(),
                    int(args.get("lookback_days", 30)),
                )
            elif tool == "mfi_flow":
                res = tool_mfi_flow(
                    str(args["ticker"]).strip(),
                    int(args.get("period", 14)),
                )
            else:
                record["status"] = "skipped"
                record["error"] = "non-technical-tool"
                results.append(record)
                continue

            record["status"] = "ok"
            record["result"] = res
        except Exception as exc:  # noqa: BLE001
            record["status"] = "error"
            record["error"] = str(exc)
        results.append(record)

    insights, summary_lines = summarize_technical_results(results)
    status = "ok"
    if not results:
        status = "empty"
    elif any(r.get("status") == "error" for r in results):
        status = "partial"

    payload = {
        "steps": steps,
        "results": results,
        "insights": insights,
        "summary_lines": summary_lines,
        "status": status,
    }
    return payload, summary_lines


def run_sentiment_block(
    ticker: str,
    article_time: str,
    *,
    channel: str = "social",
    lookback_days: int = 7,
    burst_hours: int = 6,
    conversation_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    hub = get_conversation_hub()
    if not hub:
        raise RuntimeError("conversation-hub-unavailable")

    store = getattr(hub, "store", None)
    if store is None:
        raise RuntimeError("conversation-store-unavailable")

    norm_ticker = (ticker or "").strip().upper()
    if not norm_ticker:
        raise ValueError("ticker-required")

    if not callable(compute_ticker_signal):
        raise RuntimeError("compute_ticker_signal-unavailable")

    try:
        score = compute_ticker_signal(
            store,
            norm_ticker,
            article_time,
            lookback_days=lookback_days,
            peers=None,
            channel_filter=channel,
            burst_hours=burst_hours,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "error": f"hub-score-failed: {exc}",
            "ticker": norm_ticker,
            "as_of": article_time,
            "channel": channel,
            "window_days": lookback_days,
        }

    signal = score.get("signal") if isinstance(score, dict) else None
    if not isinstance(signal, dict):
        signal = {}

    des_raw = signal.get("des_raw") if isinstance(signal.get("des_raw"), (int, float)) else None
    des_sector = signal.get("des_sector") if isinstance(signal.get("des_sector"), (int, float)) else None
    des_idio = signal.get("des_idio") if isinstance(signal.get("des_idio"), (int, float)) else None
    confidence = signal.get("confidence") if isinstance(signal.get("confidence"), (int, float)) else None
    n_deltas_raw = signal.get("n_deltas")
    n_deltas = int(n_deltas_raw) if isinstance(n_deltas_raw, (int, float)) else None

    recent_deltas = fetch_recent_deltas(store, norm_ticker, article_time, limit=6)

    payload: Dict[str, Any] = {
        "ticker": norm_ticker,
        "as_of": article_time,
        "channel": score.get("channel") if isinstance(score.get("channel"), str) else channel,
        "window_days": score.get("window_days") if isinstance(score.get("window_days"), int) else lookback_days,
        "burst_hours": score.get("burst_hours") if isinstance(score.get("burst_hours"), int) else burst_hours,
        "des_raw": des_raw,
        "des_sector": des_sector,
        "des_idio": des_idio,
        "confidence": confidence,
        "n_deltas": n_deltas,
        "DES_adj": (des_idio or 0.0) * (confidence or 0.0) if (des_idio is not None and confidence is not None) else None,
        "raw": score,
    }

    if recent_deltas:
        payload["deltas"] = recent_deltas
        payload["latest_delta"] = recent_deltas[-1]

    if isinstance(conversation_payload, dict):
        delta = conversation_payload.get("delta")
        if isinstance(delta, dict):
            who = delta.get("who") or []
            matches = norm_ticker in {str(w).strip().upper() for w in who if isinstance(w, str)}
            if matches:
                payload["article_delta"] = {
                    "t": delta.get("t"),
                    "sum": delta.get("sum"),
                    "dir": delta.get("dir"),
                    "impact": delta.get("impact"),
                    "why": delta.get("why"),
                    "chan": delta.get("chan"),
                    "cat": delta.get("cat"),
                    "src": delta.get("src"),
                    "url": delta.get("url"),
                }

    try:
        narrative = hub.ask_as_of(
            norm_ticker,
            "Summarize tone and catalysts.",
            article_time,
        )
    except Exception as exc:  # noqa: BLE001
        payload["narrative_error"] = f"hub-narrative-failed: {exc}"
    else:
        if narrative:
            payload["narrative"] = narrative

    return payload


__all__ = [
    "TECH_TOOLS",
    "SENTIMENT_TOOLS",
    "build_research_summary_text",
    "execute_technical_plan",
    "run_sentiment_block",
    "summarize_sentiment",
    "summarize_technical_results",
]
