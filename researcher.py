#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-stage research planner (Strategy -> Tool Plan) using local Ollama + Mistral.

Model:   mistral
Host:    http://localhost:11434
Stages:  1) Balanced hypotheses  2) Tool plan (timing + technical + sentiment)
"""

import json, os, re, uuid, requests
from datetime import datetime, timedelta, timezone, date
from textwrap import indent
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_CONVOS_DB = ROOT / "convos" / "conversations.sqlite"



# ----------------------------- Config (Mistral) ------------------------------
OLLAMA_HOST = "http://localhost:11434"
MODEL       = "mistral"
TEMPERATURE = 0.15
TIMEOUT     = None
try:
    OLLAMA_THREADS = max(1, int(os.getenv("WOS_OLLAMA_THREADS", "4")))
except (TypeError, ValueError):
    OLLAMA_THREADS = 4

def _ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def log_event(message: str) -> None:
    print(f"[{_ts()}] {message}")

# ------------------------------- Utilities -----------------------------------
def get_hub(*, db_path: str | Path = DEFAULT_CONVOS_DB, model: str = MODEL):
    from hub_adapter import HubClient
    # construct a NEW client (and therefore a NEW sqlite3 connection) each call
    return HubClient(db_path=str(Path(db_path)), model=model)
    
def extract_json_block(s: str):
    if not s: return None
    t = s.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t).strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    i, j = t.find("{"), t.rfind("}")
    if 0 <= i < j:
        try: return json.loads(t[i:j+1])
        except Exception: return None
    return None

def ollama_chat(messages, *, model=MODEL, session_id=None, label=None, timeout=TIMEOUT):
    print("\n" + "="*80)
    print(f"🧠  LLM CALL → {label or 'Unnamed'} (model={model}, session={session_id})")
    print("="*80)
    for m in messages:
        print(f"\n--- {m['role'].upper()} ---\n{indent(m['content'].strip(), '  ')}")
    request_kwargs = {
        "json": {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": TEMPERATURE, "num_thread": OLLAMA_THREADS},
            **({"session": session_id} if session_id else {}),
        }
    }
    if timeout is not None:
        request_kwargs["timeout"] = timeout
    resp = requests.post(f"{OLLAMA_HOST.rstrip('/')}/api/chat", **request_kwargs)
    resp.raise_for_status()
    text = (resp.json().get("message") or {}).get("content", "")
    print("\n--- RAW MODEL OUTPUT ---")
    print(indent(text.strip(), "  "))
    print("="*80 + "\n")
    return text

def _date_from_iso_z(iso_s: str) -> date:
    if not iso_s: return datetime.now(timezone.utc).date()
    try:
        dt = datetime.fromisoformat(iso_s.replace("Z", "+00:00")) if iso_s.endswith("Z") else datetime.fromisoformat(iso_s)
        if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
        return dt.date()
    except Exception:
        return datetime.now(timezone.utc).date()

# -------------------------------- Prompts ------------------------------------
GLOBAL_RULE = "Return STRICT JSON ONLY. Include a final 'why' string. Do not add commentary outside the JSON."
TOOL_PALETTE = ("Use ONLY: price_window, compute_indicators, trend_strength, volatility_state, "
                "support_resistance_check, bollinger_breakout_scan, obv_trend, mfi_flow, "
                "news_hub_score, news_hub_ask_as_of.")

SYS_STAGE1 = (
    "You are a market research strategist. Given article context + bull/bear points, write a BALANCED STRATEGY:\n"
    "- 2–3 short, NON-DUPLICATE hypotheses (≤20 words each).\n"
    "- Each hypothesis has a 'type' = continuation | reversion.\n"
    "No tools, no external facts. " + GLOBAL_RULE + r""" Schema:
{
  "hypotheses": [{"text":"string","type":"continuation|reversion"}],
  "rationale": "string",
  "why": "string"
}"""
)

SYS_STAGE2 = (
    "You are a research planner. Convert remembered hypotheses into Python tool calls. "
    + TOOL_PALETTE + " " + GLOBAL_RULE + r""" STRICT JSON Schema:
{
  "steps": [
    {"tool":"string","args":{},"covers":["timing"|"technical"|"sentiment"],"tests":[int],"pass_if":"string","fail_if":"string","why":"string"}
  ],
  "why":"string"
}
Rules:
- 3–5 steps total.
- Args MUST match exact schemas:
  - price_window: {"ticker":"T","from":"YYYY-MM-DD","to":"YYYY-MM-DD","interval":"1d"}
  - compute_indicators: {"ticker":"T","window_days":N}
  - trend_strength: {"ticker":"T","lookback_days":N}
  - volatility_state: {"ticker":"T","days":N,"baseline_days":M}
  - support_resistance_check: {"ticker":"T","days":N}
  - bollinger_breakout_scan: {"ticker":"T","days":N}
  - obv_trend: {"ticker":"T","lookback_days":N}
  - mfi_flow: {"ticker":"T","period":N}
  - news_hub_score: {"ticker":"T","as_of":"ISO","days":N,"channel":"all|news|social","peers":["T1","T2"],"burst_hours":N}
  - news_hub_ask_as_of: {"ticker":"T","as_of":"ISO","q":"short question"}
- Coverage: exactly one 'timing', ≥1 'technical', ≥1 'sentiment'.
- 'tests' index Stage-1 hypotheses order.
- Keep 'pass_if'/'fail_if' ≤15 words, domain-specific.
- Anchor dates to ARTICLE_TIME_UTC.
"""
)

# ------------------------------ Input builder --------------------------------
def hood_inputs():
    return {
        "ticker": "HOOD",
        "article_time": "2025-09-29T22:51:08Z",
        "summary_bullets": [
            "Robinhood shares increased by 12% to a record high.",
            "CEO Tenev said prediction-market trading grew strongly on Robinhood."
        ],
        "claims": [
            "Robinhood shares rose over 12%.",
            "Customers transacted over 4B event contracts.",
            "2B occurred in Q3."
        ],
        "context_bullets": [
            "Retail flow into prediction markets is strong.",
            "Timing vs article unclear; move may have preceded."
        ],
        "direction_estimate": "up",
        "bull_points": [
            "Stock up 12% on strong prediction-market growth.",
            "Unique offering suggests edge vs peers."
        ],
        "bear_points": [
            "Surge may be temporary and unsustainable.",
            "No financials; profitability of prediction markets unclear."
        ]
    }

# ----------------------------- Prompt builders -------------------------------
def build_user_stage1(inp):
    j = lambda ls: "\n• " + "\n• ".join(ls) if ls else " (none)"
    return (
        f"TICKER: {inp['ticker']}\nARTICLE_TIME_UTC: {inp['article_time']}\n"
        f"SUMMARY_BULLETS:{j(inp['summary_bullets'])}\n"
        f"CLAIMS:{j(inp['claims'])}\n"
        f"CONTEXT:{j(inp['context_bullets'])}\n"
        f"DIRECTION_ESTIMATE: {inp['direction_estimate']}\n"
        f"BULL_POINTS:{j(inp.get('bull_points', []))}\n"
        f"BEAR_POINTS:{j(inp.get('bear_points', []))}\n\n"
        "Task: Using BOTH bullish and bearish reasoning, produce 2–3 short, NON-DUPLICATE hypotheses "
        "(≤20 words each), labeling each as type=continuation or type=reversion. "
        "Add a one-line rationale. JSON only."
    )

def build_user_stage2(inp):
    at = _date_from_iso_z(inp['article_time'])
    fr = (at - timedelta(days=3)).isoformat()
    to = at.isoformat()
    return (
        f"TICKER: {inp['ticker']}\nARTICLE_TIME_UTC: {inp['article_time']}\n\n"
        "Use:\n"
        f"- timing window: from {fr} to {to} (inclusive), interval 1d\n"
        "- technical lookbacks: 20–60 days; sentiment lookback: 7 days\n\n"
        "Convert the remembered hypotheses into 3–5 tool steps using the palette, "
        "meeting the coverage rule (exactly one 'timing', ≥1 'technical', ≥1 'sentiment'). "
        "Include 'tests' (indices into Stage-1 hypotheses), and concise pass_if/fail_if. "
        "Keep args minimal. JSON only."
    )

# ------------------------------ Normalization --------------------------------
def _pf(tool):
    if tool == "price_window":           return ("Gap or large move pre-article", "No abnormal move pre-article")
    if tool == "compute_indicators":     return ("RSI<60 or near mid-band",       "RSI>70 or far above band")
    if tool == "trend_strength":         return ("ADX rising or strong trend",     "ADX weak or falling")
    if tool == "volatility_state":       return ("Volatility expansion ongoing",   "Volatility contracting")
    if tool == "support_resistance_check": return ("Room below next resistance",  "At/above recent resistance")
    if tool == "bollinger_breakout_scan":  return ("Upper band break with hold",  "Failed break / revert inside")
    if tool == "obv_trend":              return ("OBV confirms higher highs",      "OBV diverges vs price")
    if tool == "mfi_flow":               return ("MFI rising / buy flow",         "MFI falling / sell flow")
    if tool == "news_hub_score":         return ("3-day sentiment slope positive", "Sentiment turns negative")
    if tool == "news_hub_ask_as_of":     return ("Narrative flags key drivers",    "Narrative lacks clear drivers")
    return ("OK","Not OK")

def normalize_stage1(js):
    out = {"hypotheses": [], "rationale": js.get("rationale","").strip(), "why": js.get("why","").strip()}
    seen = set()
    for h in (js.get("hypotheses") or []):
        txt = (h.get("text") or "").strip()
        typ = (h.get("type") or "").strip().lower()
        if not txt or typ not in {"continuation","reversion"}: continue
        if txt.lower() in seen: continue
        seen.add(txt.lower())
        out["hypotheses"].append({"text": txt[:120], "type": typ})
        if len(out["hypotheses"]) >= 3: break
    return out

def normalize_plan(js, ticker, article_time, hyp_count):
    steps = []
    for s in (js.get("steps") or []):
        tool = (s.get("tool") or "").strip()
        args = dict(s.get("args") or {})
        covers = list(s.get("covers") or [])
        tests = list(s.get("tests") or [])
        pass_if, fail_if = (s.get("pass_if") or ""), (s.get("fail_if") or "")
        why = (s.get("why") or tool).strip() or tool

        # fix schema + inject ticker/defaults
        if tool == "price_window":
            args = {
                "ticker": ticker,
                "from": args.get("from") or args.get("start_date"),
                "to": args.get("to") or args.get("end_date"),
                "interval": args.get("interval","1d")
            }
        elif tool == "compute_indicators":
            args = {"ticker": ticker, "window_days": int(args.get("window_days") or args.get("lookback") or 60)}
        elif tool == "trend_strength":
            args = {"ticker": ticker, "lookback_days": int(args.get("lookback_days") or 30)}
        elif tool == "volatility_state":
            args = {"ticker": ticker, "days": int(args.get("days") or 20), "baseline_days": int(args.get("baseline_days") or 10)}
        elif tool in {"support_resistance_check","bollinger_breakout_scan"}:
            args = {"ticker": ticker, "days": int(args.get("days") or 20)}
        elif tool == "obv_trend":
            args = {"ticker": ticker, "lookback_days": int(args.get("lookback_days") or 30)}
        elif tool == "mfi_flow":
            args = {"ticker": ticker, "period": int(args.get("period") or 14)}
        elif tool == "news_hub_score":
            args = {
                "ticker": ticker,
                "as_of": args.get("as_of") or article_time,
                "days": int(args.get("days") or 7),
                "channel": (args.get("channel") or "all"),
                "peers": list(args.get("peers") or []),
                "burst_hours": int(args.get("burst_hours") or 6),
            }
        elif tool == "news_hub_ask_as_of":
            args = {
                "ticker": ticker,
                "as_of": args.get("as_of") or article_time,
                "q": (args.get("q") or "Summarize tone and catalysts."),
            }
        else:
            continue  # unknown tool → drop

        # covers fix (OBV/MFI are technical)
        if tool == "price_window": covers = ["timing"]
        elif tool in {"news_hub_score", "news_hub_ask_as_of"}: covers = ["sentiment"]
        elif tool in {"obv_trend","mfi_flow","compute_indicators","trend_strength","volatility_state","support_resistance_check","bollinger_breakout_scan"}:
            covers = ["technical"]
        else:
            covers = covers or ["technical"]

        # tests clamp
        if hyp_count > 0:
            tests = [min(max(0,int(i)), hyp_count-1) for i in (tests or [0])]
        else:
            tests = []

        # pass/fail domainized
        p,f = _pf(tool)
        step = {
            "tool": tool,
            "args": args,
            "covers": covers,
            "tests": tests,
            "pass_if": pass_if[:60] if pass_if else p,
            "fail_if": fail_if[:60] if fail_if else f,
            "why": why
        }
        steps.append(step)

    # coverage enforcement: exactly one timing, ≥1 technical, ≥1 sentiment
    timing = [i for i,s in enumerate(steps) if "timing" in s["covers"]]
    if len(timing) == 0:
        # add timing from article window (3-day backfill)
        at = _date_from_iso_z(article_time); fr = (at - timedelta(days=3)).isoformat(); to = at.isoformat()
        steps.insert(0, {
            "tool":"price_window","args":{"ticker":ticker,"from":fr,"to":to,"interval":"1d"},
            "covers":["timing"],"tests":[0] if hyp_count else [],
            "pass_if": _pf("price_window")[0], "fail_if": _pf("price_window")[1], "why":"added timing"
        })
    elif len(timing) > 1:
        # keep the first timing, drop the rest
        keep = timing[0]
        steps = [s for i,s in enumerate(steps) if (i==keep or "timing" not in s["covers"])]

    if not any("technical" in s["covers"] for s in steps):
        steps.append({
            "tool":"compute_indicators","args":{"ticker":ticker,"window_days":60},
            "covers":["technical"],"tests":[0] if hyp_count else [],
            "pass_if": _pf("compute_indicators")[0], "fail_if": _pf("compute_indicators")[1], "why":"added technical"
        })
    if not any(s.get("tool") == "news_hub_score" for s in steps):
        steps.append({
            "tool":"news_hub_score",
            "args":{"ticker":ticker,"as_of":article_time,"days":7,"channel":"all","peers":[],"burst_hours":6},
            "covers":["sentiment"],"tests":[0] if hyp_count else [],
            "pass_if":"3-day sentiment slope positive","fail_if":"Sentiment turns negative","why":"added sentiment"
        })

    if not any("sentiment" in s["covers"] for s in steps):
        steps.append({
            "tool":"news_hub_score",
            "args":{"ticker":ticker,"as_of":article_time,"days":7,"channel":"all","peers":[],"burst_hours":6},
            "covers":["sentiment"],"tests":[0] if hyp_count else [],
            "pass_if":"3-day sentiment slope positive","fail_if":"Sentiment turns negative","why":"added sentiment"
        })

    # trim to 3–5 steps, prefer: timing + 2 technical + 1 sentiment
    # basic greedy keep order while meeting coverage
    out = []
    has_t, has_s = False, False
    for s in steps:
        if len(out) >= 5: break
        if "timing" in s["covers"]:
            if has_t: continue
            has_t = True
            out.append(s); continue
        if "sentiment" in s["covers"]:
            if has_s: continue
            has_s = True
            out.append(s); continue
        # technical
        out.append(s)
    # ensure min length 3
    out = out[:5]
    if len(out) < 3:
        # pad with a safe technical
        out.append({
            "tool":"obv_trend","args":{"ticker":ticker,"lookback_days":30},
            "covers":["technical"],"tests":[0] if hyp_count else [],
            "pass_if": _pf("obv_trend")[0], "fail_if": _pf("obv_trend")[1], "why":"pad technical"
        })

    return {"steps": out, "why": js.get("why","").strip() or "normalized plan with coverage guarantees"}


def execute_plan_steps(steps):
    """
    Run plan steps. Builds a NEW HubClient in this thread to avoid cross-thread
    sqlite reuse. Returns a dict with timing/technical/sentiment sub-blocks.
    """
    results = {"timing": {}, "technical": {}, "sentiment": {}}
    if not isinstance(steps, list) or not steps:
        return results

    # Build a fresh hub *in this thread* (no globals / no caching).
    hub = get_hub(db_path=DEFAULT_CONVOS_DB, model=MODEL)

    # Optional: collect a tiny trace for debugging
    trace = []

    for step in steps:
        if not isinstance(step, dict):
            continue
        tool = (step.get("tool") or "").strip()
        args = step.get("args") or {}

        if tool == "news_hub_score":
            try:
                r = hub.score(
                    ticker=(args.get("ticker") or "").upper(),
                    as_of=args.get("as_of") or "",
                    days=int(args.get("days", 7)),
                    channel=(args.get("channel") or "all"),
                    peers=(args.get("peers") or None),
                    burst_hours=int(args.get("burst_hours", 6)),
                )
                sig = r.get("signal") or {}
                results["sentiment"].update({
                    "des_raw":     sig.get("des_raw"),
                    "des_sector":  sig.get("des_sector"),
                    "des_idio":    sig.get("des_idio"),
                    "confidence":  sig.get("confidence"),
                    "n_deltas":    sig.get("n_deltas"),
                    "DES_adj":     (sig.get("des_idio") or 0.0) * (sig.get("confidence") or 0.0),
                    "channel":     r.get("channel"),
                    "window_days": r.get("window_days"),
                })
                trace.append(f"score[{args.get('ticker')}@{args.get('as_of')}:{args.get('days','7')}d/{args.get('channel','all')}] ok")
            except Exception as exc:  # noqa: BLE001
                results.setdefault("sentiment", {})["error"] = f"hub-score-failed: {exc}"
                trace.append(f"score[{args.get('ticker')}] error: {exc}")

        elif tool == "news_hub_ask_as_of":
            try:
                narrative = hub.ask_as_of(
                    ticker=(args.get("ticker") or "").upper(),
                    as_of=args.get("as_of") or "",
                    q=args.get("q", "Summarize tone and catalysts."),
                )
                results["sentiment"]["narrative"] = narrative
                trace.append(f"ask_as_of[{args.get('ticker')}@{args.get('as_of')}] ok")
            except Exception as exc:  # noqa: BLE001
                results.setdefault("sentiment", {})["narrative_error"] = f"hub-ask-failed: {exc}"
                trace.append(f"ask_as_of[{args.get('ticker')}] error: {exc}")

        # ignore non-hub tools here; technical/timing are executed elsewhere

    if trace:
        # lightweight breadcrumb for UI/debug panes
        results["sentiment"]["_trace"] = trace

    return results


# -------------------------------- Pipeline -----------------------------------
def run_two_stage(inp: dict):
    session_id = f"wos-{inp['ticker']}-{uuid.uuid4().hex[:6]}"
    ticker = inp.get('ticker', 'UNKNOWN')
    log_event(f"Starting researcher session {session_id} for {ticker}.")
    log_event(
        "Stage 1: Requesting balanced hypotheses "
        f"(summary={len(inp.get('summary_bullets') or [])}, "
        f"claims={len(inp.get('claims') or [])}, "
        f"bull={len(inp.get('bull_points') or [])}, "
        f"bear={len(inp.get('bear_points') or [])})."
    )
    # Stage 1
    m1 = [{"role":"system","content":SYS_STAGE1},{"role":"user","content":build_user_stage1(inp)}]
    raw1 = ollama_chat(m1, session_id=session_id, label="Stage 1: Strategy")
    js1 = normalize_stage1(extract_json_block(raw1) or {})
    log_event(
        "Stage 1 complete: "
        f"captured {len(js1.get('hypotheses') or [])} hypotheses "
        f"and rationale length {len((js1.get('rationale') or '').strip())}."
    )
    # Stage 2
    log_event("Stage 2: Building research plan from hypotheses.")
    m2 = [{"role":"system","content":SYS_STAGE2},{"role":"user","content":build_user_stage2(inp)}]
    raw2 = ollama_chat(m2, session_id=session_id, label="Stage 2: Plan")
    js2_raw = extract_json_block(raw2) or {}
    plan = normalize_plan(js2_raw, inp["ticker"], inp["article_time"], len(js1.get("hypotheses", [])))
    steps = plan.get("steps") if isinstance(plan, dict) else []
    log_event(f"Stage 2 complete: normalized plan with {len(steps or [])} steps.")
    execution_results = execute_plan_steps(steps or [])
    if isinstance(plan, dict):
        plan["results"] = execution_results
    if isinstance(steps, list) and steps:
        for idx, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                continue
            tool = step.get("tool") or "(unknown)"
            covers = ", ".join(step.get("covers") or []) or "unspecified"
            why = (step.get("why") or "").strip()
            summary = f"Step {idx}: {tool} covering {covers}."
            if why:
                summary += f" Reason: {why}"
            log_event(summary)
    log_event(f"Researcher session {session_id} finished.")
    return js1, js2_raw, plan, session_id

# --------------------------------- Main --------------------------------------
def main():
    inp = hood_inputs()
    js1, js2, plan, sess = run_two_stage(inp)

    print("\n" + "#"*80)
    print("✅ FINAL RESULTS")
    print("#"*80)
    print("\n=== STAGE 1: STRATEGY (balanced, typed) ===")
    print(json.dumps(js1, indent=2, ensure_ascii=False))
    print("\n=== STAGE 2: PLAN (model output) ===")
    print(json.dumps(js2, indent=2, ensure_ascii=False))
    print("\n=== STAGE 2: PLAN (normalized) ===")
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    print(f"\n(Session ID: {sess})")

if __name__ == "__main__":
    main()
