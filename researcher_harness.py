#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
researcher_harness.py — Run the two-stage researcher (LLM strategy → plan),
execute the technical tools, and pull sector-aware sentiment directly from the
conversation hub for a single source article (by post_id).

Key points:
- Hard-coded, repo-relative paths (no env vars).
- Run Round Table stages IN-PROCESS via round_table.run_stages_for_post().
- Patch round_table to expose EntityTimeframeOut when missing.
- Use the ticker conversation hub for sentiment parity with researcher.py.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ──────────────────────────── Hard-wired repo paths ───────────────────────────
REPO_ROOT = Path(__file__).resolve().parent
DB_DEFAULT = REPO_ROOT / "council" / "wisdom_of_sheep.sql"
TICKERS_CSV_DEFAULT = REPO_ROOT / "tickers" / "tickers_enriched.csv"

# Ensure repo-local imports resolve (backend, researcher, round_table, hub_adapter)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Lazy import to avoid circulars during test discovery
from hub_adapter import HubClient


# ─────────────────────────── Utility: tee stdout/stderr ───────────────────────
class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        total = 0
        for s in self._streams:
            total += s.write(data)
        return total

    def flush(self) -> None:
        for s in self._streams:
            s.flush()


# ──────────────────────────────── CLI parsing ─────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the researcher pipeline for a post")
    p.add_argument("post_id", help="Post identifier from the council database (e.g., t3_1ntw4u9)")
    p.add_argument("--db-path", default=str(DB_DEFAULT), help="Path to the SQLite database")
    p.add_argument("--tickers-csv", default=str(TICKERS_CSV_DEFAULT), help="Path to tickers_enriched.csv")
    p.add_argument("--lookback-days", type=int, default=5, help="Sentiment lookback window (days)")
    p.add_argument("--skip-technical", action="store_true", help="Skip executing the technical plan")
    p.add_argument("--skip-sentiment", action="store_true", help="Skip the sentiment analyser block")
    p.add_argument("--json-output", type=Path, help="Optional path to dump the full result payload as JSON")
    p.add_argument("--show-log", action="store_true", help="Print the captured researcher log after execution")
    p.add_argument(
        "--sentiment-verbose",
        action="store_true",
        help="Enable verbose sentiment analyser logging (printed live and included in captured log)",
    )
    return p.parse_args()


def _validate_paths(db_path: Path, tickers_csv: Path) -> None:
    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}")
    if not tickers_csv.exists():
        raise SystemExit(f"Ticker CSV not found: {tickers_csv}")


# ───────────────────────────── Stage I/O helpers ──────────────────────────────
def _load_stage(conn: sqlite3.Connection, post_id: str, stage: str) -> Optional[dict]:
    cur = conn.cursor()
    row = cur.execute("SELECT payload FROM stages WHERE post_id = ? AND stage = ?", (post_id, stage)).fetchone()
    if not row:
        return None
    try:
        return json.loads(row[0])
    except Exception:
        return None


def _patch_round_table_models():
    """
    Ensure round_table module has EntityTimeframeOut in its globals (some paths miss the import).
    """
    import round_table as rt
    if not hasattr(rt, "EntityTimeframeOut"):
        try:
            from council.common import EntityTimeframeOut  # type: ignore
        except Exception:
            from council.entity_stage import EntityTimeframeOut  # type: ignore
        setattr(rt, "EntityTimeframeOut", EntityTimeframeOut)
    return rt


def _run_stage_inproc(post_id: str, stage: str, timeout: float | None = None) -> None:
    """
    Run a single Round Table stage inside the current process using
    round_table.run_stages_for_post(). Avoids brittle CLI subprocesses.
    """
    rt = _patch_round_table_models()
    rt.run_stages_for_post(
        post_id=post_id,
        title="",           # keep DB title/text
        text="",
        stages=[stage],     # e.g., "summariser", "for", "against"
        autofill_deps=True,
        verbose=False,
        timeout=timeout,
        pretty_print_stage_output=False,
    )


def _ensure_source_stages(conn: sqlite3.Connection, post_id: str) -> None:
    for s in ("summariser", "for", "against"):
        if _load_stage(conn, post_id, s) is None:
            _run_stage_inproc(post_id, s)


# ───────────────────────── Researcher input builder ───────────────────────────
def _build_researcher_input(
    backend_app: Any,
    post_row: Any,
    summariser_payload: Dict[str, Any],
    bull_payload: Dict[str, Any],
    bear_payload: Dict[str, Any],
    claims_payload: Dict[str, Any],
    context_payload: Dict[str, Any],
    direction_payload: Dict[str, Any],
    entity_payload: Dict[str, Any],
) -> Dict[str, Any]:
    article_time = backend_app._normalize_article_time(post_row["posted_at"] or post_row["scraped_at"])
    summary_bullets = backend_app._clean_strings(summariser_payload.get("summary_bullets"))
    claims = backend_app._extract_claim_texts(claims_payload)
    context_bullets = backend_app._clean_strings(context_payload.get("context_bullets"))
    bull_points = backend_app._clean_strings(bull_payload.get("bull_points"))
    bear_points = backend_app._clean_strings(bear_payload.get("bear_points"))
    direction_est = backend_app._direction_estimate(direction_payload, summariser_payload)
    ticker = backend_app._primary_ticker(summariser_payload, entity_payload)
    if not ticker:
        raise SystemExit("Unable to resolve primary ticker from summariser/entity stages")
    return {
        "ticker": ticker,
        "article_time": article_time,
        "summary_bullets": summary_bullets,
        "claims": claims,
        "context_bullets": context_bullets,
        "direction_estimate": direction_est,
        "bull_points": bull_points,
        "bear_points": bear_points,
    }


def _default_technical_payload(plan: Dict[str, Any], reason: str) -> Dict[str, Any]:
    steps = plan.get("steps") if isinstance(plan, dict) else []
    return {"steps": steps, "results": [], "insights": [], "summary_lines": [reason], "status": "skipped"}


def _print_section(title: str, payload: Any) -> None:
    print("\n" + title)
    print("-" * len(title))
    print(json.dumps(payload, indent=2, ensure_ascii=False))


# ─────────────────────────────────── Main ─────────────────────────────────────
def main() -> None:
    args = _parse_args()
    db_path = Path(args.db_path).expanduser().resolve()
    tickers_csv = Path(args.tickers_csv).expanduser().resolve()
    _validate_paths(db_path, tickers_csv)

    # Repo-local imports (after sys.path injection)
    from backend import app as backend_app
    import researcher

    # Hard-wire backend paths (no env vars)
    backend_app.DB_PATH = db_path
    backend_app.TICKERS_DIR = tickers_csv.parent

    # Ensure the source post exists and its prerequisite stages are present
    with backend_app._connect() as conn:
        post_row = backend_app._q_one(conn, "SELECT * FROM posts WHERE post_id = ?", (args.post_id,))
        if not post_row:
            raise SystemExit(f"Post not found: {args.post_id}")

        _ensure_source_stages(conn, args.post_id)

        summariser_payload = _load_stage(conn, args.post_id, "summariser") or {}
        bull_payload = _load_stage(conn, args.post_id, "for") or {}
        bear_payload = _load_stage(conn, args.post_id, "against") or {}

        # Optional stages (may be absent)
        claims_payload = _load_stage(conn, args.post_id, "claims") or {}
        context_payload = _load_stage(conn, args.post_id, "context") or {}
        direction_payload = _load_stage(conn, args.post_id, "direction") or {}
        entity_payload = _load_stage(conn, args.post_id, "entity") or {}

    # Build the minimal input the researcher expects
    researcher_input = _build_researcher_input(
        backend_app,
        post_row,
        summariser_payload,
        bull_payload,
        bear_payload,
        claims_payload,
        context_payload,
        direction_payload,
        entity_payload,
    )

    # Patch round_table models once (ensures EntityTimeframeOut exists for any later in-proc runs)
    _patch_round_table_models()

    hub = HubClient(db_path=str(REPO_ROOT / "convos" / "conversations.sqlite"), model="mistral")

    # Run Stage 1 + Stage 2 planning with captured console logs
    log_buffer = io.StringIO()
    tee_stdout = _Tee(sys.stdout, log_buffer)
    tee_stderr = _Tee(sys.stderr, log_buffer)
    with contextlib.redirect_stdout(tee_stdout), contextlib.redirect_stderr(tee_stderr):
        stage1, stage2_raw, plan, session_id = researcher.run_two_stage(researcher_input)

    # --- Split plan steps so we don't run sentiment tools in the technical executor
    TECH_TOOLS = {
        "price_window", "compute_indicators", "trend_strength", "volatility_state",
        "support_resistance_check", "bollinger_breakout_scan", "obv_trend", "mfi_flow",
    }
    SENTIMENT_TOOLS = {"news_hub_score", "news_hub_ask_as_of"}
    all_steps = (plan or {}).get("steps") or []
    technical_steps = [
        s for s in all_steps
        if (s.get("tool") in TECH_TOOLS)
           or ("technical" in (s.get("covers") or []))
           or ("timing" in (s.get("covers") or []) and s.get("tool") in TECH_TOOLS)
    ]
    sentiment_steps_leaked = [
        s for s in all_steps
        if (s.get("tool") in SENTIMENT_TOOLS) or ("sentiment" in (s.get("covers") or []))
    ]
    # Build a plan object containing only the technical/timing tools
    tech_only_plan = dict(plan or {})
    tech_only_plan["steps"] = technical_steps

    # Execute technical plan (or skip)
    if args.skip_technical:
        technical_payload = _default_technical_payload(plan, "Technical analysis skipped via CLI option")
        technical_summary: List[str] = technical_payload["summary_lines"]
    else:
        # NOTE: we intentionally pass the filtered plan here
        technical_payload, technical_summary = backend_app._execute_technical_plan(tech_only_plan)
        # If any sentiment tools were present in the plan, annotate the technical block for transparency
        if sentiment_steps_leaked:
            technical_payload = dict(technical_payload or {})
            notes = list(technical_payload.get("summary_lines") or [])
            notes.append(f"Skipped {len(sentiment_steps_leaked)} sentiment step(s) "
                         "(e.g., news_hub_score) from technical executor.")
            technical_payload["summary_lines"] = notes

    # Execute sector-aware sentiment via the conversation hub (or skip)
    if args.skip_sentiment:
        sentiment_payload = {
            "ticker": researcher_input["ticker"],
            "error": "Sentiment skipped via CLI option",
        }
    else:
        sentiment_payload = hub.score(
            ticker=researcher_input["ticker"],
            as_of=researcher_input["article_time"],
            days=int(args.lookback_days),
            channel="all",
            peers=None,
            burst_hours=6,
        )
        sentiment_payload["narrative"] = hub.ask_as_of(
            ticker=researcher_input["ticker"],
            as_of=researcher_input["article_time"],
            q="Tone, key drivers, and any near-term risks in 2–3 lines.",
        )

    # Summaries + final narrative
    sentiment_summary = backend_app._summarize_sentiment(sentiment_payload)
    summary_text = backend_app._build_research_summary_text(
        researcher_input["ticker"],
        researcher_input["article_time"],
        stage1.get("hypotheses", []) if isinstance(stage1, dict) else [],
        stage1.get("rationale", "") if isinstance(stage1, dict) else "",
        technical_summary,
        sentiment_summary,
    )

    # Console report
    print("\n" + "#" * 80)
    print(f"Researcher pipeline results for post {args.post_id}")
    print("#" * 80)
    _print_section("Stage 1 (hypotheses)", stage1)
    _print_section("Stage 2 (raw model output)", stage2_raw)
    _print_section("Stage 2 (normalized plan)", plan)
    _print_section("Technical block", technical_payload)
    _print_section("Sentiment block", sentiment_payload)

    print("\nSummary text")
    print("------------")
    print(summary_text)

    if args.show_log:
        print("\nCaptured researcher log")
        print("-----------------------")
        print(log_buffer.getvalue())

    # Optional JSON dump
    result_payload = {
        "post_id": args.post_id,
        "session_id": session_id,
        "input": researcher_input,
        "stage1": stage1,
        "stage2_raw": stage2_raw,
        "plan": plan,
        "technical": technical_payload,
        "sentiment": sentiment_payload,
        "technical_summary": technical_summary,
        "sentiment_summary": sentiment_summary,
        "summary_text": summary_text,
        "log": log_buffer.getvalue(),
    }
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote JSON payload to {args.json_output}")


if __name__ == "__main__":
    main()
