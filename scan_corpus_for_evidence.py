#!/usr/bin/env python3
"""
scan_corpus_for_evidence.py

Quickly search a large scraped corpus CSV (e.g., raw_posts_log.csv) for news/posts
relevant to a given post_id (e.g., t3_1nq4yev). It finds the reference row, gets its
scraped_at timestamp, and then scores other rows by relevance based on aliases/keywords.
By default it shows only rows DATED ON OR BEFORE the reference time to mimic real-time
trading knowledge. Optionally include "after" rows to audit the corpus.

Use this when testing round_table.py evidence detector: Is it missing articles that are in the corpus? We can find them manually with this.

Expected CSV columns (case-sensitive): platform, source, post_id, url, title, text, scraped_at

Usage examples:
    python scan_corpus_for_evidence.py --csv raw_posts_log.csv --post-id t3_1nq4yev --ticker ATCH --company "AtlasClear Holdings, Inc." --lookback-days 120 --top 25

    # To include items AFTER the post timestamp as an audit (they would not be tradable info at that time):
    python scan_corpus_for_evidence.py --csv raw_posts_log.csv --post-id t3_1nq4yev --ticker ATCH --company "AtlasClear Holdings, Inc." --include-after --top 40

Outputs:
- Prints a ranked table to stdout.
- Writes a CSV of the hits to: evidence_hits_<postid>.csv
"""

import argparse
import sys
import re
from datetime import datetime, timezone, timedelta

import pandas as pd


REQ_COLS = ["platform", "source", "post_id", "url", "title", "text", "scraped_at"]


def parse_args():
    ap = argparse.ArgumentParser(description="Scan corpus for evidence around a reference post.")
    ap.add_argument("--csv", required=True, help="Path to raw_posts_log.csv (or similar).")
    ap.add_argument("--post-id", required=True, help="Reference post_id to anchor time window (e.g., t3_1nq4yev).")
    ap.add_argument("--ticker", default=None, help="Primary ticker symbol (e.g., ATCH).")
    ap.add_argument("--company", default=None, help="Primary company/issuer name (e.g., AtlasClear Holdings, Inc.).")
    ap.add_argument("--aliases", default="", help="Comma-separated extra aliases (e.g., AtlasClear,WDCO,Wyoming Bancorp,Hanire).")
    ap.add_argument("--keywords", default="stock loan,securities lending,10-K,10K,Hanire,Wyoming Bancorp,WDCO,AtlasClear,Atlas Clear,loan revenue,high-margin",
                    help="Comma-separated extra keywords to match.")
    ap.add_argument("--lookback-days", type=int, default=120, help="How far back to search.")
    ap.add_argument("--lookahead-days", type=int, default=7, help="How far after to include when --include-after is set.")
    ap.add_argument("--include-after", action="store_true", help="Also include rows after the reference timestamp (for audit).")
    ap.add_argument("--top", type=int, default=30, help="Number of top results to print.")
    ap.add_argument("--min_score", type=float, default=1.0, help="Minimum score to keep.")
    ap.add_argument("--out", default=None, help="Optional output CSV path (default: evidence_hits_<postid>.csv)")
    return ap.parse_args()


def ensure_cols(df: pd.DataFrame):
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"ERROR: CSV missing required columns: {missing}")


def lc(s):
    return (s or "").lower()


def build_aliases(ticker: str | None, company: str | None, extra_aliases: str) -> list[str]:
    aliases = set()
    if ticker:
        t = ticker.strip().upper()
        aliases.update({t, f"${t}", f"nasdaq:{t}".lower(), f"nyse:{t}".lower(), t.lower()})
    if company:
        # include various whitespace/punctuation-normalized variants
        c = company.strip()
        aliases.add(c)
        aliases.add(c.lower())
        aliases.add(re.sub(r"[^\w]+", " ", c).strip().lower())  # "AtlasClear Holdings Inc"
        # try short forms like "AtlasClear", "Atlas Clear"
        parts = re.split(r"[, ]+", c)
        if len(parts) >= 1:
            aliases.add(parts[0])
            aliases.add(parts[0].lower())
            aliases.add(re.sub(r"(?i)inc\.?|corp\.?|ltd\.?", "", c).strip().lower())
    for a in (extra_aliases or "").split(","):
        a = a.strip()
        if a:
            aliases.add(a)
            aliases.add(a.lower())
    # Drop trivially short tokens
    aliases = {a for a in aliases if len(a) >= 3}
    return sorted(aliases)


def build_keywords(keywords_csv: str) -> list[str]:
    out = []
    for k in (keywords_csv or "").split(","):
        k = k.strip()
        if k:
            out.append(k)
            out.append(k.lower())
    return sorted(set(out))


def main():
    args = parse_args()

    # Load
    df = pd.read_csv(args.csv, dtype=str).fillna("")
    ensure_cols(df)

    # Parse times
    ts = pd.to_datetime(df["scraped_at"], errors="coerce", utc=True)
    df["_ts"] = ts

    # Find the reference row
    ref_mask = df["post_id"].astype(str) == str(args.post_id)
    if not ref_mask.any():
        raise SystemExit(f"ERROR: post_id '{args.post_id}' not found in CSV.")
    ref_row = df.loc[ref_mask].iloc[0]
    ref_ts = ref_row["_ts"]
    if pd.isna(ref_ts):
        raise SystemExit("ERROR: reference row has unparsable scraped_at timestamp.")

    # Time window
    since = ref_ts - pd.Timedelta(days=args.lookback_days)
    until = ref_ts + pd.Timedelta(days=args.lookahead_days) if args.include_after else ref_ts

    # Build search tokens
    aliases = build_aliases(args.ticker, args.company, args.aliases)
    keywords = build_keywords(args.keywords)

    # Prepare lowercase text
    bodies = (df["title"].astype(str) + " " + df["text"].astype(str)).fillna("")
    bodies_lc = bodies.str.lower()

    # Score function
    def score_row(text_lc: str, platform: str) -> float:
        score = 0.0
        # Alias hits are strongest
        for a in aliases:
            if a and a.lower() in text_lc:
                score += 3.0
        # Keywords are supportive
        for k in keywords:
            if k and k.lower() in text_lc:
                score += 1.0
        # Source preference (rss > stocktwits/x > reddit)
        p = (platform or "").lower()
        if p in ("rss", "news", "sec"):
            score += 1.5
        elif p in ("stocktwits", "x", "twitter"):
            score += 0.5
        return score

    # Candidate mask by time (and not the reference row itself)
    time_mask = (df["_ts"] >= since) & (df["_ts"] <= until)
    not_self = ~ref_mask
    cand = df.loc[time_mask & not_self].copy()

    # Compute scores
    cand["_text_lc"] = bodies_lc.loc[cand.index]
    cand["_score"] = [
        score_row(tlc, cand.loc[i, "platform"]) for i, tlc in zip(cand.index, cand["_text_lc"])
    ]

    # Filter and sort
    hits = cand.loc[cand["_score"] >= args.min_score].copy()
    hits.sort_values(["_score", "_ts"], ascending=[False, False], inplace=True)

    # Prepare output frame
    out_cols = ["platform", "source", "post_id", "url", "title", "scraped_at", "_score"]
    out = hits[out_cols].head(args.top).copy()

    # Save
    out_path = args.out or f"evidence_hits_{args.post_id}.csv"
    out.to_csv(out_path, index=False)

    # Print preview
    print(f"\nReference post_id: {args.post_id}")
    print(f"Reference time   : {ref_ts.isoformat()}")
    print(f"Window           : {since.isoformat()}  →  {until.isoformat()}  (include_after={args.include_after})")
    print(f"Aliases          : {aliases}")
    print(f"Keywords         : {keywords}\n")

    if out.empty:
        print("No hits found with the current settings. Try lowering --min_score, adding --aliases / --keywords, or enabling --include-after for audit.")
    else:
        with pd.option_context("display.max_colwidth", 120, "display.width", 160):
            print(out.to_string(index=False))

    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
