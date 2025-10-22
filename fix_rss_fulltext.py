#!/usr/bin/env python3
"""
Fix RSS rows in a raw posts log by fetching full article text — with verbose tracing
and the ability to drop unrepairable sources (default: investing.com).

Usage:
  python fix_rss_fulltext.py \
      --input "raw_posts_log copy.csv" \
      --output "raw_posts_log.fixed.csv" \
      --verbose --progress --autosave-every 100

To also save removed rows:
  python fix_rss_fulltext.py --input raw.csv --removed-out removed.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from time import perf_counter
from typing import Dict, Tuple, Optional, List
from urllib.parse import urlparse
import traceback

import pandas as pd

# optional progress bar
try:
    from tqdm import tqdm  # type: ignore
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

# --- Import your extractor ---
try:
    from rss_parser import extract_article_fulltext  # type: ignore
except Exception as e:
    print("ERROR: Could not import extract_article_fulltext from rss_parser.py\n", e, file=sys.stderr)
    sys.exit(1)


def _domain(u: str) -> str:
    try:
        return urlparse(str(u)).netloc.lower()
    except Exception:
        return ""


def _para_count(text: Optional[str]) -> int:
    if not text:
        return 0
    return sum(1 for p in text.split("\n\n") if p.strip())


def _classify_status(text: Optional[str]) -> str:
    if text and len(text.strip()) >= 400:
        return "ok"
    if text and len(text.strip()) >= 200:
        return "partial"
    if text and len(text.strip()) > 0:
        return "short"
    return "empty"


def fetch_fulltext_safe(url: str, verbose: bool = False, trace: bool = False) -> Tuple[str, Optional[str], Optional[str], str]:
    """
    Wrap extract_article_fulltext with error handling.
    Returns (final_url, title, text, status)
    """
    t0 = perf_counter()
    try:
        if verbose:
            print(f"    [fetch] start -> {url} (dom={_domain(url)})")
        out = extract_article_fulltext(url)
        t1 = perf_counter()
        final_url = out.get("final_url") or url
        title = out.get("title")
        text = out.get("text")
        status = _classify_status(text)
        if verbose:
            redir = " (redirected)" if final_url and final_url != url else ""
            print(f"    [fetch] done  <- {final_url}{redir} in {t1 - t0:.2f}s; "
                  f"len={len(text or '')} chars; paras={_para_count(text)}; status={status}")
        return final_url, title, text, status
    except Exception as ex:
        if verbose:
            print(f"    [fetch] ERROR: {type(ex).__name__}: {ex}")
            if trace:
                traceback.print_exc()
        return url, None, None, f"error:{type(ex).__name__}"


def _safe_write_csv(df: pd.DataFrame, path: str) -> None:
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)  # atomic on most platforms


def _parse_csv_list(val: Optional[str]) -> List[str]:
    if not val:
        return []
    return [s.strip() for s in val.split(",") if s.strip()]


def main():
    ap = argparse.ArgumentParser(description="Fix incomplete RSS links by fetching full article text (verbose) and drop troublesome sources.")
    ap.add_argument("--input", required=True, help="Input CSV (e.g., 'raw_posts_log copy.csv')")
    ap.add_argument("--output", default=None, help="Output CSV (default: add .fixed before extension)")
    ap.add_argument("--max", type=int, default=None, help="Optional max rows to process (rss only), for quick tests")
    ap.add_argument("--sleep", type=float, default=0.8, help="Delay between requests (seconds) to be polite")
    ap.add_argument("--verbose", action="store_true", help="Print detailed per-row tracing")
    ap.add_argument("--trace", action="store_true", help="Include exception tracebacks on errors")
    ap.add_argument("--progress", action="store_true", help="Show a tqdm progress bar (if installed)")
    ap.add_argument("--autosave-every", type=int, default=0, help="Autosave every N processed RSS rows (0 disables)")
    ap.add_argument("--only-missing", action="store_true", help="Only process rows with empty fetch_status or not 'ok'")

    # NEW: controls for dropping rows (defaults remove investing.com)
    ap.add_argument("--drop-domains", default="investing.com", help="Comma-separated list of domains to drop (match by netloc). Default: investing.com")
    ap.add_argument("--drop-sources", default="Investing.com Market News", help="Comma-separated list of source names to drop. Default: 'Investing.com Market News'")
    ap.add_argument("--removed-out", default=None, help="Optional CSV path to write removed rows (for audit)")

    args = ap.parse_args()

    in_path = args.input
    if not os.path.isfile(in_path):
        print(f"Input file not found: {in_path}", file=sys.stderr)
        sys.exit(2)

    # Default output path
    if args.output:
        out_path = args.output
    else:
        base, ext = os.path.splitext(in_path)
        out_path = f"{base}.fixed{ext or '.csv'}"

    t_start = perf_counter()
    if args.verbose:
        print(f"[load] reading CSV: {in_path}")

    df = pd.read_csv(in_path)

    # Ensure expected columns exist
    must_have = ["scraped_at", "platform", "source", "post_id", "url", "title", "text"]
    for col in must_have:
        if col not in df.columns:
            df[col] = ""
            if args.verbose:
                print(f"[columns] added missing column '{col}'")

    # Add/ensure new columns
    added_cols = []
    if "final_url" not in df.columns:
        df["final_url"] = ""
        added_cols.append("final_url")
    if "fetch_status" not in df.columns:
        df["fetch_status"] = ""
        added_cols.append("fetch_status")
    if args.verbose and added_cols:
        print(f"[columns] added new columns: {', '.join(added_cols)}")

    # Prep drop lists
    drop_domains = set(_parse_csv_list(args.drop_domains))
    drop_sources = set(_parse_csv_list(args.drop_sources))

    # Compute domain column (helps dropping & logging)
    if "domain" not in df.columns:
        df["domain"] = df["url"].astype(str).map(_domain)

    # Identify rows-to-drop (rss only)
    is_rss = df["platform"].astype(str).str.lower().eq("rss")
    to_drop_mask = (
        is_rss &
        (
            df["domain"].astype(str).isin(drop_domains) |
            df["source"].astype(str).isin(drop_sources)
        )
    )
    removed_df = df[to_drop_mask].copy()
    kept_df = df[~to_drop_mask].copy()

    if args.verbose:
        print(f"[filter] drop domains: {sorted(drop_domains)}")
        print(f"[filter] drop sources: {sorted(drop_sources)}")
        print(f"[filter] rows removed: {len(removed_df)}; rows kept: {len(kept_df)}")

    # Save removed rows if requested
    if args.removed_out:
        if args.verbose:
            print(f"[write] writing removed rows -> {args.removed_out}")
        _safe_write_csv(removed_df, args.removed_out)

    # Work only on kept_df from here
    df = kept_df.reset_index(drop=True)

    # Select RSS rows (remaining)
    is_rss = df["platform"].astype(str).str.lower().eq("rss")
    rss_idx = df[is_rss].index.tolist()

    if args.only_missing:
        rss_idx = [i for i in rss_idx if str(df.at[i, "fetch_status"]).strip().lower() != "ok"]

    if args.max is not None:
        rss_idx = rss_idx[: args.max]

    total = len(rss_idx)
    processed = 0

    # Caches and counters
    cache: Dict[str, Tuple[str, Optional[str], Optional[str], str]] = {}
    status_counts: Dict[str, int] = {}
    updates = {"text": 0, "title": 0, "final_url": 0}
    last_autosave_at = 0

    if args.verbose:
        total_rows = len(df)
        print(f"[plan] processing {total} RSS rows (of {total_rows} kept). "
              f"{'Only missing/!ok' if args.only_missing else 'All RSS'}")

    iter_indices = rss_idx
    if args.progress and _HAS_TQDM and not args.verbose:
        iter_indices = tqdm(rss_idx, total=total, unit="row")  # progress bar for non-verbose runs

    for count, idx in enumerate(iter_indices, 1):
        platform = str(df.at[idx, "platform"]).strip().lower()
        url = str(df.at[idx, "url"]).strip()
        source = str(df.at[idx, "source"]).strip()
        dom = _domain(url)

        if args.verbose:
            print(f"\n[{count}/{total}] {platform} :: {source}")
            print(f"  url: {url}")

        if platform != "rss":
            if args.verbose:
                print("  -> skip: platform is not 'rss'")
            continue

        if not url:
            df.at[idx, "fetch_status"] = "skip:no_url"
            if args.verbose:
                print("  -> skip: no URL")
            continue

        # Cache
        if url in cache:
            final_url, title, text, status = cache[url]
            if args.verbose:
                print(f"  cache: hit (status={status})")
        else:
            if args.verbose:
                print("  cache: miss")
            final_url, title, text, status = fetch_fulltext_safe(url, verbose=args.verbose, trace=args.trace)
            cache[url] = (final_url, title, text, status)
            # be polite to sources
            time.sleep(args.sleep)

        # Update row fields
        prev_text = str(df.at[idx, "text"]) if not pd.isna(df.at[idx, "text"]) else ""
        prev_title = str(df.at[idx, "title"]) if not pd.isna(df.at[idx, "title"]) else ""
        prev_final = str(df.at[idx, "final_url"]) if not pd.isna(df.at[idx, "final_url"]) else ""

        df.at[idx, "final_url"] = final_url
        df.at[idx, "fetch_status"] = status

        # Decide updates & report
        if text and len(text.strip()) > 0:
            if text != prev_text:
                df.at[idx, "text"] = text
                updates["text"] += 1
                if args.verbose:
                    print(f"  update: text (len {len(prev_text)} -> {len(text)})  paras={_para_count(text)}")
        if (not prev_title.strip()) and title:
            df.at[idx, "title"] = title
            updates["title"] += 1
            if args.verbose:
                print(f"  update: title -> {title[:96]}{'…' if len(title) > 96 else ''}")
        if final_url != prev_final:
            updates["final_url"] += 1
            if args.verbose:
                print(f"  update: final_url -> {final_url}")

        # Status counting
        status_counts[status] = status_counts.get(status, 0) + 1
        processed += 1

        # Periodic progress output when not verbose & no tqdm
        if not args.verbose and not args.progress:
            if (count % 10 == 0) or (count == total):
                print(f"[{count}/{total}] {status} :: {url}")

        # Autosave
        if args.autosave_every and processed - last_autosave_at >= args.autosave_every:
            if args.verbose:
                print(f"  autosave: writing checkpoint to {out_path}")
            _safe_write_csv(df, out_path)
            last_autosave_at = processed

    # Final write
    if args.verbose:
        print(f"\n[write] writing final CSV -> {out_path}")
    _safe_write_csv(df, out_path)

    # Summary
    elapsed = perf_counter() - t_start
    print("\n=== Summary ===")
    print(f"Rows removed (dropped): {len(removed_df)}")
    print(f"Rows processed (RSS, kept): {processed}")
    if status_counts:
        print("Statuses:")
        for k in sorted(status_counts.keys()):
            print(f"  {k:10s} : {status_counts[k]}")
    print("Field updates:")
    for k in ["text", "title", "final_url"]:
        print(f"  {k:10s} : {updates[k]}")
    print(f"Elapsed time: {elapsed:.2f}s")
    print(f"Wrote: {out_path}")
    if args.removed_out:
        print(f"Removed rows saved to: {args.removed_out}")


if __name__ == "__main__":
    main()
