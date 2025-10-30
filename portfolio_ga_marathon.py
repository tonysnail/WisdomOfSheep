#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
portfolio_ga_marathon.py — Batch GA runner that builds 'stable islands' from portfolio_probe.py outputs,
with percentile-based labeling and a baseline benchmark.

It launches portfolio_probe.py in GA mode across a set of configurations,
then aggregates the results into portfolio/islands/islands_summary.json
that Streamlit can read. It also runs a single non-GA backtest as a baseline benchmark.

Usage:
  python portfolio_ga_marathon.py \
  --db "/Users/carlhudson/Documents/Coding/WisdomOfSheep/council/wisdom_of_sheep-backtest-Oct22.sql" \
  --start 2025-09-20 --end 2025-10-22 \
  --pop 120 --gens 250 --tournament-k 5 --cx 0.9 --mut 0.25 \
  --interval auto --intraday-limit-days 59 \
  --initial-equity 100000 --verbose 1 \
  --runs 24 --seeds 42,1337,7,11,23,101,2025,3033,4044,5055,6066,7077,8088,9099,111,222,333,444,555,666,777,888,999,12345 \
  --name deep_oct_window \
  | tee "portfolio/logs/ga_marathon_$(date -u +%Y%m%dT%H%M%SZ)_deep_oct_window.log"

Artifacts:
  portfolio/islands/<run_id>/
    best_genes.json
    best_metrics.json
    ga_log.csv
    last_metrics.json
    backtest_trades.csv
    ga_progress.json

  portfolio/islands/benchmark/
    baseline_metrics.json
    baseline_trades.csv  (copied from portfolio/backtest_trades.csv at time of run)

  portfolio/islands/islands_summary.json
"""

import argparse, json, shutil, re, sys, subprocess, time, os
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
PORTFOLIO = ROOT / "portfolio"
ISLANDS_DIR = PORTFOLIO / "islands"
ISLANDS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR = PORTFOLIO / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

SEED_RE = re.compile(r"^\[GA\]\s+seed\s+(\d+)/(\d+)", re.I)
GEN_RE  = re.compile(r"^\[GA\]\s+Gen\s+(\d+)\b", re.I)

# ------------------------
# Helpers (process / IO)
# ------------------------
def _run_ga_one(cmd: list[str], run_dir: Path, run_num: int, total_runs: int, gen_goal: int = 120) -> int:
    """Run a GA process with concise progress (seed/gen emojis) and tee logs to run.log."""
    import re, sys, time, subprocess, os
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    p = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
        env=env,
    )
    seeds_seen = (0, 0)
    current_gen = -1
    last_emit = 0.0
    dots_every = 0.4
    with open(log_path, "w", encoding="utf-8") as logf:
        for line in iter(p.stdout.readline, ''):
            logf.write(line)
            # show concise live status
            if "[GA]" in line and "seed" in line:
                try:
                    # e.g. "[GA] seed 1/120"
                    m = re.search(r"seed\s+(\d+)/(\d+)", line, flags=re.I)
                    if m:
                        seeds_seen = (int(m.group(1)), int(m.group(2)))
                    sys.stdout.write(f"\r🌱 Run {run_num}/{total_runs} • seeding {seeds_seen[0]}/{seeds_seen[1]}  ")
                    sys.stdout.flush()
                    continue
                except Exception:
                    pass
            if "[GA]" in line and "Gen" in line:
                try:
                    m = re.search(r"Gen\s+(\d+)", line, flags=re.I)
                    if m:
                        current_gen = int(m.group(1))
                    sys.stdout.write(f"\r🧬 Run {run_num}/{total_runs} • Gen {current_gen}/{gen_goal}  ")
                    sys.stdout.flush()
                    continue
                except Exception:
                    pass
            now = time.time()
            if now - last_emit > dots_every:
                sys.stdout.write(".")
                sys.stdout.flush()
                last_emit = now
    p.wait()
    sys.stdout.write("\r")
    print(f"✅ Run {run_num}/{total_runs} complete.")
    return p.returncode

def _score_for_sort(m):
    # Composite for ordering inside each island
    from math import isfinite
    cagr = float(m.get("cagr", 0.0))
    sharpe = float(m.get("sharpe", 0.0))
    pf = float(m.get("profit_factor", 0.0))
    dd = float(m.get("max_dd", 0.0))
    if not isfinite(pf):
        pf = 50.0
    # Prefer high CAGR/Sharpe/PF and low DD
    return (cagr + 0.5*sharpe + 0.2*pf) - 0.7*dd

# ------------------------
# Percentile-based labeling
# ------------------------
def _percentile_labels(runs):
    """
    Compute data-driven labels per run using robust percentiles of key metrics.
    Returns dict: run_id -> label
    """
    import numpy as np

    if not runs:
        return {}

    # Extract metric vectors
    cagr  = np.array([float(r["best_metrics"].get("cagr", 0.0)) for r in runs])
    shar  = np.array([float(r["best_metrics"].get("sharpe", 0.0)) for r in runs])
    pf    = np.array([float(50.0 if r["best_metrics"].get("profit_factor")==float("inf") else r["best_metrics"].get("profit_factor", 0.0)) for r in runs])
    dd    = np.array([float(r["best_metrics"].get("max_dd", 0.0)) for r in runs])
    tr    = np.array([int(r["best_metrics"].get("trades", 0)) for r in runs])

    def P(x, q):
        try:
            return float(np.percentile(x, q))
        except Exception:
            return 0.0

    # Cut points (tuned for decent spread; adjust if you want different bucket balance)
    cuts = {
        "cagr":  (P(cagr, 60), P(cagr, 75), P(cagr, 87), P(cagr, 95)),
        "shar":  (P(shar, 55), P(shar, 70), P(shar, 85), P(shar, 93)),
        "pf":    (P(pf,   55), P(pf,   70), P(pf,   85), P(pf,   93)),
        "dd":    (P(dd,   40), P(dd,   55), P(dd,   70), P(dd,   80)),  # lower is better
        "tr":    (P(tr,   35), P(tr,   55), P(tr,   70), P(tr,   85)),
    }

    labels = {}
    for r, c, s, p, d, t in zip(runs, cagr, shar, pf, dd, tr):
        points = 0

        c1,c2,c3,c4 = cuts["cagr"]
        if c >= c1: points += 1
        if c >= c2: points += 1
        if c >= c3: points += 1
        if c >= c4: points += 1

        s1,s2,s3,s4 = cuts["shar"]
        if s >= s1: points += 1
        if s >= s2: points += 1
        if s >= s3: points += 1
        if s >= s4: points += 1

        p1,p2,p3,p4 = cuts["pf"]
        if p >= p1: points += 1
        if p >= p2: points += 1
        if p >= p3: points += 1
        if p >= p4: points += 1

        d1,d2,d3,d4 = cuts["dd"]  # lower is better
        if d <= d4: points += 1
        if d <= d3: points += 1
        if d <= d2: points += 1
        if d <= d1: points += 1

        t1,t2,t3,t4 = cuts["tr"]
        if t >= t1: points += 1
        if t >= t2: points += 1
        if t >= t3: points += 1
        if t >= t4: points += 1

        # Guard-rails to catch the spicy outliers
        if (t < max(8, t1)) or (d > max(0.35, d4 * 1.2)):
            label = "Spicy/YOLO"
        else:
            # Map “points” into your preferred labels
            # Adjust thresholds if you want a different spread
            if points >= 17:
                label = "House Deposit Fund"
            elif points >= 14:
                label = "Confident"
            elif points >= 11:
                label = "Hot"
            elif points >= 8:
                label = "Cruising"
            elif points >= 6:
                label = "A little risky"
            else:
                label = "Safe"

        labels[r["run_id"]] = label

    return labels

# ------------------------
# Benchmark (non-GA run)
# ------------------------
def _run_benchmark(db, start, end, interval, intraday_limit_days, initial_equity, verbose):
    """
    Run portfolio_probe.py *without* --ga to capture a baseline metrics / trades.
    Writes artifacts under portfolio/islands/benchmark/.
    Returns dict with paths and metrics.
    """
    import subprocess, time, sys

    bench_dir = ISLANDS_DIR / "benchmark"
    bench_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Baseline Benchmark (non-GA backtest) ===")
    cmd = [
        sys.executable, "-u", str(ROOT / "portfolio_probe.py"),
        "--db", db, "--start", start, "--end", end,
        "--interval", interval,
        "--intraday-limit-days", str(intraday_limit_days),
        "--initial-equity", str(initial_equity),
        "--verbose", str(max(1, verbose)),  # keep at least some signal
    ]
    print("CMD:", " ".join(cmd))

    # Run and tee to bench.log with a light progress ticker
    log_path = bench_dir / "bench.log"
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    p = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
        env=env,
    )
    last_emit = 0.0
    dots_every = 0.4
    with open(log_path, "w", encoding="utf-8") as logf:
        for line in iter(p.stdout.readline, ''):
            logf.write(line)
            now = time.time()
            if now - last_emit > dots_every:
                sys.stdout.write(".")
                sys.stdout.flush()
                last_emit = now
    p.wait()
    sys.stdout.write("\r")
    print("✅ Baseline complete.")

    # Pull artifacts produced by portfolio_probe.py
    last_metrics = PORTFOLIO / "last_metrics.json"
    trades_csv   = PORTFOLIO / "backtest_trades.csv"

    bench_metrics_path = bench_dir / "baseline_metrics.json"
    bench_trades_path  = bench_dir / "baseline_trades.csv"

    metrics = {}
    if last_metrics.exists():
        metrics = json.loads(last_metrics.read_text())
        shutil.copy2(last_metrics, bench_metrics_path)
    if trades_csv.exists():
        shutil.copy2(trades_csv, bench_trades_path)

    print(f"✅ Baseline saved → {bench_metrics_path}")
    return {
        "ok": True,
        "metrics": metrics,
        "metrics_path": str(bench_metrics_path) if last_metrics.exists() else None,
        "trades_csv_path": str(bench_trades_path) if trades_csv.exists() else None
    }

# ------------------------
# Main
# ------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--initial-equity", type=float, default=100000.0)
    ap.add_argument("--interval", default="auto", choices=["auto","1d","4h","1h","30m","15m","5m"])
    ap.add_argument("--intraday-limit-days", type=int, default=59)
    ap.add_argument("--verbose", type=int, default=1)

    ap.add_argument("--pop", type=int, default=80)
    ap.add_argument("--gens", type=int, default=120)
    ap.add_argument("--tournament-k", type=int, default=5)
    ap.add_argument("--cx", type=float, default=0.9)
    ap.add_argument("--mut", type=float, default=0.25)

    ap.add_argument("--runs", type=int, default=6, help="How many GA runs to perform.")
    ap.add_argument("--seeds", default="", help="Comma list of seeds. If fewer than runs, remaining are random.")
    ap.add_argument("--name", default="", help="Optional label for this marathon (used in folder names).")

    ap.add_argument("--skip-benchmark", action="store_true", help="Do not run the single non-GA baseline backtest.")

    # (Optional) console flags from your prior patch; safe to ignore if not used
    ap.add_argument("--console", default=None, help="Optional console mode string for your UI layer.")
    ap.add_argument("--gen-goal", type=int, default=None, help="Optional generations goal hint for UI.")

    args = ap.parse_args()

    # Seeds
    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    while len(seeds) < args.runs:
        seeds.append(str(int(time.time() * 1000) % 10_000_000))
        time.sleep(0.01)

    marathon_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    if args.name:
        marathon_id += f"_{args.name}"

    manifest = {
        "marathon_id": marathon_id,
        "created_utc": marathon_id,
        "db": str(Path(args.db).resolve()),
        "window": {"start": args.start, "end": args.end},
        "interval": args.interval,
        "intraday_limit_days": args.intraday_limit_days,
        "initial_equity": args.initial_equity,
        "runs": [],
        "benchmark": None,
        "params": {
            "pop": args.pop, "gens": args.gens, "tournament_k": args.tournament_k,
            "cx": args.cx, "mut": args.mut, "runs": args.runs,
            "console": args.console, "gen_goal": args.gen_goal,
        },
    }

    # ------------------------
    # Baseline benchmark first
    # ------------------------
    if not args.skip_benchmark:
        bench = _run_benchmark(
            db=args.db,
            start=args.start,
            end=args.end,
            interval=args.interval,
            intraday_limit_days=args.intraday_limit_days,
            initial_equity=args.initial_equity,
            verbose=args.verbose,
        )
        manifest["benchmark"] = bench

    # ------------------------
    # GA runs
    # ------------------------
    for i in range(args.runs):
        seed = seeds[i]
        run_id = f"run_{i+1:02d}_seed_{seed}"
        run_dir = ISLANDS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== GA RUN {i+1}/{args.runs} — {run_id} ===")
        cmd = [
            sys.executable, "-u", str(ROOT / "portfolio_probe.py"),
            "--db", args.db, "--start", args.start, "--end", args.end,
            "--ga",
            "--pop", str(args.pop),
            "--gens", str(args.gens),
            "--tournament-k", str(args.tournament_k),
            "--cx", str(args.cx),
            "--mut", str(args.mut),
            "--seed", str(seed),
            "--interval", args.interval,
            "--intraday-limit-days", str(args.intraday_limit_days),
            "--initial-equity", str(args.initial_equity),
            "--verbose", str(args.verbose),
        ]
        print("CMD:", " ".join(cmd))

        rc = _run_ga_one(cmd, run_dir, run_num=i+1, total_runs=args.runs, gen_goal=args.gens)
        if rc != 0:
            print(f"❌ GA run failed (code {rc}). Skipping.")
            continue  # <— this 'continue' is now correctly inside the loop

        # Copy artifacts from portfolio/ into this run folder
        for name in ["best_genes.json","best_metrics.json","ga_log.csv","last_metrics.json","backtest_trades.csv","ga_progress.json"]:
            src = PORTFOLIO / name
            if src.exists():
                shutil.copy2(src, run_dir / name)

        # Load metrics for this run
        metrics_path = run_dir / "best_metrics.json"
        genes_path   = run_dir / "best_genes.json"
        tr_path      = run_dir / "backtest_trades.csv"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        genes = json.loads(genes_path.read_text()) if genes_path.exists() else {}

        manifest["runs"].append({
            "run_id": run_id,
            "seed": seed,
            "best_metrics": metrics,
            "best_genes_path": str(genes_path),
            "trades_csv_path": str(tr_path) if tr_path.exists() else None,
            "score_for_sort": _score_for_sort(metrics),
            "island": "TBD",  # percentile label assigned later
        })

    # ------------------------
    # Build percentile-based islands
    # ------------------------
    # Assign labels
    label_map = _percentile_labels(manifest["runs"]) if manifest["runs"] else {}

    # Bucket runs by label with stable ordering
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in manifest["runs"]:
        lbl = label_map.get(r["run_id"], "Safe")
        r["island"] = lbl
        buckets[lbl].append(r)

    for lbl in buckets.keys():
        buckets[lbl].sort(key=lambda x: x["score_for_sort"], reverse=True)

    ORDER = ["Safe", "A little risky", "Cruising", "Confident", "Hot", "House Deposit Fund", "Spicy/YOLO"]
    islands = []
    for label in ORDER:
        rows = buckets.get(label, [])
        islands.append({
            "label": label,
            "count": len(rows),
            "runs": rows
        })

    out = {
        "marathon": manifest,
        "islands": islands
    }
    out_path = ISLANDS_DIR / "islands_summary.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n✅ Islands summary written to: {out_path}")

if __name__ == "__main__":
    main()
