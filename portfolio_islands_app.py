#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
ISLANDS_ROOT = ROOT / "portfolio" / "islands"

st.set_page_config(page_title="Portfolio Islands", layout="wide")

st.title("🏝️ Portfolio Islands — Read-Only Dashboard")
st.caption("Loads precomputed GA runs and islands from JSON artifacts")

# ---- Controls
islands_file = st.text_input(
    "Islands summary JSON",
    value=str(ISLANDS_ROOT / "islands_summary.json")
)
path = Path(islands_file)
if not path.exists():
    st.error(f"File not found: {path}")
    st.stop()

data = json.loads(path.read_text())
marathon = data.get("marathon", {})
islands = data.get("islands", [])

# ---- Marathon header
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Marathon ID", marathon.get("marathon_id", ""))
win = marathon.get("window", {})
c2.metric("Window", f"{win.get('start','?')} → {win.get('end','?')}")
c3.metric("Interval", marathon.get("interval", "auto"))
c4.metric("Runs", len(marathon.get("runs", [])))
c5.metric("Init. Equity", f"{marathon.get('initial_equity', 0):,.0f}")

# ---- Islands overview
st.subheader("Islands Overview")
for island in islands:
    with st.expander(f"{island['label']} • {island['count']} runs", expanded=True):
        rows = island.get("runs", [])
        if not rows:
            st.info("No runs in this island.")
            continue

        # summary table
        df = pd.DataFrame([{
            "run_id": r["run_id"],
            "seed": r["seed"],
            "trades": r["best_metrics"].get("trades", 0),
            "CAGR": r["best_metrics"].get("cagr", 0.0),
            "Sharpe": r["best_metrics"].get("sharpe", 0.0),
            "Sortino": r["best_metrics"].get("sortino", 0.0),
            "PF": (50.0 if r["best_metrics"].get("profit_factor") == float("inf") else r["best_metrics"].get("profit_factor", 0.0)),
            "MaxDD": r["best_metrics"].get("max_dd", 0.0),
            "Win%": r["best_metrics"].get("win_rate", 0.0),
            "Expectancy": r["best_metrics"].get("expectancy", 0.0),
            "End Equity": r["best_metrics"].get("end_equity", 0.0),
        } for r in rows])

        fmt = {
            "CAGR": "{:.1%}",
            "Sharpe": "{:.2f}",
            "Sortino": "{:.2f}",
            "PF": "{:.2f}",
            "MaxDD": "{:.1%}",
            "Win%": "{:.1%}",
            "Expectancy": "{:,.0f}",
            "End Equity": "{:,.0f}",
        }
        st.dataframe(df.style.format(fmt), use_container_width=True)

        # per-run drilldown
        sel = st.selectbox("Select a run to inspect", [r["run_id"] for r in rows], key=f"sel_{island['label']}")
        target = next((r for r in rows if r["run_id"] == sel), None)
        if target:
            left, right = st.columns([1,1])
            with left:
                st.markdown("**Best metrics**")
                bm = target["best_metrics"]
                st.json(bm, expanded=False)

                genes_path = Path(target["best_genes_path"])
                if genes_path.exists():
                    st.markdown("**Best genes**")
                    st.json(json.loads(genes_path.read_text()), expanded=False)
                else:
                    st.info("best_genes.json not found for this run.")

            with right:
                trades_csv = target.get("trades_csv_path")
                if trades_csv and Path(trades_csv).exists():
                    st.markdown("**Trades (top 200)**")
                    try:
                        tdf = pd.read_csv(trades_csv)
                        # Nice minimal subset
                        keep = [c for c in ["ticker","yf_ticker","side","signal_time","entry_time","exit_time","entry","exit","shares","pnl","equity_after","interval"] if c in tdf.columns]
                        tdf = tdf[keep].copy()
                        # basic formatting
                        for c in ["entry","exit","pnl","equity_after","shares"]:
                            if c in tdf.columns:
                                tdf[c] = tdf[c].astype(float)
                        st.dataframe(tdf.head(200), use_container_width=True)
                    except Exception as exc:
                        st.error(f"Failed to load trades CSV: {exc}")
                else:
                    st.info("No trades CSV found for this run.")

st.success("Loaded islands successfully.")
st.caption("Tip: re-run the CLI marathon whenever you want fresh results, then refresh this page.")
