Wisdom of Sheep — Theory Note (why it could work / why it might not)
Why it could work

Crowd information as fast filter. Retail and semi-pro communities surface idiosyncratic info (supplier wins, product virality, unusual order flow anecdotes) before it’s neatly packaged by mainstream media. If your LLM extracts explicit, directional statements with reasonable confidence, you’re converting noisy chatter into tradeable hypotheses quickly.

Reflexivity & attention. Viral posts can cause follow-through (more eyeballs → more orders → price impact). Capturing that attention wave intraday (minutes–hours) is a plausible edge versus slower daily/news pipelines.

Event micro-alpha. Single-name catalysts (earnings, product launches, legal rulings) often leak into social streams. Even if many signals are false, a TP/SL framework with small notional and tight horizons can create a positive skew if the winners trend quickly.

Model-on-model compounding. You can stack: (a) explicit signal extraction, (b) “mood-music” macro regime scoring, (c) staking logic that sizes up in favorable regimes and down in hostile ones. That’s classic meta-strategy design.

Why it might not work

Alpha decay & over-crowding. Anything obvious on big subs decays fast. If many bots chase the same posts, you get worse fills, slippage, and whipsaws.

Label leakage & survivorship bias. If you tune prompts/thresholds while observing recent P/L from those same sessions, you’ll overfit to noise. Robust separation of model selection and out-of-sample evaluation is mandatory.

Execution frictions. Your simulator doesn’t include commissions, bid/ask spread, partial fills, borrow costs (shorts), halts, or trading pauses. Thin names and options talk are particularly tricky.

Time alignment traps. The Reddit scrape time is not the post creation time, and the price bar time is exchange-side. A few minutes of misalignment can flip outcomes, especially around the open.

Regime dependence. Strategies that ride attention and momentum can underperform in range-bound, mean-reverting tapes or high-vol chop (post-news fade days).

Adversarial content. Pump-and-dump posts, satire, or ambiguous tickers (e.g., “ON”, “ALL”) can sneak through unless you enforce whitelists and strict parsing.

What would make the thesis stronger

Causal timing proofing. Use the earliest reliable timestamp you can for a post (created_utc via API when possible) and forbid using price information before that instant.

Realistic fills. Simulate entry at next bar open after signal time (or mid-price + half-spread), include fixed per-trade cost and slippage by ADV/liquidity bucket.

Walk-forward tuning. Fix LLM prompts/thresholds on a rolling window, validate on the next window. No peeking.

“Mood music” gating. Only trade when macro regime agrees (e.g., risk-on score > threshold) to reduce false signals in poor tape.

Adversarial tests. Shuffle timestamps; randomize ticker mapping; compare to dumb baselines (e.g., flip a coin but keep your risk settings). You want outperformance vs. naïve.

Wisdom_of_Sheep_Documentation.txt

Copy everything below into Wisdom_of_Sheep_Documentation.txt (or a README).
It explains what the program does, how it’s wired, and how to extend it.

1) Overview

Wisdom of Sheep is an experimental day-trade simulator that:

Scrapes new posts from selected subreddits.

Extracts explicit, directional trade calls with an LLM (Ollama) and a regex fallback.

Filters to real US-listed tickers via a NASDAQ/NYSE whitelist.

Logs signals and simulates trades using intraday prices (yfinance), with TP/SL and a max holding window.

Persists append-only trade events to trades_log.csv (PLACED/CLOSE).

Visualizes predictions, charts, and running P/L in a Streamlit dashboard.

This is a paper trading tool for research. It does not place live orders.

This is a private code repo for use by Carl Hudson only.

2) Data Flow (pipeline)
[Sources: Reddit subs, RSS feeds, X handles]
     ├─ reddit_scraper.scrape_sub_new() → full OP text via reddit_scraper.fetch_post_full_text()
     ├─ rss_parser.fetch_rss_feed() → news posts (title + summary)
     └─ x_scraper.scrape_x_feed() → X posts via Nitter RSS
            ↘ raw_posts_log.csv (append-only pre-LLM archive for backtesting)
            ↘ ollama_extract() → JSON {tickers, direction, target, confidence}
                 ↘ regex_extract() as fallback
                      ↘ filter_valid_signal() → check whitelist + min confidence
                           ↘ session_state.predictions (in-memory)
                                ↘ PLACED events → trades_log.csv
                                ↘ fetch_mini_prices() → intraday bars
                                     ↘ evaluate_trade_details() (TP/SL/HORIZON)
                                         ↘ CLOSE events → trades_log.csv

3) Major Components
3.1 Reddit scraping

reddit_scraper.py owns all Reddit-specific logic. scrape_sub_new(sub, max_posts) scrapes https://old.reddit.com/r/<sub>/new/ and collects post metadata.

fetch_post_full_text(permalink) lives in the same module and loads the post detail page to extract the OP’s full selftext + title (not comments).

A tiny time.sleep(0.15) pacing avoids hammering.

If Reddit responds with 429, the app logs, backs off, and rotates subs.

Why old.reddit? Simpler, stable markup that’s easy to parse with BeautifulSoup.

rss_parser.py handles generic RSS/Atom feeds (e.g., CNBC, Yahoo Finance). It normalizes titles + summaries into the same post schema the app expects.

x_scraper.py builds on rss_parser to hit public Nitter RSS endpoints for selected X/Twitter handles. Handles are configured in the sidebar; results use the same post schema.

3.2 Ticker whitelist

load_valid_tickers() downloads nasdaqlisted.txt and otherlisted.txt (cached on disk for 7 days to tickers_all.csv).

Guarantees tickers are 1–5 uppercase letters and actually listed.

You can trigger a re-cache from the sidebar button.

3.3 LLM signal extraction

ollama_extract(text, model) posts a constrained prompt to Ollama’s chat API.

The prompt enforces:

JSON-only return

explicit direction (long|short)

real US tickers (filtered later)

optional target and confidence (0–1 or %)

normalize_* helpers sanitize outputs.

regex_extract(text) backs up the LLM with simple heuristics when needed.

filter_valid_signal(sig) applies:

whitelist enforcement

minimum extraction confidence (min_conf_extract)

direction must be long/short

3.4 Trade simulation

fetch_mini_prices(ticker) retrieves recent intraday bars via yfinance, trying multiple period/interval combos, and normalizes to a UTC-naive index with a numeric Close.

evaluate_trade_details(series, ts_iso, direction, notional, tp_pct, sl_pct, horizon_min)

Entry = first available bar on/after signal timestamp.

Monitors subsequent bars until:

TP hit (entry * (1±tp_pct))

SL hit (entry * (1∓sl_pct))

HORIZON timeout (no hit within window)

Computes P/L: notional * (exit/entry - 1) for longs, inverse for shorts.

compute_trade_details_for_predictions() applies the above across eligible predictions.

Note: Simulator ignores fees, spread, and slippage. See “Limitations.”

3.5 Event logging (append-only)

trades_log.csv records immutable events:

PLACED immediately when a qualifying signal arrives (confidence ≥ min_conf_trade)

CLOSE after the simulator determines exit (TP/SL/HORIZON)

Schema columns (superset, order preserved):

event_type, trade_id, post_id, ticker, direction,
placed_at, signal_ts,
entry_time, entry_price, exit_time, exit_price, exit_reason, pnl,
confidence, target, subreddit, post_url, title, text_snippet, post_text, llm_json,
params_notional, params_tp_pct, params_sl_pct, params_horizon_min

3.6 Raw post archive

raw_posts_log.csv is an append-only log of every Reddit/RSS/X post ingested before LLM parsing. It keeps platform, source, post
ID, title, URL, and full text so you can backtest future changes to extraction logic without re-scraping the web.


trade_id is stable: <platform>-<post_id>-<ticker>-<direction>.

Loader functions maintain de-dupe sets for PLACED/CLOSE to prevent repeats.

3.6 Streamlit UI

Sidebar controls: subs list, batching/interval/pacing, confidence thresholds, TP/SL/Horizon, chart window, notional, max rows, whitelist refresh.

Left pane: activity log, “Latest extracted predictions” with Full post text expander and LLM raw JSON.

Right pane: Prediction Board

Per-ticker card with last signal, confidence, tiny intraday chart, and latest simulated P/L badge.

Optionally include tickers with zero predictions for context.

Header: current status and Total P/L badge.

4) Configuration & Parameters

Model: mistral by default (Ollama). Change in sidebar.

Confidence thresholds:

min_conf_extract (discard low-quality extractions)

min_conf_trade (require stronger conviction to place a trade)

Risk settings:

notional per trade (paper)

tp_pct, sl_pct (percent)

horizon_min (max holding minutes)

Scrape cadence:

interval (seconds between refreshes)

batch_size per refresh

pause_per_post pacing (seconds)

5) Assumptions & Limitations

Intraday data coverage: 1–5 minute bars via yfinance can be sparse, delayed, or missing for illiquid names, pre/post-market, or on outage days.

No transaction costs: Simulator ignores commissions, fees, and borrow costs. Add them for realism.

No microstructure modeling: Entries occur on the first available bar ≥ signal time; no spread/slippage model. Consider “mid + half-spread” or “open next bar + slippage”.

Timestamp alignment: We use the scrape time as signal_ts. If you switch to the Reddit API, prefer created_utc.

Content ambiguity: Despite whitelists, ambiguous tickers and sarcasm exist. LLM prompt is strict, but garbage can slip through.

Compliance: Research use only; not investment advice.

6) Backtesting Guidance (to avoid fooling yourself)

Time integrity: Use the earliest known post timestamp, never look ahead in prices.

Walk-forward: Tune prompt/thresholds on train window, evaluate on next window, then roll.

Costs & slippage: Add per-trade fee + spread model; scale slippage by ADV and price.

Liquidity filters: Drop names with ADV below a threshold; avoid halts.

Baselines: Compare to random directions, buy-and-hold for the same names/time windows, and a momentum heuristic (e.g., trade in direction of last 15-minute return).

Stat hygiene: Report Sharpe, hit-rate, avg win/loss, max drawdown, turnover, and t-stat of alpha vs baseline.

7) Extending the System
7.1 Discord & Telegram ingestion

Create new “platform adapters” mirroring the Reddit adapter:

fetch_channel_messages() → canonical post dict {id, url, title, text, source, ts}

Reuse ollama_extract(), regex_extract(), filter_valid_signal().

Tag platform in platform column; keep trade_id format stable (e.g., <platform>-<message_id>-<ticker>-<direction>).

7.2 “Mood music” (macro regime)

Add a periodic macro job that writes a regime_state.json:

Inputs: VIX level/change, credit spreads proxy, SPY/QQQ trend slope, breadth (% advancers), calendar (FOMC, CPI).

LLM summary → scalar risk_on score ∈ [0,1].

Gate trades: require risk_on ≥ threshold to place PLACED events (or scale notional by score).

7.3 Better execution model

Choice of entry: next bar open, VWAP of next N minutes, or mid + spread/2.

Slippage: slip_bps = base + k / sqrt(ADV); apply to both entry and exit.

Market hours awareness: avoid entries near close unless horizon spans next session.

8) How to Run (quick start)

Prereqs

Python 3.10+

pip install streamlit yfinance pandas beautifulsoup4 requests streamlit-autorefresh

Ollama running locally; pull your chosen model (e.g., ollama pull mistral)

Launch

streamlit run app.py

Use

Set subreddits, model, thresholds in sidebar.

Click RUN.

Watch “Latest extracted predictions,” expand Full post text to confirm complete OP text is being parsed.

Review trades_log.csv for PLACED & CLOSE events.

9) CSV & Reproducibility

trades_log.csv is append-only. Never edit in place; instead, derive analytics from it.

Version your prompt strings and parameters in the events (llm_json, params_*) so analyses can be reproduced.

10) Roadmap

✅ Full OP text scraping + pass-through to LLM

🔜 Discord/Telegram adapters

🔜 Macro regime “mood music” scorer

🔜 Execution realism (fees/spread/slippage)

🔜 Proper backtest harness (walk-forward, metrics dashboard)

🔜 Prompt A/B testing with automatic selection and logging

11) Ethics & Compliance

Do not present results as financial advice.

Respect platform ToS and rate limits.

Avoid amplifying manipulative content; consider blacklists and human review for suspicious posts.