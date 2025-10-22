🧠 Wisdom of Sheep — Agent Architecture

Version: v1.0
Author: CH Electronics, 2025
Purpose: Documentation for all major agents, database structure, and runtime flows powering Wisdom of Sheep — the news-driven financial analysis system built around the Council model.

🗺️ Overview

The Wisdom of Sheep ecosystem reads and analyses a constant stream of market news, social posts, and financial data.
It transforms this firehose into structured reasoning about tickers through an LLM-orchestrated Council of agents.

Processing flows in two main phases:

Summarise Post Process — lightweight automatic summarisation of new articles.
Runs continuously as new items enter the corpus (raw_posts_log.csv).

Council Analysis — deep LLM reasoning across multiple perspectives (Bull, Bear, Direction, Chairman), building upon technical and sentiment research.

🧩 System Pipeline
RSS/Reddit/News Feeds → SheepHarvester → raw_posts_log.csv
                                      ↓
                              Summarise Post Process
                                      ↓
                          Council Research & Analysis
                                      ↓
                               Final Verdicts (SQL)

1. Summarise Post Process

Triggered automatically when new items are added to raw_posts_log.csv.

Stages
Stage	Function	Input	Output
Spam Detector	Filters junk, ads, low-relevance content	Article text	Spam score (0–100)
Entity Extractor	Detects tickers, companies, markets	Text	List of assets
Summariser	Compresses article into 3–5 concise bullets	Text	summary_bullets, assets_mentioned, claimed_catalysts
Technical Analyser	Pulls price window + basic indicators (non-LLM)	Ticker(s)	ret1d_pct, RSI14, MACD, trend_strength, volatility_ratio, etc.
Interest Scorer	Evaluates whether article merits full Council analysis	Summariser + Technicals	interest_score (0–100) + rationale
Storage

All outputs from the Summarise Post process are stored in the Council DB (council/wisdom_of_sheep.sql)
— in the stages table with stage = entity, summariser, spam, interest_score, etc.

2. Council Analysis

Once a post passes the interest threshold, it is queued for full Council review.

Pipeline Order
Entity → Researcher (Technical + Sentiment)
→ Bull → Bear → Direction → Chairman


The Researcher now sits immediately after Entity, ensuring all downstream agents have access to its findings.

3. Researcher Stage
Purpose

Generate technical and sentiment analyses to inform the Bull, Bear, and Direction councils.

Sub-components
Sub-Stage	Function	Source
Technical Researcher	Computes extended price analytics and patterns.	technical_analyser.py
Sentiment Researcher	Derives crowd sentiment metrics using the Conversation Hub (per-ticker rolling memory).	hub_adapter.py
Output Schema
{
  "technical_research": {
    "summary_lines": ["RSI 68.5, MACD hist +1.2", "Trend up, +0.8%/day"],
    "trend_direction": "up",
    "trend_strength": 3,
    "r2": 0.71
  },
  "sentiment_research": {
    "ticker": "HOOD",
    "as_of": "2025-09-30T23:59:59Z",
    "signal": {
      "des_raw": 0.42,
      "des_sector": 0.12,
      "des_idio": 0.30,
      "confidence": 0.68,
      "n_deltas": 138
    },
    "narrative": "Tone improved after options rollout; retail flow steady, macro sentiment cautious."
  }
}


If either block is missing, Bull/Bear will raise a missing-researcher-context error and abort.

4. Bull Council (bull_case_stage.py)

Purpose: articulate why a trade could work, backed by technical and sentiment context.

Prompt Schema
{
  "bull_points": ["..."],
  "implied_catalysts": ["..."],
  "setup_quality": {
    "evidence_specificity": 0–3,
    "timeliness": 0–3,
    "edge_vs_consensus": 0–3,
    "why": "string"
  },
  "what_would_improve": ["..."],
  "why": "string"
}

Behaviour

Requires Researcher data.

Normalises all arrays; clamps quality metrics 0–3.

Freeform fallback creates minimal structured JSON when LLM drifts.

Raises bull-case-empty if nothing actionable is produced.

5. Bear Council (bear_case_stage.py)

Purpose: capture red flags, missing data, liquidity and risk factors.

Schema
{
  "bear_points": ["..."],
  "red_flags": ["..."],
  "data_gaps": ["..."],
  "liquidity_concerns": {
    "mentioned": true,
    "details": "text",
    "why": "short rationale"
  },
  "why": "string"
}

Behaviour

Uses same normalisation strategy as Bull.

Detects liquidity mentions (spread, volume, gamma, etc.).

Auto-fixes hallucinated fields.

Raises bear-case-empty if no valid content.

6. Direction Stage

Synthesises Bull and Bear arguments into a directional view.

{
  "direction": "up|down|uncertain",
  "conviction": 0–3,
  "why": "short explanation"
}


Factors considered:

balance of Bull vs Bear reasoning,

trend strength and technicals,

DES confidence.

7. Chairman Stage (chairman_stage.py)

The final synthesiser of all council outputs.

Inputs

entity, researcher, claims, context, for, against, direction, verifier

Authoritative: technical_research, sentiment_research

Optional: corpus_llm_sent, summariser

Internal steps

Harvest latest technicals and sentiment.

Build compact INPUT summary for the model.

Call LLM for structured final_metrics JSON.

Fallback call for plain_english_result.

Final Metrics Schema
{
  "direction": "up|down|neutral",
  "conviction": 0–3,
  "risk_level": 0–3,
  "tradability": 0–3,
  "technical_snapshot": {
    "trend_direction": "up",
    "rsi14": 61.4,
    "macd_hist": 0.73
  },
  "des": {
    "raw": 0.42,
    "idio": 0.30,
    "confidence": 0.68,
    "n_deltas": 138
  },
  "catalysts": ["options rollout", "Q4 retail flows"],
  "watchouts": ["liquidity compression"],
  "next_checks": ["earnings", "FOMC guidance"],
  "why": "summary of reasoning"
}


Chairman ensures numeric bounds, derives fallback risk heuristics, and stores the result as a new stages row with stage='chairman'.

8. Data Stores
8.1 Council DB — council/wisdom_of_sheep.sql

Holds everything about articles and analysis.

Table	Purpose
posts	Raw scraped items
stages	Append-only LLM outputs per stage
interest_log (optional)	Scores & flags for prioritisation

Access:
All agents open this database directly.
It replaces the older sharded design.

8.2 Conversation Hub DB — convos/conversations.sqlite

A per-ticker rolling memory, maintained automatically by the conversation hub.

Structure
conversations(id PK, ticker, ts, kind, data JSON, post_id)
convo_index(ticker PK, last_ts)


kind = "delta" → single summarised article

kind = "memory" → rolled-up note over older deltas

Why it exists

Keeps high-information deltas while trimming context.

Enables time-travel queries (as-of).

Provides input for sentiment analysis via DES scoring.

9. Conversation Hub (ticker_conversation_hub.py)
Role

Maintains a continuously updating “memory” for each ticker.
It converts summariser bullets into compressed deltas and aggregates sentiment statistics.

Core Components
Component	Description
ConversationHub	Main API: ingest, ask, ask_as_of, rollup
SQLiteStore	Thread-safe storage backend for deltas/memory
HubClient (adapter)	Lightweight import for Researcher stage
compute_ticker_signal	DES scoring function (used by Researcher)
Delta Compression

Each incoming article → 1 JSON delta:

{
  "t": "ISO time",
  "src": "seekingalpha",
  "who": ["HOOD"],
  "cat": ["product","risk"],
  "sum": "Short 2–3 sentence summary.",
  "dir": "up",
  "impact": "med",
  "why": "why it matters",
  "chan": "news"
}


Old deltas are eventually rolled into durable memory notes summarised by the hub’s LLM.

Rollups
python ticker_conversation_hub.py rollup --ticker HOOD --keep-latest 300


Summarises older deltas into a compact note:

(memory) 2025-09-01 → 2025-09-30:
- Retail optimism around options growth
- Macro headwinds fading
- Liquidity moderate

Time-Travel Queries

Leak-safe queries include only data ≤ cutoff:

python ticker_conversation_hub.py ask-as-of \
  --ticker HOOD --as-of 2025-09-30T23:59:59+00:00 \
  --q "Tone around options revenue then?"

DES Sentiment Metrics

compute_ticker_signal() calculates Directional Energy Score values used by researcher.py:

Field	Description
des_raw	Weighted mean of sentiment deltas
des_sector	Peer baseline (optional)
des_idio	Ticker-specific deviation
confidence	Agreement × coverage
n_deltas	Count of records used
Typical Workflow
# 1. Ingest summarised posts
python ticker_conversation_hub.py ingest \
  --db council/wisdom_of_sheep.sql \
  --store sqlite \
  --convos convos/conversations.sqlite \
  --model mistral --verbose

# 2. Compute DES metrics
python ticker_conversation_hub.py score \
  --ticker HOOD --as-of 2025-09-30T23:59:59+00:00 \
  --days 7 --channel all --store sqlite --convos convos/conversations.sqlite

# 3. Ask contextual questions
python ticker_conversation_hub.py ask --ticker HOOD \
  --store sqlite --convos convos/conversations.sqlite \
  --model mistral --q "Main risks now?"

10. Agent Responsibilities Summary
Agent	Reads	Writes	Purpose
Summariser	raw_posts_log.csv	stages (summariser)	Compress article text
Entity	stages	stages (entity)	Identify tickers
Researcher	Council DB + Conversation Hub	stages (technical_research, sentiment_research)	Build technical + sentiment foundations
Bull / Bear	stages (inc. Researcher)	stages (for, against)	Articulate pros/cons
Direction	stages	stages (direction)	Combine bull/bear reasoning
Chairman	all stages	stages (chairman)	Final synthesis
Conversation Hub	Council DB	convos/conversations.sqlite	Maintain ticker-level memories
11. Operational Notes

All agents store results as append-only rows in stages.

Stage readers always take the newest record per (post_id, stage).

Both SQLite files are small enough to sync to Git or cloud backup.

ticker_conversation_hub.py may run as a background daemon, periodically rolling up deltas.

researcher.py will automatically instantiate a HubClient when computing sentiment.

12. Troubleshooting
Symptom	Likely Cause	Fix
missing-prerequisites error	Researcher data absent	Run researcher.py --post-id <id>
Bull/Bear empty output	Missing Researcher context	Ensure technical_research and sentiment_research stages exist
Hub ingestion skips many posts	Summariser bullets missing tickers	Check entity extraction or cashtag parsing
Ollama chat timeout	Ollama service not running on port 11434	Restart with ollama serve
Rollup fails	Fewer deltas than --keep-latest	Lower keep count or ingest more
13. Summary

council/wisdom_of_sheep.sql → all article analysis.

convos/conversations.sqlite → rolling per-ticker conversations.

Summariser → Entity → Researcher → Bull/Bear → Direction → Chairman.

Conversation Hub supplies sentiment context and DES metrics.

Chairman consolidates everything into a trade-ready verdict.

Together, these agents form a self-contained intelligence network continuously distilling the world’s financial chatter into structured, verifiable reasoning — the true Wisdom of Sheep.