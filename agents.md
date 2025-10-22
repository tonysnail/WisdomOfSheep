# 🧠 Wisdom of Sheep — System Manual

Version: v2.0 (2025-XX)
Author: CH Electronics, 2025 — updated by gpt-5-codex
Purpose: Provide a repo-wide field manual that explains every major component, data store, and runtime flow that powers Wisdom of Sheep.

---

## 1. Architectural Overview

Wisdom of Sheep is a market-intelligence platform that monitors news, social feeds, and price action to produce trade-ready council verdicts. The ecosystem operates continuously across five pillars:

1. **Ingestion** – Harvest articles/posts from Reddit, RSS feeds, X, and Stocktwits into `raw_posts_log.csv` and the council database.
2. **Summarise Post Process** – Lightweight agents extract tickers, summarise content, compute technical snapshots, and rank interest.
3. **Conversation Memory** – The ticker conversation hub compresses article deltas into rolling sentiment memories and DES scores.
4. **Council Analysis** – A staged LLM workflow (Entity → Researcher → Bull/Bear → Direction → Chairman) reasons about catalysts, risks, and conviction.
5. **Operations & Delivery** – A FastAPI backend, React frontend, and CLI tooling orchestrate runs, surface results, and support investigations.

```
Sources → SheepHarvester/raw_posts_log.csv
                ↓
     Summariser + Entity + Interest Score
                ↓
  Researcher (technical + sentiment context)
                ↓
Bull/Bear/Direction Councils → Chairman verdict
                ↓
        Stages table in council/wisdom_of_sheep.sql
                ↓
 Conversation Hub (sentiment memory) + Frontend UI
```

Key runtime stores:

- **`raw_posts_log.csv`** – append-only log of harvested articles.
- **`council/wisdom_of_sheep.sql`** – master SQLite DB containing `posts` and `stages` tables (all agent outputs live here).
- **`convos/conversations.sqlite`** – ticker conversation hub memory store and DES metrics.

---

## 2. Data & Storage Model

### 2.1 `raw_posts_log.csv`
Produced by `sheep_harvester.py` and legacy scrapers. Columns:
`[scraped_at, platform, source, post_id, url, title, text, …]`.
Used for backfills, CSV refresh, and as a fallback text source.

### 2.2 Council Database (`council/wisdom_of_sheep.sql`)
Schema defined in `council/council_schema.sql`. Key tables:
- **`posts`** – canonical article metadata.
- **`stages`** – agent payloads (`stage` column values: summariser, entity, claims, for, against, direction, researcher, chairman, context, verifier, interest_score, etc.). Newest `(post_id, stage)` row wins.
- **`jobs`** (managed by backend) – tracks refresh/council job status.

Utilities:
- `council/init_db.py` initialises schema.
- `round_table.write_stage()` writes `stages` rows.
- Backend `app.py` exposes CRUD for posts/stages.

### 2.3 Conversation Hub (`ticker_conversation_hub.py`)
Stores article deltas and rolled-up memories per ticker in SQLite. Interfaces via:
- `ConversationHub` class – ingest, ask, ask_as_of, rollup, score.
- `SQLiteStore` – thread-safe persistence.
- `HubClient` in `hub_adapter.py` – lightweight client for researcher/backend.

---

## 3. Source Ingestion & Normalisation

### 3.1 Sheep Harvester (`sheep_harvester.py`)
Headless loop that coordinates all scrapers.
- CLI flags: `--interval`, `--max-items`, `--batch-size`, `--per-post-sleep`, `--backfill`, `--max-backfill-batches`, custom RSS/X config files.
- Deduplicates via `(platform, post_id)` and URL sets loaded from CSV.
- Appends new rows to `raw_posts_log.csv` (safe temp writes).

### 3.2 Scrapers
| Script | Role | Highlights |
| --- | --- | --- |
| `reddit_scraper.py` | Fetches new submissions for `DEFAULT_SUBS`. Uses PRAW-less HTTP/JSON to minimise deps. | Normalises posts, handles stickies/flairs, optional credentials via env. |
| `rss_parser.py` | Pulls articles from `DEFAULT_RSS_FEEDS`. | Uses `feedparser`/requests, optional full-text expansion via `fix_rss_fulltext.py`. |
| `x_scraper.py` | Scrapes X/Twitter timelines for handles in `DEFAULT_X_HANDLES`. | Leverages RSSBridge-compatible endpoints, handles pagination/backfill. |
| `stocktwits_scraper.py` | Captures Stocktwits news feed. | Filters duplicates, normalises to CSV schema. |
| `fix_rss_fulltext.py`, `enrich_tickers.py` | Helpers to expand RSS excerpts and annotate posts with enriched ticker data. |

### 3.3 Corpus Management Utilities
- `scan_corpus_for_evidence.py`, `rss_test.py`, `rss_parser.py` CLI utilities for inspecting ingestion.
- `raw_posts_log.csv` may be enriched with additional columns by downstream jobs (interest score, etc.).

---

## 4. Summarise Post Process

Stages triggered per new article (manual or via backend job):

1. **Summariser Stage (`council/summariser_stage.py`)**
   - Extracts summary bullets, catalysts, assets, claimed numbers, author stance, spam likelihood.
   - Ensures every array entry includes a `why` field (global rule).

2. **Entity Stage (`council/entity_stage.py`)**
   - Resolves tickers and trading timeframe hints using `ticker_universe.py` for aliases/sector context.
   - Outputs `EntityTimeframeOut` (assets + time hint + uncertainty).

3. **Claims Stage (`council/claims_stage.py`)**
   - Extracts structured claims referencing article facts.

4. **Context Stage (`council/context_stage.py`)**
   - Gathers supporting/contrary context from existing stage data and conversation hub memories.

5. **Verifier Stage (`council/verifier_stage.py`)**
   - Audits claims with citations, returns verdicts plus overall notes.

6. **Interest Scoring (`council/interest_score.py`)**
   - Scans summariser + technical data to assign `interest_score` (0–100) with rationale.

7. **Technical Snapshot (`technical_analyser.py`, `stock_window.py`)**
   - Provides RSI, MACD, trend, volatility, support/resistance, OBV, MFI, and price window metrics.
   - CLI: `python technical_analyser.py run-plan plan.json` or single-tool invocations.

Backend endpoint `/api/refresh-summaries/start` batches these stages for multiple posts, persisting progress under `.jobs/`.

---

## 5. Researcher Stage & Tooling

### 5.1 Researcher (`researcher.py`)
Two-phase LLM workflow:
1. Generate balanced hypotheses from summariser/bull/bear/context inputs.
2. Produce a structured tool plan covering timing, technical, and sentiment checks.

Tool palette includes price/sentiment functions from `technical_analyser.py` and `ticker_conversation_hub.py`. Uses `ollama_chat()` (Mistral by default) with temperature 0.15. Captures verbose logs.

### 5.2 Research Execution (`backend/app.py` + `researcher_harness.py`)
- Backend endpoint `/api/posts/{post_id}/research` executes researcher stage, storing output to `stages` with `stage='researcher'`.
- `researcher_harness.py` provides CLI to run the researcher end-to-end for a post, ensuring dependent stages exist (summariser/bull/bear etc.) via in-process `round_table.run_stages_for_post`.
- `hub_adapter.HubClient` mediates conversation hub access (score, ask, rollup) with optional verbose sentiment logging.

### 5.3 Technical Data Providers
- `stock_window.py` – constructs multi-resolution OHLCV windows with caching to disk (`.cache/stock_windows`).
- `yfinance_throttle.py` – rate-limits Yahoo Finance API calls.

---

## 6. Council Agents & Round Table Orchestration

### 6.1 Stage Modules (`council/*.py`)
Each stage enforces strict JSON schemas defined in `council/common.py`. Key outputs:
- **Bull Case (`bull_case_stage.py`)** – actionable bull thesis, catalysts, setup quality scores, improvement wishlist.
- **Bear Case (`bear_case_stage.py`)** – risks, liquidity warnings, missing data, catalysts likely to fail.
- **Direction (`direction_stage.py`)** – synthesises bull/bear to recommend directional bias, timeframe, conviction.
- **Chairman (`chairman_stage.py`)** – final verdict with audit trail, citations, recommended action, next checks.

### 6.2 Round Table Runner (`round_table.py`)
- CLI to run single stages or scripted batches against CSV/SQL.
- Writes to `stages` table, handles dependency autofill, JSON repair, pretty-print, dummy tests.
- Offers helpers to fetch text (`get_post_text`), normalise LLM outputs, compute council audit packages.

### 6.3 Council Flow Controller (`backend/app.py`)
- `/api/council-analysis/start` – launches multi-stage council run (Entity → Researcher → Bull → Bear → Direction → Chairman). Tracks progress, captures stdout/stderr logs.
- `/api/council-analysis/{job_id}` – poll job status, stage completions, errors.
- `/api/council-analysis/{job_id}/stop` + `/erase-all` to manage jobs.

---

## 7. Conversation Hub & Sentiment Memory

### 7.1 Core Module (`ticker_conversation_hub.py`)
- `ConversationHub.ingest` – converts summariser outputs into deltas (`{t, who, sum, dir, impact, why}`) stored in SQLite.
- `rollup` – compresses historical deltas into durable memory notes, optionally trimming to `--keep-latest`.
- `ask` / `ask_as_of` – LLM-powered Q&A over conversation memory (time-travel safe).
- `compute_ticker_signal` – DES sentiment metrics (`des_raw`, `des_sector`, `des_idio`, `confidence`, `n_deltas`).
- CLI commands: `ingest`, `score`, `ask`, `ask-as-of`, `rollup` (see inline docstrings and `docs/conversation_hub.readme.txt`).

### 7.2 Adapter (`hub_adapter.py`)
- Provides `HubClient` abstraction with context managers for ingestion/scoring.
- Handles SQLite connections, ensures DES windows align with researcher lookbacks.

### 7.3 Tests
- `tests/test_hub_adapter.py` verifies ingestion and scoring behaviour using temporary SQLite DBs.

---

## 8. Backend Service (FastAPI)

`backend/app.py` exposes the operational API. Highlights:

- **Startup** – validates environment, warms ticker universe, initialises job directories (`.jobs/`).
- **CORS & JSON** – configured via FastAPI, Pydantic models for requests/responses.
- **Stock Window Endpoint** – `/api/stocks/window` returns technical data slices (delegates to `get_stock_window`).
- **Post Management**
  - `/api/posts` (GET) – paginated list with filters (`q`, `platform`, `source`, `interest_min`, date range).
  - `/api/posts/{post_id}` (GET) – detail view with stage payloads, interest score, technical metrics, conversation signals.
  - `/api/posts/{post_id}/refresh-from-csv` – reload post body from CSV fallback.
  - `/api/posts/{post_id}/clear-analysis` – delete associated stage rows.

- **Summary Refresh Jobs** – start/poll/stop via `/api/refresh-summaries/*` endpoints. Jobs run inside background threads, executing summariser/entity/interest stages with CSV backups.

- **Council Jobs** – manage multi-stage council runs (see §6.3).

- **Research** – `/api/posts/{post_id}/research` triggers researcher stage and returns technical/sentiment payloads.

- **Health** – `/api/health` verifies DB access and optional dependencies.

- **Error Handling** – gracefully reports missing optional modules (Round Table, stock window, conversation hub) via structured JSON and flags such as `ROUND_TABLE_IMPORT_ERROR`.

Tests for backend behaviour live in `backend/test_app.py`.

---

## 9. Frontend (React + Vite)

Located in `frontend/`. Purpose: operator console for council runs and article review.

- **Stack** – TypeScript, React 18, Vite, Chakra-free custom styling (see `styles.css`, `App.css`).
- **API Wrapper (`src/api.ts`)** – Fetches backend endpoints, handles pagination, job polling, stage execution.
- **Components**
  - `PostList` – sidebar table with filters (search, platform, interest score, date). Persists UI state in `localStorage`.
  - `PostDetail` – displays stage outputs (summariser, claims, bull/bear, direction, chairman) using JSON viewers in `components/StageViewer`.
  - `TraderBar` – toolbar for launching council jobs, refreshing summaries, running researcher, copying logs.
  - Utility components for progress bars, job status, JSON drill-down.
- **State Management** – Hooks track selected post, filters, job status, logs; updates persisted between sessions.
- **Build/Run** – `npm install` (already run) and `npm run dev` with backend proxy; production via `npm run build`.

Refer to `frontend/README.md` for setup tips.

---

## 10. Analytics & Classification Utilities

- `build_ticker_universe.py` – Generates `tickers/tickers_enriched.csv` using external listings and metadata.
- `build_ticker_lexicon.py` – Constructs regex/token dictionaries for entity extraction.
- `ticker_universe.py` – Provides `TickerUniverse` dataclass for alias resolution and sector queries (used by entity stage + backend).
- `ticker_deep_classify.py` – LLM-assisted classification of ticker mentions with CLI for manual auditing.
- `stock_window.py` – Multi-interval OHLCV extraction with caching and CLI for testing windows.
- `technical_analyser.py` – See §4.7.
- `yfinance_throttle.py` – Shared throttle utility to respect Yahoo Finance limits.
- `ticker_conversation_hub.py` – See §7.

---

## 11. Automation & Research Harnesses

- `round_table.py` – Main orchestrator (see §6.2). Supports dummy tests, CSV-driven runs, JSON repair, pretty traces.
- `researcher_harness.py` – CLI wrapper for reproducible researcher runs with optional log capture and JSON dumps.
- `researcher.py` – LLM prompt/plan generator (see §5.1).
- `researcher_harness.py` + `backend/app.py` share helper methods for normalising article time, cleaning bullet strings, picking primary ticker.
- `ticker_conversation_hub.py` CLI commands integrate with `sheep_harvester` outputs for nightly maintenance (ingest → rollup → score).

---

## 12. Tests & Quality Assurance

- `tests/test_claims_stage.py` – Validates claim extraction schema and JSON normalisation.
- `tests/test_hub_adapter.py` – Ensures conversation hub client ingestion/score flows.
- `tests/test_ticker_universe.py` – Checks alias resolution, sector keywords, deduping.
- `backend/test_app.py` – FastAPI endpoint/unit tests (requires optional dependencies to be present or monkeypatched).
- Additional manual test scripts are stored under `docs/*.txt` and `prototype-v1/` for legacy reference.

Run suite via `pytest` from repo root.

---

## 13. Documentation Assets (`docs/`)

- `round_table.readme.txt`, `round_table-fixesAndStrategy.readme.txt` – Prompt engineering notes for council stages.
- `researcher.readme.txt` – Extended explanation of researcher prompt design.
- `conversation_hub.readme.txt` – Usage examples for hub ingestion, rollups, DES scoring.
- `Running Backend.txt` – Quickstart for FastAPI server.
- `wisdom_of_sheep_database.md` – Deep dive into SQLite schema and migrations.

---

## 14. Operational Playbooks

### 14.1 Running the Backend
```
cd backend
uvicorn app:app --reload --port 8000
```
Environment overrides:
- `WOS_PAGE_SIZE`, `WOS_BATCH_LIMIT`, `WOS_RUNNER_VERBOSE`, `WOS_MODEL_*` for council runner tuning.
- `WOS_CONVO_MODEL`, `WOS_EVIDENCE_LOOKBACK_DAYS`, `WOS_MAX_EVIDENCE_PER_CLAIM` for conversation hub + evidence fetch.

### 14.2 Frontend Dev Server
```
cd frontend
npm install
npm run dev -- --host
```
Proxy backend on `http://localhost:8000`.

### 14.3 Headless Harvest Loop
```
python sheep_harvester.py --interval 90 --max-items 16 --batch-size 8 --per-post-sleep 1.0
```
Provide custom RSS/X lists via `--rss-config file.txt`, `--x-config file.txt`.

### 14.4 Council Batch Run (CLI)
```
python round_table.py stage --post-id <id> --stage summariser
python round_table.py run --post-id <id> --stages summariser,entity,for,against,direction,chairman
```

### 14.5 Researcher Harness
```
python researcher_harness.py <post_id> --show-log --json-output out.json
```

### 14.6 Conversation Hub Maintenance
```
python ticker_conversation_hub.py ingest --db council/wisdom_of_sheep.sql --store sqlite --convos convos/conversations.sqlite --model mistral
python ticker_conversation_hub.py rollup --ticker HOOD --keep-latest 300
python ticker_conversation_hub.py score --ticker HOOD --days 7 --channel all
```

---

## 15. Directory Map (Quick Reference)

```
backend/            FastAPI service, API tests, job runners
frontend/           React operator console
council/            Agent stage implementations + schema + database
convos/             Conversation hub SQLite store
docs/               Extended documentation and prompts
prototype-v1/       Legacy Streamlit experiment (reference only)
researcher*.py      Researcher LLM orchestration + harness
sheep_harvester.py  Unified ingestion loop
scrapers (rss/reddit/x/stocktwits)  Source-specific harvesters
technical_analyser.py, stock_window.py, yfinance_throttle.py  Price analytics
ticker_*            Ticker metadata builders and classification utilities
tests/              Pytest suite for core modules
```

---

## 16. Troubleshooting

| Symptom | Likely Cause | Fix |
| --- | --- | --- |
| `missing-prerequisites` from bull/bear/direction | Researcher stage absent | Run `/api/posts/{id}/research` or `researcher_harness.py` first. |
| Conversation hub ingestion skips posts | Summariser bullets lack tickers | Re-run entity stage or verify ticker universe aliases. |
| Ollama chat timeout | Local model not running | Restart `ollama serve` or adjust `WOS_MODEL_TIMEOUT_SECS`. |
| Stock window endpoint errors | `stock_window.py` missing optional deps | Install `yfinance`, ensure network access. |
| Council job stuck | Stage crash mid-run | Check `.jobs/<job>/stdout.log`, use `/api/council-analysis/{job}/stop`, then rerun. |
| Rollup fails | Insufficient deltas | Lower `--keep-latest` or ingest more posts. |

---

## 17. Summary

- Harvesters ingest multi-channel market content into CSV/SQLite.
- Summariser/Entity/Interest stages triage articles for deeper analysis.
- Researcher plus technical/sentiment tooling prepare context for councils.
- Bull, Bear, Direction, Chairman agents produce audited verdicts.
- Conversation hub maintains ticker memory and sentiment metrics.
- Backend + frontend provide operational control, monitoring, and review.
- CLI utilities and tests support maintenance, verification, and debugging.

Use this manual as the authoritative reference for navigating and extending Wisdom of Sheep.
