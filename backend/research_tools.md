# `backend.research_tools`

`backend.research_tools` collects the routines used to turn researcher outputs
into actionable summaries.  It covers three main areas:

1. **Technical execution** – interprets a technical plan, calls the relevant
   `technical_analyser` helpers, and normalises the results into human-readable
   insights.
2. **Sentiment sampling** – queries the conversation hub for DES metrics and
   recent deltas, handling cases where the optional dependency is unavailable.
3. **Narrative synthesis** – composes markdown-ready research summaries that tie
   together hypotheses, rationale, technical checks, and sentiment headlines.

The module exposes the tool palettes (`TECH_TOOLS`, `SENTIMENT_TOOLS`) alongside
helpers for summarising technical runs and sentiment payloads.  This isolates the
research pipeline’s business logic from the HTTP layer, making it easier to test
and evolve the plan execution separately from the API surface.
