# `backend.conversation`

This module wraps all interaction points with the ticker conversation hub.  It
lazily initialises the optional `ConversationHub` client, caches the enriched
ticker universe sourced from `tickers/tickers_enriched.csv`, and provides helper
functions for:

- Selecting relevant tickers for a post using summariser output and fallback
  cashtag parsing
- Ingesting summarised content into the hub while recording the resulting stage
  payload
- Fetching recent deltas for use in sentiment narratives
- Filtering ticker lists against the known universe

Other subsystems import these helpers to avoid duplicating hub-specific logic or
file-system probes.  If the optional dependency is unavailable the helpers fail
softly, allowing the backend to keep serving core functionality without the
conversation hub attached.
