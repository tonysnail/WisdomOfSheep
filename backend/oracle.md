# `backend.oracle`

The `backend.oracle` module owns the state needed to coordinate with the external
Oracle summarisation service.  It handles:

- Serialising/deserialising the cursor and retry state files used for resumable
  ingestion
- Maintaining the reports of unsummarised and skipped articles with atomic
  writes
- Surfacing configuration knobs (batch sizing, backoff windows, authentication)
  for other subsystems to consume
- Lightweight helpers for joining Oracle-relative URLs

By isolating these routines we make the refresh workers agnostic to how Oracle
state is persisted on disk and concentrate the retry/backoff defaults in one
place.
