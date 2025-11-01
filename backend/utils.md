# `backend.utils`

`backend.utils` gathers lightweight helper functions shared across backend
subsystems.  The utilities focus on normalising article metadata, manipulating
strings, and preparing small pieces of derived data used in API responses.
Highlights include:

- ISO timestamp helpers (`now_iso`, `normalize_article_time`)
- Sanitising summary bullets and claim text lists
- Estimating direction, primary ticker, and market badges from stage payloads
- Parsing spam likelihood hints from arbitrary model output
- Convenience wrappers for truncating strings and formatting boolean flags for
  environment echoes

Keeping these helpers in one place avoids copy-paste between modules and keeps
`app.py` focused on routing logic.
