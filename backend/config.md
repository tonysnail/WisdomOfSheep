# `backend.config`

The `backend.config` module centralises environment-dependent settings and
filesystem locations for the Wisdom of Sheep backend.  It exposes a curated set
of helper functions for reading integers, floats, and boolean flags from the
process environment, and resolves repository-relative paths that other
subsystems rely on (database files, council assets, ticker data, etc.).

Importing this module establishes the repository root on `sys.path`, ensures the
data directory exists, and surfaces ready-to-use constants for:

- Council database files (`DB_PATH`, schema templates, council time model)
- Oracle batch processing (cursor, retry, batch tuning knobs)
- Conversation hub storage location and default model selection
- Round-table execution defaults (model, host, verbosity, evidence limits)
- Job persistence directories and logging retention

Other backend modules consume these constants instead of re-reading environment
variables.  This keeps configuration logic in one place and makes it easier to
reason about runtime behaviour across the refactored subsystems.
