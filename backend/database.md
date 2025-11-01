# `backend.database`

`backend.database` wraps all direct interactions with the council SQLite
database.  The module guarantees that the database file exists and passes
`PRAGMA quick_check`, creates tables and indices on demand, and exposes helper
functions for common CRUD operations used across the FastAPI service.

Key responsibilities include:

- Initialising or restoring the council database from the shipped template
- Providing thread-safe access to a ready `sqlite3.Connection`
- Creating and migrating schema objects for posts, stages, extras, and interest
  tracking tables
- Convenience helpers for querying, inserting stage payloads, and managing
  per-post extras

By funnelling database logic through this module we keep connection handling and
schema drift concerns out of the HTTP routing layer.  Other subsystems (jobs,
conversation hub ingestion, council analysis flows) simply import the
appropriate helpers and focus on their own domain logic.
