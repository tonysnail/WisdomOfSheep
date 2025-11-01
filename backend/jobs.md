# `backend.jobs`

`backend.jobs` provides persistence helpers for long-running backend tasks.
It abstracts reading and mutating the JSON job manifests stored in `.jobs/`,
including tail-log management and atomic updates.  The exported helpers are used
by the refresh workers and HTTP endpoints to:

- Fetch the latest snapshot for a job (`load_job`)
- Append log lines while keeping only the configured tail (`job_append_log`)
- Increment counters and update arbitrary fields (`job_increment`,
  `job_update_fields`)

Centralising these mutations ensures consistent file handling (temp-file writes,
`fsync`) and keeps filesystem concerns out of the worker orchestration logic.
