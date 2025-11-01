from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .config import JOBS_DIR, JOB_LOG_KEEP


def job_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def load_job(job_id: str) -> Optional[Dict[str, Any]]:
    path = job_path(job_id)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _job_save(job: Dict[str, Any]) -> None:
    path = job_path(job["id"])
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(job, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def _job_mutate(job_id: str, mutator: Callable[[Dict[str, Any]], None]) -> Optional[Dict[str, Any]]:
    job = load_job(job_id)
    if not job:
        return None
    mutator(job)
    job["updated_at"] = time.time()
    _job_save(job)
    return job


def job_update_fields(job_id: str, **fields: Any) -> Optional[Dict[str, Any]]:
    return _job_mutate(job_id, lambda job: job.update(fields))


def job_append_log(job_id: str, line: str, keep: int = JOB_LOG_KEEP) -> Optional[Dict[str, Any]]:
    def mutate(job: Dict[str, Any]) -> None:
        tail = list(job.get("log_tail") or [])
        tail.append(line)
        if len(tail) > keep:
            tail = tail[-keep:]
        job["log_tail"] = tail

    return _job_mutate(job_id, mutate)


def job_increment(job_id: str, field: str, amount: int = 1) -> Optional[Dict[str, Any]]:
    def mutate(job: Dict[str, Any]) -> None:
        current = int(job.get(field, 0) or 0)
        job[field] = current + amount

    return _job_mutate(job_id, mutate)


__all__ = [
    "job_append_log",
    "job_increment",
    "job_path",
    "job_update_fields",
    "load_job",
]
