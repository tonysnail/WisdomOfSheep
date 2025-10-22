"""Shared throttling helper for yfinance calls.

This project makes a number of rapid successive calls to ``yfinance`` when
computing technical indicators and interest scores.  Yahoo Finance is quick to
rate‑limit bursty traffic which results in ``Too Many Requests`` responses that
surface as fetch errors in our pipeline.  The helper below introduces a tiny
delay between outbound calls so the different modules can co‑ordinate and avoid
tripping that limit.

The throttle duration can be tuned via the ``WOS_YF_THROTTLE_SECONDS``
environment variable (default: 0.5 seconds).  Set the value to ``0`` to disable
the throttle altogether if needed.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Optional, Union

_DEFAULT_INTERVAL = 0.5

_lock = threading.Lock()
_last_request_at: float = 0.0


def _read_interval(env_override: Optional[Union[str, float, int]]) -> float:
    try:
        if env_override is None:
            return _DEFAULT_INTERVAL
        value = float(env_override)
        return max(0.0, value)
    except Exception:
        return _DEFAULT_INTERVAL


_min_interval = _read_interval(os.getenv("WOS_YF_THROTTLE_SECONDS"))


def throttle_yfinance(min_interval: Optional[float] = None) -> None:
    """Sleep briefly so that sequential Yahoo Finance calls are throttled."""

    global _last_request_at

    interval = _read_interval(min_interval) if min_interval is not None else _min_interval
    if interval <= 0:
        return

    with _lock:
        now = time.monotonic()
        wait = (_last_request_at + interval) - now
        if wait > 0:
            time.sleep(wait)
            now = time.monotonic()
        _last_request_at = now
