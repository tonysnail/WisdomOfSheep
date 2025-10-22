"""Import-only adapter for the ticker conversation hub."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import List, Optional

from ticker_conversation_hub import (
    SQLiteStore,
    build_as_of_messages,
    chat,
    compute_ticker_signal,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_CONVOS_DB = ROOT / "convos" / "conversations.sqlite"


class HubClient:
    """Pure-Python interface to the conversation hub for researcher.py."""

    def __init__(self, db_path: str | Path = DEFAULT_CONVOS_DB, model: str = "mistral"):
        resolved = Path(db_path)
        self._db_path = str(resolved)
        self._model = model
        self._store = SQLiteStore(self._db_path)

    @property
    def store(self) -> SQLiteStore:
        return self._store

    @property
    def model(self) -> str:
        return self._model

    def score(
        self,
        *,
        ticker: str,
        as_of: str,
        days: int = 7,
        channel: str = "all",
        peers: Optional[List[str]] = None,
        burst_hours: int = 6,
    ) -> dict:
        return compute_ticker_signal(
            self.store,
            ticker=ticker.upper(),
            as_of_iso=as_of,
            lookback_days=days,
            peers=peers,
            channel_filter=channel,
            burst_hours=burst_hours,
        )

    def ask_as_of(
        self,
        *,
        ticker: str,
        as_of: str,
        q: str,
        timeout_s: Optional[float] = 40.0,
    ) -> str:
        msgs = build_as_of_messages(self.store, ticker.upper(), as_of, limit=500)
        msgs.append({"role": "user", "content": q})
        return chat(msgs, model=self.model, timeout_s=timeout_s).strip()


_local = threading.local()


def get_hub(db_path: str | Path = DEFAULT_CONVOS_DB, model: str = "mistral") -> HubClient:
    hub = getattr(_local, "hub", None)
    resolved = str(Path(db_path))
    if hub is None or getattr(hub, "_db_path", None) != resolved or getattr(hub, "_model", None) != model:
        hub = HubClient(db_path=resolved, model=model)
        _local.hub = hub
    return hub


__all__ = ["HubClient", "SQLiteStore", "get_hub"]
