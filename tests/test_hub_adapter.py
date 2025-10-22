from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hub_adapter import HubClient


def _iso(ts: str) -> str:
    return datetime.fromisoformat(ts).replace(tzinfo=timezone.utc).isoformat()


def test_hub_client_score(tmp_path):
    db_path = tmp_path / "convos.sqlite"
    client = HubClient(db_path=str(db_path), model="stub")
    client.store.append_delta(
        "HOOD",
        {
            "t": _iso("2025-01-01T00:00:00"),
            "who": ["HOOD"],
            "dir": "up",
            "impact": "med",
            "sum": "Robinhood expands product reach",
            "chan": "news",
        },
    )

    res = client.score(ticker="hood", as_of=_iso("2025-01-02T00:00:00"), days=7)

    assert res["ticker"] == "HOOD"
    assert res["channel"] == "all"
    signal = res["signal"]
    assert set(signal.keys()) == {"des_raw", "des_sector", "des_idio", "confidence", "n_deltas"}


def test_hub_client_ask_as_of(monkeypatch: pytest.MonkeyPatch, tmp_path):
    db_path = tmp_path / "hub.sqlite"
    client = HubClient(db_path=str(db_path), model="stub")
    client.store.append_delta(
        "HOOD",
        {
            "t": _iso("2025-02-01T10:00:00"),
            "who": ["HOOD"],
            "dir": "up",
            "impact": "med",
            "sum": "User growth accelerates",
            "chan": "news",
        },
    )

    captured = {}

    def fake_chat(msgs, *, model, timeout_s=None):  # noqa: ANN001
        captured["msgs"] = msgs
        captured["model"] = model
        captured["timeout"] = timeout_s
        return "Narrative summary"

    monkeypatch.setattr("hub_adapter.chat", fake_chat)

    answer = client.ask_as_of(
        ticker="hood",
        as_of=_iso("2025-02-02T00:00:00"),
        q="What shifted sentiment?",
        timeout_s=12.5,
    )

    assert answer == "Narrative summary"
    assert captured["model"] == "stub"
    assert captured["timeout"] == 12.5
    msgs = captured["msgs"]
    assert msgs[-1]["role"] == "user"
    assert msgs[-1]["content"] == "What shifted sentiment?"
    assert any(m.get("content", "").startswith("AS_OF cutoff") for m in msgs if m.get("role") == "system")
