import json
import sqlite3
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

from backend import app as backend_app
from backend import jobs as backend_jobs
from council import interest_score


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_wisdom_of_sheep.sql"
    conn = sqlite3.connect(db_path)
    try:
        backend_app._ensure_schema(conn)
    finally:
        conn.close()
    monkeypatch.setattr(backend_app, "DB_PATH", db_path)
    monkeypatch.setattr(backend_app, "_DB_READY_PATH", None, raising=False)
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(backend_app, "JOBS_DIR", jobs_dir)
    monkeypatch.setattr(backend_jobs, "JOBS_DIR", jobs_dir)
    return db_path


def test_get_post_uses_latest_stage_payload(temp_db):
    post_id = "post-123"

    with backend_app._connect() as conn:
        conn.execute(
            "INSERT INTO posts (post_id, title) VALUES (?, ?)",
            (post_id, "Hello"),
        )
        conn.execute(
            """
            INSERT INTO stages (post_id, stage, created_at, payload)
            VALUES (?, ?, ?, ?)
            """,
            (
                post_id,
                "summariser",
                "2024-01-01T00:00:00Z",
                json.dumps({"spam_likelihood_pct": 5, "spam_why": "old run"}),
            ),
        )
        conn.execute(
            """
            INSERT INTO stages (post_id, stage, created_at, payload)
            VALUES (?, ?, ?, ?)
            """,
            (
                post_id,
                "summariser",
                "2024-02-01T00:00:00Z",
                json.dumps({"spam_likelihood_pct": 95, "spam_why": "latest run"}),
            ),
        )
        conn.commit()

    detail = backend_app.get_post(post_id)
    summariser_payload = detail["stages"].get("summariser")

    assert summariser_payload is not None
    assert summariser_payload["spam_likelihood_pct"] == 95
    assert summariser_payload["spam_why"] == "latest run"
    assert detail.get("interest") is None


def test_list_posts_includes_parsed_spam_fields(temp_db):
    post_id = "post-456"

    with backend_app._connect() as conn:
        conn.execute(
            "INSERT INTO posts (post_id, title) VALUES (?, ?)",
            (post_id, "Hello"),
        )
        conn.execute(
            """
            INSERT INTO stages (post_id, stage, created_at, payload)
            VALUES (?, ?, ?, ?)
            """,
            (
                post_id,
                "summariser",
                "2024-03-01T00:00:00Z",
                json.dumps(
                    {
                        "summary_bullets": [],
                        "spam_likelihood_pct": "60%",
                        "spam_why": " ",
                        "spam_reasons": [
                            "Reason one",
                            None,
                            "Reason two",
                        ],
                    }
                ),
            ),
        )
        conn.commit()

    data = backend_app.list_posts(page=1, page_size=5)
    assert len(data["items"]) == 1
    item = data["items"][0]

    assert item["spam_likelihood_pct"] == 60
    assert item["spam_why"] == "Reason one; Reason two"
    assert item.get("interest") is None


def test_list_posts_interest_filter(temp_db):
    with backend_app._connect() as conn:
        conn.execute("INSERT INTO posts (post_id, title) VALUES (?, ?)", ("p-high", "High"))
        conn.execute("INSERT INTO posts (post_id, title) VALUES (?, ?)", ("p-low", "Low"))
        conn.execute("INSERT INTO posts (post_id, title) VALUES (?, ?)", ("p-error", "Error"))
        conn.execute(
            """
            INSERT INTO council_stage_interest (post_id, status, interest_score, interest_label, interest_why, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("p-high", "ok", 82, "High", "Strong", "2024-01-01T00:00:00Z"),
        )
        conn.execute(
            """
            INSERT INTO council_stage_interest (post_id, status, interest_score, interest_label, interest_why, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("p-low", "ok", 35, "Low", "Weak", "2024-01-02T00:00:00Z"),
        )
        conn.execute(
            """
            INSERT INTO council_stage_interest (post_id, status, interest_score, interest_label, interest_why, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("p-error", "error", 90, "Err", "Issue", "2024-01-03T00:00:00Z"),
        )
        conn.commit()

    data = backend_app.list_posts(page=1, page_size=10, interest_min=70)
    ids = [item["post_id"] for item in data["items"]]

    assert ids == ["p-high"]


def test_start_council_analysis_respects_interest_minimum(temp_db, monkeypatch):
    with backend_app._connect() as conn:
        conn.execute("INSERT INTO posts (post_id, title) VALUES (?, ?)", ("p-high", "High interest"))
        conn.execute("INSERT INTO posts (post_id, title) VALUES (?, ?)", ("p-low", "Low interest"))
        conn.execute(
            """
            INSERT INTO council_stage_interest (post_id, status, interest_score, interest_label, interest_why, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("p-high", "ok", 78, "High", "Signal", "2024-03-01T00:00:00Z"),
        )
        conn.execute(
            """
            INSERT INTO council_stage_interest (post_id, status, interest_score, interest_label, interest_why, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("p-low", "ok", 32, "Low", "Weak", "2024-03-02T00:00:00Z"),
        )
        conn.commit()

    captured: dict[str, Any] = {}

    def fake_worker(job_id: str) -> None:
        captured["job_id"] = job_id

    monkeypatch.setattr(backend_app, "_worker_council_analysis", fake_worker)

    payload = backend_app.CouncilAnalysisStartRequest(interest_min=60.0, repair_missing=True)
    result = backend_app.start_council_analysis(payload)

    job_id = result["job_id"]
    assert captured.get("job_id") == job_id

    job = backend_app._load_job(job_id)
    assert job is not None
    queue_ids = [entry["post_id"] for entry in job.get("queue", [])]

    assert queue_ids == ["p-high"]
    assert job.get("skipped_below_threshold") == 1


def test_run_research_pipeline_reports_ticker_context(temp_db):
    post_id = "p-no-ticker"
    with backend_app._connect() as conn:
        conn.execute("INSERT INTO posts (post_id, title) VALUES (?, ?)", (post_id, "No ticker"))
        conn.execute(
            """
            INSERT INTO stages (post_id, stage, created_at, payload)
            VALUES (?, ?, ?, ?)
            """,
            (post_id, "summariser", "2024-04-01T00:00:00Z", json.dumps({"summary_bullets": []})),
        )
        conn.commit()

    with pytest.raises(backend_app.HTTPException) as exc_info:
        backend_app._run_research_pipeline(post_id)

    detail = exc_info.value.detail
    assert isinstance(detail, dict)
    assert detail.get("error") == "ticker-not-found"
    assert detail.get("reason") == "no candidate tickers extracted"


class DummyResponse:
    def __init__(self, status_code: int):
        self.status_code = status_code
        self.text = ""

    def json(self) -> dict[str, Any]:  # pragma: no cover - parity with requests.Response
        return {}


def _prepare_job(job_id: str) -> None:
    now = time.time()
    backend_app._job_save(
        {
            "id": job_id,
            "status": "queued",
            "log_tail": [],
            "error": "",
            "oracle_status": "connecting",
            "created_at": now,
            "updated_at": now,
        }
    )


def test_oracle_health_404_skips_failure(temp_db):
    job_id = "job-health-404"
    _prepare_job(job_id)

    should_continue = backend_app._process_oracle_health_response(job_id, DummyResponse(404))

    assert should_continue is True
    job = backend_app._load_job(job_id)
    assert job is not None
    assert job.get("status") == "queued"
    assert any("/healthz returned 404" in entry for entry in job.get("log_tail", []))


def test_oracle_health_error_marks_job_failed(temp_db):
    job_id = "job-health-error"
    _prepare_job(job_id)

    should_continue = backend_app._process_oracle_health_response(job_id, DummyResponse(503))

    assert should_continue is False
    job = backend_app._load_job(job_id)
    assert job is not None
    assert job.get("status") == "error"
    assert job.get("error") == "oracle-healthz-503"
    assert job.get("oracle_status") == "error"
    assert job.get("ended_at") is not None


def test_posts_calendar_counts_by_day(temp_db):
    with backend_app._connect() as conn:
        conn.execute(
            "INSERT INTO posts (post_id, title, posted_at) VALUES (?, ?, ?)",
            ("post-a", "A", "2024-02-05T08:00:00Z"),
        )
        conn.execute(
            "INSERT INTO posts (post_id, title, posted_at) VALUES (?, ?, ?)",
            ("post-b", "B", "2024-02-05T12:00:00Z"),
        )
        conn.execute(
            "INSERT INTO posts (post_id, title, scraped_at) VALUES (?, ?, ?)",
            ("post-c", "C", "2024-02-07T01:00:00Z"),
        )
        conn.execute(
            "INSERT INTO posts (post_id, title, posted_at) VALUES (?, ?, ?)",
            ("post-old", "Old", "2024-01-31T23:59:59Z"),
        )
        conn.commit()

    feb = backend_app.posts_calendar(year=2024, month=2)
    march = backend_app.posts_calendar(year=2024, month=3)

    assert feb.model_dump() == {
        "days": [
            {"date": "2024-02-05", "count": 2, "analysed_count": 0},
            {"date": "2024-02-07", "count": 1, "analysed_count": 0},
        ]
    }
    assert march.model_dump() == {"days": []}


def test_clear_analysis_removes_non_summary_stages(temp_db):
    post_id = "post-clear"

    with backend_app._connect() as conn:
        conn.execute(
            "INSERT INTO posts (post_id, title) VALUES (?, ?)",
            (post_id, "Hello"),
        )
        backend_app._insert_stage_payload(conn, post_id, "summariser", {"ok": True})
        backend_app._insert_stage_payload(conn, post_id, "conversation_hub", {"foo": "bar"})
        backend_app._insert_stage_payload(conn, post_id, "researcher", {"ok": True})
        backend_app._insert_stage_payload(conn, post_id, "for", {"text": "bull"})
        backend_app._insert_stage_payload(conn, post_id, "against", {"text": "bear"})
        backend_app._insert_stage_payload(conn, post_id, "chairman", {"plain_english_result": "Verdict"})

        extras = {
            "final_url": "https://example.com/article",
            "conversation_hub": {"foo": "bar"},
            "research": {"notes": "keep?"},
        }
        backend_app._save_extras_dict(conn, post_id, extras)

    res = backend_app.clear_post_analysis(post_id)

    assert res["ok"] is True
    assert res["deleted_stages"] == 4
    assert res["removed_research"] is True

    with backend_app._connect() as conn:
        rows = conn.execute(
            "SELECT stage FROM stages WHERE post_id = ? ORDER BY stage", (post_id,)
        ).fetchall()
        remaining = {row["stage"] for row in rows}
        extras_after = backend_app._load_extras_dict(conn, post_id)

    assert remaining == {"summariser", "conversation_hub"}
    assert "research" not in extras_after
    assert extras_after["conversation_hub"] == {"foo": "bar"}
    assert extras_after["final_url"] == "https://example.com/article"


def test_clear_analysis_does_not_touch_conversation_store(temp_db, tmp_path, monkeypatch):
    post_id = "post-keep-convo"

    convo_path = tmp_path / "conversations.sqlite"
    convo_path.write_text("original", encoding="utf-8")
    monkeypatch.setattr(backend_app, "CONVO_STORE_PATH", convo_path)

    with backend_app._connect() as conn:
        conn.execute(
            "INSERT INTO posts (post_id, title) VALUES (?, ?)",
            (post_id, "Hello"),
        )
        backend_app._insert_stage_payload(conn, post_id, "summariser", {"ok": True})
        backend_app._insert_stage_payload(conn, post_id, "conversation_hub", {"foo": "bar"})
        backend_app._insert_stage_payload(conn, post_id, "researcher", {"ok": True})
        backend_app._save_extras_dict(
            conn,
            post_id,
            {"conversation_hub": {"foo": "bar"}, "research": {"notes": "gone"}},
        )

    before = convo_path.read_bytes()

    res = backend_app.clear_post_analysis(post_id)

    assert res["ok"] is True
    assert convo_path.exists()
    assert convo_path.read_bytes() == before


def test_clear_analysis_missing_post_raises(temp_db):
    with pytest.raises(backend_app.HTTPException) as excinfo:
        backend_app.clear_post_analysis("missing-post")

    assert excinfo.value.status_code == 404


def test_erase_all_council_analysis_removes_non_summary(temp_db):
    posts = ("post-one", "post-two")

    with backend_app._connect() as conn:
        for pid in posts:
            conn.execute(
                "INSERT INTO posts (post_id, title) VALUES (?, ?)",
                (pid, f"Title {pid}"),
            )
            backend_app._insert_stage_payload(conn, pid, "summariser", {"ok": True})
            backend_app._insert_stage_payload(conn, pid, "conversation_hub", {"delta": {"sum": "hello"}})
            backend_app._insert_stage_payload(conn, pid, "researcher", {"ok": True})
            backend_app._insert_stage_payload(conn, pid, "for", {"bull": True})
            backend_app._insert_stage_payload(conn, pid, "chairman", {"plain_english_result": "Go"})
            backend_app._save_extras_dict(
                conn,
                pid,
                {"research": {"notes": "remove"}, "conversation_hub": {"delta": "keep"}},
            )

    result = backend_app.erase_all_council_analysis()

    assert result["ok"] is True
    assert result["deleted_stages"] == 6
    assert result["cleared_research_posts"] == 2

    with backend_app._connect() as conn:
        rows = conn.execute("SELECT post_id, stage FROM stages").fetchall()
        remaining = {(row["post_id"], row["stage"]) for row in rows}
        expected = {
            ("post-one", "summariser"),
            ("post-one", "conversation_hub"),
            ("post-two", "summariser"),
            ("post-two", "conversation_hub"),
        }
        assert remaining == expected

        for pid in posts:
            extras = backend_app._load_extras_dict(conn, pid)
            assert "research" not in extras
            assert extras.get("conversation_hub") == {"delta": "keep"}


def test_recent_texts_factory_orders_oldest_first(tmp_path):
    conv_path = tmp_path / "conversations.sqlite"
    conn = sqlite3.connect(conv_path)
    try:
        conn.execute(
            """
            CREATE TABLE conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                ts TEXT NOT NULL,
                kind TEXT NOT NULL,
                data TEXT NOT NULL,
                post_id TEXT
            )
            """
        )
        conn.executemany(
            "INSERT INTO conversations (ticker, ts, kind, data, post_id) VALUES (?, ?, ?, ?, ?)",
            [
                ("AAPL", "2024-01-02T00:00:00Z", "delta", json.dumps({"summary": "newer"}), "p2"),
                ("AAPL", "2024-01-01T00:00:00Z", "delta", json.dumps({"summary": "older"}), "p1"),
                ("AAPL", "2024-01-01T00:00:00Z", "memory", json.dumps({"note": "ignore"}), None),
            ],
        )
        conn.commit()
    finally:
        conn.close()

    cb = interest_score._get_recent_texts_for_ticker_factory(str(conv_path))
    assert cb is not None

    texts = cb("AAPL", "2024-01-03T00:00:00Z", 7)
    assert texts == ["older", "newer"]


def test_start_council_analysis_skips_existing_chairman(temp_db, tmp_path, monkeypatch):
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(backend_app, "JOBS_DIR", jobs_dir, raising=False)
    monkeypatch.setattr(backend_app, "ACTIVE_COUNCIL_JOB_ID", None, raising=False)

    started: dict[str, Any] = {}

    class DummyThread:
        def __init__(self, target=None, args=(), kwargs=None, daemon=None):
            started["target"] = target
            started["args"] = args
            started["daemon"] = daemon

        def start(self):
            started["started"] = True

    monkeypatch.setattr(backend_app.threading, "Thread", DummyThread)

    with backend_app._connect() as conn:
        posts = [
            ("post-chair", "Has verdict", "2024-02-01T00:00:00Z"),
            ("post-partial", "Needs rerun", "2024-02-02T00:00:00Z"),
            ("post-fresh", "Never run", "2024-02-03T00:00:00Z"),
        ]
        for pid, title, posted in posts:
            conn.execute(
                "INSERT INTO posts (post_id, title, posted_at) VALUES (?, ?, ?)",
                (pid, title, posted),
            )
            conn.execute(
                """
                INSERT INTO council_stage_interest (post_id, status, interest_score, created_at)
                VALUES (?, 'ok', ?, ?)
                """,
                (
                    pid,
                    {"post-chair": 92.0, "post-partial": 58.0, "post-fresh": 81.0}[pid],
                    posted,
                ),
            )

        backend_app._insert_stage_payload(conn, "post-chair", "chairman", {"ok": True})
        backend_app._insert_stage_payload(conn, "post-partial", "entity", {"ok": True})

    payload = backend_app.CouncilAnalysisStartRequest(interest_min=50.0)
    result = backend_app.start_council_analysis(payload)

    assert result["total"] == 2
    assert result["skipped_with_chairman"] == 1
    assert started.get("started") is True

    job = backend_app._load_job(result["job_id"])
    assert job is not None
    assert job.get("skipped_with_chairman") == 1
    queue_ids = [entry["post_id"] for entry in job["queue"]]
    assert queue_ids == ["post-partial", "post-fresh"]


def test_run_round_table_success(monkeypatch):
    def fake_runner(**kwargs):
        assert kwargs["post_id"] == "post-789"
        assert kwargs["stages"] == ["summariser"]
        assert kwargs["refresh_from_csv"] is False
        assert callable(kwargs["log_callback"])
        kwargs["log_callback"]("stage running")
        return {"summariser": {"ok": True}}

    monkeypatch.setattr(backend_app, "run_stages_for_post", fake_runner)

    code, out, err = backend_app._run_round_table("post-789", ["summariser"], False, False)

    assert code == 0
    payload = json.loads(out)
    assert payload == {"post_id": "post-789", "stages": ["summariser"]}
    assert "stage running" in err


def test_run_round_table_failure(monkeypatch):
    def fake_runner(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(backend_app, "run_stages_for_post", fake_runner)

    code, out, err = backend_app._run_round_table("post-000", ["summariser"], False, False)

    assert code == 1
    assert out == ""
    assert "boom" in err


def test_insert_stage_payload_records_created_at(temp_db):
    post_id = "post-insert"
    with backend_app._connect() as conn:
        created_at = backend_app._insert_stage_payload(conn, post_id, "researcher", {"ok": True})
        row = conn.execute(
            "SELECT post_id, stage, created_at, payload FROM stages WHERE post_id = ? AND stage = ?",
            (post_id, "researcher"),
        ).fetchone()

    assert isinstance(created_at, str)
    assert row is not None
    assert row["post_id"] == post_id
    assert row["stage"] == "researcher"
    assert row["created_at"] == created_at


def test_refresh_summaries_worker_ingests_conversation_hub(temp_db, tmp_path, monkeypatch):
    job_id = "job-refresh"
    csv_snapshot = tmp_path / "snapshot.csv"
    csv_snapshot.write_text(
        "post_id,title,platform,source,url,text,final_url,fetch_status,domain,scraped_at\n"
        "post-new,New post,reddit,sub,https://example.com,body,https://example.com,ok,example.com,2024-01-01T00:00:00Z\n",
        encoding="utf-8",
    )

    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(backend_app, "JOBS_DIR", jobs_dir)
    monkeypatch.setattr(backend_app, "ACTIVE_JOB_ID", job_id, raising=False)

    raw_csv = tmp_path / "raw.csv"
    raw_csv.write_text("post_id,title\n", encoding="utf-8")
    monkeypatch.setattr(backend_app, "CSV_PATH", raw_csv, raising=False)

    now = time.time()
    backend_app._job_save(
        {
            "id": job_id,
            "status": "queued",
            "snapshot": str(csv_snapshot),
            "backlog": [],
            "total": 1,
            "done": 0,
            "current": "",
            "phase": "",
            "log_tail": [],
            "error": "",
            "started_at": None,
            "ended_at": None,
            "cancelled": False,
            "created_at": now,
            "updated_at": now,
        }
    )

    ingested: list[tuple[str, str]] = []

    def fake_ingest(conn, post_id):
        ingested.append((post_id, "called"))
        return {"tickers": ["ABC"], "appended": 1}

    def fake_runner(**kwargs):
        with backend_app._connect() as conn:
            backend_app._insert_stage_payload(conn, kwargs["post_id"], "summariser", {"ok": True})
        return {"summariser": {"ok": True}}

    monkeypatch.setattr(backend_app, "_ingest_conversation_hub", fake_ingest)
    monkeypatch.setattr(backend_app, "run_stages_for_post", fake_runner)

    backend_app._worker_refresh_summaries(job_id)

    assert ingested == [("post-new", "called")]

    job = backend_app._load_job(job_id)
    assert job is not None
    assert job["status"] == "done"

    with backend_app._connect() as conn:
        rows = conn.execute(
            "SELECT stage FROM stages WHERE post_id = ?", ("post-new",)
        ).fetchall()

    assert {row["stage"] for row in rows} >= {"summariser"}


def test_refresh_summaries_active_returns_204_when_missing_job(tmp_path, monkeypatch):
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    job_id = "job-missing"
    monkeypatch.setattr(backend_app, "JOBS_DIR", jobs_dir, raising=False)
    monkeypatch.setattr(backend_app, "ACTIVE_JOB_ID", job_id, raising=False)

    response = backend_app.get_active_refresh_summaries()

    assert hasattr(response, "status_code")
    assert response.status_code == 204
    assert backend_app.ACTIVE_JOB_ID is None


def test_refresh_summaries_active_endpoint_is_not_shadowed(tmp_path, monkeypatch):
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    job_id = "job-missing"
    monkeypatch.setattr(backend_app, "JOBS_DIR", jobs_dir, raising=False)
    monkeypatch.setattr(backend_app, "ACTIVE_JOB_ID", job_id, raising=False)

    client = TestClient(backend_app.app)
    response = client.get("/api/refresh-summaries/active")

    assert response.status_code == 204
    assert backend_app.ACTIVE_JOB_ID is None


def test_refresh_summaries_active_returns_payload(tmp_path, monkeypatch):
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(backend_app, "JOBS_DIR", jobs_dir, raising=False)

    job_id = "job-active"
    now = time.time()
    backend_app._job_save(
        {
            "id": job_id,
            "status": "running",
            "total": 5,
            "done": 2,
            "phase": "processing",
            "current": "post-123",
            "log_tail": ["line"],
            "created_at": now,
            "updated_at": now,
        }
    )

    monkeypatch.setattr(backend_app, "ACTIVE_JOB_ID", job_id, raising=False)

    client = TestClient(backend_app.app)
    response = client.get("/api/refresh-summaries/active")

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == job_id
    assert data["job_id"] == job_id
    assert data["status"] == "running"
    assert data["total"] == 5
    assert data["done"] == 2
    assert data["log_tail"] == ["line"]
    assert data["message"].startswith("processing 2/5")


def test_council_active_returns_204_when_missing_job(tmp_path, monkeypatch):
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    job_id = "council-missing"
    monkeypatch.setattr(backend_app, "JOBS_DIR", jobs_dir, raising=False)
    monkeypatch.setattr(backend_app, "ACTIVE_COUNCIL_JOB_ID", job_id, raising=False)

    response = backend_app.get_active_council_analysis()

    assert hasattr(response, "status_code")
    assert response.status_code == 204
    assert backend_app.ACTIVE_COUNCIL_JOB_ID is None


def test_council_active_endpoint_returns_204_when_missing_job(tmp_path, monkeypatch):
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    job_id = "council-missing"
    monkeypatch.setattr(backend_app, "JOBS_DIR", jobs_dir, raising=False)
    monkeypatch.setattr(backend_app, "ACTIVE_COUNCIL_JOB_ID", job_id, raising=False)

    client = TestClient(backend_app.app)
    response = client.get("/api/council-analysis/active")

    assert response.status_code == 204
    assert backend_app.ACTIVE_COUNCIL_JOB_ID is None
