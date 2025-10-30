# 🧠 Wisdom of Sheep — Oracle Server Retrieval Agent

## Overview

The **Oracle Retrieval Agent** is responsible for reading the live raw news corpus from the **Oracle Server** at `/opt/wos-receiver/app.py`.  
It replaces the old CSV-based ingestion path (`raw_posts_log.csv`) with a real-time, network-safe API.

The design mirrors the structure of the legacy CSV reader — every retrieved row matches the historical CSV schema exactly:

scraped_at,platform,source,post_id,url,title,text,final_url,fetch_status,domain

pgsql
Copy code

Downstream summarisation and Council analysis can therefore process remote rows as if they were local CSV rows, with **no behavioural changes**.

---

## Architecture

| Component | Purpose | Endpoint |
|------------|----------|----------|
| **Oracle Receiver (FastAPI)** | Stores scraped posts in SQLite (`/var/wos/posts.sqlite`) and mirrors them to `raw_posts_log.ndjson`. | `/wos/raw-posts` (write) |
| **Pi Harvester** | Scrapes new articles and uploads them one by one to the Oracle Receiver. | `/wos/raw-posts` |
| **Oracle Retrieval Agent (Codex)** | Retrieves articles for summarisation and Council analysis. | `/wos/next-after`, `/wos/stats`, `/wos/ready` |

The receiver enforces **single-flight writes** using an `asyncio.Lock`, and readers are expected to poll the `/wos/ready` endpoint to determine if the server is available before making requests.

---

## Retrieval Protocol

### 1. Readiness Check

Before requesting data or attempting any write, **always check** `/wos/ready`:

```http
GET /wos/ready
Example response:

json
Copy code
{
  "ready": true,
  "write_locked": false,
  "db_ok": true,
  "db_count_posts": 32492
}
If ready is false, Codex must back off and retry later.
If write_locked is true, a writer is currently active — usually the Pi harvester uploading a post.

2. Retrieval
The retrieval process uses /wos/next-after to fetch the next unseen article(s):

http
Copy code
GET /wos/next-after?platform=reddit&post_id=t3_1nq515c&limit=1
Each returned item will match the CSV schema exactly:

json
Copy code
{
  "scraped_at": "2025-09-25T12:11:15+00:00",
  "platform": "reddit",
  "source": "stocks",
  "post_id": "t3_1nq515c",
  "url": "https://reddit.com/r/stocks/comments/1nq515c/thoughts_on_gtk_asx_gentrack_group/",
  "title": "Thoughts on GTK (ASX) Gentrack group?",
  "text": "Thoughts on GTK (ASX)...",
  "final_url": "",
  "fetch_status": "",
  "domain": "reddit.com"
}
Downstream summarisation should treat this object exactly as it would a CSV row.

3. Throttling Requirements ⚠️
Codex must never fire concurrent requests to the Oracle server.
Even though SQLite is robust under WAL, concurrent access can cause locks and slowdowns.

Rules for Safe Retrieval
Always poll /wos/ready before /wos/next-after.

If ready is false, wait at least 1 second before retrying.

Never exceed one request per second for article retrieval.

Avoid parallel requests from multiple Codex workers. Retrieval must be serialized.

Use Retry-After headers from /wos/raw-posts responses as authoritative wait durations when present.

Failure to respect these rules can overload Oracle’s I/O system and crash the SQLite process.

4. Monitoring and Stats
For visibility and diagnostics, the server exposes /wos/stats:

http
Copy code
GET /wos/stats
Example response:

json
Copy code
{
  "total_posts": 32841,
  "total_seen_keys": 32719,
  "ndjson_size_bytes": 18871239,
  "write_locked": false
}
This allows Codex and orchestration scripts to:

Monitor corpus size growth

Detect write contention

Verify DB health and liveliness

Summary of Safety Mechanisms
Safety Mechanism	Description
Write Lock (WRITE_LOCK)	Ensures only one /wos/raw-posts writer operates at a time.
WAL Mode (SQLite)	Provides concurrent read safety while keeping writes atomic.
Readiness Probe	/wos/ready guarantees the receiver is not mid-write and DB is reachable.
409 + Retry-After	Returned by /wos/raw-posts when the receiver is busy.
Backoff Logic (Client)	Codex must obey retry intervals and avoid hammering the server.

Example Retrieval Flow (Codex)
python
Copy code
import requests, time

AUTH = ("user", "pass")
BASE = "http://130.162.168.45:8000"

def wait_for_ready():
    while True:
        r = requests.get(f"{BASE}/wos/ready", auth=AUTH)
        j = r.json()
        if j.get("ready"):
            return
        time.sleep(1.0)

def fetch_next(platform=None, post_id=None):
    wait_for_ready()
    r = requests.get(f"{BASE}/wos/next-after", params={"platform": platform, "post_id": post_id, "limit": 1}, auth=AUTH)
    return r.json()["items"]

while True:
    items = fetch_next("reddit", "t3_1nq515c")
    if not items:
        break
    print("Processing:", items[0]["title"])
    time.sleep(1.0)  # throttle
⚠️ Important for Codex and Pipeline Engineers
Do not run multiple summariser or retrieval loops simultaneously.

The Oracle receiver is single-threaded at the write level and limited to one active reader for stability.

Every new Codex analysis pass should include internal sleep intervals or async await delays to maintain under 1 request/sec.

If the receiver ever responds slowly or with a 500, stop all concurrent requests immediately — it’s a sign of lock contention or memory pressure.

In summary:
Oracle Server Retrieval provides CSV-compatible article access for the WOS pipeline, but it must be treated as a finite, single-threaded resource.
Always poll, throttle, and serialize.
Codex must consider server readiness and backpressure as first-class citizens in its retrieval logic.