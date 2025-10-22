PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

-- Core table: one row per article/post in the corpus
CREATE TABLE IF NOT EXISTS posts (
    post_id      TEXT PRIMARY KEY,
    platform     TEXT,
    source       TEXT,
    url          TEXT,
    title        TEXT,
    author       TEXT,
    scraped_at   TEXT,   -- when our system fetched it
    posted_at    TEXT,   -- when the article/post was created
    score        REAL,
    text         TEXT    -- full article text
);

-- Stage outputs: store the JSON blob for each step of round_table
-- Example stages: entity, summariser, claims, context, for, against, direction, moderator, verifier
CREATE TABLE IF NOT EXISTS stages (
    post_id      TEXT NOT NULL,
    stage        TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    payload      JSON NOT NULL,
    PRIMARY KEY (post_id, stage),
    FOREIGN KEY (post_id) REFERENCES posts(post_id)
);

-- Optional: store extracted bullets separately for easy querying
CREATE TABLE IF NOT EXISTS bullets (
    post_id      TEXT NOT NULL,
    bullet_text  TEXT NOT NULL,
    ticker       TEXT,
    created_at   TEXT NOT NULL,
    PRIMARY KEY(post_id, bullet_text)
);

-- Optional: store extracted tickers separately
CREATE TABLE IF NOT EXISTS tickers (
    post_id      TEXT NOT NULL,
    ticker       TEXT NOT NULL,
    market       TEXT,
    confidence   REAL,
    PRIMARY KEY(post_id, ticker),
    FOREIGN KEY(post_id) REFERENCES posts(post_id)
);

-- Full-text search index over posts.text + title (for evidence lookups)
CREATE VIRTUAL TABLE IF NOT EXISTS posts_fts
USING fts5(post_id UNINDEXED, title, text, content='posts', content_rowid='rowid');

-- Keep FTS index in sync with posts
CREATE TRIGGER IF NOT EXISTS posts_ai AFTER INSERT ON posts BEGIN
  INSERT INTO posts_fts(rowid, post_id, title, text)
  VALUES (new.rowid, new.post_id, new.title, new.text);
END;
CREATE TRIGGER IF NOT EXISTS posts_ad AFTER DELETE ON posts BEGIN
  INSERT INTO posts_fts(posts_fts, rowid, post_id, title, text)
  VALUES ('delete', old.rowid, old.post_id, old.title, old.text);
END;
CREATE TRIGGER IF NOT EXISTS posts_au AFTER UPDATE ON posts BEGIN
  INSERT INTO posts_fts(posts_fts, rowid, post_id, title, text)
  VALUES ('delete', old.rowid, old.post_id, old.title, old.text);
  INSERT INTO posts_fts(rowid, post_id, title, text)
  VALUES (new.rowid, new.post_id, new.title, new.text);
END;
