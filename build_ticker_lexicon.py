#!/usr/bin/env python3
"""
build_ticker_lexicon.py — Learn per-ticker lexical gates from the ENTIRE DB corpus.

- Scans ALL posts in wisdom_of_sheep.sql (no lookback window)
- Uses tickers_enriched.csv to resolve/alias tickers
- Groups every post by resolved symbol (one post may count for multiple symbols)
- Pulls summariser bullets when available
- Asks a small LLM (Ollama by default; OpenAI optional) to produce a strict-JSON lexicon
- Writes to table ticker_lexicon(symbol, prompt_version, model, created_at, keywords_json)

Usage examples:
  # Upsert (default), Ollama mistral, unlimited posts per symbol
  python build_ticker_lexicon.py --model mistral

  # Skip symbols that already have entries for this (prompt_version, model)
  python build_ticker_lexicon.py --skip-existing

  # Use OpenAI (set OPENAI_API_KEY), cap to 120 samples per symbol
  USE_OPENAI=1 python build_ticker_lexicon.py --model gpt-4.1-mini --max-per-symbol 120
"""

from __future__ import annotations
import os, re, csv, json, sqlite3, argparse, subprocess
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Set

ROOT = os.path.dirname(__file__)
DB_PATH_DEFAULT = os.path.join(ROOT, "council", "wisdom_of_sheep.sql")
TICKERS_CSV_DEFAULT = os.path.join(ROOT, "tickers", "tickers_enriched.csv")

# ----------------- tiny utils -----------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def _clip(s: str, n: int) -> str:
    return " ".join((s or "").split())[:n]

def _sha1(s: str) -> str:
    import hashlib
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _parse_first_json(blob: str) -> Optional[dict]:
    try:
        s = blob.find("{"); e = blob.rfind("}")
        if s != -1 and e != -1 and e > s:
            return json.loads(blob[s:e+1])
    except Exception:
        pass
    return None

# ----------------- universe -----------------
class TickerUniverse:
    def __init__(self, metas: Dict[str, Dict[str, Any]], alias: Dict[str, str]):
        self.metas = metas
        self.alias = alias

    @classmethod
    def from_csv(cls, path: str) -> "TickerUniverse":
        metas, alias = {}, {}
        with open(path, "r", newline="", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                sym = (row.get("Symbol") or "").upper().strip()
                if not sym: 
                    continue
                metas[sym] = {
                    "longName": row.get("longName") or None,
                    "sector": (row.get("sector") or "UNKNOWN").strip(),
                    "industry": row.get("industry") or None,
                    "asset_type": row.get("asset_type") or None,
                    "marketCap": row.get("marketCap") or None,
                }
                al = (row.get("aliases") or "").strip()
                if al:
                    for a in al.split(","):
                        a = a.strip().upper()
                        if a:
                            alias.setdefault(a, sym)
        return cls(metas, alias)

    def resolve(self, raw: str) -> Optional[str]:
        s = (raw or "").upper().strip()
        if not s: 
            return None
        if s in self.metas:
            return s
        return self.alias.get(s)

# ----------------- DB I/O -----------------
def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
    CREATE TABLE IF NOT EXISTS ticker_lexicon (
        symbol TEXT NOT NULL,
        prompt_version TEXT NOT NULL,
        model TEXT NOT NULL,
        created_at TEXT NOT NULL,
        keywords_json TEXT NOT NULL,
        PRIMARY KEY(symbol, prompt_version, model)
    );
    """)
    conn.commit()

def _extract_symbols_from_stage_payload(payload: str, uni: TickerUniverse) -> Set[str]:
    """Pull tickers from JSON payloads (summariser/entity)."""
    out: Set[str] = set()
    try:
        js = json.loads(payload or "{}")
    except Exception:
        return out

    # Common locations:
    candidates = []

    # summariser schema: assets_mentioned: [{ticker, name_or_description, ...}]
    am = js.get("assets_mentioned")
    if isinstance(am, list):
        candidates.extend([it.get("ticker") for it in am if isinstance(it, dict)])

    # sometimes a flatter list:
    if isinstance(js.get("tickers"), list):
        candidates.extend(js.get("tickers"))

    # sometimes nested under "entities" / "assets"
    ent = js.get("entities") or js.get("assets")
    if isinstance(ent, list):
        for it in ent:
            if isinstance(it, dict):
                candidates.append(it.get("ticker"))

    for raw in candidates:
        if not raw:
            continue
        can = uni.resolve(str(raw))
        if can:
            out.add(can)

    return out


_CASHTAG = re.compile(r"\$[A-Z]{1,5}\b")  # crude but effective; you already have a stoplist elsewhere if needed

def _resolve_cashtags(title: str, text: str, uni: TickerUniverse) -> Set[str]:
    out: Set[str] = set()
    s = f"{title or ''} {text or ''}"
    for m in _CASHTAG.findall(s.upper()):
        raw = m[1:]  # strip $
        can = uni.resolve(raw)
        if can:
            out.add(can)
    return out


def _load_all_grouped_by_symbol(conn: sqlite3.Connection, uni: TickerUniverse, max_per_symbol: Optional[int]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Returns {symbol: [ {post_id,title,bullet,snippet}... ]} scanning ALL posts,
    deriving symbols from per-article stages (summariser/entity) or cashtags in title/text.
    """
    cur = conn.cursor()

    # 1) Load all posts (ids + compact text)
    posts: Dict[str, Dict[str, str]] = {}
    for pid, title, text in cur.execute("SELECT post_id, title, text FROM posts"):
        posts[pid] = {
            "title": _clip(title or "", 160),
            "snippet": _clip(text or "", 240),
        }

    if not posts:
        return {}

    post_ids = list(posts.keys())

    # 2) Pull summariser bullets (optional, for a better signal in prompts)
    bullets_by_post: Dict[str, str] = {}
    CHUNK = 1000
    for i in range(0, len(post_ids), CHUNK):
        sub = post_ids[i:i+CHUNK]
        ph = ",".join("?" for _ in sub)
        for pid, payload in cur.execute(
            f"SELECT post_id, payload FROM stages WHERE stage='summariser' AND post_id IN ({ph})", sub
        ):
            try:
                js = json.loads(payload)
                bs = js.get("summary_bullets") or []
                if isinstance(bs, list) and bs:
                    bullets_by_post[pid] = _clip(str(bs[0]), 200)
                else:
                    sm = (js.get("summary") or "").strip()
                    if sm:
                        bullets_by_post[pid] = _clip(sm, 200)
            except Exception:
                pass

    # 3) Derive symbols per post:
    #    a) from 'summariser' and 'entity' stage payloads
    #    b) fallback: cashtags in title/text
    symbols_by_post: Dict[str, Set[str]] = {pid: set() for pid in post_ids}

    for stage in ("summariser", "entity"):
        for i in range(0, len(post_ids), CHUNK):
            sub = post_ids[i:i+CHUNK]
            ph = ",".join("?" for _ in sub)
            for pid, payload in cur.execute(
                f"SELECT post_id, payload FROM stages WHERE stage=? AND post_id IN ({ph})",
                (stage, *sub)
            ):
                syms = _extract_symbols_from_stage_payload(payload, uni)
                if syms:
                    symbols_by_post[pid].update(syms)

    # Fallback to cashtags where still empty
    for pid, meta in posts.items():
        if symbols_by_post[pid]:
            continue
        syms = _resolve_cashtags(meta.get("title",""), meta.get("snippet",""), uni)
        if syms:
            symbols_by_post[pid].update(syms)

    # 4) Build samples grouped by symbol
    samples_by_sym: Dict[str, List[Dict[str, Any]]] = {}
    for pid, syms in symbols_by_post.items():
        if not syms:
            continue
        sample = {
            "post_id": pid,
            "title": posts[pid]["title"],
            "bullet": bullets_by_post.get(pid, ""),
            "snippet": posts[pid]["snippet"],
        }
        for sym in syms:
            arr = samples_by_sym.setdefault(sym, [])
            if max_per_symbol is None or max_per_symbol <= 0 or len(arr) < max_per_symbol:
                arr.append(sample)

    return samples_by_sym

# ----------------- LLM plumbing -----------------
class LLMCfg:
    def __init__(self, provider: str, model: str, timeout: Optional[float], prompt_version: str):
        self.provider = provider
        self.model = model
        self.timeout = timeout
        self.prompt_version = prompt_version

_PROMPT_MEMO: Dict[str, str] = {}  # prompt_hash -> raw result

def _ollama_run(model: str, prompt: str, timeout: Optional[float]) -> str:
    cmd = ["ollama", "run", model, prompt]
    kw = {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE, "check": True}
    if timeout: kw["timeout"] = timeout
    cp = subprocess.run(cmd, **kw)
    return cp.stdout.decode("utf-8", errors="ignore").strip()

def _openai_run(model: str, prompt: str, timeout: Optional[float]) -> str:
    from openai import OpenAI
    client = OpenAI()
    resp = client.responses.create(
        model=model,
        input=[{"role": "user", "content": prompt}],
        max_output_tokens=400,
        temperature=0.0,
        timeout_ms=int(timeout*1000) if timeout else None,
    )
    return (resp.output_text or "").strip()

def _compose_prompt(symbol: str, meta: Dict[str, Any], samples: List[Dict[str, Any]]) -> str:
    sector = meta.get("sector") or "UNKNOWN"
    industry = meta.get("industry") or ""
    head = (
        "You infer a lexical gate for filtering news relevant to a stock ticker.\n"
        "Given examples (titles/snippets), produce STRICT JSON with *lowercase* short keywords.\n"
        "Focus on the business model & industry; do NOT include the ticker symbol itself.\n"
        "JSON schema:\n"
        "{"
        '"must_hooks":[ "..."],'
        '"finance_core":[ "..."],'
        '"sector_terms":[ "..."],'
        '"macro_terms":[ "..."],'
        '"regulatory_terms":[ "..."],'
        '"exclusions":[ "..."],'
        '"why":"<=20 words"'
        "}\n"
        "Rules: 4–10 items per list; <=3 words; lowercase; dedup; compact.\n"
        f"Ticker: {symbol} · sector={sector} · industry={industry}\n"
        "Examples:\n"
    )
    lines: List[str] = []
    for ex in samples[:8]:
        title = _clip(ex.get("title") or "", 160)
        bullet = _clip(ex.get("bullet") or "", 200)
        snip = _clip(ex.get("snippet") or "", 200)
        use = bullet or snip
        lines.append(f"- {title} || {use}")
    return head + ("\n".join(lines) if lines else "- (no examples)") + "\nJSON only."

def _build_lexicon(symbol: str, meta: Dict[str, Any], samples: List[Dict[str, Any]], cfg: LLMCfg) -> dict:
    prompt = _compose_prompt(symbol, meta, samples)
    phash = _sha1(prompt)
    raw = _PROMPT_MEMO.get(phash)
    if not raw:
        raw = (_openai_run(cfg.model, prompt, cfg.timeout) if cfg.provider == "openai" else _ollama_run(cfg.model, prompt, cfg.timeout))
        _PROMPT_MEMO[phash] = raw
    parsed = _parse_first_json(raw) or {}

    def _norm_list(key: str) -> List[str]:
        xs = parsed.get(key) or []
        out: List[str] = []
        seen: Set[str] = set()
        for x in xs:
            if not isinstance(x, str): 
                continue
            t = " ".join(x.lower().split())
            if t and len(t) <= 24 and t not in seen:
                seen.add(t); out.append(t)
        return out[:12]

    return {
        "symbol": symbol,
        "sector": meta.get("sector") or "UNKNOWN",
        "industry": meta.get("industry"),
        "must_hooks": _norm_list("must_hooks"),
        "finance_core": _norm_list("finance_core"),
        "sector_terms": _norm_list("sector_terms"),
        "macro_terms": _norm_list("macro_terms"),
        "regulatory_terms": _norm_list("regulatory_terms"),
        "exclusions": _norm_list("exclusions"),
        "why": (parsed.get("why") or "ok")[:120],
        "prompt_hash": phash,
    }

# ----------------- main build -----------------
def build_all(db_path: str, tickers_csv: str, provider: str, model: str, prompt_version: str,
              timeout: Optional[float], max_per_symbol: Optional[int], skip_existing: bool) -> Dict[str, Any]:
    uni = TickerUniverse.from_csv(tickers_csv)
    conn = sqlite3.connect(db_path)
    _ensure_table(conn)

    samples_by_sym = _load_all_grouped_by_symbol(conn, uni, max_per_symbol=max_per_symbol)
    llm = LLMCfg(
        provider=("openai" if (provider == "openai" or os.getenv("USE_OPENAI") == "1") else "ollama"),
        model=model,
        timeout=timeout,
        prompt_version=prompt_version,
    )

    created = 0
    skipped = 0
    results = {}

    for sym, samples in sorted(samples_by_sym.items(), key=lambda kv: kv[0]):
        meta = uni.metas.get(sym) or {}
        if not samples:
            skipped += 1
            continue

        if skip_existing:
            row = conn.execute(
                "SELECT 1 FROM ticker_lexicon WHERE symbol=? AND prompt_version=? AND model=?",
                (sym, prompt_version, model),
            ).fetchone()
            if row:
                skipped += 1
                continue

        lex = _build_lexicon(sym, meta, samples, llm)
        payload = {
            "symbol": sym,
            "sector": meta.get("sector") or "UNKNOWN",
            "industry": meta.get("industry"),
            "model": model,
            "provider": llm.provider,
            "prompt_version": prompt_version,
            "lexicon": lex,
            "gate_preview": {
                "include_any": lex["must_hooks"] + lex["finance_core"] + lex["sector_terms"],
                "include_macro": lex["macro_terms"],
                "regulatory": lex["regulatory_terms"],
                "exclude_any": lex["exclusions"],
                "why": lex["why"],
            },
        }

        conn.execute(
            "INSERT OR REPLACE INTO ticker_lexicon (symbol, prompt_version, model, created_at, keywords_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (sym, prompt_version, model, _now_iso(), json.dumps(payload, ensure_ascii=False)),
        )
        conn.commit()
        results[sym] = {"count_samples": len(samples), "sector": payload["sector"], "industry": payload["industry"]}
        created += 1

    conn.close()
    return {
        "as_of": _now_iso(),
        "symbols_seen": len(samples_by_sym),
        "created_or_updated": created,
        "skipped_existing": skipped,
        "model": model,
        "provider": llm.provider,
        "prompt_version": prompt_version,
        "max_per_symbol": max_per_symbol,
        "summary": results,
    }

# ----------------- CLI -----------------
def _cli():
    ap = argparse.ArgumentParser(description="Build per-ticker lexical gates from the ENTIRE DB corpus (no time window).")
    ap.add_argument("--db", default=DB_PATH_DEFAULT, help="Path to wisdom_of_sheep.sql")
    ap.add_argument("--tickers-csv", default=TICKERS_CSV_DEFAULT, help="Path to tickers_enriched.csv")

    # LLM
    ap.add_argument("--provider", choices=["ollama", "openai"], default="ollama")
    ap.add_argument("--model", default="mistral", help="Ollama or OpenAI model name")
    ap.add_argument("--prompt-version", default="lex-v1")
    ap.add_argument("--timeout", type=float, default=15.0)

    # Sampling
    ap.add_argument("--max-per-symbol", type=int, default=0,
                    help="Cap samples per symbol (0 or <0 = unlimited).")

    # Behavior
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip symbols that already have an entry for (prompt_version, model).")

    ap.add_argument("--pretty", action="store_true")
    args = ap.parse_args()

    max_per = None if args.max_per_symbol <= 0 else int(args.max_per_symbol)

    out = build_all(
        db_path=args.db,
        tickers_csv=args.tickers_csv,
        provider=args.provider,
        model=args.model,
        prompt_version=args.prompt_version,
        timeout=args.timeout,
        max_per_symbol=max_per,
        skip_existing=args.skip_existing,
    )
    print(json.dumps(out, indent=2 if args.pretty else None, ensure_ascii=False))

if __name__ == "__main__":
    _cli()
