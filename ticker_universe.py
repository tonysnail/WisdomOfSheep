"""Shared ticker universe helpers used across tests and tooling."""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parent
TICKERS_CSV_DEFAULT = ROOT / "tickers" / "tickers_enriched.csv"


@dataclass
class TickerMeta:
    symbol: str
    long_name: Optional[str]
    sector: Optional[str]
    industry: Optional[str]
    asset_type: Optional[str]
    market_cap: Optional[float]
    aliases: Set[str]

    def sector_keywords(self) -> List[str]:
        keywords: List[str] = []
        sector = (self.sector or "UNKNOWN").strip()
        if sector:
            keywords.append(sector)
        for value in (self.industry, self.asset_type):
            if not value:
                continue
            norm = value.strip()
            if not norm:
                continue
            keywords.append(norm)
        return list(dict.fromkeys(keywords))


@dataclass
class TickerUniverse:
    metas: Dict[str, TickerMeta]
    aliases: Dict[str, str]

    def resolve(self, raw: str) -> Optional[str]:
        s = (raw or "").upper().strip()
        if not s:
            return None
        if s in self.metas:
            return s
        return self.aliases.get(s)

    def sector_for(self, raw: str) -> Optional[str]:
        sym = self.resolve(raw)
        if not sym:
            return None
        meta = self.metas.get(sym)
        return meta.sector if meta else None

    def keywords_for(self, raw: str) -> List[str]:
        sym = self.resolve(raw)
        if not sym:
            return []
        meta = self.metas.get(sym)
        if not meta:
            return []
        return meta.sector_keywords()

    def sector_members(self, sector: str) -> Set[str]:
        sector_norm = (sector or "").lower()
        if not sector_norm:
            return set()
        return {
            sym
            for sym, meta in self.metas.items()
            if meta.sector and meta.sector.lower() == sector_norm
        }


def _to_float(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _load_tickers_enriched(csv_path: str | Path) -> TickerUniverse:
    metas: Dict[str, TickerMeta] = {}
    alias_map: Dict[str, str] = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            sym = (row.get("Symbol") or "").upper().strip()
            if not sym:
                continue
            aliases_raw = (row.get("aliases") or "").strip()
            aliases = (
                {a.strip().upper() for a in aliases_raw.split(",") if a.strip()}
                if aliases_raw
                else set()
            )
            meta = TickerMeta(
                symbol=sym,
                long_name=(row.get("longName") or None),
                sector=(row.get("sector") or "UNKNOWN"),
                industry=(row.get("industry") or None),
                asset_type=(row.get("asset_type") or None),
                market_cap=_to_float(row.get("marketCap")),
                aliases=aliases,
            )
            metas[sym] = meta
            for alias in aliases:
                alias_map.setdefault(alias, sym)
    return TickerUniverse(metas=metas, aliases=alias_map)


_LONG_NAME_STOPWORDS = {
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "company",
    "co",
    "plc",
    "ltd",
    "limited",
    "the",
}


def _normalize_for_match(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _long_name_tokens(long_name: Optional[str]) -> Set[str]:
    if not long_name:
        return set()
    norm = _normalize_for_match(long_name)
    if not norm:
        return set()
    return {
        token
        for token in norm.split()
        if len(token) >= 3 and token not in _LONG_NAME_STOPWORDS
    }


def _meta_descriptors(meta: Optional[TickerMeta]) -> Set[str]:
    if not meta:
        return set()
    values: List[str] = []
    for value in (meta.sector, meta.industry, meta.asset_type):
        if not value:
            continue
        norm = value.strip().lower()
        if not norm or norm == "unknown":
            continue
        values.append(norm)
    return set(values)


def _post_tokens_and_norm(post: Dict[str, Any]) -> Tuple[Set[str], str]:
    combined = " ".join(str(post.get(field) or "") for field in ("title", "text"))
    norm = _normalize_for_match(combined) if combined else ""
    if not norm:
        return set(), ""
    return {token for token in norm.split() if token}, norm


def _post_mentions_long_name(
    long_name_tokens: Set[str],
    long_name_norm: str,
    post_tokens: Set[str],
    post_norm: str,
) -> bool:
    if not long_name_tokens:
        return False
    if long_name_norm and long_name_norm in post_norm:
        return True
    return bool(long_name_tokens.intersection(post_tokens))


def _filter_posts_by_context(
    posts: List[Dict[str, Any]],
    post_tickers: Dict[str, Set[str]],
    universe: TickerUniverse,
    source_symbol: str,
) -> Tuple[List[Dict[str, Any]], int]:
    meta = universe.metas.get(source_symbol)
    sector_norm = (meta.sector or "").strip().lower() if meta and meta.sector else ""
    industry_norm = (meta.industry or "").strip().lower() if meta and meta.industry else ""
    asset_type_norm = (meta.asset_type or "").strip().lower() if meta and meta.asset_type else ""
    source_descriptors = _meta_descriptors(meta)
    long_name_tokens = _long_name_tokens(meta.long_name if meta else None)
    long_name_norm = _normalize_for_match(meta.long_name) if meta and meta.long_name else ""

    resolver = universe.resolve

    kept: List[Dict[str, Any]] = []
    dropped = 0

    for post in posts:
        tickers = post_tickers.get(post.get("post_id"), set())
        post_tokens, post_norm = _post_tokens_and_norm(post)
        has_long_name = _post_mentions_long_name(
            long_name_tokens, long_name_norm, post_tokens, post_norm
        )

        if not tickers:
            if has_long_name:
                kept.append(post)
            else:
                dropped += 1
            continue

        include = False
        descriptor_overlap = False
        for raw in tickers:
            sym = resolver(raw)
            if not sym:
                continue
            if sym == source_symbol:
                include = True
                break
            meta_candidate = universe.metas.get(sym)
            if not meta_candidate:
                continue
            cand_sector = (meta_candidate.sector or "").strip().lower()
            cand_industry = (meta_candidate.industry or "").strip().lower()
            cand_asset_type = (meta_candidate.asset_type or "").strip().lower()
            if sector_norm and cand_sector and cand_sector == sector_norm:
                include = True
                descriptor_overlap = True
                break
            if industry_norm and cand_industry and cand_industry == industry_norm:
                include = True
                descriptor_overlap = True
                break
            if asset_type_norm and cand_asset_type and cand_asset_type == asset_type_norm:
                include = True
                descriptor_overlap = True
                break
            if source_descriptors:
                cand_descriptors = _meta_descriptors(meta_candidate)
                if cand_descriptors.intersection(source_descriptors):
                    descriptor_overlap = True

        if include:
            kept.append(post)
        else:
            if has_long_name and (descriptor_overlap or not source_descriptors):
                kept.append(post)
            else:
                dropped += 1

    return kept, dropped


__all__ = [
    "TickerMeta",
    "TickerUniverse",
    "TICKERS_CSV_DEFAULT",
    "_filter_posts_by_context",
    "_load_tickers_enriched",
]
