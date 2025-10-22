from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ticker_universe import (
    TICKERS_CSV_DEFAULT,
    TickerMeta,
    TickerUniverse,
    _filter_posts_by_context,
    _load_tickers_enriched,
)


def test_sector_lookup_handles_missing_sector():
    universe = _load_tickers_enriched(TICKERS_CSV_DEFAULT)

    symbol = universe.resolve("hood")
    assert symbol == "HOOD"

    sector = universe.sector_for("hood")
    assert sector == "UNKNOWN"

    keywords = universe.keywords_for(symbol)
    assert "UNKNOWN" in keywords
    assert "EQUITY" in keywords


def test_sector_lookup_returns_known_sector():
    universe = _load_tickers_enriched(TICKERS_CSV_DEFAULT)

    symbol = universe.resolve("AAPL")
    assert symbol == "AAPL"

    sector = universe.sector_for(symbol)
    assert sector
    assert symbol in universe.sector_members(sector)


def test_context_filter_rescues_posts_with_long_name_only():
    universe = _load_tickers_enriched(TICKERS_CSV_DEFAULT)
    posts = [
        {
            "post_id": "p1",
            "title": "Robinhood Markets expands services",
            "text": "A look at Robinhood Markets, Inc. and its new offerings.",
        }
    ]
    kept, dropped = _filter_posts_by_context(posts, {}, universe, "HOOD")
    assert len(kept) == 1
    assert dropped == 0


def test_context_filter_uses_long_name_when_descriptors_missing():
    source = TickerMeta(
        symbol="XYZ",
        long_name="Example Innovations",
        sector=None,
        industry=None,
        asset_type=None,
        market_cap=None,
        aliases=set(),
    )
    other = TickerMeta(
        symbol="AAA",
        long_name="Another Co",
        sector=None,
        industry=None,
        asset_type=None,
        market_cap=None,
        aliases=set(),
    )
    universe = TickerUniverse(metas={"XYZ": source, "AAA": other}, aliases={})
    posts = [
        {
            "post_id": "post-1",
            "title": "Example Innovations unveils roadmap",
            "text": "Analysts discuss Example Innovations and industry peers.",
        }
    ]
    post_tickers = {"post-1": {"AAA"}}

    kept, dropped = _filter_posts_by_context(posts, post_tickers, universe, "XYZ")
    assert len(kept) == 1
    assert dropped == 0
