import copy
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from council.claims_stage import _normalize_claims


def test_normalize_claims_coerces_entity_list_to_string():
    raw = {
        "claims": [
            {
                "id": "c1",
                "text": "Claim text",
                "type": "valuation",
                "entity": ["Glencore PLC", "Teck Resources"],
                "why": "because",
            }
        ],
        "why": "top-level",
    }

    normalized = _normalize_claims(copy.deepcopy(raw))

    assert normalized["claims"][0]["entity"] == "Glencore PLC"


def test_normalize_claims_strips_empty_entity_to_none():
    raw = {
        "claims": [
            {
                "id": "c1",
                "text": "Claim text",
                "type": "valuation",
                "entity": "   ",
                "why": "because",
            }
        ],
        "why": "top-level",
    }

    normalized = _normalize_claims(copy.deepcopy(raw))

    assert normalized["claims"][0]["entity"] is None
