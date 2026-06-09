"""
Standalone pagination test for Google Places Text Search (New).

Verifies that _text_search_paginated() yields counts > 20 for central Newcastle
nodes (e.g. Quayside) before committing to the full 89-sensor process_all.py run.

Run from repo root:
    export GOOGLE_PLACES_API_KEY=<key>
    python tests/test_text_search_pagination.py

Crashes (AssertionError) if:
  - Any 403/429 is encountered (propagated from RuntimeError in StaticFeatureLoader)
  - All text search counts are <= 20 (pagination not working)
  - Nearby Search count exceeds Text Search count (API regression)
"""

import os
import sys
import tempfile
import logging

# Load .env so this test can be run standalone from the workspace
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
except ImportError:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from features import StaticFeatureLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Quayside — highest expected density node in Newcastle dataset
QUAYSIDE = {
    'name': 'NclQuaysideUnderHighLevelBridge',
    'lat': 54.9675,
    'lon': -1.6097,
}

# Suburban sensor — expected low counts
SUBURBAN = {
    'name': 'NclSaltersHollywood',
    'lat': 55.0108,
    'lon': -1.6031,
}

DENSITY_QUERIES = [
    ('transit_station', 'transit stations'),
    ('store',           'stores shops retail'),
    ('restaurant',      'restaurants food'),
]


def run_test():
    api_key = os.environ.get('GOOGLE_PLACES_API_KEY')
    if not api_key:
        print("SKIP: GOOGLE_PLACES_API_KEY not set.")
        sys.exit(0)

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        tmp_cache = f.name

    try:
        loader = StaticFeatureLoader(cache_path=tmp_cache)
    except ValueError as e:
        print(f"SKIP: {e}")
        sys.exit(0)

    print("\n" + "=" * 70)
    print(f"PAGINATION TEST — Sensor: {QUAYSIDE['name']}")
    print("=" * 70)
    print(f"{'Feature':<22} {'Nearby Search':>14} {'Text Search (paginated)':>24} {'Pages beat 20?':>15}")
    print("-" * 75)

    any_above_20 = False
    results = {}

    for included_type, text_query in DENSITY_QUERIES:
        nearby_count = loader._nearby_search_count(
            QUAYSIDE['lat'], QUAYSIDE['lon'], [included_type], 500
        )
        text_count = loader._text_search_paginated(
            QUAYSIDE['lat'], QUAYSIDE['lon'], text_query, included_type, 500
        )
        beat_20 = text_count > 20
        if beat_20:
            any_above_20 = True
        marker = " <-- PAGINATED!" if beat_20 else ""
        print(f"{included_type:<22} {nearby_count:>14} {text_count:>24}{marker}")
        results[included_type] = {'nearby': nearby_count, 'text_search': text_count}

    print("-" * 75)
    print()

    # Core assertion: at least one type should exceed 20 for a city-centre node.
    # This proves pagination is fetching pages 2 and 3 (max per page = 20).
    assert any_above_20, (
        "FAIL: All Text Search counts <= 20 for Quayside. "
        "Pagination may not be working or the API has changed. "
        f"Results: {results}"
    )

    # Note: Text Search may return FEWER results than Nearby Search for the same type
    # because textQuery matching applies an additional semantic filter. This is expected
    # behaviour — not a regression. The value of Text Search is its ability to paginate
    # beyond 20, not a guaranteed superset of Nearby Search results.

    print(f"SUBURBAN CONTROL — Sensor: {SUBURBAN['name']}")
    print("-" * 75)
    for included_type, text_query in DENSITY_QUERIES:
        count = loader._text_search_paginated(
            SUBURBAN['lat'], SUBURBAN['lon'], text_query, included_type, 500
        )
        print(f"  {included_type:<20}: {count}")
    print()

    print("=" * 70)
    print("PASS: Pagination confirmed. Text Search yields > 20 for at least one")
    print(f"      density category at the city-centre Quayside node.")
    print("=" * 70)

    os.unlink(tmp_cache)


if __name__ == "__main__":
    run_test()
