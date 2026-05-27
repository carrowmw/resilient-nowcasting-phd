# Context for Architect: Removing Places Template and Code Cleanup

This document contains the gathered codebase context, relevant file paths, high-level logic summaries, and ASTs to guide the `@architect` in formulating a robust plan for removing the `places_template` path dependency and eliminating defensive programming antipatterns in [export_ui_payload.py](file:///Users/administrator/Code/ml-agent-workspace/src/export_ui_payload.py).

---

## 1. Identified References & Antipatterns

### A. places_template References
*   **File:** [export_ui_payload.py](file:///Users/administrator/Code/ml-agent-workspace/src/export_ui_payload.py)
    *   **Line 23:** `import csv`
    *   **Line 66:** `# PLACES_TEMPLATE_PATH = "data/external/google_places_template.csv"`
    *   **Lines 223–227:** `load_semantic_tags` function signature utilizes `PLACES_TEMPLATE_PATH`.
    *   **Lines 283–305:** Code block parsing the CSV template to assign `zone_type` and `nearby_poi_count` tags (Source 2).
    *   **Line 1025:** Comment referencing `# 8. Load semantic tags from Places cache + template CSV`.
*   **File:** [smoke_test.py](file:///Users/administrator/Code/ml-agent-workspace/src/smoke_test.py)
    *   **Lines 38–46:** Mock metadata generation and file export block writing `google_places_template.csv`.

### B. Defensive try-except Antipatterns in `export_ui_payload.py`

> [!CRITICAL]
> **Defensive Coding Violations (Offensive Programming Paradigm):**
> Multiple try-except blocks in `export_ui_payload.py` catch general exceptions or specific errors to fall back silently or log a warning rather than letting the script crash. Under the project's **Fail Fast / Fail Hard** mandate, these must be eliminated.

1.  **Silent Fallback to Cache Manual Read (Lines 255-261):**
    ```python
    except (ValueError, RuntimeError) as e:
        logger.warning(f"StaticFeatureLoader unavailable: {e}. Falling back to manual cache read.")
        cache = {}
        if Path(places_cache_path).exists():
            with open(places_cache_path) as f:
                cache = json.load(f)
    ```
    *   **Why it's an antipattern:** If `StaticFeatureLoader` fails to initialize (e.g. because `GOOGLE_PLACES_API_KEY` is missing or invalid), catching this exception and falling back to a direct file read hides critical configuration issues in development and production environments.
    *   **Correction:** Remove the entire `try/except` block. Allow `StaticFeatureLoader` initialization to raise exceptions directly, causing the script to crash immediately.

2.  **Silent Ignored Exception in Simulated Node Live Fetch (Lines 247-252):**
    ```python
    try:
        # Note: Using private method as requested in Task 3.2
        loader.cache[loc] = loader._fetch_location_data(meta["lat"], meta["lon"])
        loader._save_cache()
    except Exception as ex:
        logger.error(f"Failed to fetch data for simulated node {loc}: {ex}")
    ```
    *   **Why it's an antipattern:** If fetching live data from Google Places API fails during generation (due to network drops, quota, or authentication failures), it logs an error but silently continues with an incomplete cache/metadata.
    *   **Correction:** Remove the `try/except` block. If the live query fails, the script must fail hard so that engineers are immediately alerted of API/network failures.

3.  **Silent Bare Exception in W&B Metadata Retrieval (Lines 141-145):**
    ```python
    try:
        if artifact.producer:
            wandb_run_id = artifact.producer.id
    except Exception:
        pass
    ```
    *   **Why it's an antipattern:** Silently suppresses all exceptions with a bare `except Exception: pass`.
    *   **Correction:** While obtaining optional metadata, if it fails, it should either raise or be handled explicitly, rather than using bare swallow patterns.

4.  **Silent Config Parsing Fallback (Lines 749-750):**
    ```python
    except Exception as e:
        logger.error(f"Failed to parse co-located configuration: {e}. Falling back to default architecture...")
    ```
    *   **Why it's an antipattern:** Silently absorbs configuration parsing errors and substitutes defaults, increasing the risk of subtle shape mismatches during downstream loading.
    *   **Correction:** Crash immediately on failing to parse configuration files.

---

## 2. Decoupling & Cleanup Strategy

### A. Semantic Tag Extraction Migration
*   **Remove CSV template dependency entirely:** `google_places_template.csv` is redundant.
*   **Migrate POI Density Tags:** Transfer POI density tag assignment (`high_poi_density`, `medium_poi_density`, `low_poi_density`) to use the cached value (`nearby_poi_count` from `places_cache.json`) inside the cache loop.
*   **Discard `zone_type` Tags:** Since `zone_type` is a mock field only present in the template CSV, discontinue these tags.

### B. Code Execution Without Try-Except Block
By removing the `try/except` block in `load_semantic_tags`, the code becomes incredibly clean and strictly adheres to the offensive programming guidelines:

```python
def load_semantic_tags(
    nodes_meta: List[Dict],
    places_cache_path: str = PLACES_CACHE_PATH,
) -> Dict[str, List[str]]:
    """
    Build a {location_name: [tag, ...]} mapping from the Places cache.

    Source:
      1. StaticFeatureLoader (Places API) — live API features for simulated
         nodes + cached features for real nodes.
    """
    tags: Dict[str, List[str]] = {m["location_name"]: [] for m in nodes_meta}

    # Initialize loader; will raise ValueError/RuntimeError and crash on auth/quota/missing key issues
    loader = StaticFeatureLoader(cache_path=places_cache_path)

    # Live Query for simulated nodes missing from cache
    for meta in nodes_meta:
        loc = meta["location_name"]
        if meta["is_simulated"] and loc not in loader.cache:
            logger.info(f"Fetching live data for simulated node: {loc}")
            # Note: Using private method as requested in Task 3.2
            loader.cache[loc] = loader._fetch_location_data(meta["lat"], meta["lon"])
            loader._save_cache()

    cache = loader.cache
    
    # Process features and append tags from Cache
    for loc, feats in cache.items():
        if loc not in tags:
            continue
        t = tags[loc]
        
        # Commercial density
        cd = int(feats.get("commercial_density", 0))
        if cd >= 15:
            t.append("high_commercial_density")
        elif cd >= 5:
            t.append("medium_commercial_density")
        elif cd > 0:
            t.append("low_commercial_density")
            
        # School zone
        if feats.get("is_school_zone"):
            t.append("school_zone")
            
        # Transit density
        hp = int(feats.get("highway_proximity", 0))
        if hp >= 15:
            t.append("high_transit_density")
        elif hp >= 5:
            t.append("medium_transit_density")
        elif hp > 0:
            t.append("low_transit_density")

        # POI density (Transferred from template CSV to live cache)
        poi = int(feats.get("nearby_poi_count", 0))
        if poi >= 20:
            t.append("high_poi_density")
        elif poi >= 8:
            t.append("medium_poi_density")
        elif poi >= 1:
            t.append("low_poi_density")

    return tags
```

---

## 3. AST / File Changes Checklist

### [MODIFY] [export_ui_payload.py](file:///Users/administrator/Code/ml-agent-workspace/src/export_ui_payload.py)
*   [ ] Remove `import csv` on line 23.
*   [ ] Remove commented `PLACES_TEMPLATE_PATH` constant.
*   [ ] Remove `template_path: str = PLACES_TEMPLATE_PATH` from `load_semantic_tags` signature.
*   [ ] Remove try-except blocks from `load_semantic_tags` and allow exceptions to bubble up naturally.
*   [ ] Migrate `nearby_poi_count` logic into the cache iteration block.
*   [ ] Remove Source 2 parsing block.
*   [ ] Remove try-except block when resolving `loaded_config_data` (line 749), letting it fail hard on syntax/file errors.

### [MODIFY] [smoke_test.py](file:///Users/administrator/Code/ml-agent-workspace/src/smoke_test.py)
*   [ ] Remove Section 2 (`Populate the Google Places Template`) completely.
