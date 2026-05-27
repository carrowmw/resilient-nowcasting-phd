# Architect Plan: Removing Places Template and Code Cleanup

This document outlines the high-level, step-by-step architectural design to guide the `@implementer` in removing the `places_template` path dependency and cleaning up the codebase, specifically eliminating defensive programming try-except blocks in favor of the **Offensive Programming (Fail Fast)** paradigm.

---

## 1. High-Level Design Principles

1.  **Enforce the North Star Guidelines (Offensive Programming):**
    *   **Fail Fast / Fail Hard / Let it Burn:** Eliminate defensive `try-except` blocks that catch general exceptions or specific errors to fall back silently. If the environment setup or external resources fail (e.g. missing API keys, connection issues, parse errors), allow the pipeline to crash immediately with a clear traceback.
    *   **Assert Inputs:** Validate all incoming variables (such as checking coordinates are valid numbers and features are integers) using defensive assertions.
2.  **Decoupled Systems:**
    *   Completely eliminate the redundant, untracked `google_places_template.csv`.
    *   Migrate all necessary semantic tags (specifically `nearby_poi_count`) to load directly from the Google Places API cache (`places_cache.json` / `places_cache_test.json`), fully decoupling the system from dynamic CSV structures.

---

## 2. Step-by-Step Architectural Tasks

### Task 1: Refactor `load_semantic_tags` in `src/export_ui_payload.py`
*   **Remove Default Argument:** Drop the `template_path: str = PLACES_TEMPLATE_PATH` argument from the signature of `load_semantic_tags`.
*   **Eliminate Defensive Cache Fallback:**
    *   Remove the `try/except (ValueError, RuntimeError) as e` block around `StaticFeatureLoader` initialization. Allow any initialization error to crash the script immediately.
*   **Eliminate Defensive Simulated Node Live Query Error Swallowing:**
    *   Remove the inner `try/except Exception as ex` block around simulated node live fetching. If the live query fails, the script must fail hard.
*   **Integrate POI Density Tags directly from Cache (Source 1):**
    *   Within the cache loading iteration, retrieve `nearby_poi_count` from `feats` and append the respective `high_poi_density`, `medium_poi_density`, or `low_poi_density` tag.
*   **Completely Drop Source 2 (CSV Template):**
    *   Delete the entire logic block parsing the CSV template.
    *   Discard the unused synthetic `zone_type` tags.
*   **Clean Up Unused Import:**
    *   Delete `import csv` on line 23 since CSV parsing is no longer used in this file.

### Task 2: Refactor Configuration Loading in `src/export_ui_payload.py`
*   **Eliminate Silent Config Fallback:**
    *   Remove the `try/except Exception as e` around co-located configuration yaml reading (`config.yaml`). If the file exists but has syntax or file errors, the script must crash immediately.

### Task 3: Simplify `src/smoke_test.py`
*   **Eliminate Redundant CSV Creation:**
    *   Remove Section 2 (`Populate the Google Places Template`) entirely.
    *   The smoke test no longer needs to create or write the `google_places_template.csv` file since the PreprocessingPipeline and StaticFeatureLoader do not consume it.

---

## 3. Mandated "Desk Checks"

The `@implementer` must perform the following "Desk Checks" before submitting any functional code to verify correctness:

### A. Semantic Tag Extraction Desk Check
Simulate the output of `load_semantic_tags` for the following mock data cases:
1.  **Case 1 (Missing Key):** A cached feature dict `{"commercial_density": 20, "is_school_zone": true}` (missing `highway_proximity` and `nearby_poi_count`).
    *   *Expected behavior:* `feats.get("nearby_poi_count", 0)` should default to `0` and not raise `KeyError`. The tags should evaluate to `["high_commercial_density", "school_zone"]`.
2.  **Case 2 (Boundary Thresholds):**
    *   `commercial_density = 15` -> Expected tag: `"high_commercial_density"`
    *   `commercial_density = 5` -> Expected tag: `"medium_commercial_density"`
    *   `commercial_density = 4` -> Expected tag: `"low_commercial_density"`
    *   `nearby_poi_count = 20` -> Expected tag: `"high_poi_density"`
    *   `nearby_poi_count = 8` -> Expected tag: `"medium_poi_density"`
    *   `nearby_poi_count = 1` -> Expected tag: `"low_poi_density"`
    *   `nearby_poi_count = 0` -> Expected tag: None.

### B. Exception Bubbling Check
Trace execution flow when `GOOGLE_PLACES_API_KEY` is not present in `.env`:
*   *Expected behavior:* `StaticFeatureLoader` initialization throws `ValueError("GOOGLE_PLACES_API_KEY not found in environment.")`. The exception must bubble up and crash `export_ui_payload.py` with a full traceback, instead of catching and falling back to direct JSON read.

---

## 4. Documentation & Verification Mandate

1.  **Verification Run:** The `@implementer` must run `python src/smoke_test.py` and `python src/export_ui_payload.py` inside the container to ensure no runtime syntax errors exist and compilation works correctly.
2.  **Update the Ledger (CRITICAL):**
    The `@implementer` is strictly instructed to append a new entry to the `EXPERIMENTS.md` ledger table upon successful completion of the changes. The entry must record:
    *   **Date:** Current date.
    *   **Branch:** The active git branch.
    *   **Prompt/Goal:** "Remove places template path dependency and clean up code in export_ui_payload.py and smoke_test.py, eliminating defensive programming try-except blocks."
    *   **Mathematical/Functional Outcome:** Summary of cleaned-up code, successful removal of `google_places_template.csv`, and validation via smoke test execution.
