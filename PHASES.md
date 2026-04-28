# Project Roadmap: Decoupled Parallelism

This document outlines the macro-phases of the Resilient IoT Spatiotemporal Pipeline. We strictly adhere to a "Decoupled Parallelism" strategy: the frontend and backend are developed independently via a static data bridge until the mathematical architecture is completely stable.

## Phase 1: The Static Frontend Prototype & UX Validation

- **Focus:** Visualizing complex ML outputs efficiently in the browser.
- Generate a comprehensive, static `ui_payload.json` from the current "Iteration 0" PyTorch model.
- **Strict Constraint:** The payload MUST be generated exclusively from a quarantined **Visual Holdout Set** (e.g., the final 3 months of data) to prevent data leakage. Every payload must be stamped with 'Artifact Twinning' metadata (W&B Run ID, Git Hash, YAML config path) for traceability.
- Build the Next.js and Deck.gl frontend to ingest this static file.
- Validate the Zustand global clock, WebGL data filtering, and React rendering performance (60fps target).

_Status: In Progress._

## Phase 2: Backend Ablation Studies & Architecture Stabilization

- **Focus:** Freezing the ML architecture through rigorous scientific testing.
- Once the Phase 1 UI is rendering correctly, frontend development halts.
- Pivot back to the GPU server and utilize the Optuna/W&B framework to run ablation studies.
- **Strict Constraint:** All ablation studies must be executed purely by swapping centralized YAML configurations (`configs/train.yaml`).
  - _Study 1 (Topology):_ Disable the `DynamicGraphLearner`. Does the imputation MSE spike?
  - _Study 2 (Attention):_ Swap `GATv2Conv` for a standard GCN. Do we lose the temporal explainability?
  - _Study 3 (Target Expansion):_ Inject cross-modal features (e.g., weather) directly into the decoders.
- **Output:** A mathematically sound, final `best_pipeline.pt` with locked tensor shapes.

## Phase 3: Live Inference API & Spatial Integration

- **Focus:** Productionizing the frozen pipeline.
- Wrap the stable PyTorch pipeline in a highly concurrent FastAPI application.
- **Integrational Bottlenecks:** - Implement a Redis in-memory **Rolling Window Cache** to maintain the 12-timestep tensor requirements for instant inference.
  - Implement a strict **UUID Hashing Dictionary** to safely map shifting PyTorch integer indices to static physical sensor coordinates.
- Spin up Supabase (PostgreSQL + PostGIS) and ingest OpenStreetMap road networks.
- Connect the Next.js frontend to the live API, dropping the static `ui_payload.json` bridge.
- Implement PostGIS spatial interpolations to map point predictions onto continuous road segments.
