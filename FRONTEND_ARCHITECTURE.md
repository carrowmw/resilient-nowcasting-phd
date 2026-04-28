# FRONTEND_ARCHITECTURE.md

## Objective

Implement a Geospatial Digital Twin for the Resilient IoT network. This frontend must visualize complex spatiotemporal graph inferences, including Zero-Inflated Negative Binomial (ZINB) predictions, mathematical uncertainty bounds, dynamic graph attention weights, and deep explainability metrics (e.g., self-healing imputation logs). It must be highly performant, capable of rendering thousands of animating nodes and edges at 60fps across a sliding time window.

## 1. Core Tech Stack

- **Framework:** Next.js (React) with TypeScript (Strict mode enabled).
- **Geospatial Engine:** **Deck.gl** (by Uber) running over a **MapLibre GL JS** base map. Standard mapping libraries (Leaflet, Mapbox standard) are banned for data layers to prevent DOM overload. Deck.gl WebGL layers must be used for all data rendering.
- **State Management:** **Zustand**. Required to decouple the rapid time-slider updates from the React render cycle, preventing the entire DOM from re-rendering when only WebGL properties need to change.
- **Database & Spatial Backend (Future Phase):** Supabase (PostgreSQL + PostGIS) for road-network mapping and spatial interpolations.
- **Live Inference API (Future Phase):** FastAPI (Python) wrapping the frozen PyTorch model.

## 2. Architectural Components (The Visual Layers)

The UI is built on independent, composable Deck.gl layers and HTML overlays that consume the backend payload.

**CRITICAL - Defensive Rendering:** The frontend must be strictly defensive. If an ablation study on the backend removes a component (e.g., removing the GAT removes attention weights), the frontend must gracefully degrade by hiding the affected layer, rather than crashing on undefined properties.

### A. The Node Layer (Predictions & Uncertainty)

- **Component**: Deck.gl ScatterplotLayer.
- **Data Source**: Sensor locations (real) and Simulated Nodes.
- **Visual Mapping**:
  - **Fill Color (Uncertainty)**: Mapped to the 95% Confidence Interval derived from the ZINB $\theta$ (dispersion) parameter.
  - **Stroke/Halo Radius (Prediction)**: Mapped to the ZINB $\mu$ parameter (mean predicted traffic). A wider radius indicates higher predicted values. Uses a logarithmic scale to accommodate the wide range of traffic volumes.
  - **Simulated Nodes**: Visually distinct (e.g., using a dashed stroke or distinct shape) to indicate the prediction is a zero-shot topological inference, not a physical sensor.

### B. The Topology Layer (Attention Weights)

- **Component**: Deck.gl `ArcLayer` or `LineLayer`.
- **Data Source**: The attention_sequence extracted from the GATv2Conv layers.
- **Visual Mapping:** Opacity/Width mapped to the attention weight ($\alpha$). Must be aggressively filtered to top _N_ edges to prevent visual clutter.

### C. The Road-Constrained Heatmap (Spatial Interpolation)

- **Component:** Deck.gl `PathLayer` rendering GeoJSON LineStrings.
- **Mechanism:** PostGIS functions (`ST_ClosestPoint`, `ST_Distance`) interpolate the ZINB $\mu$ values between adjacent sensors to color-code continuous road segments representing traffic flow.

## 3. State Management & Time Travel

- **The Global Clock:** A single integer state (`currentTimestep`) managed by `useTimeStore`.
- **Interpolation Logic:** Deck.gl layers must linearly interpolate positions, colors, and radii between `timestep[t]` and `timestep[t+1]` for smooth animation.
- **Data Slicing:** The React components do not filter the massive data payload. Deck.gl layers receive the full data arrays and use **DataFilterExtension** (a WebGL shader extension) to efficiently hide/show data on the GPU based on `currentTimestep`.

## 4. Phased Implementation Strategy

### Phase 1: The Static Prototype (Memory-Bound)

### Phase 1: The Static Prototype (Memory-Bound)

- **Goal:** Validate the visual paradigm without setting up databases or APIs.
- **Mechanism:** The Next.js app loads `ui_payload.json` directly into browser memory.
- **Deliverables:** A fully functioning, interactive Deck.gl frontend that operates completely independently of the active Python training environment.

### Phase 2: PostGIS & Live API Integration (Post-Ablation)

- **Goal:** Live inference and Road-Constrained Heatmaps.
- **Mechanism:** ONLY executed after the PyTorch architecture is mathematically frozen via rigorous ablation studies.
- **Integrational Bottlenecks to Address:**
  - **The Rolling Window Cache:** The FastAPI server cannot scan raw Parquet files for live sequences. An in-memory cache (e.g., Redis) must be used to buffer exactly 12 rolling timesteps for instant tensor generation.
  - **Tensor-to-Geospatial Alignment:** The API must maintain a strict, version-controlled hashing dictionary mapping PyTorch integer indices to physical sensor UUIDs/Coordinates to prevent visualization shifting if sensors go offline.
