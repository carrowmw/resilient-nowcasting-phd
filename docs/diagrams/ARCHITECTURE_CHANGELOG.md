# Architecture Evolution Changelog

## Update: 2026-08-14 11:08

- **Edge Feature Enrichment Pipeline**: Added OSM walking network cache (`edge_features_cache.json`) loading and 10-dimensional `edge_attr` construction (`edge_feature_table` with 9 physical features ⊕ learned similarity).
- **Physical Road & Distance Features**: Mapped normalized shortest path distance, euclidean distance, directness ratio, and 6 road category fractions into the dynamic graph learner.
- **Enriched Spatial GATv2Conv**: Updated `MaskedSpatiotemporalEncoder` to show `GATv2Conv` with `edge_dim=10` and learnable $\Theta_e$ projection.
- **Structural Missingness (MNAR)**: Added explicit structural missingness extraction (Total MNAR + Night MNAR masks) concatenated with dynamic time series to form 5-channel encoder inputs.
- **Scale-Invariant MSLE Loss**: Diagrammed separate scaling for traffic (channel 0) vs weather (channels 1-2) with inverse-transformation to raw count domain for scale-invariant MSLE loss via registered `traffic_mean` and `traffic_std` buffers.
- **Spatial Holdout & Kriging Head**: Added `make_spatial_holdout_mask` and `apply_spatial_holdout` alongside the dedicated `SpatialKrigingDecoder` (2-layer MLP) and three-term joint training objective (`w_imp * L_imp + w_fore * L_fore + w_krig * L_krig`).
- **Inductive Zero-Shot Gating**: Included `loss_node_mask` ensuring zero-shot evaluation without target leakage.
- **Strict Topology Masking Alignment**: Diagrammed row-aligned filtering of `edge_index`, `edge_weight`, and `edge_attr` to eliminate outgoing edges from simulated and held-out nodes.
- **MLOps & Evaluation Harness**: Updated training block to include early GPU selection (`gpu_selector.py`), evaluation harness (`harness/evaluate.py`), holdout manifests, and FP32 autocast loss execution.

---

## Update: 2026-06-05 16:43

- **Refactored Model Detail**: Replaced the high-level `pipeline.py` block with an explicit component-level breakdown of the `ResilientIoTPipeline`.
- **Dynamic Graph Learner**: Mapped the linear projection to similarity adjacency, sparsity thresholding, and the generation of `edge_index` and `edge_weight`.
- **Spatiotemporal Encoder**: Diagrammed the transition from `GATv2Conv` layers to flattened representations and finally the `TransformerEncoder`.
- **Decoupled Decoders**: Split the latent embedding flow into the `Imputation Decoder MLP` (MSE Loss) and `Forecasting Decoder MLP` (ZINB NLL Loss).
- **Optimizer Mapping**: Updated the training orchestrator to explicitly show the `optimizer.step` pushing gradients to the detailed sub-modules.

---

## Update: 2026-06-05 16:36

- **Refactored Data Pipeline Diagram**: Replaced the code-centric file layout with a logical mapping of Data Ingestion -> Feature Engineering -> Preprocessing -> Model Integration, precisely aligning with the draw.io architecture reference.
- **Added Preprocessing Granularity**: Clarified the separation of Static Scaling vs. Artificial Masking + Dynamic Scaling.
- **Captured MLOps Constraints**: Added explicit diagram nodes showing that `Target Variables` bypass scaling logic, and highlighted the `Coordinates excluded to prevent spatial leakage` directive.

---

## Update: 2026-06-05 16:26

- **Added Data Pipeline components**: `traffic_sensors.json`, `Google Places API`, `Dynamic Time-Series Readings`, `dataset.py: RealIoTDataset`, and `preprocess.py: StandardScaler & Masking`.
- **Added Model components**: `pipeline.py: ResilientIoTPipeline` orchestrator, `graph_learner.py: DynamicGraphLearner` topology inference, `encoder.py: MaskedSpatiotemporalEncoder` (GATv2 + Transformer), and `decoders.py: DecoupledDecoders` (Imputation + ZINB Forecasting).
- **Added Loss & Optimization components**: MSE Imputation Loss, ZINB NLL Loss, Joint Loss, `train.py: Orchestrator` with W&B logging, and `tune.py` Optuna sweeping.
- **Added Static Data Bridge**: Quarantined `Visual Holdout Set`, `export_ui_payload.py` generating `master_index.json` and sharded `nodes/node_*.json` details, and metadata stamp for `Artifact Twinning`.
- **Added Frontend Geospatial Twin components**: Next.js App, Zustand state store (`useTimeStore`), Deck.gl and MapLibre layers (`nodes-real`, `nodes-simulated`, `topology-layer`, `ego-graph-layer`), and React views (`MapVisualizer`, `NodeInspector`, `AttentionControls`, `TimeSlider`).

---

