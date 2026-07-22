# Architecture Evolution Changelog

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

