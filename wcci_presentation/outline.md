# Resilient Spatiotemporal Forecasting

## A Digital Twin Paradigm for Imperfect Urban Sensor Networks

### PhD Presentation WCCI Conference, Maastricht

**Note**:
Today, I will be presenting my thesis on resilient spatiotemporal forecasting for urban Internet of Things (IoT) sensor networks. While many of you are deeply familiar with advanced AI architectures, applying these models to the messy, incomplete reality of physical urban environments introduces unique systemic challenges. We will focus heavily on the problem space, the data reality, and the engineering novelty required to bridge the gap between theoretical AI and operational situational awareness.

## Agenda

* The Problem & Data Reality: Sensing paradoxes and missingness
* Intrinsic Resilience: Bypassing traditional imputation
* RIoT Architecture: Joint spatial-temporal learning
* Spatial Estimation: Edge-enriched kriging
* Uncertainty & MLOps: Building operational trust

**Note:**
This structure deliberately fronts the problem and the data. We first examine the physical reality of the Newcastle Urban Observatory dataset. Then, we explore why traditional imputation fails in this context, leading into the design of the Resilient Graph Neural Network (RIoT) pipeline. We will dive into our novel spatial kriging methodologies for missing nodes, and finally, validate the engineering through our MLOps scaling and uncertainty quantification strategies.

## 1. The Urban Sensing Paradox
--
### Abundant Data, Limited Intelligence

* Urban networks function as complex adaptive systems.
* Increasing spatial coverage relies on low-cost, high-variance sensing.
* Result: Intermittent connectivity, gaps, and outright failures.
* The Rapid Deployment Problem: Pre-processing pipelines fail at the edge.

**Note:** Cities function as complex adaptive systems, and state-of-the-art approaches increasingly rely on digital twins for situational awareness. However, we face a paradox: deploying vast networks of low-cost sensors yields high spatial coverage but introduces severe data quality issues like missing records and delays. Traditional models require complete datasets, forcing a reliance on brittle imputation pre-processing. This creates a "Rapid Deployment Problem," where edge networks fail when disconnected from centralised enterprise servers.

--
### The Newcastle Observatory Dataset
* Source: Pedestrian monitoring stations (2018-2026).
* Heterogeneity: Zero sensors maintain 100% completeness.
* Ambiguity: Zero explicit "0" counts in the data.
* Result: Total Null-Zero Conflation.

**Note:** Our empirical lens is the Newcastle Urban Observatory dataset, specifically pedestrian count data from 2018 to 2026 (although only at a usable scale from 2022). The network is extremely heterogeneous; over the full archive period, there are no sensors with 100% completeness. The most critical data reality is "Total Null-Zero Conflation". The dataset contains strictly positive count values. Therefore, a missing 15-minute record could mean a genuine zero-activity interval or a complete operational outage.

## 2. Intrinsic Resilience
--
### The Failure of Imputation
* Traditional pipelines separate imputation and forecasting.
* Distorts true spatio-temporal dynamics during massive outages.
* Imputed zeros are misinterpreted as genuine spatial correlations.
* We must treat missing data as a learnable signal.

**Note:** Traditional imputation pipelines, like k-NN or mean substitution, treat data recovery as an isolated pre-processing step. When networks experience massive non-random data loss, these local statistical assumptions break down completely. If multiple sensors drop offline, external imputation might feed zeros to the model, causing the graph neural network to learn spurious, fabricated relationships between failed nodes. Our framework abandons sequential imputation; it treats missing data as an informative operational state, generating intrinsic resilience through Masked Auto-Encoding.

## 3. The RIoT Pipeline Architecture
--
### Resilient IoT Pipeline (RIoT)
* Dynamic Graph Learner: Infers topology on-the-fly.
* Masked Spatiotemporal Encoder: Combines GATv2Conv and Transformers.
* Triple Decoders: Multi-task spatial and temporal learning.
* Target Constraint: Strict univariate processing per run.

**Note:** The RIoT pipeline abandons the Independent and Identically Distributed (I.I.D.) assumption. It is built on three modules: First, the Dynamic Graph Learner infers topology from static features, yielding an adaptive adjacency matrix. Second, the Masked Spatiotemporal Encoder processes the graph spatially via PyTorch Geometric GATv2Conv layers and temporally via Transformer encoders. Third, Decoupled Decoders execute multi-task learning for imputation, forecasting, and spatial kriging. To prevent gradient interference across vehicle classes, the pipeline remains strictly univariate per run.

--
### Dynamic Topology Inference
* Sensor graphs lack explicit, reliable routing topologies.
* DynamicGraphLearner uses static environmental covariates.
* Applies ReLU and sparsity thresholds to dense adjacency matrices.
* Constraint: Raw spatial coordinates are explicitly discarded.

**Note:** Because urban networks lack explicit routing topologies, the DynamicGraphLearner dynamically infers graph structure using static environmental covariates like point-of-interest density. It uses multi-head GAT or linear embeddings with pairwise similarity to learn a dense adjacency matrix, which is converted to sparse edge indices via a ReLU and a sparsity threshold. Crucially, to prevent trivial spatial leakage, raw spatial coordinates are deliberately discarded prior to model ingestion and are never passed to the model as features.

## 4. Spatial Estimation & Kriging
--
### The Spatial Holdout Problem
* Baseline models learn temporal self-repair at known locations.
* Cannot reconstruct data for entirely unknown, simulated nodes.
* Solution: IGNNK-style node masking during training.
* Validated via a dedicated Spatial Kriging Decoder.

**Note:** Our diagnosis revealed that while the base model learned imputation and forecasting, it failed at spatial kriging—reconstructing data for an unknown location given only its spatial context. If we drop a simulated node into the graph, it produces a bland, mean-regressed forecast. To fix this, we implemented an IGNNK-style spatial holdout strategy. We randomly sample ~15% of nodes per batch, replace their dynamic features with zeros, and train a dedicated kriging decoder to reconstruct their full time series from graph context alone.

--
### Edge Feature Enrichment
* Original GATv2Conv relied on scalar feature similarity weights.
* Now injects [E, edge_dim] physical edge features.
* Features: Shortest path distance, Euclidean directness.
* Features: OSMnx road type composition (pedestrian, primary).

**Note:** Previously, our attention mechanism used a 1-dimensional similarity score, which couldn't differentiate between nodes 50m apart on a physical street versus 5km apart with similar POI profiles. We resolved this via Edge Feature Enrichment. For every edge, we pre-compute network distance along the walking graph, path directness, and road type composition from OSMnx highway tags. This multi-dimensional edge attribute tensor enables the GATv2Conv layer to understand the physical reality of the connections.

--
### Live Spatial Estimation

**Note:** This live Next.js digital twin dashboard visualizes the model's spatial kriging outputs in real time. Powered by Deck.gl over MapLibre, it renders complex spatial interpolations using WebGL layers without DOM overload. Notice how the predictions for entirely simulated nodes adjust dynamically based on the newly integrated OSMnx road network constraints and the attention weights from their nearest topological neighbors.

## 5. Uncertainty & Scaling Verification
--Zero-Inflated Negative Binomial
* Predicts parameters: $\mu$ (mean), $\theta$ (dispersion), $\pi$ (dropout probability).* Separates genuine inactivity from operational outages.
* Target variables strictly remain as raw, unscaled counts.

**Note:** To manage the Total Null-Zero Conflation, we forecast using a Zero-Inflated Negative Binomial (ZINB) loss. This models a probabilistic mixture: the $\pi$ parameter captures the likelihood of technical dropouts, while the Negative Binomial component captures genuine environmental zeros. Crucially, while dynamic and static features are globally normalized via StandardScaler to prevent gradient starvation, the target variables must strictly remain as raw, unscaled counts to preserve the domain required mathematically by the ZINB NLL loss.

--
### Probabilistic Forecasting Boundaries
* Quantifies Aleatoric (environmental) vs Epistemic (model) uncertainty.
* Flags novel events to human operators ("I do not know").
* Enables rapid trust calibration and classification-with-rejection workflows.

**Note:** Deterministic point predictions suffer from mode collapse during anomalies. By outputting probabilistic boundaries via Gaussian Negative Log-Likelihood, the model separates aleatoric uncertainty (inherent noise) from epistemic uncertainty (lack of knowledge). When epistemic uncertainty spikes, the model explicitly flags its own ignorance to the operator. This physical grounding is critical for "AI-Readiness", allowing human operators to confidently implement classification-with-rejection workflows at the tactical edge.

--
Model Predictions & Uncertainty

**Note:** Here we see the forecast dashboard. Notice the ScatterplotLayer plotting both the ZINB point predictions and the mathematical uncertainty bounds. The global Zustand clock decouples rapid time-slider updates from the React render cycle, preventing the entire DOM from re-rendering and executing WebGL data filtering directly on the GPU.

## 6. MLOps & Rigorous Engineering--Strict Environment Isolation
* Zero Host Download Policy: No local virtual environments.
* Dockerized Execution: Segmented CPU, GPU, and Frontend containers.
* Decoupled Parallelism: Static payload bridges isolate UI development.

**Note:** To ensure reproducible, publication-grade results, the entire pipeline is governed by a strict MLOps blueprint. We enforce a "Zero Host Download" policy—all code execution and dependency management occurs strictly inside designated Docker containers. We utilize "Decoupled Parallelism," independently developing the Next.js frontend via a static ui_payload.json generated from a strict visual holdout set to prevent data leakage.

--
## Optuna Ablation & Traceability
* Hyperparameters tuned via SQLite-backed Optuna sweeps.
* Every payload is stamped with "Artifact Twinning" metadata.
* High-resolution analysis scripts extract parameter importance.

**Note:** The architecture was rigorously validated through Optuna hyperparameter sweeps managed via centralized YAML configurations. To guarantee experiment traceability, every generated UI payload is embedded with 'Artifact Twinning' metadata, including the Weights & Biases Run ID and the specific Git Commit Hash. This ensures we can explicitly link visual behavior in the dashboard back to the exact mathematical state of the underlying model.

## 7. Conclusion--Achieving AI-Readiness at the Edge
* Demonstrated intrinsic resilience against massive missing data.
* Bridged spatial gaps via physical edge-enriched kriging.
* Built operational trust through explicit uncertainty quantification.
* Deployed via a highly scalable, isolated MLOps pipeline.

**Note:** In conclusion, this research successfully bridges the gap between theoretical GNNs and the messy reality of urban sensor networks. By abandoning brittle imputation, enriching spatial kriging with physical road typologies, and exposing mathematical uncertainty bounds, we transition AI from a fragile oracle into a resilient, "AI-Ready" decision-support tool. Thank you, and I look forward to your questions.


Slide Target | Figure Concept | Frontend Implementation Strategy
|---------------|---------------|--------------------------------------|
| 1. The Urban Sensing Paradox | Missingness & Null-Zero Conflation Matrix | A D3.js or canvas-based heatmap rendering time on the X-axis and Sensor IDs on the Y-axis. Blocks of sensors turning orange/black simultaneously will visually demonstrate systemic, non-random network outages.|
| 2. The RIoT Pipeline | Dynamic Topology Graph | A dynamic network graph component showing sensors as nodes. Let the user toggle between a baseline Euclidean distance graph and the model's dynamically inferred matrix to prove how the GAT layers reshape the network topology based on environmental covariates. |
| 4. Spatial Estimation | Edge-Enriched Kriging Map | MapLibre map utilizing a light basemap style. Renders your walking path segments using a color gradient driven by the kriging decoder's predictions for simulated, out-of-sample nodes. |
| 5. Uncertainty Boundaries | ZINB Probabilistic Confidence Bands | A high-performance time-series chart component. Plots raw vehicle counts as black points, the ZINB mean prediction as an orange line, and shading for the dispersion ($\theta$) and dropout ($\pi$) probabilities. |