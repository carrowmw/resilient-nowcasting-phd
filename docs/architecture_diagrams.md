# Resilient IoT Architecture Diagrams

These Mermaid diagrams represent the 4 architectural figures you can use as a baseline for your `draw.io` methodology figures.

## Figure 1: Data Pipeline
```mermaid
graph TD
    subgraph Data Sources
        SENSORS[traffic_sensors.json: Lat/Lon Coords]
        API[Google Places API New]
        C[Dynamic Time-Series Data: Sensor Readings]
        G[Target Variables: Raw Vehicle Counts]
    end

    subgraph StaticFeatureLoader
        SENSORS -- "lat, lon, radius (250m-500m)" --> API
        API -- "includedTypes (e.g. store, school)" --> STATS[Static Features: POI, Commercial Density, etc.]
        note[Note: Coords are NOT passed to model] -.-> STATS
    end

    subgraph Preprocessing
        B(StandardScaler Normalization)
        D(Artificial Masking)
        E(StandardScaler Normalization)
    end

    STATS --> B
    C --> D
    D --> E
    
    B --> F{PyTorch Dataset Loader}
    E --> F
    G --> F
    
    F --> H[Batch Dictionary]
    H --> I((ResilientIoTPipeline Model))
```

## Figure 2: Model Architecture
```mermaid
graph TD
    subgraph Inputs
        SF[Static Features: API Location Metadata]
        MTS[Masked Time-Series Data]
    end

    subgraph 1. Dynamic Graph Learner
        SF --> GAT[Multi-head GAT / Pairwise Similarity]
        GAT --> ADJ[Dynamic Adjacency Matrix]
        ADJ --> THRES[ReLU & Sparsity Threshold]
        THRES --> EI[edge_index & edge_weight]
    end

    subgraph 2. Masked Spatiotemporal Encoder
        MTS --> ST_ENC[Spatial: GATv2Conv Layers]
        EI --> ST_ENC
        ST_ENC --> FLAT[Flatten Spatial Output]
        FLAT --> T_ENC[Temporal: TransformerEncoder]
        T_ENC --> LS[Shared Latent Space Embedding]
    end

    subgraph 3. Decoupled Decoders
        LS --> IMP_DEC[Imputation Decoder: MLP]
        IMP_DEC --> REC[Reconstructed Masked Data]
        
        LS --> FOR_DEC[Forecasting Decoder: MLP]
        FOR_DEC --> ZINB[ZINB Parameters: μ, θ, π]
    end
    
    subgraph Loss Functions
        REC --> L1[MSE Imputation Loss]
        ZINB --> L2[ZINB NLL Loss]
        L1 -.-> JOINT_LOSS((Joint Training Objective))
        L2 -.-> JOINT_LOSS
    end
```

## Figure 3: MLOps Architecture
```mermaid
graph TD
    subgraph Centralized Configuration
        TRAIN_YML[configs/train.yaml]
        SWEEP_YML[configs/sweep.yaml]
    end

    subgraph Dockerized Execution Environments
        AGENT[Agent Workspace: Development]
        CPU[Mac Host: CPU Debugging]
        GPU[Remote Server: GPU Training / Sweeps]
        
        AGENT -.-> CPU
        AGENT -.-> GPU
    end

    TRAIN_YML --> TRAIN_PY[src/train.py: Orchestrator]
    SWEEP_YML --> TUNE_PY[src/tune.py: Optuna Sweeps]
    TUNE_PY --> TRAIN_PY

    TRAIN_PY --> WANDB[(Weights & Biases)]
    WANDB --> OPTUNA[(Optuna SQLite DB)]

    TRAIN_PY --> EXP_PY[export_ui_payload.py]
    EXP_PY --> UI_PAYLOAD[ui_payload.json with Traceability Metadata]
    EXP_PY -.-> VISUAL_HOLDOUT[Strict Visual Holdout Set]
```

## Figure 4: Frontend & Database Architecture
```mermaid
graph TD
    subgraph Browser Client
        NEXTJS[Next.js + React]
        ZUSTAND[Zustand State Management: useTimeStore]
        DECKGL[Deck.gl over MapLibre GL JS]
        
        NEXTJS --> ZUSTAND
        ZUSTAND --> DECKGL
        
        subgraph Deck.gl Visual Layers
            NODE[ScatterplotLayer: Predictions / Uncertainty]
            EDGE[ArcLayer/LineLayer: Attention Topology]
            HEAT[PathLayer: PostGIS Road Heatmaps]
        end
        DECKGL --> NODE
        DECKGL --> EDGE
        DECKGL --> HEAT
    end

    subgraph Backend Infrastructure
        UI_PAYLOAD[Phase 1: Static ui_payload.json]
        
        subgraph Phase 2: Live Integration
            FASTAPI[FastAPI Model Wrapper]
            CACHE[(Redis: 12-Timestep Rolling Cache)]
            DB[(Supabase: PostgreSQL + PostGIS)]
            
            FASTAPI --> CACHE
            FASTAPI --> DB
        end
    end

    UI_PAYLOAD --> NEXTJS
    FASTAPI -.-> NEXTJS
```

---

# Experimental Branch Architecture Diagrams

The following diagrams document the architectural changes on the three `exp/` branches.
Each builds incrementally on the previous (see [Branch Lineage](#branch-lineage)).

## Branch Lineage

```mermaid
gitGraph
    commit id: "main"
    branch exp/scale-invariant-loss
    commit id: "Phase 3: dormant MSLE + kriging head" type: HIGHLIGHT
    commit id: "Phase 3: activate MSLE on traffic channel"
    checkout main
    branch exp/spatial-kriging-head
    commit id: "Phase 4: wire spatial-holdout kriging" type: HIGHLIGHT
    branch exp/enriched-gat-kriging
    commit id: "Phase 5: wire enriched GATv2 edge_attr" type: HIGHLIGHT
    commit id: "Phase 5: normalise distances + fail-fast"
```

> **Note:** `exp/scale-invariant-loss` branches from a different common ancestor
> than the kriging branches. `exp/spatial-kriging-head` and
> `exp/enriched-gat-kriging` share a linear lineage — the latter contains all
> commits from the former plus the Phase 5 edge enrichment.

---

## Figure 5: `exp/scale-invariant-loss` — Channel-Split MSLE Imputation

**Branch:** `exp/scale-invariant-loss` (1 commit over main)
**Files:** `decoders.py`, `pipeline.py`, `train.py`, `train.yaml`

### What changed vs. main

| Area | Main (baseline) | exp/scale-invariant-loss |
|------|----------------|--------------------------|
| Imputation loss | Global MSE over all channels | Channel-split: **MSLE** on traffic (channel 0), MSE on weather channels |
| Domain for traffic loss | StandardScaler-normalised | Inverse-transformed to **raw count domain** via `x * std + mean` |
| Scaler stats | Not stored on model | Registered as `traffic_mean` / `traffic_std` buffers on `ResilientIoTPipeline` |
| `compute_loss` signature | `(imputed, target, mu, theta, pi, ...)` | + `traffic_mean`, `traffic_std`, `traffic_channel` args |
| `train.py` | No scaler extraction | Extracts `scaler.mean_[0]`, `scaler.scale_[0]` from dataset |
| `train.yaml` | `early_stop_patience: 10` | `early_stop_patience: 5` |

### Diagram — Model Architecture (Loss Detail)

```mermaid
graph TD
    subgraph Inputs
        SF[Static Features]
        MTS[Masked Time-Series Data]
    end

    subgraph 1. Dynamic Graph Learner
        SF --> GAT[Linear Projection → Pairwise Similarity]
        GAT --> ADJ[Dynamic Adjacency Matrix]
        ADJ --> THRES[ReLU & Sparsity Threshold]
        THRES --> EI[edge_index & edge_weight]
    end

    subgraph 2. Masked Spatiotemporal Encoder
        MTS --> ST_ENC[Spatial: GATv2Conv Layers]
        EI --> ST_ENC
        ST_ENC --> FLAT[Flatten Spatial Output]
        FLAT --> T_ENC[Temporal: TransformerEncoder]
        T_ENC --> LS[Latent Embeddings]
    end

    subgraph 3. Decoupled Decoders
        LS --> IMP_DEC[Imputation Decoder MLP]
        IMP_DEC --> REC[Reconstructed Masked Data]

        LS --> FOR_DEC[Forecasting Decoder MLP]
        FOR_DEC --> ZINB[ZINB Parameters: μ, θ, π]
    end

    subgraph "Loss Functions (Scale-Invariant)"
        REC --> SPLIT{Channel Split}

        SPLIT -->|"channel 0: traffic"| INV["Inverse Scaler: x × std + mean"]
        INV --> MSLE["MSLE: mean( log1p pred − log1p target )²"]

        SPLIT -->|"channels 1..D: weather"| MSE_W[MSE: Weather Channels]

        MSLE --> L_IMP[loss_impute = loss_traffic + loss_weather]
        MSE_W --> L_IMP

        ZINB --> L2[ZINB NLL Loss]
        L_IMP -.-> JOINT((Joint Training Objective))
        L2 -.-> JOINT
    end

    subgraph Registered Buffers
        BUF_M["traffic_mean (from dataset scaler)"] -.-> INV
        BUF_S["traffic_std (from dataset scaler)"] -.-> INV
    end
```

### Scale-Invariant MSLE — Data Flow

```mermaid
flowchart LR
    A["Imputed Features<br/>(StandardScaler domain)"] --> B["Split by Channel"]
    B -->|"ch0: traffic"| C["× traffic_std + traffic_mean"]
    C --> D["clamp ≥ 0"]
    D --> E["log1p(pred), log1p(target)"]
    E --> F["MSE in log domain"]
    B -->|"ch1..D: weather"| G["Standard MSE"]
    F --> H["loss_impute"]
    G --> H
```

---

## Figure 6: `exp/spatial-kriging-head` — IGNNK-Style Spatial Kriging

**Branch:** `exp/spatial-kriging-head` (2 commits over main)
**Files:** `dataset.py`, `decoders.py`, `pipeline.py`, `train.py`, `train.yaml`, `test_spatial_holdout.py`

### What changed vs. main

| Area | Main (baseline) | exp/spatial-kriging-head |
|------|----------------|--------------------------|
| Dataset | No spatial holdout | + `make_spatial_holdout_mask(N, p)` and `apply_spatial_holdout(batch, mask)` |
| Holdout mechanism | N/A | ~15% of nodes' dynamic features zeroed; static features **preserved** |
| Topology masking | Only `simulated_nodes_mask` | **Union**: `simulated_nodes_mask ∪ spatial_holdout_mask` |
| Model components | 3 (graph_learner, encoder, decoders) | + **SpatialKrigingDecoder** (MLP: H→2H→GELU→D) |
| Forward return | 8-tuple | **9-tuple** (+ `loss_kriging`) |
| Loss function | 2-term: `w_imp * L_imp + w_fore * L_fore` | **3-term**: `+ w_krig * L_krig` |
| Kriging loss | N/A | Hybrid: MSLE on traffic channel, MSE on weather channels |
| `compute_loss` | Returns `(total, imp, fore)` | Returns `(total, imp, fore, krig)` |
| `train.yaml` | No kriging config | + `spatial_holdout_prob: 0.15`, `w_kriging: 0.5` |
| W&B logging | No kriging metrics | + `train_loss_kriging`, `val_kriging_loss` |
| Tests | N/A | + `test_spatial_holdout.py` (7 tests) |

### Diagram — Spatial Holdout & Kriging Path

```mermaid
graph TD
    subgraph "Data Pipeline (Spatial Holdout)"
        LOADER[PyTorch Dataset Loader] --> MASK_FN["make_spatial_holdout_mask(N, 0.15)"]
        MASK_FN --> APPLY["apply_spatial_holdout(batch, mask)"]
        NOTE_IGNNK["IGNNK: static features intact<br/>dynamic → 0 for held-out nodes"] -.-> APPLY
        APPLY --> BATCH["Batch Dict + spatial_holdout_mask"]
    end

    subgraph "ResilientIoTPipeline (Phase 4)"
        BATCH --> GL[Dynamic Graph Learner]

        subgraph "Strict Topology Masking"
            GL --> TOPO["Drop outgoing edges from masked nodes"]
            BATCH -->|"spatial_holdout_mask"| UNION["combined = simulated ∪ held-out"]
            BATCH -->|"simulated_nodes_mask"| UNION
            UNION --> TOPO
        end

        TOPO --> ENC[Masked Spatiotemporal Encoder]
        BATCH -->|"dynamic_features (zeroed at holdout)"| ENC
        ENC --> LATENT[Latent Embeddings]

        LATENT --> DEC_IMP[Imputation Decoder]
        LATENT --> DEC_FOR[Forecasting Decoder]

        subgraph "Spatial Kriging Decoder (NEW)"
            LATENT --> KRIG_SEL["Select held-out nodes:<br/>latent[:, holdout_mask, :, :]"]
            KRIG_SEL --> KRIG_MLP["MLP: H → 2H → GELU → Dropout → D"]
            KRIG_MLP --> KRIG_OUT["Reconstructed series [B, K, T, D]"]
        end
    end

    subgraph "Three-Term Joint Loss"
        DEC_IMP --> L_IMP[MSE Imputation Loss]
        DEC_FOR --> L_FORE[ZINB NLL Loss]
        KRIG_OUT --> L_KRIG["Kriging Loss:<br/>MSLE (traffic) + MSE (weather)"]
        L_IMP --> JOINT["L = w_imp × L_imp<br/>+ w_fore × L_fore<br/>+ w_krig × L_krig"]
        L_FORE --> JOINT
        L_KRIG --> JOINT
    end
```

### Spatial Holdout — Information Flow

```mermaid
flowchart LR
    subgraph "Before Holdout"
        A["Batch: all nodes have dynamic features"]
    end
    subgraph "After apply_spatial_holdout"
        B["~15% nodes: dynamic = 0.0<br/>static = preserved"]
        C["Remaining nodes: untouched"]
    end
    subgraph "Edge Blocking"
        D["Held-out nodes cannot broadcast<br/>messages as SOURCE nodes"]
        E["They CAN receive messages<br/>from observable neighbours"]
    end
    subgraph "Kriging Head"
        F["Latent embeddings at held-out nodes<br/>(purely from message-passing)"]
        G["MLP reconstructs full series"]
    end
    A --> B
    A --> C
    B --> D
    D --> E
    E --> F
    F --> G
```

---

## Figure 7: `exp/enriched-gat-kriging` — OSM Edge Feature Enrichment + Kriging

**Branch:** `exp/enriched-gat-kriging` (5 commits over main, includes all Phase 4 commits)
**Files:** `graph_learner.py`, `encoder.py`, `decoders.py`, `pipeline.py`, `train.py`, `dataset.py`, `train.yaml`, `test_edge_enrichment.py`, `test_spatial_holdout.py`

### What changed vs. `exp/spatial-kriging-head`

| Area | exp/spatial-kriging-head | exp/enriched-gat-kriging |
|------|--------------------------|--------------------------|
| Graph Learner output | `(edge_index, edge_weight)` | `(edge_index, edge_weight, edge_attr [E,10])` |
| Edge features | Scalar learned similarity only | **9 pre-computed OSM features ⊕ 1 learned similarity** |
| Edge feature source | N/A | `edge_features_cache.json` (Phase 1 pre-computed) |
| Feature table | N/A | Registered buffer `edge_feature_table [N,N,9]` |
| Distance normalisation | N/A | Min-Max normalised `shortest_path` and `euclidean_distance` to [0,1] |
| Fail-fast enforcement | N/A | **Every** off-diagonal pair must exist in cache — no silent zero-fill |
| Encoder `edge_dim` | `1` (scalar weight) | `10` (enriched edge_attr) |
| GATv2Conv | `edge_dim=1` | `edge_dim=10` → learns Θ_e projection over all channels |
| Encoder forward arg | `edge_weight: [E]` | `edge_attr: [E, 10]` with backward-compat 1D→[E,1] promotion |
| Topology masking | Filters `edge_index`, `edge_weight` | Also filters `edge_attr` row-aligned |
| Pipeline constructor | No `node_names` / `edge_feature_cache_path` | + `node_names`, `edge_feature_cache_path` args |
| Analysis mode output | `edge_index`, `edge_weight`, `attention_weights` | + `edge_attr [E, 10]` |
| Tests | N/A | + `test_edge_enrichment.py` (enrichment + end-to-end gradient) |

### Diagram — Full Architecture (Enriched)

```mermaid
graph TD
    subgraph "Data Pipeline"
        SENSORS[traffic_sensors.json] --> PLACES[Google Places API]
        PLACES --> STATIC[Static Features]
        DYN[Dynamic Readings] --> MASK[Artificial Masking]
        MASK --> SCALER[StandardScaler]
        LOADER[Dataset Loader]
        STATIC --> LOADER
        SCALER --> LOADER
        LOADER --> HOLDOUT["Spatial Holdout (p=0.15)"]
        HOLDOUT --> BATCH[Batch Dict]
    end

    subgraph "External Data"
        OSM[OSM Walking Network] --> CACHE["edge_features_cache.json<br/>9 features × N(N−1) directed pairs"]
    end

    subgraph "ResilientIoTPipeline (Phase 5)"
        BATCH --> GL_IN[Static Features]
        BATCH --> ENC_IN[Masked Dynamic Features]

        subgraph "Dynamic Graph Learner (Enriched)"
            GL_IN --> PROJ[Linear Projection]
            PROJ --> SIM[Pairwise Similarity → Adjacency]
            SIM --> SPARSE[ReLU & Sparsity Threshold]
            SPARSE --> EI_EW["edge_index [2,E] + edge_weight [E]"]

            subgraph "Edge Enrichment Pipeline"
                TABLE["Buffer: edge_feature_table [N,N,9]<br/>Min-Max normalised distances"]
                CACHE --> TABLE
                EI_EW --> GATHER["Gather physical features [E,9]"]
                TABLE --> GATHER
                GATHER --> CAT["cat(physical [E,9], similarity [E,1])"]
                EI_EW -->|edge_weight| CAT
                CAT --> EA["edge_attr [E, 10]"]
            end
        end

        subgraph "Topology Masking"
            EI_EW --> TOPO[Drop masked source edges]
            EA --> TOPO
            TOPO --> FILT_OUT["Filtered: edge_index, edge_weight, edge_attr"]
        end

        subgraph "Encoder (edge_dim=10)"
            ENC_IN --> GAT["GATv2Conv(edge_dim=10)<br/>Θ_e projection over all 10 channels"]
            FILT_OUT --> GAT
            GAT --> FLAT[Flatten]
            FLAT --> TRANS[Temporal TransformerEncoder]
            TRANS --> LATENT[Latent Embeddings]
        end

        LATENT --> IMP[Imputation Decoder]
        LATENT --> FORE[Forecasting Decoder]
        LATENT --> KRIG[Kriging Decoder]
    end

    subgraph "Three-Term Loss"
        IMP --> L1[MSE Impute]
        FORE --> L2[ZINB NLL]
        KRIG --> L3[Kriging MSLE+MSE]
        L1 --> JOINT["w₁L₁ + w₂L₂ + w₃L₃"]
        L2 --> JOINT
        L3 --> JOINT
    end
```

### Edge Feature Composition

The 10-dimensional `edge_attr` vector for each edge `(src → dst)`:

```mermaid
block-beta
    columns 10
    block:physical:9
        columns 9
        A["shortest_path<br/>distance"]
        B["euclidean<br/>distance"]
        C["directness<br/>ratio"]
        D["residential"]
        E["footway"]
        F["service"]
        G["tertiary"]
        H["path"]
        I["unclassified"]
    end
    block:learned:1
        J["learned<br/>similarity"]
    end
```

| Index | Feature | Domain | Normalisation |
|-------|---------|--------|---------------|
| 0 | `shortest_path_distance` | metres | Min-Max → [0, 1] |
| 1 | `euclidean_distance` | metres | Min-Max → [0, 1] |
| 2 | `directness` | ratio | Already [0, 1] |
| 3–8 | Road category fractions | fraction | Already [0, 1] |
| 9 | Learned similarity | dot product | ReLU-thresholded |
