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
