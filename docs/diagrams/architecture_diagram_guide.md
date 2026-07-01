# Architecture Diagram Guide — IGNNK-Style Thesis Figures

> Your architecture is substantially more complex than IGNNK's single-task graph imputation model. This guide describes what each diagram should show, how the data representations differ from IGNNK, and practical advice for creating the figures.

## Reference: Your IGNNK Images

````carousel
![IGNNK Figure 1 — Node masking and message-passing reconstruction](/Users/administrator/.gemini/antigravity-ide/brain/tempmediaStorage/media__1782818680519.png)
<!-- slide -->
![IGNNK Figure 2 — Training/kriging data split with spatio-temporal matrix](/Users/administrator/.gemini/antigravity-ide/brain/tempmediaStorage/media__1782818684940.png)
````

---

## 1. Key Conceptual Differences from IGNNK

Before designing the figures, it's critical to understand why your architecture can't be a direct visual port of IGNNK's diagrams:

| Aspect | IGNNK | Your Pipeline (`ResilientIoTPipeline`) |
|--------|-------|---------------------------------------|
| **Input to GNN** | Raw spatio-temporal values (direct node features) | StandardScaler-normalised 3-channel time series `[traffic, humidity, solar]` |
| **Graph** | Fixed adjacency matrix (given) | **Learned** topology from static features via `DynamicGraphLearner` |
| **Edge features** | None (unweighted or scalar) | 10-dim: 9 physical OSM features + 1 learned similarity |
| **Latent space** | Implicit (single GNN output) | Explicit `[B, N, T, H]` latent embedding shared by 3 heads |
| **Masking** | IGNNK masks nodes spatially only | **Dual masking**: temporal NaN injection (~15%) + spatial holdout (~15%) |
| **Outputs** | Single reconstruction task | **Three decoder heads**: imputation, ZINB forecasting, spatial kriging |
| **Loss** | Single MSE reconstruction | Weighted sum of 3 losses with channel-specific treatment (MSLE on raw traffic counts, MSE on weather, ZINB NLL on forecasts) |

> [!IMPORTANT]
> The central visual challenge is representing the **latent embedding tensor** `[B, N, T, H]` — IGNNK's diagrams can show raw values flowing through the graph because their GNN operates directly on observed data. Your pipeline has an explicit **encoder → latent space → decoder** structure where the latent representation is abstract.

---

## 2. Proposed Diagram Set (4 Figures)

### Figure 1: High-Level Pipeline Overview (Full System)

**Purpose**: Show the end-to-end data flow from raw inputs to the three loss terms.

**Visual structure** (top-to-bottom or left-to-right flow):

```
┌──────────────────────────────────────────────────────────────┐
│  RAW INPUTS                                                   │
│                                                               │
│  ┌───────────────┐        ┌──────────────────────────────┐   │
│  │ Static Features│        │ Dynamic Features              │   │
│  │ [N, 15]        │        │ [B, N, T, 3]                  │   │
│  │ (POI counts,   │        │ (traffic, humidity, solar)    │   │
│  │  centrality,   │        │                               │   │
│  │  dist, etc.)   │        │  ┌─ ~15% NaN temporal mask    │   │
│  └───────┬───────┘        │  └─ ~15% spatial holdout (→0) │   │
│          │                 └─────────────┬────────────────┘   │
│          ▼                               │                    │
│  ┌──────────────────┐                    │                    │
│  │ DynamicGraphLearner│                   │                    │
│  │ Linear → Dot-Prod → │                   │                    │
│  │ ReLU → Sparsify     │                   │                    │
│  └────────┬─────────┘                    │                    │
│           │                              │                    │
│    edge_index [2,E]                      │                    │
│    edge_attr [E,10]                      │                    │
│           │                              │                    │
│           └──────────┐ ┌─────────────────┘                    │
│                      ▼ ▼                                      │
│           ┌─────────────────────────┐                         │
│           │ MaskedSpatiotemporalEncoder │                      │
│           │ GATv2Conv (edge_dim=10)    │                       │
│           │ TransformerEncoder          │                       │
│           └──────────┬──────────────┘                         │
│                      │                                        │
│              LATENT EMBEDDINGS                                │
│               [B, N, T, H]                                    │
│                      │                                        │
│          ┌───────────┼───────────┐                            │
│          ▼           ▼           ▼                             │
│  ┌───────────┐ ┌──────────┐ ┌──────────────┐                 │
│  │ Imputation │ │ Forecast │ │ Spatial      │                 │
│  │ Decoder    │ │ Decoder  │ │ Kriging Dec. │                 │
│  │ MLP→GELU  │ │ MLP→GELU │ │ MLP→GELU     │                 │
│  │ →Drop→Lin │ │ →Drop→Lin│ │ →Drop→Lin    │                 │
│  └─────┬─────┘ └────┬─────┘ └──────┬───────┘                 │
│        │            │              │                          │
│        ▼            ▼              ▼                          │
│   L_impute     L_forecast      L_kriging                      │
│   (MSLE+MSE)   (ZINB NLL)     (MSLE+MSE)                     │
│        └────────────┼──────────────┘                          │
│                     ▼                                         │
│         L_total = w_i·L_i + w_f·L_f + w_k·L_k                │
└──────────────────────────────────────────────────────────────┘
```

**IGNNK parallel**: This corresponds loosely to IGNNK's second image (the data matrix diagram), but instead of showing a flat spatio-temporal matrix, you need to show the **3D tensor stack** and the **triple decoder fan-out**.

**Visual tips**:
- Use the same warm yellow/gold colour scheme from IGNNK for the node circles on the graph
- Use a **grey dashed arrow** for the graph learner → encoder connection (topology flows separately from features)
- Use **solid blue arrows** for message passing inside the encoder (like IGNNK's blue arrows)
- Colour the three decoder boxes differently: e.g. light blue (imputation), light green (forecast), light orange (kriging)

---

### Figure 2: Encoder Detail — Spatial + Temporal Processing

**Purpose**: Show how the GATv2Conv and TransformerEncoder interact inside the encoder.

**Key representation challenge**: The encoder reshapes the input tensor multiple times. The diagram should show the **two processing stages** without drowning the reader in reshape operations.

**Recommended visual**:

```
                    Dynamic Features
                    [B, N, T, 3]
                         │
                    ┌────▼────┐
                    │ Reshape │  → [B·T·N, 3]
                    └────┬────┘
                         │
    edge_index ──────►┌──▼───────────────┐
    [2, E·B·T]       │    GATv2Conv     │◄── edge_attr [E·B·T, 10]
    (batched)        │  (4 attn heads)  │
                     │  in=3, out=H/4   │
                     └────────┬─────────┘
                              │  [B·T·N, H]
                         ┌────▼────┐
                         │ Reshape │  → [T, B·N, H]
                         └────┬────┘
                              │
                     ┌────────▼──────────┐
                     │ TransformerEncoder │
                     │ (2 layers, 4 heads)│
                     │ dim_ff = 4·H       │
                     └────────┬──────────┘
                              │  [T, B·N, H]
                         ┌────▼────┐
                         │ Reshape │  → [B, N, T, H]
                         └────┬────┘
                              │
                      Latent Embeddings
                       [B, N, T, H]
```

**IGNNK parallel**: This is analogous to the **message-passing arrows** in IGNNK's first image (the blue dashed arrows between nodes). But instead of showing message passing on raw values, your diagram shows it happening on **multivariate sensor features** that are then sequenced through a Transformer.

**Visual approach inspired by IGNNK**:
- Draw a **small graph** (5–7 nodes, like IGNNK's examples) with yellow circles
- Show one timestep's GATv2 message-passing with blue arrows between nodes
- Then show the **same nodes unrolled into a temporal sequence** feeding into the Transformer block
- This two-panel approach (spatial panel | temporal panel) effectively communicates the dual-stage processing

**Edge Feature Inset**: Consider a small inset box showing the 10-dim edge feature vector decomposition:

```
edge_attr [E, 10]
├─ [0] shortest_path_distance (normalised)
├─ [1] euclidean_distance     (normalised)
├─ [2] directness             (ratio)
├─ [3-8] road type fractions  (6 categories)
└─ [9] learned similarity     (from graph learner)
```

---

### Figure 3: Decoder Heads — IGNNK-Style Node Diagram

**Purpose**: Show what each decoder head receives and produces, using a node-graph visualisation in the IGNNK style.

This is where you can most directly replicate the IGNNK visual language. Create **three sub-panels**, each showing the same graph but with different node highlighting:

#### Panel A: Imputation Decoder
```
Input:  ALL nodes' latent embeddings [B, N, T, H]
Output: Reconstructed full sequence [B, N, T, 3]

Visual: Show the graph with ~15% of nodes' time-series bars having white gaps
        (like IGNNK's "missing" blocks). Arrow → decoder → same graph with
        gaps filled in (use pink/red to mark reconstructed values, like IGNNK's
        pink nodes).
```

#### Panel B: Forecasting Decoder (ZINB)
```
Input:  LAST timestep's embedding only [B, N, H]  (from latent[:, :, -1, :])
Output: Three ZINB parameters per node: μ, θ, π

Visual: Show the graph with a time-axis. The last timestep is highlighted.
        Arrow → decoder → each node gets three small output bars
        (μ = mean count, θ = dispersion, π = zero-inflation probability).
        Use the IGNNK kriging-data (Y) highlighting style for the forecast horizon.
```

#### Panel C: Spatial Kriging Decoder
```
Input:  HELD-OUT nodes' latent embeddings [B, K, T, H]
        (shaped by message-passing from neighbours ONLY)
Output: Reconstructed series for held-out nodes [B, K, T, 3]

Visual: THIS IS THE CLOSEST TO IGNNK's FIGURE 1.
        Show the graph with K nodes blacked out (masked, like IGNNK's black nodes).
        Blue message-passing arrows from yellow neighbours → black nodes.
        Arrow → decoder → same graph with masked nodes now showing
        reconstructed values in pink/red (like IGNNK's red "reconstructed" nodes).
```

> [!TIP]
> **Panel C is the killer diagram for your thesis.** It directly extends the IGNNK concept but shows that your model reconstructs *multivariate time series* (not just scalar values) for spatially unseen nodes. The pink reconstructed nodes should each show a tiny 3-channel time series (traffic + humidity + solar) rather than a single scalar.

---

### Figure 4: Dual Masking Strategy (Data-Level View)

**Purpose**: Show how your training data preparation differs from IGNNK's single node-masking approach.

**IGNNK parallel**: This directly maps to IGNNK's second image (the spatio-temporal matrix). But you need **two masking panels** instead of one:

```
┌─────────────────────────────────────────────────────────┐
│                   TEMPORAL MASKING (~15%)                 │
│                                                          │
│  Node 1: ████████░░████████░░██████████████████████      │
│  Node 2: ██████████████████████░░████████████████░░      │
│  Node 3: ████░░██████████████████████████████████████    │
│  Node 4: ██████████████████████████░░░░████████████      │
│  Node 5: ████████████████░░██████████████████████████    │
│      ...                                                 │
│  Node N: ██████████████████████████████░░████████████    │
│                                                          │
│  ░░ = NaN injection (temporal mask, ~15% per element)    │
│  Target: Imputation Decoder reconstructs these positions │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                   SPATIAL HOLDOUT (~15%)                  │
│                                                          │
│  Node 1: ████████████████████████████████████████████    │
│  Node 2: ████████████████████████████████████████████    │
│  Node 3: ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓    │  ← HELD OUT
│  Node 4: ████████████████████████████████████████████    │
│  Node 5: ████████████████████████████████████████████    │
│  Node 6: ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓    │  ← HELD OUT
│      ...                                                 │
│  Node N: ████████████████████████████████████████████    │
│                                                          │
│  ▓▓ = Entire series zeroed (spatial mask, ~15% of nodes) │
│  Target: Kriging Decoder reconstructs these nodes        │
└─────────────────────────────────────────────────────────┘
```

Use the same gold-bar aesthetic as IGNNK's second figure, but show that the bars are **3-channel** (a thin colour stripe for each of traffic/humidity/solar within each bar).

---

## 3. Representing the Latent Embedding (The Hard Part)

The central difficulty vs. IGNNK is that your architecture operates in a **latent space**, not directly on observables. Here are three strategies:

### Strategy A: "Black Box with Shape Annotations" (Simplest)
- Show the encoder as a labelled box with input/output tensor shapes
- Don't try to visualise the latent space itself
- **Pros**: Clean, unambiguous. **Cons**: Loses intuition.

### Strategy B: "Compressed Feature Map" (Recommended)
- Show the latent embedding as a **heat-map grid** `[N × T]` where each cell's colour represents the `H`-dimensional embedding magnitude
- Annotate that each cell contains an `H`-dim vector (e.g. `H=64`)
- Show the decoder heads "reading" different slices of this grid:
  - Imputation: reads all cells `[N, T]`
  - Forecast: reads only the rightmost column `[N, 1]`  (last timestep)
  - Kriging: reads only the held-out rows `[K, T]`
- **Pros**: Intuitive, shows what each decoder "sees". **Cons**: Requires careful colour design.

### Strategy C: "Per-Node Embedding Trace" (Most Detailed)
- Show a small set of nodes (3-4) with their per-timestep embedding vectors as small bar charts or sparklines
- Colour observable nodes' embeddings in gold, held-out nodes' embeddings in grey/faded
- Show how the decoder heads map these to outputs
- **Pros**: Very informative. **Cons**: Can be cluttered for many nodes.

> [!TIP]
> **Recommendation**: Use **Strategy B** for Figure 1 (overview), and **Strategy C** selectively for Figure 3, Panel C (kriging), where it's most impactful to show that held-out node embeddings are shaped purely by neighbourhood message-passing.

---

## 4. Tensor Shape Reference Card

For your own reference when drawing the diagrams — every key tensor in the forward pass:

| Tensor | Shape | Where | Notes |
|--------|-------|-------|-------|
| `static_features` | `[N, 15]` | Graph Learner input | POI, centrality, distance features |
| `dynamic_features` | `[B, N, T, 3]` | Encoder input | traffic (ch0), humidity (ch1), solar (ch2) |
| `edge_index` | `[2, E]` | Graph Learner → Encoder | Sparse learned topology |
| `edge_attr` | `[E, 10]` | Graph Learner → Encoder | 9 physical + 1 learned |
| `batched_edge_index` | `[2, E·B·T]` | Inside Encoder | Replicated for each graph in batch |
| `x_spatial` (post-GAT) | `[B·T·N, H]` | Inside Encoder | After GATv2Conv |
| `x_temporal` (pre-Transformer) | `[T, B·N, H]` | Inside Encoder | Reshaped for Transformer |
| `latent_embeddings` | `[B, N, T, H]` | Encoder output | **Shared representation** |
| `imputed_features` | `[B, N, T, 3]` | Imputation output | Full sequence reconstruction |
| `last_step_embedding` | `[B, N, H]` | Forecast input | `latent[:, :, -1, :]` |
| `forecast_logits` | `[B, N, 3]` | Forecast output | Raw μ, θ, π before activation |
| `mu, theta, pi` | `[B, N]` each | Forecast output | After softplus/sigmoid |
| `holdout_mask` | `[N]` bool | Kriging input | True = held-out node |
| `held_out` | `[B, K, T, H]` | Kriging input | K = holdout_mask.sum() |
| `kriging_pred` | `[B, K, T, 3]` | Kriging output | Reconstructed held-out series |

---

## 5. Tools and Format Recommendations for Thesis Figures

### Recommended Tools
1. **draw.io / diagrams.net** — Free, produces clean vector SVGs. Good for the box-and-arrow pipeline diagrams (Figures 1–2).
2. **Inkscape** or **Adobe Illustrator** — For the IGNNK-style node graphs (Figures 3–4). You need fine control over node positioning, arrow styling, and the small time-series insets.
3. **TikZ/PGF (LaTeX)** — If your thesis is in LaTeX, TikZ produces publication-quality diagrams that match your document's typography perfectly. Good templates exist for neural network architecture diagrams.
4. **Matplotlib / Manim** — For the data-level masking visualisation (Figure 4), you could programmatically generate the gold-bar matrix with actual data patterns.

### Colour Palette (Matching IGNNK)
- **Node fill (observable)**: `#FBBF24` (warm gold/yellow)
- **Node fill (masked/held-out)**: `#1F2937` (dark charcoal/black)
- **Node fill (reconstructed)**: `#EF4444` (red/coral, like IGNNK's pink)
- **Message-passing arrows**: `#3B82F6` (blue, like IGNNK's blue arrows)
- **Data bars (observed)**: `#F59E0B` (amber, like IGNNK's data bars)
- **Data bars (missing)**: `#FFFFFF` (white gaps, like IGNNK)
- **Background**: White or very light grey

### Sizing
- Aim for figures that are **single-column width** (~3.5 inches) for Figures 2–3 panels
- Figure 1 (full pipeline) and Figure 4 (dual masking) can be **double-column width** (~7 inches)

---

## 6. Diagram-by-Diagram Checklist

### ☐ Figure 1: Pipeline Overview
- [ ] Show static features → Graph Learner → edge_index + edge_attr
- [ ] Show dynamic features → Encoder (with graph structure fed in)
- [ ] Show latent embedding as a central block/heatmap
- [ ] Fan out to three decoder heads
- [ ] Show three loss terms combining into weighted total
- [ ] Annotate all tensor shapes

### ☐ Figure 2: Encoder Internals
- [ ] Show GATv2Conv processing one timestep across all nodes
- [ ] Show the edge feature vector decomposition (10-dim inset)
- [ ] Show TransformerEncoder processing the temporal sequence per node
- [ ] Show attention heads (4 heads in both GAT and Transformer)
- [ ] Annotate the spatial→temporal reshape transition

### ☐ Figure 3: Three Decoder Heads (IGNNK-Style)
- [ ] Panel A (Imputation): Graph with temporal NaN gaps → filled gaps
- [ ] Panel B (Forecast): Graph with last-timestep highlight → μ, θ, π outputs
- [ ] Panel C (Kriging): Graph with blacked-out nodes → reconstructed nodes
- [ ] Use IGNNK colour conventions: black=masked, red/pink=reconstructed, yellow=observed
- [ ] Show message-passing arrows for Panel C

### ☐ Figure 4: Dual Masking Strategy
- [ ] Spatio-temporal matrix in IGNNK bar style
- [ ] Top panel: temporal masking (random NaN gaps within bars)
- [ ] Bottom panel: spatial holdout (entire node rows greyed out)
- [ ] 3-channel colour stripe within each bar
- [ ] Graph insets on left/right showing training vs. kriging topology

---

## 7. Source Files for Reference

| Component | Source File | Key Lines |
|-----------|-----------|-----------|
| Graph Learner | [graph_learner.py](file:///Users/administrator/Code/ml-agent-workspace/src/graph_learner.py) | L32–229 |
| Encoder | [encoder.py](file:///Users/administrator/Code/ml-agent-workspace/src/encoder.py) | L7–154 |
| Decoders (Imputation + Forecast) | [decoders.py](file:///Users/administrator/Code/ml-agent-workspace/src/decoders.py) | L6–85 |
| Kriging Decoder | [decoders.py](file:///Users/administrator/Code/ml-agent-workspace/src/decoders.py#L87-L129) | L87–129 |
| Full Pipeline | [pipeline.py](file:///Users/administrator/Code/ml-agent-workspace/src/pipeline.py) | L15–224 |
| Dataset + Masking | [dataset.py](file:///Users/administrator/Code/ml-agent-workspace/src/dataset.py) | L15–356 |
| Architecture Spec | [NSTAR_BACKEND_ARCHITECTURE.md](file:///Users/administrator/Code/ml-agent-workspace/NSTAR_BACKEND_ARCHITECTURE.md) | Full doc |
| Spatial Estimation Design | [spatial_estimation_architecture.md](file:///Users/administrator/Code/ml-agent-workspace/docs/model/spatial_estimation_architecture.md) | Full doc |
