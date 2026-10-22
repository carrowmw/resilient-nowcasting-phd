# Chapter 6 — Spatial Dependence Track: Full Implementation Plan

> [!IMPORTANT]
> **Objective:** Build a self-contained `src/spatial/` experiment package (mirroring the `src/quality/` architecture) that answers RO2: *does explicitly modelling inter-sensor spatial dependencies via GATv2 improve short-term pedestrian-count forecasting over purely temporal baselines — and by how much?* All legacy GAT notebook work (`docs/context/GAT/`) is context only; no code is reused. Outputs are thesis tables and figures only (no dashboard).

Companion prompt: [`gat_experiment_prompt.md`](file:///Users/administrator/Code/ml-agent-workspace/docs/spatial-depedencies/gat_experiment_prompt.md)
Style reference: [`docs/mdpi_sensors/implementation_plan.md`](file:///Users/administrator/Code/ml-agent-workspace/docs/mdpi_sensors/implementation_plan.md)

---

## 1. Goal

Re-implement Experiment 2 (RO2, thesis Chapter 6) as a config-driven, resumable, statistically rigorous harness. The experiment provides a **controlled, progressive comparison**:

1. **Temporal-only baselines** — persistence, seasonal naive (daily/weekly), historical average, per-node LSTM, per-node Transformer.
2. **Spatial models** — GATv2 message passing over static graphs (Gabriel / kNN / fully-connected), with and without a temporal encoder, with and without physical edge features, and with learned topology.
3. **Ablations** — isolating exactly which spatial ingredient (message passing itself, graph structure, edge attributes, attention capacity) contributes.

The design borrows the `src/quality/` philosophy (capability-gated forecasters, offensive asserts, config-SHA resumability, per-job manifests, DM tests) but defines its own contract for the multi-node graph-batch domain `[B, N, T, D]`.

---

## 2. Improvements Over the Original Prompt

These changes strengthen the experimental design for a thesis chapter; each is folded into the Decisions below with rationale.

1. **Seeds are the replication unit.** The quality track had ~100 independent per-sensor fits as natural replication. Graph-level models are trained *once* on the whole network — a single fit means seed noise could swamp a small spatial delta. All trainable models run at **3 seeds**; results report mean ± sd across seeds, and DM tests are run per seed and summarised.
2. **Matched-capacity control (`identity` graph).** The strongest possible isolation of "does message passing help" is the *same GAT architecture with self-loops only* — identical parameter count, identical decoder, identical training recipe, zero cross-node information flow. This replaces vague "with vs without graph" comparisons and gives the thesis an unambiguous causal contrast.
3. **Loss held constant for the primary comparison.** Comparing GAT+ZINB against LSTM+MSE confounds loss with architecture (a flaw in the legacy notebooks). The primary comparison uses **masked MSE in scaled space for every model**. ZINB becomes a clearly-labelled secondary study (it is Chapter 7's territory anyway).
4. **Direct multi-horizon decoder.** One model emits all horizons `[B, N, H]` (standard in the ST-forecasting literature) instead of one job per horizon. This guarantees identical window sets across horizons (paired DM tests), cuts the job count ~3×, and removes the per-horizon retraining confound.
5. **Graphs built from cached pairwise distances.** The Gabriel test (`d(u,v)² < min_w d(u,w)² + d(w,v)²`) is scale-invariant and needs only distances — which already exist in `data/external/edge_features_cache.json` (`euclidean_distance` per directed pair). No new geometry code paths, one source of truth for both topology and edge attributes.
6. **Learned topology is a model variant, not a graph variant.** A learned graph is not a static `GraphSpec`; it is a module inside the forecaster (wrapping the existing `DynamicGraphLearner`). Treating it as a fourth "graph" (as the prompt implies) would break the static-graph contract everywhere.
7. **Scoped config SHA.** The quality runner hashes the *whole* config, so touching `wandb.project` invalidates every finished job. Here each job is keyed by the SHA of only the config subsets that affect it (`data`, `nodes`, `temporal`, `inputs`, `graph.<variant>`, `models.<key>`, seed).
8. **Shared torch training loop.** The quality track duplicates ~120 lines of fit/early-stopping across three wrappers (flagged in its own docstrings). One `TorchGraphForecaster` base owns the loop; concrete models only build `nn.Module`s.
9. **Capability gating, not name-prefix gating.** The quality runner special-cases models via `model_key.startswith(...)`. Here the runner consults `SpatialCapabilities` (e.g. only `emits_attention` models get an attention pass).
10. **MASE added, observed-targets-only evaluation.** Node scales differ wildly (legacy Test-5: mean node validity ~33%, extreme count heterogeneity). MASE (vs in-sample daily-seasonal-naive) makes cross-node aggregation meaningful; all metrics and losses are computed only where the target was actually observed.
11. **Quantified attention analysis.** Legacy Analysis-2/4/5 were qualitative PNG inspections. Here: attention entropy per node, attention-vs-distance decay curves, and attention stability across seeds — numbers that go in the thesis, not just pictures.
12. **Historical-average baseline added**, per `NSTAR_BACKEND_ARCHITECTURE.md`'s "benchmark against dumb baselines first" rule.
13. **Cross-node leakage unit test.** A perturbation test asserts that on the `identity` graph, changing node *j*'s inputs leaves node *i*'s prediction bit-identical — mechanically proving the baseline contrast is what we claim it is.

---

## 3. Decisions (Resolved)

### Package name: `src/spatial/`
Mirrors `src/quality/`; scripts and configs follow (`scripts/run_spatial_experiments.py`, `configs/spatial/`). Absolute imports within the package (`from spatial.contract import ...`), scripts do `sys.path.insert(0, <repo>/src)` — identical to the quality convention.

### Branch: `feature/ch6-spatial-gat-rebuild` from `main`
`feature/ch6-spatial-dependence` is currently 0 commits ahead of `main`, so branching from `main` and from HEAD are equivalent today. Per the NSTAR git protocol the harness build is `feat`-tier work (infrastructure, merged to main when verified); subsequent hypothesis runs can fork `exp/` branches if configs need to diverge. This plan document lives on the new branch too.

### Decoupling: new contract, shared pure utilities only
The package defines its own `GraphForecaster` ABC and does **not** import `src/harness/`. Permitted imports of existing code:
- `src/graph_learner.py::DynamicGraphLearner` — reused as the learned-topology module (it already handles the edge-feature cache, fail-fast on missing pairs, `[E,10]` edge attributes).
- `src/quality/stats.py::diebold_mariano`, `bootstrap_ci` — pure statistical functions with no per-sensor assumptions; re-implemented wrappers live in `spatial/stats.py`.
- `data/external/edge_features_cache.json` — read-only source for distances and physical edge features.

### Fairness protocol (the load-bearing design decision)
Every model — persistence through learned-topology GAT — consumes the **same** `GraphSplit` windows, the same train-only per-node scaling, the same masked-MSE loss (trainable models), and emits predictions for the same `[n_windows, N, H]` set. Temporal baselines are implemented *inside* this contract (shared-weight per-node networks that simply never see edges), not borrowed from the quality track's per-sensor machinery. This is what makes per-node paired DM tests valid.

### Data & node set
- Source: `data/processed/master_dataset_15T.parquet` via a `ParquetConnector`-style pushdown load (`veh_class == "person"`, directions summed per location — matches `RealIoTDataset(target_dir='all')` and the legacy GAT work). DB connector supported through the same `create_connector(config)` factory.
- Node = location. Real sensors only — **no simulated nodes** (those exist for Exp 3 kriging) and no spatial holdout (kriging/inductive evaluation is Chapter 7's question).
- Node inclusion: completeness ≥ `nodes.min_completeness` over the study period **and** present in the edge-feature cache (both directions for all retained pairs; asserted). Exclusions recorded with reasons in `node_report.csv` — nodes are dropped loudly, never silently.
- Node ordering: `sorted(node_ids)` is the order contract for every `[N, ...]` tensor, recorded in the graph manifest.

### Missingness policy
Inputs: NaN → 0 with a **missingness mask channel** appended (matches RIoT and legacy practice; the mask is information, not noise). Targets: never imputed — the loss and all metrics are masked to observed targets (`target_observed [n, N, H]`). A window is usable if its origin exists; per-(node, horizon) targets are individually maskable. This keeps low-quality nodes in the study (the alternative — complete-case windows across 80+ nodes — would leave almost no data, per legacy Test-5's sparsity stats).

### Temporal parameters
`window=12` (3 h at 15 min, legacy reference), `horizons=[1, 2, 4]` (15/30/60 min — literature-standard short-term bracket; legacy's 45 min sits inside it), `stride=1`, chronological 70/15/15 split on the global time axis with boundary purge of `window + max(horizon)` steps at each split boundary. All config-driven. Optional Stage D ablation: `window ∈ {12, 48}` (legacy Test-3 suggested long context helps rich-feature models).

### Loss
Primary: masked MSE in scaled space, all models. Secondary (Study C, optional): masked ZINB NLL on raw counts for the champion GAT vs its MSE twin — with PIT and zero-reliability diagnostics carried over from the Analysis-5 methodology. ZINB heads follow the NSTAR tripwire rules (`softplus + 1e-6` on μ, θ; clamped σ(π); `CRITICAL:` NaN/Inf asserts on every loss term).

### Graph variants (static) + topology (model-level)
- `identity` — self-loops only; the matched-capacity control.
- `gabriel` — Gabriel graph from cached euclidean distances (scale-invariant test, no coordinates needed).
- `knn8` — directed kNN (k configurable) from the same distances, symmetrised.
- `fc` — fully connected (legacy Test-6 reference point).
- Learned topology — a **model** (`gat_learned`), wrapping `DynamicGraphLearner` on the 15-dim static features; emits `[E,1]` (similarity only) or `[E,10]` (physical + similarity) edge attributes.

Edge attributes for static graphs: `none` (edge_dim absent) or `physical` (edge_dim=9, columns per `features.EDGE_FEATURE_KEYS`: distances, directness, 6 road-category fractions). The prompt's "edge_dim=10" applies only to learned topology (col 9 is the learned similarity, which does not exist for static graphs).

### Job matrix (staged, not full cross-product)

| Stage | Question | Jobs | Count |
|---|---|---|---|
| **A** — Model comparison | Do spatial models beat temporal baselines on the *same* graph? | 4 non-trainable baselines ×1 seed; {lstm, transformer} ×3 seeds; {gat_plain, gat_lstm, gat_transformer} × gabriel ×3 seeds | 4 + 6 + 9 = **19** |
| **B** — Graph structure | Which topology, and is message passing itself the ingredient? | champion GAT × {identity, knn8, fc} ×3 seeds; gat_learned ×3 seeds | 9 + 3 = **12** |
| **C** — Component ablations | Heads, depth, edge features (champion config ± one factor) | heads {1, 8}, layers {1, 3}, edge_attr {none} — 5 variants ×3 seeds | **15** |
| **D** (optional) — Window & ZINB | Long context; likelihood choice | window=48 champion ×3; zinb champion ×3 | **6** |

≈ 46–52 jobs, each a full-network fit (minutes to tens of minutes on the GPU runner) — comfortably a few GPU-days, fully resumable. The champion for Stages B/C is chosen on **validation** metrics from Stage A (test set touched only in final aggregation, per the methodology chapter's tuning-discipline commitment).

### Statistics
- Per-(node, horizon) RMSE/MAE/sMAPE/MASE on raw counts, observed targets only; aggregate median + IQR across nodes; mean ± sd across seeds.
- **Diebold–Mariano** per node (paired on `target_dt` within horizon, test split, squared loss, HAC + Harvey–Leybourne–Newbold — reusing `quality.stats.diebold_mariano`), with **BH-FDR across nodes** and a "fraction of nodes significantly improved" summary per model pair.
- Bootstrap CIs (nodes resampled) on the median metric delta between focal pairs.
- Ablation tables: Δmetric vs champion with seed spread.

### Runner & reproducibility
Config-SHA-scoped resumable registry; per-job `predictions.parquet` + `results.csv` + `manifest.json` (+ `attention.parquet` for capable models); graph manifests (edge lists + SHA) persisted once per variant; per-fit reseeding (`torch.manual_seed` + `np.random.seed` at every `fit()`); scaler means/stds recorded in the manifest; git hash in every manifest; W&B offline-gated exactly as the quality runner. No dashboard artefacts of any kind.

### Execution: strict container isolation
Unit tests and desk checks in `agent-workspace`/`cpu-runner`; smoke config on `cpu-runner`; full matrix via `docker compose run --rm gpu-runner ...` on the server (separate clone if parallel with other work, per the Parallel Execution Rule). Zero host installs; PyG and scipy are already in the images (verify `scipy.spatial` import in the smoke test).

---

## 4. Proposed Changes

Phases are sequential; each phase ends in a commit and its tests pass before the next begins.

---

### Phase 1 — Contract, loader, splits, dataset

#### [NEW] [`src/spatial/__init__.py`](file:///Users/administrator/Code/ml-agent-workspace/src/spatial/__init__.py)
Package docstring stating: decoupled from `src/harness/` and `src/quality/` by design (shared philosophy, own contract); deviation register vs the legacy GAT notebooks (loss held constant, seeds as replication, observed-only evaluation). Re-exports the contract symbols.

#### [NEW] [`src/spatial/contract.py`](file:///Users/administrator/Code/ml-agent-workspace/src/spatial/contract.py)

```python
@dataclass(frozen=True)
class SpatialCapabilities:
    trainable: bool = False          # real fit stage
    uses_graph: bool = False         # consumes edge_index (False ⇒ provably per-node)
    uses_edge_attr: bool = False     # consumes edge_attr
    learns_topology: bool = False    # owns an internal graph learner
    emits_attention: bool = False    # can return GATv2 attention weights

@dataclass
class FitReport:
    fitted: bool
    epochs_run: int = 0
    best_val_loss: Optional[float] = None
    notes: Dict = field(default_factory=dict)

@dataclass
class GraphContext:
    node_ids: List[str]              # sorted; THE order contract for every [N, ...] axis
    grid_index: pd.DatetimeIndex     # [T_total], continuous at freq
    values_raw: np.ndarray           # [T_total, N] float32, NaN at gaps
    values_scaled: np.ndarray        # [T_total, N], NaN at gaps
    observed_mask: np.ndarray        # [T_total, N] bool
    static_features: np.ndarray      # [N, 15] scaled (train-window stats)
    scaler_mean: np.ndarray          # [N] — per-node, train-only fit
    scaler_std: np.ndarray           # [N] (std < 1e-8 ⇒ warn, use 1.0)
    window: int
    horizons: Tuple[int, ...]        # ascending; asserted
    freq: str

    def steps_per_day(self) -> int
    def inverse_transform(self, y_scaled: np.ndarray) -> np.ndarray   # [..., N, ...] → raw, clip ≥ 0

@dataclass
class GraphSplit:
    context: GraphContext
    split: str                       # 'train' | 'val' | 'test' (asserted)
    window_start_idx: np.ndarray     # [n] int — lazy, windows never materialised

    # derived properties (no storage):
    origin_idx: np.ndarray           # [n] = start + window - 1
    target_idx: np.ndarray           # [n, H] = origin + horizons
    y_scaled: np.ndarray             # [n, N, H], NaN where unobserved
    y_raw: np.ndarray                # [n, N, H]
    target_observed: np.ndarray      # [n, N, H] bool
    origin_dt / target_dt            # DatetimeIndex / [n, H] datetimes

@dataclass(frozen=True)
class GraphSpec:
    method: str                      # 'identity' | 'gabriel' | 'knn' | 'fully_connected'
    edge_index: np.ndarray           # [2, E] int64, includes self-loops
    edge_attr: Optional[np.ndarray]  # [E, edge_dim] float32 or None
    edge_dim: int                    # 0 when edge_attr is None
    params: Dict                     # construction params (k, cache path, sha, …)

class GraphForecaster(ABC):
    capabilities: SpatialCapabilities = SpatialCapabilities()
    name: str = "abstract"

    @abstractmethod
    def fit(self, train: GraphSplit, val: GraphSplit, graph: GraphSpec) -> FitReport
    @abstractmethod
    def predict(self, split: GraphSplit, graph: GraphSpec) -> np.ndarray
        # [n, N, H] scaled space; NaN forbidden (assert)
    def predict_attention(self, split, graph, max_windows: int) -> "AttentionRecord"
        # default: raise NotImplementedError; overridden iff emits_attention
    def get_params(self) -> Dict     # manifest payload; default {}
```

Contract semantics documented in the module docstring: window `i` covers grid steps `[start, start+window)`, origin is the last input step, targets sit at `origin + h` for each `h` in `horizons`; predictions are scaled; evaluation inverse-transforms; **all models must predict the identical window set** (DM pairing depends on it — same offensive assert as the quality track).

#### [NEW] [`src/spatial/loader.py`](file:///Users/administrator/Code/ml-agent-workspace/src/spatial/loader.py)

```python
class GraphSeriesStore:
    def __init__(self, config: dict, connector=None)
    def build_context(self) -> GraphContext
    def node_report(self) -> pd.DataFrame   # per candidate node: completeness, in_edge_cache,
                                            # included, exclusion_reason
```

- Pushdown-filtered parquet load (`veh_class`, date range) or DB connector; sums directions per location; asserts non-empty, no negative counts, no duplicate `(location, dt)`.
- Pivots to `[T_total, N]` on a continuous `date_range` grid; asserts zero observations lost by reindexing.
- Node selection: completeness threshold ∩ edge-cache coverage (both directed pairs for all retained node pairs; hard assert with the missing pair named).
- Per-node scaler fit **only on grid steps covered by train-assigned windows** (the quality track's leakage fix, ported).
- Static features: from `data/external/static_features_cache.parquet` keyed by location (assert all 15 `STATIC_FEATURE_COLS` present per node), standardised with train-period stats — legacy Test-6's gradient-explosion fix, institutionalised.

#### [NEW] [`src/spatial/splits.py`](file:///Users/administrator/Code/ml-agent-workspace/src/spatial/splits.py)

```python
def assign_split_windows(T_total: int, window: int, max_horizon: int, stride: int,
                         train_frac: float, val_frac: float,
                         purge_boundaries: bool) -> Dict[str, np.ndarray]
def build_graph_splits(context: GraphContext, stride: int, train_frac: float,
                       val_frac: float, purge_boundaries: bool) -> Dict[str, GraphSplit]
def count_usable_targets(split: GraphSplit) -> pd.DataFrame   # per (node, horizon) observed-target counts
```

Single source of truth for split arithmetic (the quality track's best structural idea, kept). Chronological on the global time axis; purge drops the first `window + max_horizon - 1` windows of val and test. Asserts: no train target index ≥ first val input index, and symmetrically for val/test.

#### [NEW] [`src/spatial/dataset.py`](file:///Users/administrator/Code/ml-agent-workspace/src/spatial/dataset.py)

```python
class GraphWindowDataset(torch.utils.data.Dataset):
    def __init__(self, split: GraphSplit, mask_channel: bool, time_features: Sequence[str])
    def __getitem__(self, i) -> Dict[str, torch.Tensor]
        # x:      [N, T, D]  (scaled count, NaN→0; + mask channel; + cyclical time encodings)
        # y:      [N, H]     (scaled, NaN→0 — dead values, see y_mask)
        # y_mask: [N, H]     bool, target_observed
def collate_graph(batch: List[dict]) -> Dict[str, torch.Tensor]   # stacks to [B, N, T, D] etc.
def masked_mse(pred: Tensor, y: Tensor, y_mask: Tensor) -> Tensor
    # mean over observed entries only; assert y_mask.any(); assert loss finite ("CRITICAL: ...")
```

Time features (`day`, `week` sin/cos pairs) computed from `grid_index` — same encodings as `quality.per_sensor_dataset.build_feature_matrix`, reimplemented for the `[N, T, D]` layout (shared across nodes). D is derived, never hardcoded: `1 + mask_channel + 2·len(time_features)`.

---

### Phase 2 — Graph construction

#### [NEW] [`src/spatial/graphs.py`](file:///Users/administrator/Code/ml-agent-workspace/src/spatial/graphs.py)

```python
EDGE_FEATURE_KEYS_PHYSICAL: Tuple[str, ...]   # the 9 cached keys, column order contract

def load_distance_matrix(node_ids: Sequence[str], cache_path: str) -> np.ndarray
    # [N, N] euclidean metres from edge_features_cache.json; asserts symmetry within
    # tolerance, positivity off-diagonal, every directed pair present

def build_identity_graph(n: int) -> GraphSpec
def build_gabriel_graph(dist: np.ndarray, node_ids) -> GraphSpec
    # edge (u,v) iff ∀w: d(u,w)² + d(w,v)² ≥ d(u,v)²  — pure distance form, scale-invariant
def build_knn_graph(dist: np.ndarray, node_ids, k: int, symmetrise: bool = True) -> GraphSpec
def build_fc_graph(n: int, node_ids) -> GraphSpec

def build_graph(variant_key: str, variant_cfg: dict, node_ids, cache_path) -> GraphSpec
    # dispatcher; adds self-loops; attaches physical edge_attr when
    # variant_cfg['edge_attr'] == 'physical' (edge_dim=9, self-loop rows zeroed
    # except directness=1) via load_physical_edge_attr(...)

def load_physical_edge_attr(edge_index, node_ids, cache_path) -> np.ndarray  # [E, 9]
def graph_manifest(spec: GraphSpec, node_ids) -> dict    # method, params, E, degree stats,
                                                         # edge list as id pairs, sha16
```

Offensive asserts: Gabriel graph connected (warn + record if not — it is provably connected for points in general position, so a violation flags cache corruption); no NaN in edge_attr; road-category fractions sum to ~1 or ~0 (mirrors `DynamicGraphLearner`'s build-time checks). Desk check verifies the Gabriel builder against an O(N³) brute-force reference on random point sets.

---

### Phase 3 — Models & registry

#### [NEW] [`src/spatial/models/__init__.py`](file:///Users/administrator/Code/ml-agent-workspace/src/spatial/models/__init__.py)
Re-exports concrete classes.

#### [NEW] [`src/spatial/models/base.py`](file:///Users/administrator/Code/ml-agent-workspace/src/spatial/models/base.py)

```python
class TorchGraphForecaster(GraphForecaster):
    """Owns the entire fit/predict machinery; subclasses only build nn.Modules."""
    def __init__(self, lr, batch_size, epochs, patience, scheduler_step_size,
                 scheduler_gamma, grad_clip, seed, device, mask_channel, time_features)
    @abstractmethod
    def _build_module(self, input_dim: int, n_nodes: int, window: int,
                      n_horizons: int, graph: GraphSpec) -> nn.Module
    def fit(self, train, val, graph) -> FitReport
        # torch.manual_seed(seed) + np.random.seed(seed) at fit start (order-independence);
        # Adam + StepLR(per-epoch); masked_mse; grad-clip; early stopping with CPU-cloned
        # best-state snapshot and mandatory restore (assert best_state is not None);
        # non-finite train loss is an assert, never a skip
    def predict(self, split, graph) -> np.ndarray   # [n, N, H]; assert no NaN
```

Graph tensors (`edge_index`, `edge_attr`, `static_features`) are hoisted to device once per fit; batching over the graph axis uses the proven offset-arithmetic pattern from `src/encoder.py` (disjoint-union replication `offsets = arange(B)·N`), not `torch_geometric.data.Batch`.

#### [NEW] [`src/spatial/models/temporal.py`](file:///Users/administrator/Code/ml-agent-workspace/src/spatial/models/temporal.py)
Non-trainable baselines (pure NumPy, capabilities all-False):

```python
class PersistenceForecaster(GraphForecaster)      # last OBSERVED value in window per node;
                                                  # all-missing window ⇒ node train mean,
                                                  # count surfaced in self.diagnostics
class SeasonalNaiveForecaster(GraphForecaster)    # value at target - season; asserts
                                                  # season > max horizon (future-leakage guard);
                                                  # missing reference ⇒ persistence fallback,
                                                  # fraction in diagnostics
class HistoricalAverageForecaster(GraphForecaster) # per-(node, hour-of-week) train mean
```

Trainable per-node baselines (capabilities: `trainable=True`, `uses_graph=False`):

```python
class PerNodeRNNForecaster(TorchGraphForecaster)         # type: lstm
    # module reshapes [B, N, T, D] → [B·N, T, D]; weight-shared LSTM
    # (hidden_dim, num_layers, dropout, bidirectional) → last hidden → Linear(→ H)
class PerNodeTransformerForecaster(TorchGraphForecaster) # type: transformer
    # same reshape; learned positional embedding, TransformerEncoder
    # (d_model, nhead, num_encoder_layers, dim_feedforward), mean-pool → Linear(→ H)
```

Weight sharing across nodes is deliberate and documented: it matches the GAT models' parameterisation (one network for all nodes), so the spatial delta is not confounded with per-node capacity. (The quality track's per-sensor fits answer a different question and are not comparable here.)

#### [NEW] [`src/spatial/models/gat.py`](file:///Users/administrator/Code/ml-agent-workspace/src/spatial/models/gat.py)
One configurable family (capabilities: `trainable`, `uses_graph`, `emits_attention`, `uses_edge_attr` per spec):

```python
class GATForecaster(TorchGraphForecaster)   # type: gat
    # spec: temporal_encoder ∈ {none, lstm, transformer}; gat_layers; heads; hidden_dim;
    #       edge policy from GraphSpec.edge_dim; use_static_features: bool

class _GATForecastNet(nn.Module):
    def forward(self, x [B,N,T,D], edge_index, edge_attr, static) -> [B,N,H]  (+ alpha list)
    # 1. Temporal encode per node → h [B, N, F]
    #    none: flatten window → MLP;  lstm/transformer: as in temporal.py, minus head
    # 2. Spatial: stack of gat_layers × GATv2Conv(F → hidden//heads, heads, edge_dim,
    #    add_self_loops=False — self-loops are in the GraphSpec), ELU, residual, dropout;
    #    applied per batch item via offset-replicated edge_index (B graphs, not B·T —
    #    temporal encoding first means one graph pass per window, unlike src/encoder.py's
    #    per-timestep pass; documented as a deliberate, cheaper design)
    # 3. Decoder: MLP [F_spatial (+15 static)] → H
```

> [!NOTE]
> The legacy `src/encoder.py` applies a *single* GATv2 layer per timestep before the Transformer; its `num_layers` controls the Transformer only (config comments there are misleading). This package makes GAT depth a real, honest ablation axis (`gat_layers`), and applies spatial mixing *after* temporal encoding — the dominant pattern in recent ST-GNN work and ~T× cheaper per step.

```python
class LearnedTopologyGATForecaster(GATForecaster)   # type: gat_learned
    # capabilities += learns_topology; ignores the static GraphSpec (asserts it is
    # `identity` to keep the runner contract uniform); owns a DynamicGraphLearner
    # (imported from src/graph_learner.py) over context.static_features;
    # edge_attr_mode ∈ {'learned_only' (edge_dim=1), 'physical_plus_learned' (edge_dim=10)};
    # topology re-derived each forward (embeddings train jointly); final resolved
    # edge_index persisted into the manifest at fit end.
```

Attention: `predict_attention(split, graph, max_windows)` runs `return_attention_weights=True` on a seeded subsample of test windows, averaging α over windows → `AttentionRecord(edge_index, alpha_mean [E, heads], alpha_by_hour [24, E])`.

#### [NEW] [`src/spatial/registry.py`](file:///Users/administrator/Code/ml-agent-workspace/src/spatial/registry.py)

```python
KNOWN_TYPES = ("persistence", "seasonal_naive", "historical_average",
               "lstm", "transformer", "gat", "gat_learned")

def build_forecaster(model_key: str, spec: Dict, seed: int,
                     device: Optional[str]) -> GraphForecaster
```

Asserts `type` known; required hyperparameters read with `spec["key"]` (KeyError-loud, zero Python defaults for anything scientific); sets `forecaster.name = model_key` post-construction (config key is the run label — quality convention kept).

---

### Phase 4 — Evaluation, statistics, attention, plotting

#### [NEW] [`src/spatial/evaluate.py`](file:///Users/administrator/Code/ml-agent-workspace/src/spatial/evaluate.py)

```python
def mase_scale(context: GraphContext, train: GraphSplit) -> np.ndarray   # [N]: in-sample
    # daily-seasonal-naive MAE per node on observed train pairs; assert > 0 or node flagged
def evaluate_split(split: GraphSplit, preds_scaled: np.ndarray,
                   mase_scale: np.ndarray) -> pd.DataFrame
    # one row per (node_id, horizon): rmse_raw, mae_raw, smape_raw, mase, n_targets,
    # rmse_scaled, mae_scaled, mean_count_raw — observed targets only;
    # metric keys generated from ONE shared list (no hand-enumerated NaN branches)
def aggregate_nodes(results: pd.DataFrame) -> pd.DataFrame
    # per horizon: median + IQR + mean across nodes, n_nodes
def predictions_frame(split, preds_scaled, model, graph_variant, seed) -> pd.DataFrame
    # long format: node_id, model, graph_variant, seed, horizon, origin_dt, target_dt,
    # y_true_raw, y_pred_raw, y_true_scaled, y_pred_scaled, target_observed
```

sMAPE denominator floored at 1e-8; MAPE deliberately dropped in favour of MASE (zero-heavy person counts make MAPE meaningless even with the max(|y|,1) fix — deviation documented in the docstring).

#### [NEW] [`src/spatial/stats.py`](file:///Users/administrator/Code/ml-agent-workspace/src/spatial/stats.py)

```python
from quality.stats import diebold_mariano, bootstrap_ci   # pure functions, reused

def dm_by_node(preds_a: pd.DataFrame, preds_b: pd.DataFrame, horizon: int,
               loss: str = "squared", min_n: int = 30) -> pd.DataFrame
    # per node: inner-merge on (node_id, target_dt), ASSERT merged length equals both
    # sides ("models did not predict the identical window set"); observed-only;
    # nodes with n < min_n excluded with reason column, never silently
def dm_summary(dm_df: pd.DataFrame, alpha: float = 0.05) -> dict
    # BH-FDR across nodes → n_nodes, frac_significant_fdr, frac_a_better,
    # median_dm_stat, mean_loss_diff
def paired_delta_ci(results_a, results_b, metric: str, n_bootstrap: int = 2000,
                    seed: int = 42) -> dict   # resample NODES jointly; median delta + CI
def seed_stability(results: pd.DataFrame, metric: str) -> pd.DataFrame
    # per (model, graph_variant, horizon): mean, sd, range across seeds
def ablation_table(results, champion_key: str, factor: str) -> pd.DataFrame
```

#### [NEW] [`src/spatial/attention.py`](file:///Users/administrator/Code/ml-agent-workspace/src/spatial/attention.py)

```python
@dataclass
class AttentionRecord:
    edge_index: np.ndarray; alpha_mean: np.ndarray; alpha_by_hour: np.ndarray
    n_windows: int; seed: int

def attention_entropy(record, n_nodes) -> pd.DataFrame     # per-node inbound entropy
    # (uniform vs peaked attention) normalised by log(in-degree)
def attention_vs_distance(record, dist_matrix, node_ids) -> pd.DataFrame
    # α vs metres — does attention recover spatial decay, or ignore it?
def attention_seed_agreement(records: List[AttentionRecord]) -> dict
    # Spearman ρ of edge α rankings across seeds — is "which edges matter" stable
    # enough to interpret? (the credibility gate for the thesis's attention claims)
def top_edges_table(record, node_ids, k: int = 20) -> pd.DataFrame
```

#### [NEW] [`src/spatial/plotting.py`](file:///Users/administrator/Code/ml-agent-workspace/src/spatial/plotting.py)
Reuses the quality track's `PAPER_RC` idiom (`apply_paper_style()`, dpi 300, tight bbox):

```python
def plot_model_comparison(agg, metric, out_path)        # per-horizon box/strip across nodes
def plot_spatial_delta(results, baseline_key, focal_key, metric, out_path)
                                                        # per-node paired deltas, DM-sig marked
def plot_graph_ablation(agg, out_path)                  # identity → gabriel → knn → fc → learned
def plot_attention_graph(record, dist_matrix, node_ids, out_path)   # spatial layout via
                                                        # classical MDS on the distance matrix
                                                        # (no coordinates required)
def plot_attention_distance_decay(df, out_path)
def plot_seed_stability(df, out_path)
```

---

### Phase 5 — Runner, aggregator, configs

#### [NEW] [`scripts/run_spatial_experiments.py`](file:///Users/administrator/Code/ml-agent-workspace/scripts/run_spatial_experiments.py)

```python
def job_sha(config: dict, model_key: str, graph_variant: str, seed: int) -> str
    # sha256[:16] over canonical yaml.safe_dump(sort_keys=True) of ONLY:
    # {data, nodes, temporal, inputs, graph: {shared keys + this variant},
    #  models: {this key}, seed}  — scoped invalidation (Improvement 7)

def expand_jobs(config) -> List[Job]      # from run.matrix stages; deterministic order;
                                          # Job = (job_key, stage_tag, model_key, graph_variant, seed)
                                          # job_key = f"{model_key}_{graph_variant}_s{seed}"
def run_job(job, store_context, splits, graphs, config) -> dict   # manifest
def main() -> None                        # --config path
```

Flow (mirrors `run_quality_experiments.py`, single sequential process):
1. Load YAML, mkdir output, load/parse `registry.json` (`{job_key: sha}`).
2. Build `GraphContext` once (assert node count ≥ config minimum); write `node_report.csv`, `context_manifest.json` (node order, scaler stats, split boundaries, git hash).
3. Build every static `GraphSpec` once; write `graphs/<variant>_manifest.json`.
4. Per job: skip iff `registry[job_key] == job_sha`; else build forecaster (fresh, seeded), `fit(train, val, graph)`, `predict(test)` (+ `predict(val)` — champion selection uses val), `evaluate_split`, `predictions_frame`; if `capabilities.emits_attention` and config asks: `predict_attention` → `attention.parquet`.
5. Persist `runs/<job_key>/{predictions.parquet, results.csv, manifest.json}`; manifest fields: `job_key, stage, model, model_spec, graph_variant, graph_sha, seed, horizons, epochs_run, best_val_loss, n_nodes, diagnostics, git_hash, job_sha, wall_seconds` (failures impossible to swallow: a job that raises is *not* registered and the traceback propagates — one network fit per job means no per-sensor failure list is needed; the whole job either succeeded or must be rerun).
6. Registry rewritten after each job (crash loses at most the in-flight job); W&B logged **after** registry marking (fixes the quality-track ordering quirk).

#### [NEW] [`scripts/aggregate_spatial_results.py`](file:///Users/administrator/Code/ml-agent-workspace/scripts/aggregate_spatial_results.py)
Reads `runs/*/`, produces under `<output_dir>/aggregated/`:
- `model_comparison.csv` — per (model, graph, horizon): median/IQR across nodes, mean ± sd across seeds.
- `dm_<focal>_vs_<baseline>.csv` + `dm_summary.csv` — per configured focal pairs (config-driven pair list, **not** name-prefix heuristics), per seed and pooled.
- `ablation_{graph,heads,layers,edge_attr}.csv` — Δ vs champion.
- `attention_{entropy,distance,stability,top_edges}.csv`.
- `figures/*.png` via `spatial.plotting`.
- `champion.json` — the val-selected champion and the selection rule applied.

#### [NEW] [`configs/spatial/spatial_full.yaml`](file:///Users/administrator/Code/ml-agent-workspace/configs/spatial/spatial_full.yaml)

```yaml
# NSTAR rule: zero hyperparameters hardcoded in Python.
data_source: "parquet"
data:
  parquet_path: "data/processed/master_dataset_15T.parquet"
  start_date: "2022-01-01"       # align with quality_full_v2 study period
  end_date:   "2024-01-01"
  veh_class: "person"
  direction: "all"               # summed per location
  freq: "15min"

nodes:
  min_completeness: 0.30         # observed fraction over study period
  edge_features_cache: "data/external/edge_features_cache.json"
  static_features_cache: "data/external/static_features_cache.parquet"
  min_nodes: 30                  # hard floor; assert after selection

temporal:
  window: 12                     # 3 h lookback (legacy reference)
  horizons: [1, 2, 4]            # 15 / 30 / 60 min
  stride: 1
  train_frac: 0.70
  val_frac: 0.15
  purge_boundaries: true

inputs:
  mask_channel: true
  time_features: [day, week]     # sin/cos pairs

graph:
  variants:
    identity: {method: identity}
    gabriel:  {method: gabriel, edge_attr: physical}
    knn8:     {method: knn, k: 8, symmetrise: true, edge_attr: physical}
    fc:       {method: fully_connected, edge_attr: physical}

models:
  persistence:      {type: persistence}
  snaive_daily:     {type: seasonal_naive, seasonality_days: 1}
  snaive_weekly:    {type: seasonal_naive, seasonality_days: 7}
  hist_avg:         {type: historical_average}
  lstm:             {type: lstm, hidden_dim: 64, num_layers: 2, dropout: 0.1,
                     lr: 1.0e-3, batch_size: 32, epochs: 60, patience: 8,
                     scheduler_step_size: 5, scheduler_gamma: 0.7, grad_clip: 1.0}
  transformer:      {type: transformer, d_model: 64, nhead: 4, num_encoder_layers: 2,
                     dim_feedforward: 128, dropout: 0.1, lr: 1.0e-3, batch_size: 32,
                     epochs: 60, patience: 8, scheduler_step_size: 5,
                     scheduler_gamma: 0.7, grad_clip: 1.0}
  gat_plain:        {type: gat, temporal_encoder: none,  gat_layers: 2, heads: 4,
                     hidden_dim: 64, use_static_features: false, <training block as lstm>}
  gat_lstm:         {type: gat, temporal_encoder: lstm,  gat_layers: 2, heads: 4, ...}
  gat_transformer:  {type: gat, temporal_encoder: transformer, gat_layers: 2, heads: 4, ...}
  gat_learned:      {type: gat_learned, temporal_encoder: transformer, gat_layers: 2,
                     heads: 4, embedding_dim: 32, sparsity_threshold: 0.5,
                     edge_attr_mode: physical_plus_learned, ...}
  # Stage C variants: gat_h1, gat_h8, gat_l1, gat_l3, gat_noedgeattr — champion spec ± one factor

run:
  output_dir: "artifacts/spatial_full"
  matrix:
    - {tag: A_baselines, models: [persistence, snaive_daily, snaive_weekly, hist_avg],
       graphs: [identity], seeds: [42]}
    - {tag: A_temporal,  models: [lstm, transformer], graphs: [identity], seeds: [42, 43, 44]}
    - {tag: A_spatial,   models: [gat_plain, gat_lstm, gat_transformer],
       graphs: [gabriel], seeds: [42, 43, 44]}
    - {tag: B_graphs,    models: [gat_transformer],   # replace with val champion before Stage B
       graphs: [identity, knn8, fc], seeds: [42, 43, 44]}
    - {tag: B_learned,   models: [gat_learned], graphs: [identity], seeds: [42, 43, 44]}
    - {tag: C_ablations, models: [gat_h1, gat_h8, gat_l1, gat_l3, gat_noedgeattr],
       graphs: [gabriel], seeds: [42, 43, 44]}
  attention: {enabled: true, max_windows: 256, models: [gat_transformer, gat_learned]}
  device: "cuda:0"

analysis:
  dm_pairs:                      # aggregator input — explicit, no name heuristics
    - {focal: gat_transformer_gabriel, baseline: lstm_identity}
    - {focal: gat_transformer_gabriel, baseline: gat_transformer_identity}
    - {focal: gat_transformer_gabriel, baseline: transformer_identity}
  champion_metric: "mae_raw"     # median across nodes, val split, h=1

wandb: {enabled: false, project: "ch6-spatial-gat", mode: "offline"}
```

#### [NEW] [`configs/spatial/spatial_smoke.yaml`](file:///Users/administrator/Code/ml-agent-workspace/configs/spatial/spatial_smoke.yaml)
3-month window, `min_completeness: 0.5`, `min_nodes: 10`, tiny models (`epochs: 2`, `hidden_dim: 16`, `heads: 2`), matrix = one baseline + `lstm` + `gat_plain` on `{identity, gabriel}`, single seed, `output_dir: artifacts/spatial_smoke`, CPU-safe (`device: null`).

---

### Phase 6 — Tests & desk checks

#### [NEW] `tests/test_spatial_contract.py`
Split-name assertion; lazy property shapes; `inverse_transform` round-trip + clip-at-zero; horizon ordering assert.

#### [NEW] `tests/test_spatial_graphs.py`
- Gabriel vs O(N³) brute force on 50 random planar point sets (property-based).
- Gabriel scale invariance (`dist` vs `3.7·dist` → identical edges).
- kNN symmetrisation; identity graph = self-loops only; FC edge count = N².
- Physical edge_attr: shape `[E, 9]`, no NaN, road fractions sum to ~1/~0, loud KeyError on a missing cache pair.

#### [NEW] `tests/test_spatial_splits.py`
Chronological ordering; **leakage test**: max train target grid-index < min val input grid-index (and val/test), with and without purge; `count_usable_targets` consistency with `GraphSplit.target_observed`.

#### [NEW] `tests/test_spatial_dataset.py`
`x` channel-count arithmetic; mask channel equals `~observed`; NaN-free tensors; `masked_mse` ignores unobserved entries (known-answer) and asserts on an all-masked batch.

#### [NEW] `tests/test_spatial_models.py`
- Shape contracts for every registry type on a toy context (N=6, T_total=400).
- **Cross-node leakage test** (Improvement 13): `gat_plain` on `identity` + each temporal baseline — perturb node *j*'s inputs, assert node *i*'s predictions bit-identical. On `gabriel`, assert they *do* change (message passing is live).
- Determinism: same seed twice → identical predictions; `predict` before `fit` raises.
- Seasonal-naive future-leakage assert fires when `seasonality ≤ max horizon`.

#### [NEW] `tests/test_spatial_evaluate.py` / `tests/test_spatial_stats.py`
Metric known-answers (incl. MASE against hand computation); observed-only masking; `dm_by_node` known-answer + the identical-window-set merge assert; BH-FDR monotonicity; bootstrap CI contains point estimate.

#### [NEW] `tests/test_spatial_runner.py`
`job_sha` scoping (change `wandb.project` → sha unchanged; change `models.gat_plain.heads` → only that model's jobs invalidate); `expand_jobs` determinism and counts; registry skip logic via tmpdir round-trip.

#### [NEW] `scripts/desk_checks/desk_check_spatial_splits.py`
Prints split boundaries, purge windows, per-split usable-target counts on the real dataset — human-verifiable arithmetic before any GPU time.

#### [NEW] `scripts/desk_checks/desk_check_spatial_graphs.py`
Builds all variants on the real node set; prints |E|, degree distribution, Gabriel connectivity, edge-attr summary stats; renders a quick MDS layout figure per graph.

#### [NEW] `scripts/desk_checks/desk_check_spatial_stats.py`
DM on synthetic AR(1) errors with known superiority (positive control) and on identical-model split-half predictions (negative control: FDR-significant fraction ≈ α).

---

### Existing files to modify

| File | Change |
|---|---|
| [`EXPERIMENTS.md`](file:///Users/administrator/Code/ml-agent-workspace/EXPERIMENTS.md) | Ledger entry per phase completion (format: Date / Branch / Goal / **Outcome** with Verification + Files added, matching the `feat/quality-track` entry precedent). |
| `requirements.txt` | Only if the smoke test reveals a missing import (scipy and torch-geometric are already present; no additions expected). |

**Nothing in `src/quality/`, `src/harness/`, `src/graph_learner.py`, or the training pipeline is modified.** `spatial` imports `quality.stats.{diebold_mariano, bootstrap_ci}` and `graph_learner.DynamicGraphLearner` read-only.

---

## 5. Execution Sequence

1. ✏️ `git checkout main && git pull && git checkout -b feature/ch6-spatial-gat-rebuild`; commit this plan.
2. ✏️ Phase 1 (contract, loader, splits, dataset) + `tests/test_spatial_{contract,splits,dataset}.py` → run in container: `docker compose exec agent-workspace pytest tests/test_spatial_*.py -q` → commit.
3. ✏️ Phase 2 (graphs) + `tests/test_spatial_graphs.py` + `desk_check_spatial_graphs.py` (CPU, real cache — read-only) → commit.
4. ✏️ Phase 3 (models, registry) + `tests/test_spatial_models.py` → commit.
5. ✏️ Phase 4 (evaluate, stats, attention, plotting) + remaining unit tests + `desk_check_spatial_stats.py` → commit.
6. ✏️ Phase 5 (runner, aggregator, both configs) + `tests/test_spatial_runner.py` → commit.
7. 🖥️ **On server (or locally on CPU):** smoke run —
   `docker compose run --rm cpu-runner python scripts/run_spatial_experiments.py --config configs/spatial/spatial_smoke.yaml`
   then `... aggregate_spatial_results.py --config configs/spatial/spatial_smoke.yaml`. Verify outputs (see §6), rerun to confirm the registry skips every job.
8. 🖥️ **On server:** Stage A —
   `docker compose run --rm gpu-runner python scripts/run_spatial_experiments.py --config configs/spatial/spatial_full.yaml` (Stage A rows enabled in `run.matrix`).
9. ✏️ Inspect `aggregated/champion.json` (val metrics); set Stage B/C model keys to the champion spec in the config; commit the config change (never a branch — NSTAR §4.2).
10. 🖥️ **On server:** Stages B + C (same command — registry re-runs only new jobs); optional Stage D.
11. 🖥️ Final aggregation; pull `aggregated/` tables + figures for the thesis chapter.
12. ✏️ EXPERIMENTS.md ledger entry; merge `feature/ch6-spatial-gat-rebuild` → `main` once end-to-end verified.

Steps 1–6 and 9/12 are local code changes; steps 7–8, 10–11 are container execution (GPU server for the full matrix).

---

## 6. Verification Plan

### Automated
- All `tests/test_spatial_*.py` pass in the `agent-workspace` container (target: ~60+ assertions across 7 files; the leakage, cross-node-isolation, and DM known-answer tests are the non-negotiables).
- Smoke run completes end-to-end on CPU; second invocation skips 100% of jobs via the registry; changing one model's hyperparameter invalidates exactly that model's jobs.
- Desk checks: Gabriel = brute force; DM positive/negative controls behave; split arithmetic printout matches hand calculation.

### Manual
- `node_report.csv` — sensible node count (expect ~60–90 of the 104 cache locations after completeness filtering), no surprising exclusions.
- Smoke `aggregated/model_comparison.csv` row count = models × graphs × horizons; persistence MASE ≈ its theoretical value; `gat_plain` on `identity` within seed noise of a per-node MLP (sanity: message passing off ⇒ no spatial advantage).
- Stage A sanity ordering: persistence ≤ snaive ≤ neural models on MAE at h=1 for most nodes; anything else demands investigation before Stages B/C.
- Attention outputs: `attention_seed_agreement` ρ reported *before* any interpretive claims are drafted for the chapter.
- Every run directory contains a manifest with git hash + job SHA; `context_manifest.json` records scaler constants and node order (Artifact-Twinning rule).

### Ledger
Per NSTAR §8 the work is not complete until `EXPERIMENTS.md` carries the entry: Date, `feature/ch6-spatial-gat-rebuild`, objective (RO2 harness rebuild), **Outcome** with test counts, desk-check names, smoke-run result, and the files-added list.

---

## 7. Risks & Open Items

- **Edge-cache coverage** — if completeness-viable nodes are missing from `edge_features_cache.json`, the intersection shrinks. The node report surfaces this immediately (step 7); the fallback is regenerating the cache for missing pairs with the existing `features.compute_pairwise_edge_features`, not weakening the assert.
- **Study period** — `start_date`/`end_date` above are placeholders aligned to the quality track; confirm the exact window used in Chapter 5 so the thesis chapters share a data narrative.
- **Champion selection** — Stage B/C keys are set manually after Stage A (step 9). This is deliberate (a human decision, logged in git), not automated.
- **GPU memory** — B=32 × N≈80 × FC graph (≈6.4k edges) is small by RIoT standards (which batches B·T graphs); no AMP needed, FP32 throughout keeps the loss path simple.
