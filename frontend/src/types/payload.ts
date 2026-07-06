// Types for master_index.json and nodes/node_{id}.json produced by export_ui_payload.py

// ---------------------------------------------------------------------------
// Shared primitives
// ---------------------------------------------------------------------------

/**
 * Node display taxonomy (matches export_ui_payload.py node_role):
 *   observed  — normal sensor: prediction + ground truth
 *   holdout   — real sensor masked throughout training: genuine zero-shot
 *               prediction WITH ground truth available for comparison
 *   simulated — zero-baseline synthetic node: prediction, no ground truth
 */
export type NodeRole = "observed" | "holdout" | "simulated";

export interface TimestepEntry {
  t: number;
  prediction: number; // μ (point forecast)
  uncertainty: number | null; // σ (std_dev); null for models without a ZINB head (baselines)
}

export interface PayloadMetadata {
  num_nodes: number;
  num_real_nodes: number;
  num_simulated_nodes: number;
  num_holdout_nodes?: number;            // inductive zero-shot nodes; absent in pre-harness payloads
  holdout_manifest_sha?: string | null;  // cross-model comparability key; null when no holdout
  timesteps: number;
  seq_len: number;
  veh_class: string;
  dir: string;
  start_date: string;
  end_date: string;
  // --- ARTIFACT TWINNING ---
  wandb_run_id?: string;
  git_commit_hash?: string;
  hyperparameters?: any;
}

export interface EdgeEntry {
  source: number;
  target: number;
  attention_sequence: number[];
}

// ---------------------------------------------------------------------------
// ZINB Calibration types  (Iteration 4)
// ---------------------------------------------------------------------------

/** One bin of the Probability Integral Transform histogram. */
export interface PITBin {
  bin_center: number; // midpoint of the [0,1] bin (e.g. 0.05, 0.15 …)
  density: number;    // fraction of PIT values in this bin; uniform at ~0.1 when well-calibrated
}

/** One bin of the zero-inflation reliability diagram. */
export interface ZeroInflationBin {
  pi_center: number;               // midpoint of the predicted-π bin
  observed_zero_fraction: number;  // actual fraction of true zeros in this bin
  count: number;                   // number of observations contributing to this bin
}

/** Per-node Two-Extremes missingness bracket. */
export interface TwoExtremes {
  mae_drop_nan: number | null;  // MAE after dropping NaN ground-truth rows; null when none exist
  rmse_drop_nan: number | null;
  mae_fill_nan: number;         // MAE after filling NaN ground-truth rows with 0
  rmse_fill_nan: number;
}

// ---------------------------------------------------------------------------
// master_index.json schema
// ---------------------------------------------------------------------------

export interface NodeEntry {
  id: number;
  location_name: string;
  lat: number;
  lon: number;
  node_role?: NodeRole;       // absent in pre-harness payloads → derive from is_simulated
  is_simulated: boolean;
  timeseries: TimestepEntry[];
  semantic_tags?: string[];   // flattened POI/land-use tags; absent in pre-sharding payloads
  two_extremes?: TwoExtremes; // absent in payloads generated before Iteration 4
}

export interface MasterIndex {
  metadata: PayloadMetadata;
  nodes: NodeEntry[];
  edges: EdgeEntry[];
  // null (not just absent) for models without the corresponding capability:
  // a baseline reports N/A ZINB calibration and no attention leaderboard.
  pit_histogram?: PITBin[] | null;                          // null for models with no ZINB head
  zero_inflation_reliability?: ZeroInflationBin[] | null;   // null for models with no zero-inflation head
  global_imputation_burden?: number[];
  global_healing_quality?: number[];
  attention_entropy_series?: number[];                      // empty for graph-less baselines
  tripwire_audit_trail?: Record<string, any>[];
  alpha_node_leaderboard?: Record<string, any>[] | null;    // null for graph-less baselines
}

// Backward-compat alias used by MapVisualizer / NodeInspector
export type UIPayload = MasterIndex;

// ---------------------------------------------------------------------------
// nodes/node_{id}.json schema  (deep per-node analytics)
// ---------------------------------------------------------------------------

export interface ZINBEntry {
  t: number;
  mu: number;
  theta: number | null;   // dispersion; null for models without a ZINB uncertainty head
  std_dev: number | null;  // σ; null for models without a ZINB uncertainty head
  pi: number | null;       // zero-inflation probability ∈ (0,1); null for models without the head
  is_missing: boolean;     // true = this timestep was artificially masked before imputation
}

/** One directed GAT attention edge in the localized ego-graph. */
export interface AttentionEdge {
  source_id: number;
  target_id: number;
  direction: "incoming" | "outgoing";
  weight_sequence: number[];  // per-timestep GAT attention weights, length = num_timesteps
  source_lat: number;
  source_lon: number;
  target_lat: number;
  target_lon: number;
}

export interface HopEdge {
  neighbor_id: number;
  direction: "outgoing" | "incoming";
  attention_sequence: number[];
}

export interface NodeDetailPayload {
  id: number;
  location_name: string;
  lat: number;
  lon: number;
  node_role?: NodeRole;             // absent in pre-harness payloads → derive from is_simulated
  is_simulated: boolean;
  zinb_series: ZINBEntry[];         // theta/std_dev/pi are null for graph-less/ZINB-less baselines
  ground_truth_series: (number | null)[];
  imputed_series: number[];
  mask_series: boolean[];
  edges_1hop: HopEdge[];            // empty for graph-less baselines
  two_extremes?: TwoExtremes;       // absent in payloads generated before Iteration 4
  ego_graph_edges?: AttentionEdge[]; // empty/absent for graph-less baselines
}
