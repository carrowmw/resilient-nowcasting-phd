// Types for master_index.json and nodes/node_{id}.json produced by export_ui_payload.py

// ---------------------------------------------------------------------------
// Shared primitives
// ---------------------------------------------------------------------------

export interface TimestepEntry {
  t: number;
  prediction: number; // μ (ZINB mean)
  uncertainty: number; // σ (std_dev)
}

export interface PayloadMetadata {
  num_nodes: number;
  num_real_nodes: number;
  num_simulated_nodes: number;
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
  is_simulated: boolean;
  timeseries: TimestepEntry[];
  semantic_tags?: string[];   // flattened POI/land-use tags; absent in pre-sharding payloads
  two_extremes?: TwoExtremes; // absent in payloads generated before Iteration 4
}

export interface MasterIndex {
  metadata: PayloadMetadata;
  nodes: NodeEntry[];
  edges: EdgeEntry[];
  pit_histogram?: PITBin[];                          // absent in pre-Iteration-4 payloads
  zero_inflation_reliability?: ZeroInflationBin[];   // absent in pre-Iteration-4 payloads
}

// Backward-compat alias used by MapVisualizer / NodeInspector
export type UIPayload = MasterIndex;

// ---------------------------------------------------------------------------
// nodes/node_{id}.json schema  (deep per-node analytics)
// ---------------------------------------------------------------------------

export interface ZINBEntry {
  t: number;
  mu: number;
  theta: number;      // dispersion parameter
  std_dev: number;
  pi: number;         // zero-inflation probability ∈ (0,1) — absent in pre-Iteration-4b payloads
  is_missing: boolean; // true = this timestep was artificially masked before imputation
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
  is_simulated: boolean;
  zinb_series: ZINBEntry[];
  ground_truth_series: (number | null)[];
  imputed_series: number[];
  mask_series: boolean[];
  edges_1hop: HopEdge[];
  two_extremes?: TwoExtremes;       // absent in payloads generated before Iteration 4
  ego_graph_edges?: AttentionEdge[]; // absent in payloads generated before Iteration 4b
}
