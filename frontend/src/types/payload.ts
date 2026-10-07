// RIoT v5 payload schema (schema_version 5), produced by src/export_paginated_payload.py.
//
//   runs/run_registry.json                 → RunRegistry          (types/registry.ts)
//   runs/{run_id}/spatial.json             → SpatialIndex         (types/registry.ts)
//   runs/{run_id}/{date}/page.json         → TemporalPage         (this file)
//   runs/{run_id}/{date}/nodes/node_{id}.json → NodeDetailPayload (this file)
//
// Conventions
//   * All counts are RAW vehicle/person counts per 15-minute slot (never scaled).
//   * Per-timestep arrays have length TemporalPage.timesteps (96 for a full day);
//     index t is the forecast for TemporalPage.timestamps[t].
//   * null means "not available" — a missing observation (truth) or a capability
//     the model does not have (e.g. outage_q for a ZINB model). Never zero-filled.
//   * Each forecast is a one-step-ahead prediction made from the preceding
//     seq_len slots; holdout nodes are blanked for the whole look-back window
//     (genuine zero-shot / kriging forecasts with ground truth available).

export const PAYLOAD_SCHEMA_VERSION = 5;

// ---------------------------------------------------------------------------
// Shared primitives
// ---------------------------------------------------------------------------

/**
 *   observed — the model sees this sensor's history (when it is online)
 *   holdout  — inductive zero-shot sensor: never trained on and blanked at
 *              inference; ground truth is still shown for comparison
 */
export type NodeRole = "observed" | "holdout";

export type Likelihood = "zinb" | "nb" | "ztnb" | "outage_mixture";

/** What the model can genuinely produce (harness/contract.py Capabilities). */
export interface Capabilities {
  uncertainty: boolean;     // predictive distribution → lo/hi interval, PIT, CRPS
  zero_inflation: boolean;  // genuine ZINB pi
  imputation: boolean;      // imputed_last in node detail
  kriging: boolean;
  graph: boolean;           // edge_attention / alpha leaderboard
  outage: boolean;          // outage_q = P(sensor offline) (outage mixture)
  quantiles: boolean;       // q50 / q90 quantile heads
}

/** Forecast accuracy and real-gap behaviour over a set of target slots. */
export interface WindowMetrics {
  n_targets: number;               // target slots scored (node × timestep)
  n_observed: number;              // of which had ground truth
  observed_frac: number;           // n_observed / n_targets
  obs_all_rmse: number | null;
  obs_all_mae: number | null;
  obs_trans_rmse: number | null;   // observed (transductive) nodes only
  obs_trans_mae: number | null;
  obs_ind_rmse: number | null;     // holdout (inductive, zero-shot) nodes only
  obs_ind_mae: number | null;
  bias: number | null;             // mean(point − truth) on observed targets
  interval80_coverage: number | null; // P(lo ≤ truth ≤ hi); null without uncertainty
  q50_coverage: number | null;     // P(truth ≤ q50 head); null without quantile heads
  q90_coverage: number | null;     // P(truth ≤ q90 head)
  crps: number | null;             // mean CRPS on observed targets; null without uncertainty
  collapse_rate: number | null;    // P(point < 1 | truth missing, own profile > 10)
  real_gap_mass_ratio: number | null; // Σ point / Σ profile over missing targets (1 = no collapse)
}

/** One bin of the mid-PIT histogram (10 bins; uniform ≈ 0.1 when calibrated). */
export interface PITBin {
  bin_center: number;
  density: number;
}

/** Reliability bin: predicted probability vs observed event frequency. */
export interface ReliabilityBin {
  bin_center: number;     // midpoint of the predicted-probability bin
  mean_predicted: number; // mean predicted probability inside the bin
  observed_frac: number;  // observed event frequency inside the bin
  count: number;
}

export interface Calibration {
  pit_histogram: PITBin[] | null;               // null without uncertainty
  /** p_zero vs event "slot reads zero or is missing". null when the model has no p_zero. */
  p_zero_reliability: ReliabilityBin[] | null;
  /** outage_q vs event "slot is missing". null unless capabilities.outage. */
  outage_reliability: ReliabilityBin[] | null;
}

export interface TripwireEntry {
  timestep: number;
  event_type?: string;
  [key: string]: unknown;
}

export interface AlphaNodeEntry {
  node_id: number;
  location_name: string;
  total_outgoing_attention: number; // Σ over timesteps and outgoing edges
}

/** Per-node accuracy over one day (observed targets only). */
export interface NodeDayMetrics {
  n_observed: number;
  rmse: number | null;
  mae: number | null;
  bias: number | null;
  interval80_coverage: number | null;
  peak_point: number;
  mean_point: number;
  mean_truth: number | null;
}

// ---------------------------------------------------------------------------
// page.json — one day of one run
// ---------------------------------------------------------------------------

export interface NodeSeries {
  point: number[];                 // scored point forecast
  lo: number[] | null;             // 10th percentile of the predictive count distribution
  hi: number[] | null;             // 90th percentile (lo/hi = 80% predictive interval)
  q50: number[] | null;            // quantile-head forecasts (capabilities.quantiles)
  q90: number[] | null;
  p_zero: number[] | null;         // P(slot reads zero / is empty)
  outage_q: number[] | null;       // P(sensor outage) (capabilities.outage)
  profile: number[];               // node's seasonal slot-of-week prior (train slice)
  truth: (number | null)[];        // observed count; null = sensor offline
  gap_steps: number[];             // slots since the sensor last reported (0 = reporting)
}

export interface EdgeAttention {
  source: number;
  target: number;
  /** GAT attention at the last look-back step of each forecast, mean over heads.
   *  null where the edge was dropped because its source was missing / blanked. */
  weights: (number | null)[];
}

export interface NetworkSeries {
  observed_frac: number[];                   // fraction of nodes with ground truth
  mean_point: number[];
  mean_truth: (number | null)[];             // over observed nodes
  real_gap_mass_ratio: (number | null)[];    // per-timestep Σ point / Σ profile over offline nodes
  attention_entropy: (number | null)[] | null; // mean per-target-node entropy; null without graph
}

export interface TemporalPage {
  schema_version: 5;
  date: string;              // "2026-05-01"
  timesteps: number;
  seq_len: number;
  step_minutes: number;
  timestamps: string[];      // ISO local timestamps of the forecast targets
  node_series: Record<string, NodeSeries>;   // keyed by String(node.id)
  node_metrics: Record<string, NodeDayMetrics>;
  network: NetworkSeries;
  day_metrics: WindowMetrics;
  calibration: Calibration;
  edge_attention: EdgeAttention[] | null;    // null without graph
  alpha_node_leaderboard: AlphaNodeEntry[] | null;
  tripwire_audit_trail: TripwireEntry[];
}

// ---------------------------------------------------------------------------
// nodes/node_{id}.json — deep per-node detail for one day
// ---------------------------------------------------------------------------

export interface DistributionSeries {
  likelihood: Likelihood;
  mu: number[];               // NB mean (counterfactual count for the outage mixture)
  theta: number[] | null;     // NB dispersion
  pi: number[] | null;        // ZINB zero-inflation probability
  q: number[] | null;         // outage-mixture outage probability
}

/** Directed GAT attention edge in the node's ego graph. */
export interface AttentionEdge {
  source_id: number;
  target_id: number;
  direction: "incoming" | "outgoing";
  weight_sequence: (number | null)[];
  source_lat: number;
  source_lon: number;
  target_lat: number;
  target_lon: number;
}

export interface HopEdge {
  neighbor_id: number;
  direction: "incoming" | "outgoing";
  attention_sequence: (number | null)[];
}

export interface NodeDetailPayload {
  schema_version: 5;
  id: number;
  location_name: string;
  lat: number;
  lon: number;
  node_role: NodeRole;
  date: string;
  distribution: DistributionSeries;
  /** Self-healing view: model reconstruction of the LAST look-back slot of each
   *  forecast window vs what the sensor actually reported there. */
  imputed_last: number[] | null;
  input_truth_last: (number | null)[];
  input_missing_frac: number[];   // fraction of the look-back window hidden from the model
  edges_1hop: HopEdge[];
  ego_graph_edges: AttentionEdge[];
  metrics: NodeDayMetrics;
}
