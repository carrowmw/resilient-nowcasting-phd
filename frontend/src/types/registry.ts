// RIoT v5 run registry + per-run spatial index (schema_version 5).
//
//   runs/run_registry.json   → RunRegistry
//   runs/{run_id}/spatial.json → SpatialIndex
// Per-day payloads are in types/payload.ts.

import type { Capabilities, Likelihood, NodeRole, WindowMetrics } from "@/types/payload";

/** Training-time held-out TEST metrics (docs/experiments/riot_v5 results), plus
 *  the validation RMSE the top-k selection ranks by. null = not produced. */
export interface TestMetrics {
  val_obs_all_rmse: number | null;
  test_obs_all_rmse: number | null;
  test_obs_all_mae: number | null;
  test_obs_trans_rmse: number | null;
  test_obs_ind_rmse: number | null;
  test_peak_rmse: number | null;
  test_so_12_forecast_rmse: number | null;   // 12-step (3 h) synthetic outage
  test_crps: number | null;
  test_collapse_rate: number | null;
  test_real_gap_mass_ratio: number | null;
  test_q90_coverage: number | null;
  test_pzero_missing_auc: number | null;
}

export interface RegistryPlacement {
  rank: number;              // 1 = reference run, then by val_obs_all_rmse
  is_reference: boolean;     // the full v5 configuration (v5_full-s42)
  selection_rule: string;
}

export interface RunInfo {
  run_id: string;            // e.g. "v5_full-s42"
  experiment: string;        // e.g. "v5_full"
  group: string;             // experiment phase, e.g. "p7_v5_suite"
  description: string;       // one-line ablation description
  model_type: string;        // "gnn"
  forecast_likelihood: Likelihood;
  quantile_levels: number[]; // [] without quantile heads
  capabilities: Capabilities;
  seq_len: number;
  wandb_run_id: string | null;
  wandb_artifact: string | null;
  git_commit_hash: string | null;
  generated_at: string;      // ISO UTC
}

// ---------------------------------------------------------------------------
// run_registry.json
// ---------------------------------------------------------------------------

export interface RunRegistryEntry extends RunInfo {
  registry: RegistryPlacement;
  analysis_period: { start_date: string; end_date: string };  // end exclusive
  available_dates: string[];
  num_nodes: number;
  num_holdout_nodes: number;
  num_ghost_nodes: number;   // nodes with no observation anywhere in the analysis window
  holdout_manifest_sha: string | null;
  test_metrics: TestMetrics;
  window_metrics: WindowMetrics;   // exporter-computed over the whole analysis window
  hyperparameters: Record<string, unknown>;
}

export interface RunRegistry {
  schema_version: 5;
  generated_at: string;
  default_run_id: string | null;
  selection_rule: string;
  runs: RunRegistryEntry[];        // ordered by registry.rank
}

// ---------------------------------------------------------------------------
// spatial.json (per run)
// ---------------------------------------------------------------------------

export interface SpatialNode {
  id: number;
  location_name: string;
  lat: number;
  lon: number;
  node_role: NodeRole;
  semantic_tags: string[];
  /** Observed target slots over the whole analysis window. 0 = "ghost" node:
   *  forecast every slot but never scorable (all accuracy metrics null). */
  window_observed: number;
  /** Last observation at or before the window end (ISO); null = never in the dataset. */
  last_observed: string | null;
}

export interface EdgeTopology {
  source: number;
  target: number;
}

export interface SpatialMetadata extends RunInfo {
  schema_version: 5;
  pipeline_version: 5;
  registry: RegistryPlacement;
  hyperparameters: Record<string, unknown>;
  num_nodes: number;
  num_holdout_nodes: number;
  num_ghost_nodes: number;   // nodes with no observation anywhere in the analysis window
  holdout_manifest_sha: string | null;
  available_dates: string[];
  start_date: string;
  end_date: string;          // exclusive
  veh_class: string;
  dir: string;
  step_minutes: number;
  test_metrics: TestMetrics;
  window_metrics: WindowMetrics;
}

export interface SpatialIndex {
  metadata: SpatialMetadata;
  nodes: SpatialNode[];
  edge_topology: EdgeTopology[];   // learned graph; [] for graph_mode=identity
}
