// Types for the multi-run paginated payload layout (Phase 1 / Phase 3).
//
//   runs/run_registry.json          → RunRegistry
//   runs/{run_id}/spatial.json       → SpatialIndex   (nodes + topology + metadata)
//   runs/{run_id}/{date}/page.json   → TemporalPage   (per-day timeseries + attention)
//   runs/{run_id}/{date}/nodes/*     → NodeDetailPayload (unchanged, see payload.ts)

import type {
  TimestepEntry,
  EdgeEntry,
  PITBin,
  ZeroInflationBin,
  NodeRole,
  TripwireEntry,
  AlphaNodeEntry,
} from "@/types/payload";

// ---------------------------------------------------------------------------
// run_registry.json
// ---------------------------------------------------------------------------

export interface RunRegistryEntry {
  run_id: string;
  wandb_run_id: string;
  wandb_run_name: string;
  git_commit: string;
  generated_at: string;
  analysis_period: {
    start_date: string | null;
    end_date: string | null;
  };
  train_config: Record<string, unknown>;
  metrics_summary: {
    val_rmse?: number;
    val_mae?: number;
    val_zero_acc?: number;
    [k: string]: number | undefined;
  };
  available_dates: string[];
  num_nodes: number;
  num_simulated_nodes: number;
}

export interface RunRegistry {
  default_run_id: string | null;
  runs: RunRegistryEntry[];
}

// ---------------------------------------------------------------------------
// spatial.json (per run)
// ---------------------------------------------------------------------------

export interface SpatialNode {
  id: number;
  location_name: string;
  lat: number;
  lon: number;
  node_role?: NodeRole;   // absent in pre-harness payloads → derive from is_simulated
  is_simulated: boolean;
  semantic_tags: string[];
}

export interface EdgeTopology {
  source: number;
  target: number;
}

export interface SpatialMetadata {
  run_id: string;
  wandb_run_id: string;
  git_commit_hash: string;
  generated_at: string;
  hyperparameters: Record<string, unknown>;
  num_nodes: number;
  num_real_nodes: number;
  num_simulated_nodes: number;
  num_holdout_nodes?: number;
  holdout_manifest_sha?: string | null;
  available_dates: string[];
  veh_class: string;
  dir: string;
  seq_len: number;
  metrics_summary?: Record<string, number>;
}

export interface SpatialIndex {
  metadata: SpatialMetadata;
  nodes: SpatialNode[];
  edge_topology: EdgeTopology[];
}

// ---------------------------------------------------------------------------
// page.json (per day)
// ---------------------------------------------------------------------------

export interface TemporalPage {
  date: string;
  timesteps: number;
  seq_len: number;
  // Per-node timeseries keyed by stringified node id.
  node_timeseries: Record<string, TimestepEntry[]>;
  // Per-edge attention (same shape as the legacy EdgeEntry). Empty for baselines.
  edge_attention: EdgeEntry[];
  pit_histogram?: PITBin[] | null;                        // null for models with no ZINB head
  zero_inflation_reliability?: ZeroInflationBin[] | null; // null for models with no zero-inflation head
  global_imputation_burden?: number[];
  global_healing_quality?: number[];
  attention_entropy_series?: number[];                    // empty for graph-less baselines
  tripwire_audit_trail?: TripwireEntry[];
  alpha_node_leaderboard?: AlphaNodeEntry[] | null; // null for graph-less baselines
}
