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
}

export interface EdgeEntry {
  source: number;
  target: number;
  attention_sequence: number[];
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
  semantic_tags?: string[]; // flattened POI/land-use tags; absent in pre-sharding payloads
}

export interface MasterIndex {
  metadata: PayloadMetadata;
  nodes: NodeEntry[];
  edges: EdgeEntry[];
}

// Backward-compat alias used by MapVisualizer / NodeInspector
export type UIPayload = MasterIndex;

// ---------------------------------------------------------------------------
// nodes/node_{id}.json schema  (deep per-node analytics)
// ---------------------------------------------------------------------------

export interface ZINBEntry {
  t: number;
  mu: number;
  theta: number; // dispersion parameter
  std_dev: number;
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
}
