// Types for ui_payload.json schema produced by export_ui_payload.py

export interface TimestepEntry {
  t: number;
  prediction: number;
  uncertainty: number;
}

export interface NodeEntry {
  id: number;
  location_name: string;
  lat: number;
  lon: number;
  is_simulated: boolean;
  timeseries: TimestepEntry[];
}

export interface EdgeEntry {
  source: number;
  target: number;
  attention_sequence: number[];
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

export interface UIPayload {
  metadata: PayloadMetadata;
  nodes: NodeEntry[];
  edges: EdgeEntry[];
}
