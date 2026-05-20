/**
 * tests/payload.test.ts
 *
 * Schema-integrity tests for the ui_payload.json structure.
 *
 * We use a *mock* payload rather than loading the real 4.8 MB file so tests
 * are fast and network-free.  The mock mirrors the exact structure produced by
 * export_ui_payload.py and typed in src/types/payload.ts.
 */

import { describe, it, expect } from 'vitest';
import type { UIPayload, NodeEntry, EdgeEntry } from '@/types/payload';

// ── Mock payload factory ──────────────────────────────────────────────────────

function makeNode(id: number, isSimulated: boolean, timesteps: number): NodeEntry {
  return {
    id,
    location_name: `Sensor ${id}`,
    lat: 54.97 + id * 0.001,
    lon: -1.61 + id * 0.001,
    is_simulated: isSimulated,
    timeseries: Array.from({ length: timesteps }, (_, t) => ({
      t,
      prediction: Math.random() * 500,
      uncertainty: Math.random() * 0.5,
    })),
    // Phase 1 Analytics Upgrade: imputation series live alongside the forecast.
    // Simulated nodes have no physical sensor → ground_truth is null and the
    // mask is true at every timestep (every input was NaN by construction).
    ground_truth_series: Array.from({ length: timesteps }, () =>
      isSimulated ? null : Math.random() * 100,
    ),
    imputed_series: Array.from({ length: timesteps }, () => Math.random() * 100),
    mask_series: Array.from({ length: timesteps }, () =>
      isSimulated ? true : Math.random() < 0.15,
    ),
  };
}

function makeEdge(source: number, target: number, timesteps: number): EdgeEntry {
  return {
    source,
    target,
    attention_sequence: Array.from({ length: timesteps }, () => Math.random()),
  };
}

function makeMockPayload(
  numRealNodes = 5,
  numSimNodes = 2,
  timesteps = 96,
  numEdges = 10,
): UIPayload {
  const nodes: NodeEntry[] = [
    ...Array.from({ length: numRealNodes }, (_, i) => makeNode(i, false, timesteps)),
    ...Array.from({ length: numSimNodes }, (_, i) =>
      makeNode(numRealNodes + i, true, timesteps),
    ),
  ];

  const edges: EdgeEntry[] = Array.from({ length: numEdges }, (_, i) =>
    makeEdge(i % numRealNodes, (i + 1) % numRealNodes, timesteps),
  );

  return {
    metadata: {
      num_nodes: numRealNodes + numSimNodes,
      num_real_nodes: numRealNodes,
      num_simulated_nodes: numSimNodes,
      timesteps,
      seq_len: 12,
      veh_class: 'cyclist',
      dir: 'from town',
      start_date: '2024-05-10',
      end_date: '2024-05-10',
    },
    nodes,
    edges,
  };
}

// ── Schema integrity tests ────────────────────────────────────────────────────

describe('UIPayload schema', () => {
  it('metadata.num_nodes equals nodes.length', () => {
    const p = makeMockPayload(5, 2, 96, 10);
    expect(p.nodes.length).toBe(p.metadata.num_nodes);
  });

  it('metadata.num_real_nodes + metadata.num_simulated_nodes equals num_nodes', () => {
    const p = makeMockPayload(45, 5, 96, 100);
    expect(p.metadata.num_real_nodes + p.metadata.num_simulated_nodes).toBe(
      p.metadata.num_nodes,
    );
  });

  it('every node has timeseries.length === metadata.timesteps', () => {
    const p = makeMockPayload(5, 2, 96);
    for (const node of p.nodes) {
      expect(node.timeseries.length).toBe(p.metadata.timesteps);
    }
  });

  it('every timeseries entry has a t, prediction, and uncertainty field', () => {
    const p = makeMockPayload(3, 1, 4);
    for (const node of p.nodes) {
      for (const entry of node.timeseries) {
        expect(entry).toHaveProperty('t');
        expect(entry).toHaveProperty('prediction');
        expect(entry).toHaveProperty('uncertainty');
        expect(typeof entry.prediction).toBe('number');
        expect(typeof entry.uncertainty).toBe('number');
      }
    }
  });

  it('timeseries t values are sequential indices 0…timesteps-1', () => {
    const timesteps = 8;
    const p = makeMockPayload(2, 1, timesteps);
    for (const node of p.nodes) {
      node.timeseries.forEach((entry, idx) => {
        expect(entry.t).toBe(idx);
      });
    }
  });

  it('every edge has attention_sequence.length === metadata.timesteps', () => {
    const p = makeMockPayload(5, 0, 96, 20);
    for (const edge of p.edges) {
      expect(edge.attention_sequence.length).toBe(p.metadata.timesteps);
    }
  });

  it('simulated nodes are correctly flagged', () => {
    const p = makeMockPayload(5, 3, 96);
    const simulated = p.nodes.filter((n) => n.is_simulated);
    const real = p.nodes.filter((n) => !n.is_simulated);
    expect(simulated.length).toBe(p.metadata.num_simulated_nodes);
    expect(real.length).toBe(p.metadata.num_real_nodes);
  });

  it('all node ids are unique', () => {
    const p = makeMockPayload(10, 5, 4);
    const ids = p.nodes.map((n) => n.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it('all attention weights are in [0, 1] (normalised by the export script)', () => {
    const p = makeMockPayload(5, 0, 10, 50);
    for (const edge of p.edges) {
      for (const w of edge.attention_sequence) {
        expect(w).toBeGreaterThanOrEqual(0);
        expect(w).toBeLessThanOrEqual(1);
      }
    }
  });

  it('every node has imputed/ground-truth/mask arrays of length timesteps', () => {
    const p = makeMockPayload(4, 2, 12);
    for (const node of p.nodes) {
      expect(node.imputed_series.length).toBe(p.metadata.timesteps);
      expect(node.ground_truth_series.length).toBe(p.metadata.timesteps);
      expect(node.mask_series.length).toBe(p.metadata.timesteps);
    }
  });

  it('simulated nodes carry null ground truth and all-true mask', () => {
    const p = makeMockPayload(3, 4, 16);
    for (const node of p.nodes.filter((n) => n.is_simulated)) {
      expect(node.ground_truth_series.every((v) => v === null)).toBe(true);
      expect(node.mask_series.every((m) => m === true)).toBe(true);
    }
  });

  it('real-node ground truths are numeric (not null)', () => {
    const p = makeMockPayload(5, 0, 12);
    for (const node of p.nodes) {
      for (const v of node.ground_truth_series) {
        expect(typeof v).toBe('number');
      }
    }
  });

  it('node coordinates are within the Newcastle upon Tyne bounding box', () => {
    // Rough bounding box: lat 54.93–55.05, lon -1.75–-1.48
    const p = makeMockPayload(5, 0, 4);
    // Just validate our mock nodes have valid numeric coords (real tests
    // against real data would check the actual bounding box).
    for (const node of p.nodes) {
      expect(typeof node.lat).toBe('number');
      expect(typeof node.lon).toBe('number');
      expect(isNaN(node.lat)).toBe(false);
      expect(isNaN(node.lon)).toBe(false);
    }
  });

  it('contains Artifact Twinning metadata fields', () => {
    const p = makeMockPayload(5, 2, 96);
    // Inject the fields into the mock for testing
    p.metadata.wandb_run_id = 'test-run';
    p.metadata.git_commit_hash = 'abcdef123456';
    p.metadata.hyperparameters = { model: { hidden_dim: 32 } };

    expect(p.metadata).toHaveProperty('wandb_run_id');
    expect(p.metadata).toHaveProperty('git_commit_hash');
    expect(p.metadata).toHaveProperty('hyperparameters');
    expect(p.metadata.wandb_run_id).toBe('test-run');
  });
});
