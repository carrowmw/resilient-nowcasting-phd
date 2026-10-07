/**
 * tests/fixtures/v5.ts
 *
 * Typed, deterministic builders for the v5 payload contract
 * (src/types/payload.ts + src/types/registry.ts). Every builder returns a
 * fully-typed object, so a contract change breaks the fixtures at compile time.
 *
 * Capability gating mirrors the exporter: fields a model cannot produce are
 * null, never zero-filled.
 */

import type {
  AttentionEdge,
  Calibration,
  Capabilities,
  EdgeAttention,
  Likelihood,
  NodeDayMetrics,
  NodeDetailPayload,
  NodeSeries,
  TemporalPage,
  WindowMetrics,
} from '@/types/payload';
import type {
  RunRegistry,
  RunRegistryEntry,
  SpatialIndex,
  SpatialNode,
  TestMetrics,
} from '@/types/registry';

// ── Deterministic PRNG (mulberry32) ──────────────────────────────────────────

export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const round = (v: number, d = 3) => Math.round(v * 10 ** d) / 10 ** d;

// ── Capabilities / metrics ───────────────────────────────────────────────────

export const FULL_CAPABILITIES: Capabilities = {
  uncertainty: true,
  zero_inflation: false,
  imputation: true,
  kriging: true,
  graph: true,
  outage: true,
  quantiles: true,
};

/** A point-forecast baseline: no distribution, no graph, no heads. */
export const BASELINE_CAPABILITIES: Capabilities = {
  uncertainty: false,
  zero_inflation: false,
  imputation: false,
  kriging: false,
  graph: false,
  outage: false,
  quantiles: false,
};

export function makeCapabilities(overrides: Partial<Capabilities> = {}): Capabilities {
  return { ...FULL_CAPABILITIES, ...overrides };
}

export function makeWindowMetrics(
  caps: Capabilities = FULL_CAPABILITIES,
  overrides: Partial<WindowMetrics> = {},
): WindowMetrics {
  return {
    n_targets: 960,
    n_observed: 816,
    observed_frac: 0.85,
    obs_all_rmse: 21.4,
    obs_all_mae: 13.2,
    obs_trans_rmse: 20.1,
    obs_trans_mae: 12.5,
    obs_ind_rmse: 27.9,
    obs_ind_mae: 17.0,
    bias: -1.3,
    interval80_coverage: caps.uncertainty ? 0.78 : null,
    q50_coverage: caps.quantiles ? 0.52 : null,
    q90_coverage: caps.quantiles ? 0.88 : null,
    crps: caps.uncertainty ? 9.8 : null,
    collapse_rate: 0.04,
    real_gap_mass_ratio: 0.93,
    ...overrides,
  };
}

export function makeTestMetrics(
  caps: Capabilities = FULL_CAPABILITIES,
  overrides: Partial<TestMetrics> = {},
): TestMetrics {
  return {
    val_obs_all_rmse: 20.5,
    test_obs_all_rmse: 22.1,
    test_obs_all_mae: 13.9,
    test_obs_trans_rmse: 21.0,
    test_obs_ind_rmse: 28.4,
    test_peak_rmse: 41.2,
    test_so_12_forecast_rmse: 30.3,
    test_crps: caps.uncertainty ? 10.2 : null,
    test_collapse_rate: 0.05,
    test_real_gap_mass_ratio: 0.91,
    test_q90_coverage: caps.quantiles ? 0.87 : null,
    test_pzero_missing_auc: caps.outage ? 0.81 : null,
    ...overrides,
  };
}

// ── Run info / spatial ───────────────────────────────────────────────────────

export interface RunOpts {
  runId?: string;
  rank?: number;
  isReference?: boolean;
  likelihood?: Likelihood;
  capabilities?: Capabilities;
  dates?: string[];
  numObserved?: number;
  numHoldout?: number;
  seqLen?: number;
  /** Node ids with no observation in the analysis window (all-null truth). */
  ghostIds?: number[];
}

const DEFAULT_DATES = ['2026-05-01', '2026-05-02', '2026-05-03'];

function runInfo(o: Required<RunOpts>) {
  return {
    run_id: o.runId,
    experiment: o.runId.replace(/-s\d+$/, ''),
    group: 'p7_v5_suite',
    description: `${o.runId} ablation`,
    model_type: 'gnn',
    forecast_likelihood: o.likelihood,
    quantile_levels: o.capabilities.quantiles ? [0.5, 0.9] : [],
    capabilities: o.capabilities,
    seq_len: o.seqLen,
    wandb_run_id: 'abc123',
    wandb_artifact: null,
    git_commit_hash: '00888eb0123456789',
    generated_at: '2026-10-01T00:00:00Z',
  };
}

function withDefaults(opts: RunOpts): Required<RunOpts> {
  return {
    runId: opts.runId ?? 'v5_full-s42',
    rank: opts.rank ?? 1,
    isReference: opts.isReference ?? (opts.rank ?? 1) === 1,
    likelihood: opts.likelihood ?? 'outage_mixture',
    capabilities: opts.capabilities ?? FULL_CAPABILITIES,
    dates: opts.dates ?? DEFAULT_DATES,
    numObserved: opts.numObserved ?? 4,
    numHoldout: opts.numHoldout ?? 2,
    seqLen: opts.seqLen ?? 12,
    ghostIds: opts.ghostIds ?? [],
  };
}

const HYPERPARAMETERS = {
  model: { hidden_dim: 64, num_layers: 3, num_heads: 4, embedding_dim: 32, dropout: 0.1 },
  training: { epochs: 40, batch_size: 32, learning_rate: 0.001 },
};

export function makeSpatial(opts: RunOpts = {}): SpatialIndex {
  const o = withDefaults(opts);
  const n = o.numObserved + o.numHoldout;
  const nodes: SpatialNode[] = Array.from({ length: n }, (_, i) => ({
    id: i,
    location_name: `NclSensor${i}`,
    lat: 54.97 + i * 0.001,
    lon: -1.61 + i * 0.001,
    node_role: i < o.numObserved ? 'observed' : 'holdout',
    semantic_tags: i % 2 === 0 ? ['school'] : [],
    window_observed: o.ghostIds.includes(i) ? 0 : 90 * o.dates.length,
    last_observed: o.ghostIds.includes(i) ? (i % 2 === 0 ? '2025-11-03T08:15:00+00:00' : null) : `${o.dates[o.dates.length - 1]}T23:45:00+00:00`,
  }));
  const end = new Date(`${o.dates[o.dates.length - 1]}T00:00:00Z`);
  end.setUTCDate(end.getUTCDate() + 1);
  return {
    metadata: {
      ...runInfo(o),
      schema_version: 5,
      pipeline_version: 5,
      registry: { rank: o.rank, is_reference: o.isReference, selection_rule: 'reference + top-k by val_obs_all_rmse' },
      hyperparameters: HYPERPARAMETERS,
      num_nodes: n,
      num_holdout_nodes: o.numHoldout,
      num_ghost_nodes: o.ghostIds.length,
      holdout_manifest_sha: 'deadbeefcafe',
      available_dates: o.dates,
      start_date: o.dates[0],
      end_date: end.toISOString().slice(0, 10),
      veh_class: 'person',
      dir: 'all',
      step_minutes: 15,
      test_metrics: makeTestMetrics(o.capabilities),
      window_metrics: makeWindowMetrics(o.capabilities),
    },
    nodes,
    edge_topology: o.capabilities.graph
      ? nodes.slice(1).map((nd) => ({ source: 0, target: nd.id }))
      : [],
  };
}

export function makeRegistryEntry(opts: RunOpts = {}): RunRegistryEntry {
  const o = withDefaults(opts);
  const sp = makeSpatial(o);
  return {
    ...runInfo(o),
    registry: sp.metadata.registry,
    analysis_period: { start_date: sp.metadata.start_date, end_date: sp.metadata.end_date },
    available_dates: o.dates,
    num_nodes: sp.metadata.num_nodes,
    num_holdout_nodes: o.numHoldout,
    num_ghost_nodes: o.ghostIds.length,
    holdout_manifest_sha: sp.metadata.holdout_manifest_sha,
    test_metrics: sp.metadata.test_metrics,
    window_metrics: sp.metadata.window_metrics,
    hyperparameters: HYPERPARAMETERS,
  };
}

export function makeRegistry(entries: RunRegistryEntry[] = [makeRegistryEntry()]): RunRegistry {
  return {
    schema_version: 5,
    generated_at: '2026-10-01T00:00:00Z',
    default_run_id: entries[0]?.run_id ?? null,
    selection_rule: 'reference + top-k by val_obs_all_rmse',
    runs: entries,
  };
}

// ── Page ─────────────────────────────────────────────────────────────────────

export interface PageOpts {
  date?: string;
  timesteps?: number;
  /** Slots where every node is offline (truth null), in addition to holdout-free noise. */
  outageSlots?: number[];
  seed?: number;
}

export function makeTimestamps(date: string, T: number, stepMinutes = 15): string[] {
  return Array.from({ length: T }, (_, t) => {
    const mins = t * stepMinutes;
    const hh = String(Math.floor(mins / 60)).padStart(2, '0');
    const mm = String(mins % 60).padStart(2, '0');
    return `${date}T${hh}:${mm}:00`;
  });
}

export function makeNodeSeries(
  T: number,
  caps: Capabilities,
  rand: () => number,
  outage: Set<number> = new Set(),
): NodeSeries {
  const profile = Array.from({ length: T }, (_, t) => round(20 + 15 * Math.sin((t / T) * Math.PI * 2)));
  const point = profile.map((p) => round(Math.max(0, p + (rand() - 0.5) * 6)));
  const truth = point.map((p, t) => (outage.has(t) ? null : Math.round(Math.max(0, p + (rand() - 0.5) * 10))));
  let gap = 0;
  const gap_steps = truth.map((v) => (v == null ? ++gap : (gap = 0)));
  const prob = () => point.map(() => round(rand() * 0.3));
  return {
    point,
    lo: caps.uncertainty ? point.map((p) => round(Math.max(0, p * 0.6))) : null,
    hi: caps.uncertainty ? point.map((p) => round(p * 1.5 + 2)) : null,
    q50: caps.quantiles ? point.map((p) => round(p * 0.98)) : null,
    q90: caps.quantiles ? point.map((p) => round(p * 1.35)) : null,
    p_zero: caps.uncertainty ? prob() : null,
    outage_q: caps.outage ? truth.map((v) => (v == null ? 0.8 : round(rand() * 0.1))) : null,
    profile,
    truth,
    gap_steps,
  };
}

function nodeMetrics(s: NodeSeries): NodeDayMetrics {
  const pairs = s.truth
    .map((v, t) => (v == null ? null : ([s.point[t], v] as const)))
    .filter((x): x is readonly [number, number] => x != null);
  const n = pairs.length;
  const err = pairs.map(([p, y]) => p - y);
  const mean = (a: readonly number[]) => a.reduce((x, y) => x + y, 0) / a.length;
  return {
    n_observed: n,
    rmse: n ? round(Math.sqrt(mean(err.map((e) => e * e)))) : null,
    mae: n ? round(mean(err.map(Math.abs))) : null,
    bias: n ? round(mean(err)) : null,
    interval80_coverage:
      n && s.lo && s.hi
        ? round(s.truth.filter((v, t) => v != null && v >= s.lo![t] && v <= s.hi![t]).length / n)
        : null,
    peak_point: Math.max(...s.point),
    mean_point: round(mean(s.point)),
    mean_truth: n ? round(mean(pairs.map(([, y]) => y))) : null,
  };
}

function makeCalibration(caps: Capabilities): Calibration {
  const bins = (f: (c: number) => number) =>
    Array.from({ length: 5 }, (_, i) => {
      const c = 0.1 + i * 0.2;
      return { bin_center: c, mean_predicted: round(c + 0.01), observed_frac: round(f(c)), count: 50 + i };
    });
  return {
    pit_histogram: caps.uncertainty
      ? Array.from({ length: 10 }, (_, i) => ({ bin_center: 0.05 + i * 0.1, density: round(0.1 + (i % 2 ? 0.01 : -0.01)) }))
      : null,
    p_zero_reliability: caps.uncertainty ? bins((c) => c * 0.9) : null,
    outage_reliability: caps.outage ? bins((c) => Math.min(1, c * 1.1)) : null,
  };
}

export function makePage(spatial: SpatialIndex, opts: PageOpts = {}): TemporalPage {
  const caps = spatial.metadata.capabilities;
  const date = opts.date ?? spatial.metadata.available_dates[0];
  const T = opts.timesteps ?? 96;
  const rand = rng(opts.seed ?? 7);
  const outage = new Set(opts.outageSlots ?? [10, 11, 12, 40]);

  const node_series: Record<string, NodeSeries> = {};
  const node_metrics: Record<string, NodeDayMetrics> = {};
  const allSlots = new Set(Array.from({ length: T }, (_, t) => t));
  for (const n of spatial.nodes) {
    const s = makeNodeSeries(T, caps, rand, n.window_observed === 0 ? allSlots : outage);
    node_series[String(n.id)] = s;
    node_metrics[String(n.id)] = nodeMetrics(s);
  }

  const all = Object.values(node_series);
  const observed_frac = Array.from({ length: T }, (_, t) => all.filter((s) => s.truth[t] != null).length / all.length);
  const edge_attention: EdgeAttention[] | null = caps.graph
    ? spatial.edge_topology.map((e) => ({
        source: e.source,
        target: e.target,
        // The source is blanked during outages → edge dropped (null).
        weights: Array.from({ length: T }, (_, t) => (outage.has(t) ? null : round(rand()))),
      }))
    : null;

  return {
    schema_version: 5,
    date,
    timesteps: T,
    seq_len: spatial.metadata.seq_len,
    step_minutes: spatial.metadata.step_minutes,
    timestamps: makeTimestamps(date, T, spatial.metadata.step_minutes),
    node_series,
    node_metrics,
    network: {
      observed_frac,
      mean_point: Array.from({ length: T }, (_, t) => round(all.reduce((a, s) => a + s.point[t], 0) / all.length)),
      mean_truth: Array.from({ length: T }, (_, t) => {
        const v = all.map((s) => s.truth[t]).filter((x): x is number => x != null);
        return v.length ? round(v.reduce((a, b) => a + b, 0) / v.length) : null;
      }),
      real_gap_mass_ratio: Array.from({ length: T }, (_, t) => (outage.has(t) ? 0.92 : null)),
      attention_entropy: caps.graph ? Array.from({ length: T }, (_, t) => (outage.has(t) ? null : round(1 + rand()))) : null,
    },
    day_metrics: makeWindowMetrics(caps),
    calibration: makeCalibration(caps),
    edge_attention,
    alpha_node_leaderboard: caps.graph
      ? spatial.nodes.slice(0, 3).map((n, i) => ({
          node_id: n.id,
          location_name: n.location_name,
          total_outgoing_attention: round(30 - i * 5),
        }))
      : null,
    tripwire_audit_trail: [
      { timestep: 10, timestep_end: 12, node_id: 0, event_type: 'imputation_clamp', location_name: spatial.nodes[0]?.location_name },
    ],
  };
}

// ── Node detail ──────────────────────────────────────────────────────────────

export function makeNodeDetail(spatial: SpatialIndex, page: TemporalPage, nodeId = 0): NodeDetailPayload {
  const caps = spatial.metadata.capabilities;
  const node = spatial.nodes.find((n) => n.id === nodeId);
  if (!node) throw new Error(`fixture: no node ${nodeId}`);
  const s = page.node_series[String(nodeId)];
  const T = page.timesteps;
  const likelihood = spatial.metadata.forecast_likelihood;
  const neighbors = spatial.nodes.filter((n) => n.id !== nodeId).slice(0, 3);
  const ego: AttentionEdge[] = caps.graph
    ? neighbors.map((nb, i) => ({
        source_id: i % 2 === 0 ? nb.id : nodeId,
        target_id: i % 2 === 0 ? nodeId : nb.id,
        direction: i % 2 === 0 ? 'incoming' : 'outgoing',
        weight_sequence: Array.from({ length: T }, (_, t) => (t === 0 ? null : round(0.2 + 0.1 * i))),
        source_lat: i % 2 === 0 ? nb.lat : node.lat,
        source_lon: i % 2 === 0 ? nb.lon : node.lon,
        target_lat: i % 2 === 0 ? node.lat : nb.lat,
        target_lon: i % 2 === 0 ? node.lon : nb.lon,
      }))
    : [];
  return {
    schema_version: 5,
    id: node.id,
    location_name: node.location_name,
    lat: node.lat,
    lon: node.lon,
    node_role: node.node_role,
    date: page.date,
    distribution: {
      likelihood,
      mu: s.point,
      theta: caps.uncertainty ? s.point.map(() => 4.2) : null,
      pi: likelihood === 'zinb' ? s.point.map(() => 0.1) : null,
      q: likelihood === 'outage_mixture' ? (s.outage_q ?? null) : null,
    },
    imputed_last: caps.imputation ? s.point.map((p) => round(p * 0.95)) : null,
    input_truth_last: node.node_role === 'holdout' ? s.truth.map(() => null) : s.truth,
    input_missing_frac: s.truth.map((v) => (node.node_role === 'holdout' ? 1 : v == null ? 0.5 : 0)),
    edges_1hop: ego.map((e) => ({
      neighbor_id: e.direction === 'incoming' ? e.source_id : e.target_id,
      direction: e.direction,
      attention_sequence: e.weight_sequence,
    })),
    ego_graph_edges: ego,
    metrics: page.node_metrics[String(nodeId)],
  };
}

/** Convenience: a coherent registry + spatial + page + node detail bundle. */
export function makeBundle(opts: RunOpts & PageOpts = {}) {
  const spatial = makeSpatial(opts);
  const page = makePage(spatial, opts);
  const entry = makeRegistryEntry(opts);
  return { spatial, page, entry, registry: makeRegistry([entry]), node: makeNodeDetail(spatial, page, 0) };
}
