/**
 * tests/payload.test.ts
 *
 * v5 payload contract: fixture invariants, schema guards, pure helpers
 * (src/lib/v5.ts) and the analytics row builders. Fixtures come from the
 * typed builder in tests/fixtures/v5.ts.
 */

import { describe, it, expect, vi, afterEach } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';
import { PAYLOAD_SCHEMA_VERSION } from '@/types/payload';
import type { EdgeAttention } from '@/types/payload';
import {
  attentionConcentration,
  assertNodeDetail,
  assertPage,
  assertRegistry,
  assertSpatial,
  contiguousRuns,
  fetchPayload,
  fmt,
  fmtPct,
  intervalColor,
  isGhost,
  lastReportedText,
  meanDefined,
  NEUTRAL_COLOR,
  observedFraction,
  relativeIntervalWidth,
  SchemaError,
  timestepLabel,
  topEdgesByMeanWeight,
} from '@/lib/v5';
import { forecastRows, slotAxis } from '@/components/analytics/ForecastTab';
import { leaderboardRows } from '@/components/analytics/RunLeaderboard';
import { nodeRows } from '@/components/analytics/NodeGrid';
import { tripwireColumns } from '@/components/analytics/NetworkTab';
import {
  BASELINE_CAPABILITIES,
  makeBundle,
  makeRegistry,
  makeRegistryEntry,
  makeTimestamps,
} from './fixtures/v5';

// ── Fixture invariants (the contract the exporter must honour) ──────────────

describe('v5 fixture invariants', () => {
  const { spatial, page, node } = makeBundle();
  const T = page.timesteps;

  it('carries schema_version 5 everywhere', () => {
    expect(PAYLOAD_SCHEMA_VERSION).toBe(5);
    expect(spatial.metadata.schema_version).toBe(5);
    expect(page.schema_version).toBe(5);
    expect(node.schema_version).toBe(5);
  });

  it('has one series and one metrics entry per spatial node, keyed by String(id)', () => {
    for (const n of spatial.nodes) {
      expect(page.node_series[String(n.id)]).toBeDefined();
      expect(page.node_metrics[String(n.id)]).toBeDefined();
    }
  });

  it('every per-timestep array has length timesteps', () => {
    expect(page.timestamps).toHaveLength(T);
    for (const s of Object.values(page.node_series)) {
      for (const arr of [s.point, s.lo, s.hi, s.q50, s.q90, s.p_zero, s.outage_q, s.profile, s.truth, s.gap_steps]) {
        if (arr) expect(arr).toHaveLength(T);
      }
    }
    for (const e of page.edge_attention ?? []) expect(e.weights).toHaveLength(T);
    expect(page.network.observed_frac).toHaveLength(T);
  });

  it('gap_steps counts slots since the sensor last reported', () => {
    const s = page.node_series['0'];
    s.truth.forEach((v, t) => {
      if (v != null) expect(s.gap_steps[t]).toBe(0);
      else expect(s.gap_steps[t]).toBe(t > 0 ? s.gap_steps[t - 1] + 1 : 1);
    });
  });

  it('marks holdout nodes in spatial and counts them in metadata', () => {
    const holdout = spatial.nodes.filter((n) => n.node_role === 'holdout');
    expect(holdout).toHaveLength(spatial.metadata.num_holdout_nodes);
    expect(spatial.nodes.every((n) => n.node_role === 'observed' || n.node_role === 'holdout')).toBe(true);
  });

  it('a baseline model nulls capability-gated fields instead of zero-filling them', () => {
    const b = makeBundle({ capabilities: BASELINE_CAPABILITIES, likelihood: 'nb' });
    const s = b.page.node_series['0'];
    expect([s.lo, s.hi, s.q50, s.q90, s.p_zero, s.outage_q]).toEqual([null, null, null, null, null, null]);
    expect(b.page.edge_attention).toBeNull();
    expect(b.page.alpha_node_leaderboard).toBeNull();
    expect(b.page.network.attention_entropy).toBeNull();
    expect(b.page.calibration).toEqual({ pit_histogram: null, p_zero_reliability: null, outage_reliability: null });
    expect(b.page.day_metrics.crps).toBeNull();
    expect(b.node.imputed_last).toBeNull();
    expect(b.node.distribution.theta).toBeNull();
  });
});

// ── Schema guards ────────────────────────────────────────────────────────────

describe('schema guards', () => {
  const { spatial, page, node, registry } = makeBundle();

  it('accept v5 payloads unchanged', () => {
    expect(assertRegistry(registry)).toBe(registry);
    expect(assertSpatial(spatial)).toBe(spatial);
    expect(assertPage(page)).toBe(page);
    expect(assertNodeDetail(node)).toBe(node);
  });

  it('reject v4 payloads (no schema_version) with a re-export hint', () => {
    const v4Spatial = { metadata: { run_id: 'x', num_real_nodes: 3, num_simulated_nodes: 1 }, nodes: [] };
    expect(() => assertSpatial(v4Spatial)).toThrow(SchemaError);
    expect(() => assertSpatial(v4Spatial)).toThrow(/expected 5 — re-export/);
    expect(() => assertPage({ date: '2026-05-01', node_series: {} })).toThrow(/schema_version missing/);
    expect(() => assertRegistry({ runs: [] })).toThrow(SchemaError);
    expect(() => assertNodeDetail({ id: 0 })).toThrow(SchemaError);
  });

  it('reject spatial.json exported before ghost nodes (no window_observed)', () => {
    const { spatial } = makeBundle();
    const stale = { ...spatial, nodes: spatial.nodes.map(({ window_observed: _w, ...n }) => (void _w, n)) };
    expect(() => assertSpatial(stale)).toThrow(/re-export/);
  });

  it('reject the wrong version and non-objects', () => {
    expect(() => assertPage({ ...page, schema_version: 4 })).toThrow(/schema_version 4/);
    expect(() => assertRegistry(null)).toThrow(/not a JSON object/);
    expect(() => assertRegistry({ schema_version: 5, runs: 'nope' })).toThrow(/runs is not an array/);
  });
});

describe('fetchPayload', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('applies the guard to the JSON body', async () => {
    const { page } = makeBundle();
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify(page))));
    await expect(fetchPayload('/data/runs/x/2026-05-01/page.json', assertPage)).resolves.toEqual(page);
  });

  it('throws on HTTP errors', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response('nope', { status: 404 })));
    await expect(fetchPayload('/missing', assertPage)).rejects.toThrow('HTTP 404');
  });
});

// ── Formatting / series helpers ──────────────────────────────────────────────

describe('formatting never fabricates zeros', () => {
  it('fmt / fmtPct render null and non-finite as an em dash', () => {
    expect(fmt(null)).toBe('—');
    expect(fmt(undefined)).toBe('—');
    expect(fmt(NaN)).toBe('—');
    expect(fmt(0)).toBe('0.000');
    expect(fmt(1.23456, 2)).toBe('1.23');
    expect(fmtPct(null)).toBe('—');
    expect(fmtPct(0.785)).toBe('78.5%');
  });

  it('meanDefined skips nulls and returns null when nothing is defined', () => {
    expect(meanDefined([1, null, 3])).toBe(2);
    expect(meanDefined([null, null])).toBeNull();
    expect(meanDefined(null)).toBeNull();
  });

  it('observedFraction counts non-null truth', () => {
    expect(observedFraction([1, null, 0, null])).toBe(0.5);
    expect(observedFraction([])).toBeNull();
  });

  it('contiguousRuns finds inclusive gap runs, including at the edges', () => {
    expect(contiguousRuns([null, 1, null, null, 2, null], (v) => v == null)).toEqual([
      [0, 0],
      [2, 3],
      [5, 5],
    ]);
    expect(contiguousRuns([1, 2], (v) => v == null)).toEqual([]);
  });
});

describe('timestepLabel', () => {
  it('reads HH:MM lexically from ISO-local timestamps', () => {
    const ts = makeTimestamps('2026-05-01', 96);
    expect(timestepLabel(0, ts)).toBe('00:00');
    expect(timestepLabel(37, ts)).toBe('09:15');
    expect(timestepLabel(95, ts)).toBe('23:45');
  });

  it('falls back to t × step_minutes without timestamps', () => {
    expect(timestepLabel(5)).toBe('01:15');
    expect(timestepLabel(3, null, 30)).toBe('01:30');
    expect(timestepLabel(96)).toBe('00:00');
  });
});

// ── Map colouring ────────────────────────────────────────────────────────────

describe('relative interval width colouring', () => {
  it('computes (hi − lo)/(point + 1)', () => {
    const { page } = makeBundle();
    const s = page.node_series['0'];
    expect(relativeIntervalWidth(s, 3)).toBeCloseTo((s.hi![3] - s.lo![3]) / (s.point[3] + 1));
  });

  it('is null (neutral colour) when the model has no interval', () => {
    const { page } = makeBundle({ capabilities: BASELINE_CAPABILITIES });
    const s = page.node_series['0'];
    expect(relativeIntervalWidth(s, 3)).toBeNull();
    expect(relativeIntervalWidth(undefined, 3)).toBeNull();
    expect(intervalColor(null)).toEqual(NEUTRAL_COLOR);
  });

  it('runs green → red and saturates', () => {
    expect(intervalColor(0)).toEqual([0, 200, 0, 220]);
    expect(intervalColor(2)).toEqual([255, 0, 0, 220]);
    expect(intervalColor(100)).toEqual(intervalColor(2));
  });
});

describe('topEdgesByMeanWeight', () => {
  const edge = (source: number, weights: (number | null)[]): EdgeAttention => ({ source, target: 99, weights });

  it('ranks by mean of non-null weights and caps the list', () => {
    const edges = [edge(1, [0.1, 0.1]), edge(2, [null, 0.9]), edge(3, [0.5, 0.5]), edge(4, [null, null])];
    expect(topEdgesByMeanWeight(edges, 3).map((e) => e.source)).toEqual([2, 3, 1]);
  });

  it('returns [] for models without a graph', () => {
    expect(topEdgesByMeanWeight(null, 10)).toEqual([]);
  });
});

// ── Analytics row builders ───────────────────────────────────────────────────

describe('leaderboardRows', () => {
  it('orders runs by registry.rank and flattens test + window metrics', () => {
    const reg = makeRegistry([
      makeRegistryEntry({ runId: 'v5_no_graph-s42', rank: 3, capabilities: BASELINE_CAPABILITIES }),
      makeRegistryEntry({ runId: 'v5_full-s42', rank: 1 }),
      makeRegistryEntry({ runId: 'v5_zinb-s42', rank: 2, likelihood: 'zinb' }),
    ]);
    const rows = leaderboardRows(reg.runs);
    expect(rows.map((r) => r.id)).toEqual(['v5_full-s42', 'v5_zinb-s42', 'v5_no_graph-s42']);
    expect(rows[0].is_reference).toBe(true);
    expect(rows[1].likelihood).toBe('zinb');
    expect(rows[0].test_so_12_forecast_rmse).toBe(reg.runs[1].test_metrics.test_so_12_forecast_rmse);
    expect(rows[0].w_obs_ind_rmse).toBe(reg.runs[1].window_metrics.obs_ind_rmse);
    // Baseline: uncertainty metrics stay null.
    expect(rows[2].test_crps).toBeNull();
    expect(rows[2].w_interval80_coverage).toBeNull();
  });
});

describe('nodeRows', () => {
  it('joins spatial nodes with day metrics and observed fraction', () => {
    const { spatial, page } = makeBundle();
    const rows = nodeRows(spatial, page);
    expect(rows).toHaveLength(spatial.nodes.length);
    const r = rows.find((x) => x.node_role === 'holdout')!;
    expect(r.rmse).toBe(page.node_metrics[String(r.id)].rmse);
    expect(r.observed_frac).toBe(observedFraction(page.node_series[String(r.id)].truth));
  });

  it('leaves metrics null when a node is missing from the page', () => {
    const { spatial, page } = makeBundle();
    const { ['0']: _dropped, ...rest } = page.node_metrics;
    void _dropped;
    const rows = nodeRows(spatial, { ...page, node_metrics: rest, node_series: {} });
    expect(rows[0].rmse).toBeNull();
    expect(rows[0].observed_frac).toBeNull();
  });
});

describe('ghost nodes', () => {
  // Node 1 (observed) and node 4 (holdout) have no data in the window.
  const bundle = () => makeBundle({ ghostIds: [1, 4] });

  it('are exactly the nodes with zero observed slots in the window, in both roles', () => {
    const { spatial } = bundle();
    expect(spatial.nodes.filter(isGhost).map((n) => [n.id, n.node_role])).toEqual([
      [1, 'observed'],
      [4, 'holdout'],
    ]);
    expect(spatial.metadata.num_ghost_nodes).toBe(2);
    expect(isGhost(null)).toBe(false);
  });

  it('have no ground truth, so accuracy metrics are null but forecast stats remain', () => {
    const { spatial, page } = bundle();
    for (const n of spatial.nodes.filter(isGhost)) {
      expect(page.node_series[String(n.id)].truth.every((v) => v == null)).toBe(true);
      const m = page.node_metrics[String(n.id)];
      expect([m.rmse, m.mae, m.bias, m.interval80_coverage, m.mean_truth]).toEqual([null, null, null, null, null]);
      expect(m.mean_point).not.toBeNull();
    }
  });

  it('describe when the sensor last reported', () => {
    expect(lastReportedText({ last_observed: '2025-11-03T08:15:00+00:00' })).toBe('last reported 2025-11-03');
    expect(lastReportedText({ last_observed: null })).toBe('never reported in the dataset');
  });

  it('are flagged in the node grid rows with the last-reported tooltip', () => {
    const { spatial, page } = bundle();
    const rows = nodeRows(spatial, page);
    expect(rows.filter((r) => r.ghost != null).map((r) => r.id)).toEqual([1, 4]);
    expect(rows.find((r) => r.id === 1)!.ghost).toContain('never reported');
    expect(rows.find((r) => r.id === 4)!.ghost).toContain('last reported 2025-11-03');
  });
});

describe('forecastRows', () => {
  it('keeps truth gaps as null and builds the [lo, hi] band', () => {
    const { page } = makeBundle({ outageSlots: [2, 3] });
    const s = page.node_series['0'];
    const rows = forecastRows(s);
    expect(rows).toHaveLength(page.timesteps);
    expect(rows[2].truth).toBeNull();
    expect(rows[3].truth).toBeNull();
    expect(rows[0].band).toEqual([s.lo![0], s.hi![0]]);
    expect(rows[2].outage_q).toBe(s.outage_q![2]);
  });

  it('has no band or heads for a baseline model', () => {
    const { page } = makeBundle({ capabilities: BASELINE_CAPABILITIES });
    const r = forecastRows(page.node_series['0'])[0];
    expect([r.band, r.q50, r.q90, r.p_zero, r.outage_q]).toEqual([null, null, null, null, null]);
    expect(r.point).not.toBeNull();
  });

  it('slotAxis pads half a slot and labels ticks from timestamps', () => {
    const { page } = makeBundle();
    const ax = slotAxis(page);
    expect(ax.domain).toEqual([-0.5, page.timesteps - 0.5]);
    expect(ax.tickFormatter(4)).toBe('01:00');
  });
});

describe('tripwireColumns', () => {
  it('adds columns for extra keys present in the trail', () => {
    const { page } = makeBundle();
    const trail = [...page.tripwire_audit_trail, { timestep: 5, event_type: 'clamp', magnitude: 3.2 }];
    const fields = tripwireColumns(trail, page).map((c) => c.field);
    expect(fields).toEqual(['timestep', 'timestep_end', 'node_id', 'location_name', 'event_type', 'magnitude']);
  });
});

describe('attentionConcentration', () => {
  // 3 sensors, 2 slots. Sensor 0 hears 1 and 2; the self-loop takes 1 − Σ incoming.
  const edges: EdgeAttention[] = [
    { source: 1, target: 0, weights: [0.25, 0.5] },
    { source: 2, target: 0, weights: [0.25, null] },   // sensor 2 offline at t=1
  ];

  it('rebuilds each distribution with the implicit self-loop', () => {
    const c = attentionConcentration(edges, 3, 2)!;
    // t=0: sensor 0 is [0.5 self, 0.25, 0.25]; sensors 1 and 2 only attend to themselves.
    const h0 = -(0.5 * Math.log(0.5) + 2 * 0.25 * Math.log(0.25));
    expect(c.evenness[0]).toBeCloseTo(h0 / Math.log(3), 10);       // single-input sensors skipped
    expect(c.effective[0]).toBeCloseTo((Math.exp(h0) + 1 + 1) / 3, 10);
    expect(c.available[0]).toBeCloseTo((3 + 1 + 1) / 3, 10);
    expect(c.selfShare[0]).toBeCloseTo((0.5 + 1 + 1) / 3, 10);
    expect(c.entropy[0]).toBeCloseTo(h0 / 3, 10);
    // t=1: [0.5 self, 0.5] is perfectly even over its 2 available inputs.
    expect(c.evenness[1]).toBeCloseTo(1, 10);
    expect(c.available[1]).toBeCloseTo((2 + 1 + 1) / 3, 10);
  });

  it('is null without a graph and when no sensor has a reporting neighbour', () => {
    expect(attentionConcentration(null, 3, 2)).toBeNull();
    expect(attentionConcentration([], 3, 2)!.evenness).toEqual([null, null]);
  });
});

// ── Real staged payloads (when present) ──────────────────────────────────────

const STAGING = path.resolve(__dirname, '../../artifacts/payload_staging/runs');

describe.skipIf(!fs.existsSync(STAGING))('staged payloads', () => {
  const runs = fs.existsSync(STAGING)
    ? fs
        .readdirSync(STAGING, { withFileTypes: true })
        .filter((d) => d.isDirectory())
        .map((d) => d.name)
    : [];

  it.each(runs)('%s spatial.json + first page.json parse and match the v5 guard', (runId) => {
    const spatial = JSON.parse(fs.readFileSync(path.join(STAGING, runId, 'spatial.json'), 'utf8'));
    const version = spatial?.metadata?.schema_version;
    if (version !== 5) {
      // Stale pre-v5 export: the guard must reject it rather than half-render.
      expect(() => assertSpatial(spatial)).toThrow(SchemaError);
      return;
    }
    const s = assertSpatial(spatial);
    const date = s.metadata.available_dates[0];
    const page = assertPage(JSON.parse(fs.readFileSync(path.join(STAGING, runId, date, 'page.json'), 'utf8')));
    expect(page.date).toBe(date);
    for (const n of s.nodes) expect(page.node_series[String(n.id)]?.point).toHaveLength(page.timesteps);

    // Ghosts (re-exported payloads only): counted in metadata, and dark on every day.
    if (s.nodes.every((n) => typeof n.window_observed === 'number')) {
      expect(s.metadata.num_ghost_nodes).toBe(s.nodes.filter(isGhost).length);
      for (const n of s.nodes.filter(isGhost)) {
        expect(page.node_series[String(n.id)].truth.every((v) => v == null)).toBe(true);
      }
    }

    // The client-side reconstruction (implicit self-loop = 1 − Σ incoming) must
    // reproduce the exporter's entropy, which saw the self-loop weights directly.
    const stored = page.network.attention_entropy;
    if (page.edge_attention && stored) {
      const c = attentionConcentration(page.edge_attention, s.nodes.length, page.timesteps)!;
      stored.forEach((h, t) => h != null && expect(Math.abs(c.entropy[t]! - h)).toBeLessThan(2e-3));
      c.evenness.forEach((v) => v != null && expect(v).toBeLessThanOrEqual(1 + 1e-9));
      c.effective.forEach((v, t) => expect(v!).toBeLessThanOrEqual(c.available[t]! + 1e-9));
    }
  });
});
