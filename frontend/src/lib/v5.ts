// Pure helpers for the v5 payload contract (types/payload.ts, types/registry.ts).
// No React here, so everything in this module is unit-testable.

import {
  PAYLOAD_SCHEMA_VERSION,
  type EdgeAttention,
  type NodeSeries,
  type TemporalPage,
  type NodeDetailPayload,
} from "@/types/payload";
import type { RunRegistry, SpatialIndex, SpatialNode } from "@/types/registry";

export type RGBA = [number, number, number, number];

// ---------------------------------------------------------------------------
// Formatting: null / non-finite always renders as an em dash, never as 0.
// ---------------------------------------------------------------------------

export function fmt(v: number | null | undefined, digits = 3): string {
  return v == null || !Number.isFinite(v) ? "—" : v.toFixed(digits);
}

export function fmtPct(v: number | null | undefined, digits = 1): string {
  return v == null || !Number.isFinite(v) ? "—" : `${(v * 100).toFixed(digits)}%`;
}

/** Mean of the non-null entries; null when there are none. */
export function meanDefined(values: readonly (number | null)[] | null | undefined): number | null {
  if (!values) return null;
  let sum = 0;
  let n = 0;
  for (const v of values) {
    if (v != null && Number.isFinite(v)) {
      sum += v;
      n += 1;
    }
  }
  return n > 0 ? sum / n : null;
}

/** Fraction of slots with ground truth (null when the series is empty). */
export function observedFraction(truth: readonly (number | null)[] | undefined): number | null {
  if (!truth || truth.length === 0) return null;
  return truth.filter((v) => v != null).length / truth.length;
}

/** Contiguous [start, end] (inclusive) runs of indices where `pred` holds. */
export function contiguousRuns<T>(values: readonly T[], pred: (v: T) => boolean): [number, number][] {
  const runs: [number, number][] = [];
  let start: number | null = null;
  values.forEach((v, i) => {
    if (pred(v)) {
      if (start === null) start = i;
    } else if (start !== null) {
      runs.push([start, i - 1]);
      start = null;
    }
  });
  if (start !== null) runs.push([start, values.length - 1]);
  return runs;
}

// ---------------------------------------------------------------------------
// Time labels
// ---------------------------------------------------------------------------

/** "HH:MM" for a slot index, from the ISO-local timestamp when present
 *  (read lexically: no timezone conversion), else from t × stepMinutes. */
export function timestepLabel(
  t: number,
  timestamps?: readonly string[] | null,
  stepMinutes = 15,
): string {
  const ts = timestamps?.[t];
  const m = ts ? /T(\d{2}):(\d{2})/.exec(ts) : null;
  if (m) return `${m[1]}:${m[2]}`;
  const total = t * stepMinutes;
  const hh = Math.floor(total / 60) % 24;
  return `${String(hh).padStart(2, "0")}:${String(total % 60).padStart(2, "0")}`;
}

// ---------------------------------------------------------------------------
// Map colouring: relative 80% interval width (hi − lo) / (point + 1)
// ---------------------------------------------------------------------------

/** Ratio at which the colour scale saturates to red. An 80% interval is ≈2.56σ
 *  wide, so 2.0 corresponds to a coefficient of variation of roughly 0.8. */
export const INTERVAL_RATIO_SATURATION = 2;

/** Neutral slate for models without an interval: nothing to heat-map. */
export const NEUTRAL_COLOR: RGBA = [100, 116, 139, 200];

/** Green → yellow → red for c ∈ [0, 1]. */
export function heatColor(c: number): RGBA {
  const x = Math.min(1, Math.max(0, c));
  if (x < 0.5) return [Math.round(x * 2 * 255), 200, 0, 220];
  return [255, Math.round(200 * (1 - (x - 0.5) * 2)), 0, 220];
}

/** (hi − lo)/(point + 1) at slot t; null when the model has no interval. */
export function relativeIntervalWidth(series: NodeSeries | undefined, t: number): number | null {
  if (!series || !series.lo || !series.hi) return null;
  const lo = series.lo[t];
  const hi = series.hi[t];
  const p = series.point[t];
  if (lo == null || hi == null || p == null) return null;
  return Math.max(0, hi - lo) / (Math.max(0, p) + 1);
}

export function intervalColor(ratio: number | null): RGBA {
  if (ratio == null || !Number.isFinite(ratio)) return NEUTRAL_COLOR;
  return heatColor(ratio / INTERVAL_RATIO_SATURATION);
}

// ---------------------------------------------------------------------------
// Attention helpers (null weight = edge dropped at that step)
// ---------------------------------------------------------------------------

/** Mean over the non-null weights; 0 for ranking when an edge was never active. */
export function meanWeight(weights: readonly (number | null)[]): number {
  return meanDefined(weights) ?? 0;
}

/** Top-`cap` edges by mean non-null attention (stable topology-layer data). */
export function topEdgesByMeanWeight(edges: EdgeAttention[] | null, cap: number): EdgeAttention[] {
  if (!edges || edges.length === 0) return [];
  return edges
    .map((e) => ({ e, m: meanWeight(e.weights) }))
    .sort((a, b) => b.m - a.m)
    .slice(0, cap)
    .map(({ e }) => e);
}

// ---------------------------------------------------------------------------
// Ghost nodes: in the inference graph but with no observation anywhere in the
// analysis window. Forecast every slot, but nothing can be checked against them.
// ---------------------------------------------------------------------------

export function isGhost(node: Pick<SpatialNode, "window_observed"> | null | undefined): boolean {
  return node != null && node.window_observed === 0;
}

/** "last reported 2025-11-03" / "never reported in the dataset". */
export function lastReportedText(node: Pick<SpatialNode, "last_observed">): string {
  return node.last_observed ? `last reported ${node.last_observed.slice(0, 10)}` : "never reported in the dataset";
}

/** Per-timestep network means describing how concentrated each sensor's attention is. */
export interface AttentionConcentration {
  /** Mean over sensors of entropy / ln(inputs): 1 = spread evenly over every available input. */
  evenness: (number | null)[];
  /** Mean over sensors of exp(entropy): how many inputs the attention is effectively spread over. */
  effective: (number | null)[];
  /** Mean over sensors of the inputs available (reporting neighbours + the sensor itself). */
  available: (number | null)[];
  /** Mean share of attention a sensor keeps on its own state (the GAT self-loop). */
  selfShare: (number | null)[];
  /** Mean entropy (nats); equals the exporter's network.attention_entropy. */
  entropy: (number | null)[];
}

/**
 * Rebuilds each sensor's full attention distribution at every timestep from the
 * edge weights. Every sensor also attends to itself (GAT self-loop); that weight
 * is not stored, but it is 1 − Σ incoming because each head's softmax sums to 1.
 * Evenness skips sensors with no reporting neighbour (a single input has no spread).
 */
export function attentionConcentration(
  edges: EdgeAttention[] | null,
  numNodes: number,
  T: number,
): AttentionConcentration | null {
  if (!edges || numNodes === 0 || T === 0) return null;
  const sum = new Float64Array(T * numNodes);
  const plogp = new Float64Array(T * numNodes);
  const count = new Uint16Array(T * numNodes);
  for (const e of edges) {
    if (e.source === e.target) continue;
    e.weights.forEach((w, t) => {
      if (w == null) return;
      const i = t * numNodes + e.target;
      sum[i] += w;
      count[i] += 1;
      if (w > 0) plogp[i] -= w * Math.log(w);
    });
  }

  const out: AttentionConcentration = { evenness: [], effective: [], available: [], selfShare: [], entropy: [] };
  for (let t = 0; t < T; t++) {
    let even = 0, nEven = 0, eff = 0, avail = 0, self = 0, ent = 0;
    for (let n = 0; n < numNodes; n++) {
      const i = t * numNodes + n;
      const s = Math.max(0, 1 - sum[i]);
      const h = plogp[i] - (s > 0 ? s * Math.log(s) : 0);
      const k = count[i] + 1;
      ent += h;
      eff += Math.exp(h);
      avail += k;
      self += s;
      if (k > 1) {
        even += h / Math.log(k);
        nEven += 1;
      }
    }
    out.evenness.push(nEven > 0 ? even / nEven : null);
    out.effective.push(eff / numNodes);
    out.available.push(avail / numNodes);
    out.selfShare.push(self / numNodes);
    out.entropy.push(ent / numNodes);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Schema guards: fail loudly on stale (pre-v5) payloads instead of rendering
// a half-broken view.
// ---------------------------------------------------------------------------

export class SchemaError extends Error {}

function checkVersion(obj: unknown, what: string, version: unknown): void {
  if (obj == null || typeof obj !== "object") throw new SchemaError(`${what}: not a JSON object`);
  if (version !== PAYLOAD_SCHEMA_VERSION) {
    throw new SchemaError(
      `${what}: schema_version ${String(version ?? "missing")}, expected ${PAYLOAD_SCHEMA_VERSION} — re-export the payloads`,
    );
  }
}

export function assertRegistry(obj: unknown): RunRegistry {
  const r = obj as RunRegistry;
  checkVersion(obj, "run_registry.json", r?.schema_version);
  if (!Array.isArray(r.runs)) throw new SchemaError("run_registry.json: runs is not an array");
  return r;
}

export function assertSpatial(obj: unknown): SpatialIndex {
  const s = obj as SpatialIndex;
  checkVersion(obj, "spatial.json", s?.metadata?.schema_version);
  if (!Array.isArray(s.nodes)) throw new SchemaError("spatial.json: nodes is not an array");
  // Pre-ghost exports lack window_observed; treating that as "has data" would hide every ghost.
  if (s.nodes.some((n) => typeof n.window_observed !== "number")) {
    throw new SchemaError("spatial.json: nodes lack window_observed (stale export) — re-export the payloads");
  }
  return s;
}

export function assertPage(obj: unknown): TemporalPage {
  const p = obj as TemporalPage;
  checkVersion(obj, "page.json", p?.schema_version);
  if (!p.node_series || typeof p.node_series !== "object") {
    throw new SchemaError("page.json: node_series missing");
  }
  return p;
}

export function assertNodeDetail(obj: unknown): NodeDetailPayload {
  const n = obj as NodeDetailPayload;
  checkVersion(obj, "node detail", n?.schema_version);
  return n;
}

/** fetch → HTTP check → JSON → schema guard. */
export async function fetchPayload<T>(
  url: string,
  guard: (obj: unknown) => T,
  signal?: AbortSignal,
): Promise<T> {
  const r = await fetch(url, { signal });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return guard(await r.json());
}

export function isAbortError(err: unknown): boolean {
  return err instanceof Error && err.name === "AbortError";
}

export function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}
