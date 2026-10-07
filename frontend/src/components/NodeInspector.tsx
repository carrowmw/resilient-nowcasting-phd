"use client";

import { useEffect, useMemo, useState } from "react";
import useTimeStore from "@/store/useTimeStore";
import type { AttentionEdge, NodeDetailPayload } from "@/types/payload";
import { nodeUrl } from "@/lib/payloadSource";
import {
  assertNodeDetail,
  contiguousRuns,
  errorMessage,
  fetchPayload,
  fmt,
  fmtPct,
  isAbortError,
  isGhost,
  lastReportedText,
  timestepLabel,
} from "@/lib/v5";

// ---------------------------------------------------------------------------
// Generic SVG mini chart (no charting dependency inside the map overlay)
// ---------------------------------------------------------------------------

const PLOT_W = 320;
const IMPUTED_LINE = "rgb(255, 165, 0)";
const GHOST_LINE = "rgb(148, 163, 184)";
const PAD = { top: 8, right: 8, bottom: 18, left: 30 };
const INNER_W = PLOT_W - PAD.left - PAD.right;

interface MiniSeries {
  values: readonly (number | null)[];
  color: string;
  dash?: string;
}

interface MiniChartProps {
  series: MiniSeries[];
  T: number;
  height: number;
  cursor: number;
  ariaLabel: string;
  /** Fixed y-domain (e.g. [0, 1] for probabilities); auto from data otherwise. */
  domain?: [number, number];
  /** Per-slot flag: the sensor had no reading at the plotted slot (shaded as merged bands). */
  offline?: readonly boolean[];
}

function MiniChart({ series, T, height, cursor, ariaLabel, domain, offline }: MiniChartProps) {
  const innerH = height - PAD.top - PAD.bottom;

  const [yMin, yMax] = useMemo((): [number, number] => {
    if (domain) return domain;
    const vals = series.flatMap((s) => s.values.filter((v): v is number => v != null));
    if (vals.length === 0) return [0, 1];
    const lo = Math.min(...vals);
    const hi = Math.max(...vals);
    const pad = Math.max(1, hi - lo) * 0.05;
    return [Math.max(0, lo - pad), hi + pad];
  }, [series, domain]);

  const xOf = (t: number) => PAD.left + (T <= 1 ? 0 : (t / (T - 1)) * INNER_W);
  const yOf = (v: number) => PAD.top + innerH - ((v - yMin) / (yMax - yMin || 1)) * innerH;
  const bandW = T <= 1 ? INNER_W : INNER_W / (T - 1);

  // Null breaks a line into segments: gaps are drawn as gaps.
  const segments = (values: readonly (number | null)[]) =>
    contiguousRuns(values, (v) => v != null).map(([a, b]) =>
      values.slice(a, b + 1).map((v, i) => [xOf(a + i), yOf(v as number)] as const),
    );

  return (
    <svg width={PLOT_W} height={height} viewBox={`0 0 ${PLOT_W} ${height}`} role="img" aria-label={ariaLabel}>
      {offline &&
        contiguousRuns(offline, (o) => o).map(([a, b]) => {
          const x1 = Math.max(PAD.left, xOf(a) - bandW / 2);
          const x2 = Math.min(PAD.left + INNER_W, xOf(b) + bandW / 2);
          return <rect key={a} x={x1} y={PAD.top} width={x2 - x1} height={innerH} fill="rgba(239, 68, 68, 0.18)" />;
        })}
      <rect x={PAD.left} y={PAD.top} width={INNER_W} height={innerH} fill="none" stroke="rgba(255,255,255,0.2)" />
      <text x={PAD.left - 4} y={PAD.top + 4} fill="rgba(255,255,255,0.6)" fontSize="9" textAnchor="end">
        {yMax.toFixed(domain ? 0 : 1)}
      </text>
      <text x={PAD.left - 4} y={PAD.top + innerH} fill="rgba(255,255,255,0.6)" fontSize="9" textAnchor="end">
        {yMin.toFixed(domain ? 0 : 1)}
      </text>
      <text x={PAD.left} y={height - 4} fill="rgba(255,255,255,0.6)" fontSize="9">
        t=0
      </text>
      <text x={PAD.left + INNER_W} y={height - 4} fill="rgba(255,255,255,0.6)" fontSize="9" textAnchor="end">
        t={T - 1}
      </text>
      {series.map((s, i) =>
        segments(s.values).map((pts, j) =>
          pts.length > 1 ? (
            <polyline
              key={`${i}-${j}`}
              points={pts.map(([x, y]) => `${x},${y}`).join(" ")}
              fill="none"
              stroke={s.color}
              strokeWidth="1.5"
              strokeDasharray={s.dash}
            />
          ) : (
            // Isolated value: a dot, since a one-point polyline is invisible.
            <circle key={`${i}-${j}`} cx={pts[0][0]} cy={pts[0][1]} r={1.5} fill={s.color} />
          ),
        ),
      )}
      <line
        x1={xOf(cursor)}
        y1={PAD.top}
        x2={xOf(cursor)}
        y2={PAD.top + innerH}
        stroke="rgba(255,255,255,0.55)"
        strokeWidth="1"
        strokeDasharray="2 2"
      />
    </svg>
  );
}

function LegendSwatch({ color, label }: { color: string; label: string }) {
  return (
    <span className="flex items-center gap-1">
      <span className="inline-block w-3" style={{ borderTop: `2px solid ${color}` }} />
      {label}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Attention edge rank list (null weight = edge absent at this step)
// ---------------------------------------------------------------------------

function EdgeList({ edges, currentTimestep }: { edges: AttentionEdge[]; currentTimestep: number }) {
  const active = useMemo(() => {
    const withW = edges
      .map((e) => ({ e, w: e.weight_sequence[currentTimestep] ?? null }))
      .filter((x): x is { e: AttentionEdge; w: number } => x.w != null);
    const top = (dir: AttentionEdge["direction"]) =>
      withW
        .filter((x) => x.e.direction === dir)
        .sort((a, b) => b.w - a.w)
        .slice(0, 5);
    return [...top("incoming"), ...top("outgoing")];
  }, [edges, currentTimestep]);

  if (active.length === 0) {
    return <div className="mt-1 text-[10px] text-slate-500">No active edges at this step.</div>;
  }

  const maxW = Math.max(...active.map((x) => x.w), 0.001);

  return (
    <ul className="mt-1 space-y-1">
      {active.map(({ e, w }) => {
        const isIncoming = e.direction === "incoming";
        const neighborId = isIncoming ? e.source_id : e.target_id;
        return (
          <li key={`${e.direction}-${neighborId}`} className="flex items-center gap-2 text-[10px]">
            <span
              className={`w-3 text-center font-bold ${isIncoming ? "text-blue-400" : "text-rose-400"}`}
              title={isIncoming ? "Incoming" : "Outgoing"}
            >
              {isIncoming ? "←" : "→"}
            </span>
            <span className="w-14 font-mono text-slate-300 truncate">Node {neighborId}</span>
            <span
              className="inline-block rounded-sm"
              style={{
                width: Math.round((w / maxW) * 80),
                minWidth: 2,
                height: 3,
                background: isIncoming ? "rgb(96, 165, 250)" : "rgb(251, 113, 133)",
              }}
            />
            <span className="ml-auto font-mono text-slate-400">{w.toFixed(3)}</span>
          </li>
        );
      })}
    </ul>
  );
}

function Row({ label, value, accent, hint }: { label: string; value: string; accent?: string; hint?: string }) {
  return (
    <>
      <dt className="text-slate-400" title={hint}>
        {label}
      </dt>
      <dd className="text-right font-mono" style={accent ? { color: accent } : undefined}>
        {value}
      </dd>
    </>
  );
}

// ---------------------------------------------------------------------------
// Main NodeInspector component
// ---------------------------------------------------------------------------

export default function NodeInspector() {
  const selectedNodeId = useTimeStore((s) => s.selectedNodeId);
  const setSelectedNodeId = useTimeStore((s) => s.setSelectedNodeId);
  const currentTimestep = useTimeStore((s) => s.currentTimestep);
  const setEgoGraphEdges = useTimeStore((s) => s.setEgoGraphEdges);
  const activeRunId = useTimeStore((s) => s.activeRunId);
  const activeDate = useTimeStore((s) => s.activeDate);
  const dayPayload = useTimeStore((s) => s.dayPayload);

  const [nodeDetail, setNodeDetail] = useState<NodeDetailPayload | null>(null);
  const [fetchError, setFetchError] = useState<string | null>(null);

  // Lazy per-node fetch of the day-scoped node_{id}.json; ego_graph_edges go
  // into the store so MapVisualizer can draw the ego-graph ArcLayer.
  useEffect(() => {
    if (selectedNodeId === null || !activeRunId || !activeDate) {
      setEgoGraphEdges(null);
      return;
    }
    const controller = new AbortController();
    fetchPayload(nodeUrl(activeRunId, activeDate, selectedNodeId), assertNodeDetail, controller.signal)
      .then((d) => {
        setFetchError(null);
        setNodeDetail(d);
        setEgoGraphEdges(d.ego_graph_edges.length > 0 ? d.ego_graph_edges : null);
      })
      .catch((err) => {
        if (isAbortError(err)) return;
        setFetchError(`Failed to load node detail: ${errorMessage(err)}`);
        setNodeDetail(null);
        setEgoGraphEdges(null);
      });
    return () => controller.abort();
  }, [selectedNodeId, activeRunId, activeDate, setEgoGraphEdges]);

  if (selectedNodeId === null) return null;

  if (fetchError) {
    return (
      <aside
        style={{ position: "absolute", top: 16, right: 16, width: 360, zIndex: 10 }}
        className="rounded-lg border border-red-500/30 bg-neutral-950/85 p-4 text-xs text-red-300 shadow-xl backdrop-blur"
      >
        <div className="flex items-center justify-between gap-2">
          <span>{fetchError}</span>
          <button
            type="button"
            onClick={() => setSelectedNodeId(null)}
            className="rounded px-2 py-0.5 text-slate-300 hover:bg-white/10"
            aria-label="Close inspector"
          >
            ✕
          </button>
        </div>
      </aside>
    );
  }

  // Stale detail from the previous selection/day is never shown.
  if (!nodeDetail || nodeDetail.id !== selectedNodeId || nodeDetail.date !== activeDate) return null;

  const t = currentTimestep;
  const dist = nodeDetail.distribution;
  const series = dayPayload?.page.node_series[String(nodeDetail.id)];
  const timestamps = dayPayload?.page.timestamps;
  const stepMinutes = dayPayload?.page.step_minutes ?? 15;
  const T = dist.mu.length || nodeDetail.input_truth_last.length;
  const m = nodeDetail.metrics;
  const spatialNode = dayPayload?.spatial.nodes.find((n) => n.id === nodeDetail.id);
  const ghost = isGhost(spatialNode);

  // Red bands mark slots where the sensor itself reported nothing: the self-healing
  // chart plots the last look-back slot, the probability chart the forecast target.
  const inputOffline = nodeDetail.input_truth_last.map((v) => v == null);
  const targetOffline = series?.truth.map((v) => v == null);
  const probSeries: MiniSeries[] = [];
  if (dist.pi) probSeries.push({ values: dist.pi, color: "rgb(167, 139, 250)" });
  if (dist.q) probSeries.push({ values: dist.q, color: "rgb(245, 158, 11)" });

  const healSeries: MiniSeries[] = [
    { values: nodeDetail.input_truth_last, color: "rgb(96, 165, 250)" },
  ];
  if (nodeDetail.imputed_last) {
    // Ghosts: grey, like v4's simulated sensors — a plausible-looking pattern, not a checkable one.
    healSeries.push({ values: nodeDetail.imputed_last, color: ghost ? GHOST_LINE : IMPUTED_LINE });
  }

  return (
    <aside
      style={{
        position: "absolute",
        top: 16,
        right: 16,
        width: 360,
        zIndex: 10,
        maxHeight: "calc(100vh - 32px)",
      }}
      className="rounded-lg border border-white/10 bg-neutral-950/85 p-4 text-xs text-slate-200 shadow-xl backdrop-blur overflow-y-auto"
    >
      <header className="flex items-center justify-between gap-2">
        <div className="min-w-0">
          <div className="truncate font-semibold text-sm">{nodeDetail.location_name}</div>
          <div className="text-[10px] text-slate-400">
            id={nodeDetail.id} · lat={nodeDetail.lat.toFixed(4)} · lon={nodeDetail.lon.toFixed(4)}
          </div>
        </div>
        <div className="flex items-center gap-2">
          {ghost && spatialNode && (
            <span
              className="rounded border border-slate-500/60 px-1.5 py-0.5 text-[10px] font-medium text-slate-400"
              title={`Ghost: no observations anywhere in the analysis window (${lastReportedText(spatialNode)}). The model still forecasts it from its neighbours, but nothing can be checked.`}
            >
              GHOST
            </span>
          )}
          {nodeDetail.node_role === "holdout" && (
            <span
              className="rounded bg-violet-600/20 px-1.5 py-0.5 text-[10px] font-medium text-violet-300"
              title="Zero-shot: never trained on and blanked for the whole look-back window; compared against held-out ground truth."
            >
              HOLDOUT
            </span>
          )}
          <button
            type="button"
            onClick={() => setSelectedNodeId(null)}
            className="rounded px-2 py-0.5 text-slate-300 hover:bg-white/10"
            aria-label="Close inspector"
          >
            ✕
          </button>
        </div>
      </header>

      {/* ── Forecast at the cursor ── */}
      <div className="mt-3 text-[10px] uppercase tracking-wide text-blue-400">
        Forecast · {timestepLabel(t, timestamps, stepMinutes)} (t = {t})
      </div>
      <dl className="mt-1 grid grid-cols-2 gap-1 text-[11px]">
        <Row label="point" value={fmt(series?.point[t], 2)} />
        <Row
          label="80% interval"
          value={series?.lo && series?.hi ? `[${fmt(series.lo[t], 1)}, ${fmt(series.hi[t], 1)}]` : "n/a"}
        />
        <Row label="truth" value={series ? (series.truth[t] == null ? "offline" : fmt(series.truth[t], 0)) : "—"} />
        <Row label="gap steps" value={series ? String(series.gap_steps[t] ?? "—") : "—"} />
        <Row label={`μ (${dist.likelihood})`} value={fmt(dist.mu[t], 2)} />
        <Row label="θ (dispersion)" value={dist.theta ? fmt(dist.theta[t], 2) : "n/a"} />
        <Row label="π (zero-inflation)" value={dist.pi ? fmt(dist.pi[t], 3) : "n/a"} accent="#c4b5fd" />
        <Row label="q (outage prob.)" value={dist.q ? fmt(dist.q[t], 3) : "n/a"} accent="#fcd34d" />
        <Row
          label="look-back hidden"
          value={fmtPct(nodeDetail.input_missing_frac[t] ?? null, 0)}
          hint="Share of the input window behind this forecast that the model could not see (sensor offline, or the whole window for a zero-shot node)."
        />
      </dl>

      {/* ── Self-healing: imputed_last vs input_truth_last ── */}
      <div className="mt-4 border-t border-white/10 pt-3">
        <div className="mb-1 text-[10px] text-slate-400">
          {ghost ? "Model estimate" : "Self-healing"} — last look-back slot of each window
        </div>
        <MiniChart
          series={healSeries}
          T={T}
          height={140}
          cursor={t}
          offline={ghost ? undefined : inputOffline}
          ariaLabel={`Imputation chart for ${nodeDetail.location_name}`}
        />
        <div className="mt-1 flex flex-wrap items-center gap-3 text-[10px] text-slate-400">
          {!ghost && <LegendSwatch color="rgb(96, 165, 250)" label="Sensor input" />}
          {nodeDetail.imputed_last ? (
            <LegendSwatch color={ghost ? GHOST_LINE : IMPUTED_LINE} label={ghost ? "Model estimate (no data to check)" : "Imputed"} />
          ) : (
            <span className="text-slate-500">Imputation not available for this model</span>
          )}
          {!ghost && (
            <span className="flex items-center gap-1">
              <span className="inline-block h-2 w-3 bg-rose-400/30" /> Sensor offline
            </span>
          )}
        </div>
        {ghost && spatialNode && (
          <div className="mt-1 text-[10px] text-slate-400">
            Ghost sensor — no readings in this window ({lastReportedText(spatialNode)}). The grey line is the
            model&apos;s estimate from neighbouring sensors; judge it on how plausible the pattern looks.
          </div>
        )}
        {nodeDetail.node_role === "holdout" && !ghost && (
          <div className="mt-1 text-[10px] text-violet-300">
            Zero-shot node: the model never sees this sensor, so the sensor input line is ground truth shown for
            comparison only and the imputation is reconstructed from neighbours.
          </div>
        )}
        <dl className="mt-2 grid grid-cols-2 gap-1 text-[11px]">
          <Row
            label="input at cursor"
            value={nodeDetail.input_truth_last[t] == null ? "missing" : fmt(nodeDetail.input_truth_last[t], 0)}
          />
          <Row label="imputed at cursor" value={nodeDetail.imputed_last ? fmt(nodeDetail.imputed_last[t], 2) : "n/a"} />
        </dl>
      </div>

      {/* ── Probability heads (π / q) ── */}
      {probSeries.length > 0 && (
        <div className="mt-4 border-t border-white/10 pt-3">
          <div className="mb-1 flex items-center gap-3 text-[10px] text-slate-400">
            {dist.pi && <LegendSwatch color="rgb(167, 139, 250)" label="π zero-inflation" />}
            {dist.q && <LegendSwatch color="rgb(245, 158, 11)" label="q outage" />}
          </div>
          <MiniChart
            series={probSeries}
            T={T}
            height={90}
            cursor={t}
            domain={[0, 1]}
            offline={ghost ? undefined : targetOffline}
            ariaLabel={`Probability timeline for ${nodeDetail.location_name}`}
          />
        </div>
      )}

      {/* ── Day metrics ── */}
      <div className="mt-4 border-t border-white/10 pt-3">
        <div className="text-[10px] uppercase tracking-wide text-blue-400">Day metrics</div>
        {ghost && (
          <div className="mt-1 text-[10px] text-slate-500">No ground truth: accuracy metrics are blank, forecast stats still apply.</div>
        )}
        <dl className="mt-1 grid grid-cols-2 gap-1 text-[11px]">
          <Row label="observed slots" value={String(m.n_observed)} />
          <Row label="RMSE" value={fmt(m.rmse, 2)} />
          <Row label="MAE" value={fmt(m.mae, 2)} />
          <Row label="bias" value={fmt(m.bias, 2)} />
          <Row label="80% coverage" value={fmtPct(m.interval80_coverage)} />
          <Row label="mean point / truth" value={`${fmt(m.mean_point, 1)} / ${fmt(m.mean_truth, 1)}`} />
          <Row label="peak point" value={fmt(m.peak_point, 1)} />
        </dl>
      </div>

      {/* ── Ego-graph attention ── */}
      {nodeDetail.ego_graph_edges.length > 0 && (
        <div className="mt-4 border-t border-white/10 pt-3">
          <div className="mb-1 flex items-center justify-between text-[10px]">
            <span className="font-medium text-slate-300">Top Attention Edges</span>
            <span className="text-slate-500">
              <span className="text-blue-400">←</span> in <span className="text-rose-400">→</span> out
            </span>
          </div>
          <EdgeList edges={nodeDetail.ego_graph_edges} currentTimestep={t} />
        </div>
      )}
    </aside>
  );
}
