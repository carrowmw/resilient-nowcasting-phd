"use client";

import { useEffect, useMemo } from "react";
import useTimeStore from "@/store/useTimeStore";
import type { NodeDetailPayload, AttentionEdge } from "@/types/payload";
import { useState } from "react";

// ---------------------------------------------------------------------------
// SVG layout constants (shared by both charts)
// ---------------------------------------------------------------------------

const PLOT_W = 320;
const PAD = { top: 8, right: 8, bottom: 18, left: 30 };
const INNER_W = PLOT_W - PAD.left - PAD.right;

// Imputation chart height
const IMPUTE_H = 140;
const IMPUTE_INNER_H = IMPUTE_H - PAD.top - PAD.bottom;

// Pi MNAR timeline height (shorter — pi is bounded [0,1])
const PI_H = 90;
const PI_INNER_H = PI_H - PAD.top - PAD.bottom;

// ---------------------------------------------------------------------------
// Imputation chart (existing — unchanged)
// ---------------------------------------------------------------------------

interface ImpChartProps {
  node: NodeDetailPayload;
  currentTimestep: number;
}

function ImpuationChart({ node, currentTimestep }: ImpChartProps) {
  const T = node.imputed_series.length;

  const { yMin, yMax } = useMemo(() => {
    const vals: number[] = [];
    for (const v of node.ground_truth_series) if (v !== null) vals.push(v);
    for (const v of node.imputed_series) vals.push(v);
    if (vals.length === 0) return { yMin: 0, yMax: 1 };
    const lo = Math.min(...vals);
    const hi = Math.max(...vals);
    const span = Math.max(1, hi - lo) * 0.05;
    return { yMin: Math.max(0, lo - span), yMax: hi + span };
  }, [node]);

  const xOf = (t: number) => PAD.left + (T <= 1 ? 0 : (t / (T - 1)) * INNER_W);
  const yOf = (v: number) =>
    PAD.top + IMPUTE_INNER_H - ((v - yMin) / (yMax - yMin || 1)) * IMPUTE_INNER_H;

  const gtSegments = useMemo(() => {
    const segs: string[] = [];
    let cur: string[] = [];
    for (let t = 0; t < T; t++) {
      const v = node.ground_truth_series[t];
      if (v === null || v === undefined) {
        if (cur.length) segs.push(cur.join(" "));
        cur = [];
      } else {
        cur.push(`${xOf(t)},${yOf(v)}`);
      }
    }
    if (cur.length) segs.push(cur.join(" "));
    return segs;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [node, yMin, yMax]);

  const imputedPoints = useMemo(() => {
    return node.imputed_series.map((v, t) => `${xOf(t)},${yOf(v)}`).join(" ");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [node, yMin, yMax]);

  const maskBands = useMemo(() => {
    const bandW = T <= 1 ? INNER_W : INNER_W / (T - 1);
    return node.mask_series
      .map((m, t) => (m ? t : -1))
      .filter((t) => t >= 0)
      .map((t) => ({ x: xOf(t) - bandW / 2, w: bandW }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [node]);

  const cursorX = xOf(currentTimestep);

  return (
    <svg
      width={PLOT_W}
      height={IMPUTE_H}
      viewBox={`0 0 ${PLOT_W} ${IMPUTE_H}`}
      role="img"
      aria-label={`Imputation chart for ${node.location_name}`}
    >
      {maskBands.map((b, i) => (
        <rect
          key={i}
          x={b.x}
          y={PAD.top}
          width={b.w}
          height={IMPUTE_INNER_H}
          fill="rgba(255, 99, 132, 0.18)"
        />
      ))}
      <rect
        x={PAD.left}
        y={PAD.top}
        width={INNER_W}
        height={IMPUTE_INNER_H}
        fill="none"
        stroke="rgba(255,255,255,0.2)"
      />
      <text x={PAD.left - 4} y={PAD.top + 4} fill="rgba(255,255,255,0.6)" fontSize="9" textAnchor="end">
        {yMax.toFixed(1)}
      </text>
      <text x={PAD.left - 4} y={PAD.top + IMPUTE_INNER_H} fill="rgba(255,255,255,0.6)" fontSize="9" textAnchor="end">
        {yMin.toFixed(1)}
      </text>
      <text x={PAD.left} y={IMPUTE_H - 4} fill="rgba(255,255,255,0.6)" fontSize="9">
        t=0
      </text>
      <text x={PAD.left + INNER_W} y={IMPUTE_H - 4} fill="rgba(255,255,255,0.6)" fontSize="9" textAnchor="end">
        t={T - 1}
      </text>
      <polyline points={imputedPoints} fill="none" stroke="rgb(255, 165, 0)" strokeWidth="1.5" />
      {gtSegments.map((seg, i) => (
        <polyline key={i} points={seg} fill="none" stroke="rgb(96, 165, 250)" strokeWidth="1.5" />
      ))}
      <line
        x1={cursorX} y1={PAD.top}
        x2={cursorX} y2={PAD.top + IMPUTE_INNER_H}
        stroke="rgba(255,255,255,0.55)"
        strokeWidth="1"
        strokeDasharray="2 2"
      />
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Pi MNAR Ghosting Timeline (custom SVG — no external charting dependency)
//
// Renders π (zero-inflation probability) over time as a violet polyline.
// Timesteps where is_missing === true are shaded red before the line is drawn
// (same visual idiom as the imputation chart's mask bands).
// ---------------------------------------------------------------------------

interface PiChartProps {
  node: NodeDetailPayload;
  currentTimestep: number;
}

function PiChart({ node, currentTimestep }: PiChartProps) {
  const series = node.zinb_series;
  const T = series.length;

  // Guard: old payloads without pi field degrade gracefully
  if (T === 0 || series[0].pi === undefined) return null;

  const xOf = (t: number) => PAD.left + (T <= 1 ? 0 : (t / (T - 1)) * INNER_W);
  // pi is always in (0,1) — fixed Y axis
  const yOf = (v: number) => PAD.top + PI_INNER_H - v * PI_INNER_H;

  // Contiguous runs of is_missing === true for background shading
  const missingRuns = useMemo(() => {
    const runs: { x1: number; x2: number }[] = [];
    let start: number | null = null;
    for (const { t, is_missing } of series) {
      if (is_missing && start === null) start = t;
      if (!is_missing && start !== null) {
        runs.push({ x1: xOf(start), x2: xOf(t - 1) });
        start = null;
      }
    }
    if (start !== null) runs.push({ x1: xOf(start), x2: xOf(T - 1) });
    return runs;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [series]);

  const piPoints = series.map(({ t, pi }) => `${xOf(t)},${yOf(pi)}`).join(" ");
  const cursorX = xOf(currentTimestep);
  const currentPi = series[currentTimestep]?.pi;

  return (
    <svg
      width={PLOT_W}
      height={PI_H}
      viewBox={`0 0 ${PLOT_W} ${PI_H}`}
      role="img"
      aria-label={`Pi MNAR timeline for ${node.location_name}`}
    >
      {/* Red shading for masked timesteps (rendered first — behind the line) */}
      {missingRuns.map((r, i) => (
        <rect
          key={i}
          x={r.x1}
          y={PAD.top}
          width={Math.max(1, r.x2 - r.x1)}
          height={PI_INNER_H}
          fill="rgba(239, 68, 68, 0.22)"
        />
      ))}
      {/* Chart border */}
      <rect
        x={PAD.left}
        y={PAD.top}
        width={INNER_W}
        height={PI_INNER_H}
        fill="none"
        stroke="rgba(255,255,255,0.2)"
      />
      {/* Y-axis labels: 0 and 1 */}
      <text x={PAD.left - 4} y={PAD.top + 4} fill="rgba(255,255,255,0.5)" fontSize="9" textAnchor="end">1</text>
      <text x={PAD.left - 4} y={PAD.top + PI_INNER_H} fill="rgba(255,255,255,0.5)" fontSize="9" textAnchor="end">0</text>
      {/* 0.5 reference line */}
      <line
        x1={PAD.left} y1={PAD.top + PI_INNER_H / 2}
        x2={PAD.left + INNER_W} y2={PAD.top + PI_INNER_H / 2}
        stroke="rgba(255,255,255,0.08)"
        strokeWidth="1"
        strokeDasharray="3 3"
      />
      {/* X-axis labels */}
      <text x={PAD.left} y={PI_H - 4} fill="rgba(255,255,255,0.5)" fontSize="9">t=0</text>
      <text x={PAD.left + INNER_W} y={PI_H - 4} fill="rgba(255,255,255,0.5)" fontSize="9" textAnchor="end">
        t={T - 1}
      </text>
      {/* Pi polyline */}
      <polyline points={piPoints} fill="none" stroke="rgb(167, 139, 250)" strokeWidth="1.5" />
      {/* Current timestep cursor */}
      <line
        x1={cursorX} y1={PAD.top}
        x2={cursorX} y2={PAD.top + PI_INNER_H}
        stroke="rgba(255,255,255,0.55)"
        strokeWidth="1"
        strokeDasharray="2 2"
      />
      {/* Pi value at cursor */}
      {currentPi !== undefined && (
        <text
          x={Math.min(cursorX + 3, PAD.left + INNER_W - 22)}
          y={PAD.top + 9}
          fill="rgb(167,139,250)"
          fontSize="8"
        >
          {currentPi.toFixed(3)}
        </text>
      )}
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Attention Edge rank list
// ---------------------------------------------------------------------------

function EdgeList({ edges }: { edges: AttentionEdge[] }) {
  const currentTimestep = useTimeStore((s) => s.currentTimestep);

  const { topIncoming, topOutgoing } = useMemo(() => {
    const byWeight = (a: AttentionEdge, b: AttentionEdge) =>
      (b.weight_sequence[currentTimestep] ?? 0) - (a.weight_sequence[currentTimestep] ?? 0);
    return {
      topIncoming: edges.filter((e) => e.direction === "incoming").sort(byWeight).slice(0, 5),
      topOutgoing: edges.filter((e) => e.direction === "outgoing").sort(byWeight).slice(0, 5),
    };
  }, [edges, currentTimestep]);

  const sorted = [...topIncoming, ...topOutgoing];
  const maxW = Math.max(...sorted.map((e) => e.weight_sequence[currentTimestep] ?? 0), 0.001);

  return (
    <ul className="mt-1 space-y-1">
      {sorted.map((e, i) => {
        const isIncoming = e.direction === "incoming";
        const neighborId = isIncoming ? e.source_id : e.target_id;
        const w = e.weight_sequence[currentTimestep] ?? 0;
        const barW = Math.round((w / maxW) * 80);
        return (
          <li key={i} className="flex items-center gap-2 text-[10px]">
            {/* Direction arrow */}
            <span
              className={`w-3 text-center font-bold ${isIncoming ? "text-blue-400" : "text-amber-400"}`}
              title={isIncoming ? "Incoming" : "Outgoing"}
            >
              {isIncoming ? "←" : "→"}
            </span>
            {/* Neighbor ID */}
            <span className="w-14 font-mono text-slate-300 truncate">
              Node {neighborId}
            </span>
            {/* Weight bar */}
            <span
              className="inline-block rounded-sm"
              style={{
                width: barW,
                minWidth: 2,
                height: 3,
                background: isIncoming
                  ? "rgb(96, 165, 250)"
                  : "rgb(251, 146, 60)",
              }}
            />
            {/* Numeric weight */}
            <span className="ml-auto font-mono text-slate-400">
              {w.toFixed(3)}
            </span>
          </li>
        );
      })}
    </ul>
  );
}

// ---------------------------------------------------------------------------
// Main NodeInspector component
// ---------------------------------------------------------------------------

export default function NodeInspector() {
  const selectedNodeId = useTimeStore((s) => s.selectedNodeId);
  const setSelectedNodeId = useTimeStore((s) => s.setSelectedNodeId);
  const currentTimestep = useTimeStore((s) => s.currentTimestep);
  const attentionThreshold = useTimeStore((s) => s.attentionThreshold);
  const setAttentionThreshold = useTimeStore((s) => s.setAttentionThreshold);
  const setEgoGraphEdges = useTimeStore((s) => s.setEgoGraphEdges);

  const [nodeDetail, setNodeDetail] = useState<NodeDetailPayload | null>(null);

  // Lazy per-node fetch: load node_{id}.json whenever the selected node changes.
  // Write ego_graph_edges into the store so MapVisualizer can render the ArcLayer.
  useEffect(() => {
    if (selectedNodeId === null) {
      setNodeDetail(null);
      setEgoGraphEdges(null);
      return;
    }
    fetch(`/data/nodes/node_${selectedNodeId}.json`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d: NodeDetailPayload | null) => {
        setNodeDetail(d);
        setEgoGraphEdges(d?.ego_graph_edges ?? null);
      })
      .catch(() => {
        setNodeDetail(null);
        setEgoGraphEdges(null);
      });
  }, [selectedNodeId, setEgoGraphEdges]);

  if (selectedNodeId === null) return null;
  if (!nodeDetail) return null;

  const hasImputation =
    Array.isArray(nodeDetail.imputed_series) &&
    Array.isArray(nodeDetail.ground_truth_series) &&
    Array.isArray(nodeDetail.mask_series) &&
    nodeDetail.imputed_series.length > 0;

  const hasPi =
    Array.isArray(nodeDetail.zinb_series) &&
    nodeDetail.zinb_series.length > 0 &&
    nodeDetail.zinb_series[0].pi !== undefined;

  const hasEgoEdges =
    Array.isArray(nodeDetail.ego_graph_edges) &&
    nodeDetail.ego_graph_edges.length > 0;

  const currentZINB = nodeDetail.zinb_series?.[currentTimestep];
  const currentMasked = nodeDetail.mask_series?.[currentTimestep] ?? false;
  const currentImputed = nodeDetail.imputed_series?.[currentTimestep];
  const currentGT = nodeDetail.ground_truth_series?.[currentTimestep];

  return (
    <aside
      style={{ position: "absolute", top: 16, right: 16, width: 360, zIndex: 10, maxHeight: "calc(100vh - 32px)" }}
      className="rounded-lg border border-white/10 bg-neutral-950/85 p-4 text-xs text-slate-200 shadow-xl backdrop-blur overflow-y-auto"
    >
      {/* Header */}
      <header className="flex items-center justify-between gap-2">
        <div className="min-w-0">
          <div className="truncate font-semibold text-sm">{nodeDetail.location_name}</div>
          <div className="text-[10px] text-slate-400">
            id={nodeDetail.id} · lat={nodeDetail.lat.toFixed(4)} · lon={nodeDetail.lon.toFixed(4)}
          </div>
        </div>
        <div className="flex items-center gap-2">
          {nodeDetail.is_simulated && (
            <span className="rounded bg-amber-600/20 px-1.5 py-0.5 text-[10px] font-medium text-amber-300">
              SIMULATED
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

      {/* ── Imputation chart ── */}
      {hasImputation ? (
        <>
          <div className="mt-3">
            <ImpuationChart node={nodeDetail} currentTimestep={currentTimestep} />
          </div>
          <div className="mt-2 flex items-center gap-3 text-[10px] text-slate-400">
            <span className="flex items-center gap-1">
              <span className="inline-block h-0.5 w-3 bg-blue-400" /> Ground truth
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block h-0.5 w-3 bg-orange-400" /> Imputed
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block h-2 w-3 bg-rose-400/30" /> Masked
            </span>
          </div>
          <dl className="mt-3 grid grid-cols-2 gap-1 text-[11px]">
            <dt className="text-slate-400">t = {currentTimestep}</dt>
            <dd className="text-right">{currentMasked ? "masked" : "observed"}</dd>
            <dt className="text-slate-400">ground truth</dt>
            <dd className="text-right">
              {currentGT === null || currentGT === undefined ? "—" : currentGT.toFixed(2)}
            </dd>
            <dt className="text-slate-400">imputed</dt>
            <dd className="text-right">
              {currentImputed === undefined ? "—" : currentImputed.toFixed(2)}
            </dd>
            {currentZINB && (
              <>
                <dt className="text-slate-400">µ (forecast)</dt>
                <dd className="text-right">{currentZINB.mu.toFixed(2)}</dd>
                <dt className="text-slate-400">θ (dispersion)</dt>
                <dd className="text-right">{currentZINB.theta.toFixed(2)}</dd>
                <dt className="text-slate-400">σ (uncertainty)</dt>
                <dd className="text-right">{currentZINB.std_dev.toFixed(2)}</dd>
                {currentZINB.pi !== undefined && (
                  <>
                    <dt className="text-slate-400">π (zero-inflation)</dt>
                    <dd className="text-right text-violet-300">{currentZINB.pi.toFixed(3)}</dd>
                  </>
                )}
              </>
            )}
          </dl>
        </>
      ) : (
        <div className="mt-3 text-slate-400">Imputation data unavailable in this payload.</div>
      )}

      {/* ── MNAR Ghosting Timeline (π over time) ── */}
      {hasPi && (
        <div className="mt-4 border-t border-white/10 pt-3">
          <div className="mb-1 flex items-center justify-between text-[10px] text-slate-400">
            <span>
              <span className="font-medium text-violet-300">π</span>
              {" "}zero-inflation probability
            </span>
            <span className="text-rose-400/70">red = masked</span>
          </div>
          <PiChart node={nodeDetail} currentTimestep={currentTimestep} />
        </div>
      )}

      {/* ── Top Attention Edges (Ego-Graph) ── */}
      {hasEgoEdges && (
        <div className="mt-4 border-t border-white/10 pt-3">
          <div className="mb-1 flex items-center justify-between text-[10px]">
            <span className="font-medium text-slate-300">Top Attention Edges</span>
            <span className="text-slate-500">
              <span className="text-blue-400">←</span> in{" "}
              <span className="text-amber-400">→</span> out
            </span>
          </div>
          <EdgeList edges={nodeDetail.ego_graph_edges!} />
        </div>
      )}

      {/* ── Attention threshold slider ── */}
      <div className="mt-4 border-t border-white/10 pt-3">
        <label className="flex items-center justify-between text-[11px] text-slate-300">
          <span>Attention threshold</span>
          <span className="font-mono text-slate-400">{attentionThreshold.toFixed(2)}</span>
        </label>
        <input
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={attentionThreshold}
          onChange={(e) => setAttentionThreshold(Number(e.target.value))}
          className="mt-1 w-full accent-amber-400"
          aria-label="Attention edge weight threshold"
        />
      </div>
    </aside>
  );
}
