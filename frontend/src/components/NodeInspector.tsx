"use client";

import { useEffect, useMemo, useState } from "react";
import useTimeStore from "@/store/useTimeStore";
import type { NodeDetailPayload } from "@/types/payload";

// SVG plot dimensions (purely visual; no responsive resize needed for Phase 1).
const PLOT_W = 320;
const PLOT_H = 140;
const PAD = { top: 8, right: 8, bottom: 18, left: 30 };
const INNER_W = PLOT_W - PAD.left - PAD.right;
const INNER_H = PLOT_H - PAD.top - PAD.bottom;

interface ChartProps {
  node: NodeDetailPayload;
  currentTimestep: number;
}

function NodeChart({ node, currentTimestep }: ChartProps) {
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
    PAD.top + INNER_H - ((v - yMin) / (yMax - yMin || 1)) * INNER_H;

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
      height={PLOT_H}
      viewBox={`0 0 ${PLOT_W} ${PLOT_H}`}
      role="img"
      aria-label={`Imputation chart for ${node.location_name}`}
    >
      {maskBands.map((b, i) => (
        <rect key={i} x={b.x} y={PAD.top} width={b.w} height={INNER_H} fill="rgba(255, 99, 132, 0.18)" />
      ))}
      <rect x={PAD.left} y={PAD.top} width={INNER_W} height={INNER_H} fill="none" stroke="rgba(255,255,255,0.2)" />
      <text x={PAD.left - 4} y={PAD.top + 4} fill="rgba(255,255,255,0.6)" fontSize="9" textAnchor="end">{yMax.toFixed(1)}</text>
      <text x={PAD.left - 4} y={PAD.top + INNER_H} fill="rgba(255,255,255,0.6)" fontSize="9" textAnchor="end">{yMin.toFixed(1)}</text>
      <text x={PAD.left} y={PLOT_H - 4} fill="rgba(255,255,255,0.6)" fontSize="9">t=0</text>
      <text x={PAD.left + INNER_W} y={PLOT_H - 4} fill="rgba(255,255,255,0.6)" fontSize="9" textAnchor="end">t={T - 1}</text>
      <polyline points={imputedPoints} fill="none" stroke="rgb(255, 165, 0)" strokeWidth="1.5" />
      {gtSegments.map((seg, i) => (
        <polyline key={i} points={seg} fill="none" stroke="rgb(96, 165, 250)" strokeWidth="1.5" />
      ))}
      <line x1={cursorX} y1={PAD.top} x2={cursorX} y2={PAD.top + INNER_H} stroke="rgba(255,255,255,0.55)" strokeWidth="1" strokeDasharray="2 2" />
    </svg>
  );
}

export default function NodeInspector() {
  const selectedNodeId = useTimeStore((s) => s.selectedNodeId);
  const setSelectedNodeId = useTimeStore((s) => s.setSelectedNodeId);
  const currentTimestep = useTimeStore((s) => s.currentTimestep);
  const attentionThreshold = useTimeStore((s) => s.attentionThreshold);
  const setAttentionThreshold = useTimeStore((s) => s.setAttentionThreshold);

  const [nodeDetail, setNodeDetail] = useState<NodeDetailPayload | null>(null);

  // Lazy per-node fetch: load node_{id}.json whenever the selected node changes.
  useEffect(() => {
    if (selectedNodeId === null) {
      setNodeDetail(null);
      return;
    }
    fetch(`/data/nodes/node_${selectedNodeId}.json`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d: NodeDetailPayload | null) => setNodeDetail(d))
      .catch(() => setNodeDetail(null));
  }, [selectedNodeId]);

  if (selectedNodeId === null) return null;
  if (!nodeDetail) return null;

  const hasImputation =
    Array.isArray(nodeDetail.imputed_series) &&
    Array.isArray(nodeDetail.ground_truth_series) &&
    Array.isArray(nodeDetail.mask_series) &&
    nodeDetail.imputed_series.length > 0;

  const currentZINB = nodeDetail.zinb_series?.[currentTimestep];
  const currentMasked = nodeDetail.mask_series?.[currentTimestep] ?? false;
  const currentImputed = nodeDetail.imputed_series?.[currentTimestep];
  const currentGT = nodeDetail.ground_truth_series?.[currentTimestep];

  return (
    <aside
      style={{ position: "absolute", top: 16, right: 16, width: 360, zIndex: 10 }}
      className="rounded-lg border border-white/10 bg-neutral-950/85 p-4 text-xs text-slate-200 shadow-xl backdrop-blur"
    >
      <header className="flex items-center justify-between gap-2">
        <div className="min-w-0">
          <div className="truncate font-semibold text-sm">{nodeDetail.location_name}</div>
          <div className="text-[10px] text-slate-400">
            id={nodeDetail.id} · lat={nodeDetail.lat.toFixed(4)} · lon={nodeDetail.lon.toFixed(4)}
          </div>
        </div>
        <div className="flex items-center gap-2">
          {nodeDetail.is_simulated && (
            <span className="rounded bg-amber-600/20 px-1.5 py-0.5 text-[10px] font-medium text-amber-300">SIMULATED</span>
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

      {hasImputation ? (
        <>
          <div className="mt-3">
            <NodeChart node={nodeDetail} currentTimestep={currentTimestep} />
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
            <dd className="text-right">{currentImputed === undefined ? "—" : currentImputed.toFixed(2)}</dd>
            {currentZINB && (
              <>
                <dt className="text-slate-400">µ (forecast)</dt>
                <dd className="text-right">{currentZINB.mu.toFixed(2)}</dd>
                <dt className="text-slate-400">θ (dispersion)</dt>
                <dd className="text-right">{currentZINB.theta.toFixed(2)}</dd>
                <dt className="text-slate-400">σ (uncertainty)</dt>
                <dd className="text-right">{currentZINB.std_dev.toFixed(2)}</dd>
              </>
            )}
          </dl>
        </>
      ) : (
        <div className="mt-3 text-slate-400">Imputation data unavailable in this payload.</div>
      )}

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
