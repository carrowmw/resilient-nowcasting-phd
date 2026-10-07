"use client";

import { useMemo } from "react";
import Box from "@mui/material/Box";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ReferenceArea,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { Capabilities, NodeDayMetrics, NodeSeries, TemporalPage } from "@/types/payload";
import type { SpatialNode } from "@/types/registry";
import { contiguousRuns, fmt, fmtPct, timestepLabel } from "@/lib/v5";
import { ACCENT, axisProps, gridStroke, NotAvailable, RoleChip, SectionTitle, StatCard, tooltipStyle, tooltipValue } from "./shared";

export interface ForecastPoint {
  t: number;
  point: number | null;
  band: [number, number] | null;   // [lo, hi]; null without uncertainty
  q50: number | null;
  q90: number | null;
  profile: number | null;
  truth: number | null;
  p_zero: number | null;
  outage_q: number | null;
}

export function forecastRows(s: NodeSeries): ForecastPoint[] {
  return s.point.map((p, t) => {
    const lo = s.lo?.[t];
    const hi = s.hi?.[t];
    return {
      t,
      point: p ?? null,
      band: lo != null && hi != null ? [lo, hi] : null,
      q50: s.q50?.[t] ?? null,
      q90: s.q90?.[t] ?? null,
      profile: s.profile[t] ?? null,
      truth: s.truth[t] ?? null,
      p_zero: s.p_zero?.[t] ?? null,
      outage_q: s.outage_q?.[t] ?? null,
    };
  });
}

/** Shared x-axis: numeric slots, padded half a slot so gap shading fits. */
export function slotAxis(page: TemporalPage) {
  const T = page.timesteps;
  const every = Math.max(1, Math.round(T / 12));
  return {
    dataKey: "t",
    type: "number" as const,
    domain: [-0.5, T - 0.5] as [number, number],
    ticks: Array.from({ length: Math.ceil(T / every) }, (_, i) => i * every),
    tickFormatter: (t: number) => timestepLabel(t, page.timestamps, page.step_minutes),
    allowDataOverflow: true,
    ...axisProps,
  };
}

export function GapShading({ runs }: { runs: [number, number][] }) {
  return (
    <>
      {runs.map(([a, b]) => (
        <ReferenceArea key={`gap-${a}`} x1={a - 0.5} x2={b + 0.5} fill="#ef4444" fillOpacity={0.09} stroke="none" ifOverflow="hidden" />
      ))}
    </>
  );
}

interface Props {
  page: TemporalPage;
  node: SpatialNode | null;
  capabilities: Capabilities;
}

export default function ForecastTab({ page, node, capabilities }: Props) {
  const series = node ? page.node_series[String(node.id)] : undefined;
  const metrics: NodeDayMetrics | undefined = node ? page.node_metrics[String(node.id)] : undefined;
  const data = useMemo(() => (series ? forecastRows(series) : []), [series]);
  const gaps = useMemo(() => (series ? contiguousRuns(series.truth, (v) => v == null) : []), [series]);

  if (!node) return <NotAvailable>Select a node in the table above to plot its forecast.</NotAvailable>;
  if (!series) return <NotAvailable>No forecast series for node {node.id} on {page.date}.</NotAvailable>;

  const hasBand = series.lo != null && series.hi != null;
  const hasProb = series.p_zero != null || series.outage_q != null;
  const xAxis = slotAxis(page);
  const labelFmt = (t: unknown) => timestepLabel(Number(t), page.timestamps, page.step_minutes);

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 1.5 }}>
      <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
        <SectionTitle hint={`id ${node.id} · red bands = sensor offline`}>{node.location_name}</SectionTitle>
        <RoleChip role={node.node_role} />
      </Box>
      {metrics && (
        <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1 }}>
          <StatCard label="RMSE" value={fmt(metrics.rmse, 2)} hint={`${metrics.n_observed} observed slots`} />
          <StatCard label="MAE" value={fmt(metrics.mae, 2)} />
          <StatCard label="Bias" value={fmt(metrics.bias, 2)} />
          <StatCard label="80% coverage" value={fmtPct(metrics.interval80_coverage)} hint={capabilities.uncertainty ? "target 80%" : "no interval"} />
          <StatCard label="Mean point / truth" value={`${fmt(metrics.mean_point, 1)} / ${fmt(metrics.mean_truth, 1)}`} />
        </Box>
      )}

      <Box sx={{ height: 320 }} aria-label="Forecast chart">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 4, right: 20, left: 0, bottom: 4 }} syncId="forecast">
            <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} />
            <XAxis {...xAxis} />
            <YAxis {...axisProps} width={48} />
            <Tooltip contentStyle={tooltipStyle} labelFormatter={labelFmt} formatter={(v: unknown, name: string) => [tooltipValue(v), name]} />
            <Legend wrapperStyle={{ color: "#64748b", fontSize: "0.72rem" }} />
            <GapShading runs={gaps} />
            {hasBand && (
              <Area
                type="linear"
                dataKey="band"
                name="80% interval"
                fill={ACCENT}
                fillOpacity={0.15}
                stroke="none"
                activeDot={false}
                isAnimationActive={false}
                legendType="square"
              />
            )}
            <Line type="linear" dataKey="profile" name="profile" stroke="#64748b" strokeDasharray="2 3" dot={false} isAnimationActive={false} />
            {series.q50 && (
              <Line type="linear" dataKey="q50" name="q50" stroke="#34d399" strokeDasharray="6 3" dot={false} isAnimationActive={false} />
            )}
            {series.q90 && (
              <Line type="linear" dataKey="q90" name="q90" stroke="#f59e0b" strokeDasharray="6 3" dot={false} isAnimationActive={false} />
            )}
            <Line type="linear" dataKey="point" name="point" stroke={ACCENT} strokeWidth={2} dot={false} isAnimationActive={false} />
            <Line
              type="linear"
              dataKey="truth"
              name="truth"
              stroke="#e2e8f0"
              strokeWidth={0}
              dot={{ r: 2, fill: "#e2e8f0", strokeWidth: 0 }}
              activeDot={{ r: 3 }}
              connectNulls={false}
              isAnimationActive={false}
              legendType="circle"
            />
          </ComposedChart>
        </ResponsiveContainer>
      </Box>

      {hasProb && (
        <Box sx={{ height: 140 }} aria-label="Probability chart">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={data} margin={{ top: 4, right: 20, left: 0, bottom: 4 }} syncId="forecast">
              <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} />
              <XAxis {...xAxis} />
              <YAxis {...axisProps} width={48} domain={[0, 1]} />
              <Tooltip contentStyle={tooltipStyle} labelFormatter={labelFmt} formatter={(v: unknown, name: string) => [tooltipValue(v), name]} />
              <Legend wrapperStyle={{ color: "#64748b", fontSize: "0.72rem" }} />
              <GapShading runs={gaps} />
              {series.p_zero && (
                <Line type="linear" dataKey="p_zero" name="p_zero" stroke="#a78bfa" dot={false} isAnimationActive={false} />
              )}
              {series.outage_q && (
                <Line type="linear" dataKey="outage_q" name="outage_q" stroke="#f87171" dot={false} isAnimationActive={false} />
              )}
            </ComposedChart>
          </ResponsiveContainer>
        </Box>
      )}
    </Box>
  );
}
