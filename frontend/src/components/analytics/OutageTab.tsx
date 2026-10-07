"use client";

import { useMemo } from "react";
import Box from "@mui/material/Box";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { TemporalPage } from "@/types/payload";
import type { SpatialNode } from "@/types/registry";
import { contiguousRuns, fmt, fmtPct, timestepLabel } from "@/lib/v5";
import { ACCENT, axisProps, gridStroke, NotAvailable, SectionTitle, StatCard, tooltipStyle, tooltipValue } from "./shared";
import { GapShading, slotAxis } from "./ForecastTab";

interface Props {
  page: TemporalPage;
  node: SpatialNode | null;
}

export default function OutageTab({ page, node }: Props) {
  const net = page.network;
  const dm = page.day_metrics;
  const series = node ? page.node_series[String(node.id)] : undefined;

  const networkData = useMemo(
    () =>
      net.observed_frac.map((f, t) => ({
        t,
        observed_frac: f,
        real_gap_mass_ratio: net.real_gap_mass_ratio[t] ?? null,
      })),
    [net],
  );

  const nodeData = useMemo(
    () =>
      series
        ? series.gap_steps.map((g, t) => ({
            t,
            gap_steps: g,
            point: series.point[t] ?? null,
            profile: series.profile[t] ?? null,
          }))
        : [],
    [series],
  );
  const gaps = useMemo(() => (series ? contiguousRuns(series.truth, (v) => v == null) : []), [series]);

  const xAxis = slotAxis(page);
  const labelFmt = (t: unknown) => timestepLabel(Number(t), page.timestamps, page.step_minutes);
  const formatter = (v: unknown, name: string) => [tooltipValue(v), name];

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 1.5 }}>
      <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1 }}>
        <StatCard label="Observed" value={fmtPct(dm.observed_frac)} hint={`${dm.n_observed} / ${dm.n_targets} targets`} />
        <StatCard label="Collapse rate" value={fmtPct(dm.collapse_rate)} hint="P(point < 1 | offline, profile > 10)" />
        <StatCard label="Gap mass ratio" value={fmt(dm.real_gap_mass_ratio)} hint="Σ point / Σ profile offline · 1 = no collapse" />
        <StatCard label="RMSE all / ind" value={`${fmt(dm.obs_all_rmse, 2)} / ${fmt(dm.obs_ind_rmse, 2)}`} />
      </Box>

      <SectionTitle hint="network-wide, per timestep">Sensor availability vs real-gap mass ratio</SectionTitle>
      <Box sx={{ height: 260 }} aria-label="Network outage chart">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={networkData} margin={{ top: 4, right: 8, left: 0, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} />
            <XAxis {...xAxis} />
            <YAxis yAxisId="frac" {...axisProps} width={48} domain={[0, 1]} tickFormatter={(v: number) => `${Math.round(v * 100)}%`} />
            <YAxis yAxisId="ratio" orientation="right" {...axisProps} width={48} />
            <Tooltip contentStyle={tooltipStyle} labelFormatter={labelFmt} formatter={formatter} />
            <Legend wrapperStyle={{ color: "#64748b", fontSize: "0.72rem" }} />
            <ReferenceLine yAxisId="ratio" y={1} stroke="#f59e0b" strokeDasharray="4 4" />
            <Area
              yAxisId="frac"
              type="step"
              dataKey="observed_frac"
              name="observed fraction"
              fill="#34d399"
              fillOpacity={0.12}
              stroke="#34d399"
              isAnimationActive={false}
            />
            <Line
              yAxisId="ratio"
              type="linear"
              dataKey="real_gap_mass_ratio"
              name="real-gap mass ratio"
              stroke="#f87171"
              strokeWidth={2}
              dot={false}
              connectNulls={false}
              isAnimationActive={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </Box>

      <SectionTitle hint={node ? `${node.location_name} · red bands = offline` : undefined}>
        Node gap length and forecast during outages
      </SectionTitle>
      {!node ? (
        <NotAvailable>Select a node in the table above to see its outage gaps.</NotAvailable>
      ) : !series ? (
        <NotAvailable>No series for node {node.id} on {page.date}.</NotAvailable>
      ) : (
        <Box sx={{ height: 240 }} aria-label="Node gap chart">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={nodeData} margin={{ top: 4, right: 8, left: 0, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} />
              <XAxis {...xAxis} />
              <YAxis yAxisId="gap" {...axisProps} width={48} allowDecimals={false} />
              <YAxis yAxisId="count" orientation="right" {...axisProps} width={48} />
              <Tooltip contentStyle={tooltipStyle} labelFormatter={labelFmt} formatter={formatter} />
              <Legend wrapperStyle={{ color: "#64748b", fontSize: "0.72rem" }} />
              <GapShading runs={gaps} />
              <Area
                yAxisId="gap"
                type="step"
                dataKey="gap_steps"
                name="gap steps"
                fill="#ef4444"
                fillOpacity={0.15}
                stroke="#ef4444"
                isAnimationActive={false}
              />
              <Line yAxisId="count" type="linear" dataKey="profile" name="profile" stroke="#64748b" strokeDasharray="2 3" dot={false} isAnimationActive={false} />
              <Line yAxisId="count" type="linear" dataKey="point" name="point" stroke={ACCENT} strokeWidth={2} dot={false} isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </Box>
      )}
    </Box>
  );
}
