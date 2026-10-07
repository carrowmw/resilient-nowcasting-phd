"use client";

import Box from "@mui/material/Box";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { Calibration, Capabilities, ReliabilityBin, WindowMetrics } from "@/types/payload";
import { fmt, fmtPct } from "@/lib/v5";
import { ACCENT, axisProps, gridStroke, NotAvailable, SectionTitle, StatCard, tooltipStyle } from "./shared";

function ReliabilityChart({ bins, color, xLabel }: { bins: ReliabilityBin[]; color: string; xLabel: string }) {
  return (
    <ResponsiveContainer width="100%" height={260}>
      <ComposedChart data={bins} margin={{ top: 8, right: 16, left: 0, bottom: 20 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} />
        <XAxis
          dataKey="mean_predicted"
          type="number"
          domain={[0, 1]}
          {...axisProps}
          tickFormatter={(v: number) => v.toFixed(1)}
          label={{ value: xLabel, position: "insideBottom", offset: -10, fill: "#64748b", fontSize: 10 }}
        />
        <YAxis type="number" domain={[0, 1]} {...axisProps} width={44} />
        <Tooltip
          contentStyle={tooltipStyle}
          labelFormatter={(v: unknown) => `predicted ${fmt(Number(v), 3)}`}
          formatter={(v: unknown, _name: string, item: { payload?: ReliabilityBin }) => [
            `${fmt(Number(v), 3)} (n = ${item.payload?.count ?? "—"})`,
            "observed",
          ]}
        />
        {/* Presentational y = x: never hoverable data. */}
        <ReferenceLine
          segment={[
            { x: 0, y: 0 },
            { x: 1, y: 1 },
          ]}
          stroke="#f59e0b"
          strokeDasharray="6 3"
        />
        <Line
          type="linear"
          dataKey="observed_frac"
          stroke={color}
          strokeWidth={1}
          dot={{ r: 4, fill: color, stroke: color }}
          activeDot={{ r: 6 }}
          isAnimationActive={false}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

interface Props {
  calibration: Calibration;
  dayMetrics: WindowMetrics;
  windowMetrics: WindowMetrics | null;
  capabilities: Capabilities;
}

export default function CalibrationTab({ calibration, dayMetrics, windowMetrics, capabilities }: Props) {
  const pit = calibration.pit_histogram;
  const pZero = calibration.p_zero_reliability;
  const outage = capabilities.outage ? calibration.outage_reliability : null;
  const uniform = pit && pit.length > 0 ? 1 / pit.length : null;
  const both = (pick: (w: WindowMetrics) => number | null, f: (v: number | null) => string) =>
    `${f(pick(dayMetrics))} · ${windowMetrics ? f(pick(windowMetrics)) : "—"}`;

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <Box>
        <SectionTitle hint="day · analysis window">Coverage</SectionTitle>
        <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1 }}>
          <StatCard label="80% interval" value={both((w) => w.interval80_coverage, (v) => fmtPct(v))} hint="target 80%" />
          <StatCard label="q50 coverage" value={both((w) => w.q50_coverage, (v) => fmtPct(v))} hint="target 50%" />
          <StatCard label="q90 coverage" value={both((w) => w.q90_coverage, (v) => fmtPct(v))} hint="target 90%" />
          <StatCard label="CRPS" value={both((w) => w.crps, (v) => fmt(v))} hint="lower is better" />
        </Box>
      </Box>

      <Box sx={{ display: "flex", gap: 2, flexWrap: "wrap" }}>
        <Box sx={{ flex: 1, minWidth: 300 }}>
          <SectionTitle hint="flat bars = calibrated">PIT histogram</SectionTitle>
          {capabilities.uncertainty && pit && pit.length > 0 ? (
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={pit} margin={{ top: 8, right: 16, left: 0, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} />
                <XAxis
                  dataKey="bin_center"
                  {...axisProps}
                  tickFormatter={(v: number) => v.toFixed(2)}
                  label={{ value: "mid-PIT", position: "insideBottom", offset: -10, fill: "#64748b", fontSize: 10 }}
                />
                <YAxis {...axisProps} width={44} />
                <Tooltip contentStyle={tooltipStyle} formatter={(v: unknown) => [fmt(Number(v), 4), "density"]} />
                {uniform != null && (
                  <ReferenceLine y={uniform} stroke="#f59e0b" strokeDasharray="4 4" />
                )}
                <Bar dataKey="density" fill={ACCENT} fillOpacity={0.8} isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <NotAvailable>PIT histogram not available for this model.</NotAvailable>
          )}
        </Box>

        <Box sx={{ flex: 1, minWidth: 300 }}>
          <SectionTitle hint="event: slot reads zero or is missing">p_zero reliability</SectionTitle>
          {pZero && pZero.length > 0 ? (
            <ReliabilityChart bins={pZero} color="#a78bfa" xLabel="predicted p_zero" />
          ) : (
            <NotAvailable>p_zero reliability not available for this model.</NotAvailable>
          )}
        </Box>

        <Box sx={{ flex: 1, minWidth: 300 }}>
          <SectionTitle hint="event: slot is missing">Outage reliability</SectionTitle>
          {outage && outage.length > 0 ? (
            <ReliabilityChart bins={outage} color="#f87171" xLabel="predicted outage_q" />
          ) : (
            <NotAvailable>Outage reliability not available for this model.</NotAvailable>
          )}
        </Box>
      </Box>
    </Box>
  );
}
