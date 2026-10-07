"use client";

import { useMemo } from "react";
import Box from "@mui/material/Box";
import { DataGrid } from "@mui/x-data-grid";
import type { GridColDef } from "@mui/x-data-grid";
import {
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { AlphaNodeEntry, Capabilities, TemporalPage, TripwireEntry } from "@/types/payload";
import { attentionConcentration, fmt, fmtPct, meanDefined, timestepLabel } from "@/lib/v5";
import { ACCENT, gridSx, NotAvailable, SectionTitle, tooltipStyle, tooltipValue, gridStroke, axisProps } from "./shared";
import { slotAxis } from "./ForecastTab";

const alphaColumns: GridColDef<AlphaNodeEntry & { id: number }>[] = [
  { field: "node_id", headerName: "ID", width: 64, type: "number" },
  { field: "location_name", headerName: "Location", flex: 1, minWidth: 140 },
  {
    field: "total_outgoing_attention",
    headerName: "Outgoing attention",
    width: 150,
    type: "number",
    valueFormatter: (v: number | null) => fmt(v, 3),
  },
];

// Neutral reference line (inputs available); validated against the #0a0a0a surface
// alongside ACCENT, and dashed so it never relies on colour alone.
const AVAILABLE_COLOR = "#64748b";

const KNOWN_TRIPWIRE_KEYS = ["timestep", "timestep_end", "node_id", "location_name", "event_type"];

function cellText(v: unknown): string {
  if (v == null) return "—";
  return typeof v === "object" ? JSON.stringify(v) : String(v);
}

/** Columns for the known tripwire fields plus any extra keys present. */
export function tripwireColumns(trail: TripwireEntry[], page: TemporalPage): GridColDef[] {
  const keys = new Set<string>();
  for (const e of trail) for (const k of Object.keys(e)) keys.add(k);
  const extra = [...keys].filter((k) => !KNOWN_TRIPWIRE_KEYS.includes(k)).sort();
  const time = (v: unknown) =>
    typeof v === "number" ? `${timestepLabel(v, page.timestamps, page.step_minutes)} (t=${v})` : "—";
  const cols: GridColDef[] = [
    { field: "timestep", headerName: "From", width: 110, valueFormatter: time },
  ];
  if (keys.has("timestep_end")) cols.push({ field: "timestep_end", headerName: "To", width: 110, valueFormatter: time });
  if (keys.has("node_id")) cols.push({ field: "node_id", headerName: "Node", width: 70, valueFormatter: cellText });
  if (keys.has("location_name")) cols.push({ field: "location_name", headerName: "Location", width: 160, valueFormatter: cellText });
  cols.push({ field: "event_type", headerName: "Safeguard", flex: 1, minWidth: 160, valueFormatter: cellText });
  for (const k of extra) cols.push({ field: k, headerName: k, width: 120, valueFormatter: cellText });
  return cols;
}

interface Props {
  page: TemporalPage;
  capabilities: Capabilities;
  onSelectNode: (id: number) => void;
}

export default function NetworkTab({ page, capabilities, onSelectNode }: Props) {
  const alpha = capabilities.graph ? page.alpha_node_leaderboard : null;
  const trail = page.tripwire_audit_trail;

  const concentration = useMemo(
    () =>
      capabilities.graph
        ? attentionConcentration(page.edge_attention, Object.keys(page.node_series).length, page.timesteps)
        : null,
    [capabilities.graph, page],
  );
  const concentrationData = useMemo(
    () =>
      (concentration?.evenness ?? []).map((evenness, t) => ({
        t,
        evenness,
        effective: concentration?.effective[t] ?? null,
        available: concentration?.available[t] ?? null,
      })),
    [concentration],
  );
  const hasConcentration = concentration != null && concentration.evenness.some((v) => v != null);
  const slotLabel = (t: unknown) => timestepLabel(Number(t), page.timestamps, page.step_minutes);
  const trailRows = useMemo(() => trail.map((e, i) => ({ ...e, id: i })), [trail]);
  const trailCols = useMemo(() => tripwireColumns(trail, page), [trail, page]);

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <Box sx={{ display: "flex", gap: 2, flexWrap: "wrap" }}>
        <Box sx={{ flex: 1, minWidth: 300 }}>
          <SectionTitle hint="Σ outgoing attention over the day">Alpha-node leaderboard</SectionTitle>
          {alpha && alpha.length > 0 ? (
            <Box sx={{ height: 300 }}>
              <DataGrid
                rows={alpha.map((r) => ({ ...r, id: r.node_id }))}
                columns={alphaColumns}
                density="compact"
                hideFooter
                disableRowSelectionOnClick
                onRowClick={(p) => onSelectNode(Number(p.id))}
                aria-label="Alpha-node leaderboard"
                sx={gridSx}
              />
            </Box>
          ) : (
            <NotAvailable>Attention leaderboard not available for this model.</NotAvailable>
          )}
        </Box>
        <Box sx={{ flex: 1, minWidth: 300 }}>
          <SectionTitle
            hint={
              hasConcentration
                ? `1 = spread evenly over every input · day mean ${fmt(meanDefined(concentration.evenness), 2)}, ` +
                  `${fmtPct(meanDefined(concentration.selfShare), 0)} kept on itself`
                : undefined
            }
          >
            Attention evenness
          </SectionTitle>
          {hasConcentration ? (
            <ResponsiveContainer width="100%" height={300}>
              <ComposedChart data={concentrationData} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} />
                <XAxis {...slotAxis(page)} />
                <YAxis {...axisProps} width={44} domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} />
                <Tooltip
                  contentStyle={tooltipStyle}
                  labelFormatter={slotLabel}
                  formatter={(v: unknown) => [tooltipValue(v), "evenness"]}
                />
                <Line type="linear" dataKey="evenness" stroke={ACCENT} strokeWidth={2} dot={false} connectNulls={false} isAnimationActive={false} />
              </ComposedChart>
            </ResponsiveContainer>
          ) : (
            <NotAvailable>Attention not available for this model.</NotAvailable>
          )}
        </Box>
      </Box>

      {hasConcentration && (
        <Box>
          <SectionTitle hint="mean per sensor; inputs = reporting neighbours + the sensor itself">
            Inputs attended vs available
          </SectionTitle>
          <ResponsiveContainer width="100%" height={240}>
            <ComposedChart data={concentrationData} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} />
              <XAxis {...slotAxis(page)} />
              <YAxis {...axisProps} width={44} domain={[0, "auto"]} allowDecimals={false} />
              <Tooltip
                contentStyle={tooltipStyle}
                labelFormatter={slotLabel}
                formatter={(v: unknown, name: unknown) => [typeof v === "number" ? fmt(v, 1) : "—", String(name)]}
              />
              <Legend wrapperStyle={{ color: "#64748b", fontSize: "0.72rem" }} />
              <Line
                type="linear"
                dataKey="available"
                name="Inputs available"
                stroke={AVAILABLE_COLOR}
                strokeWidth={2}
                strokeDasharray="5 4"
                dot={false}
                isAnimationActive={false}
              />
              <Line
                type="linear"
                dataKey="effective"
                name="Effectively attended"
                stroke={ACCENT}
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </Box>
      )}

      <Box>
        <SectionTitle hint={`${trail.length} events`}>Tripwire audit trail</SectionTitle>
        {trail.length > 0 ? (
          <Box sx={{ height: 280 }}>
            <DataGrid
              rows={trailRows}
              columns={trailCols}
              density="compact"
              disableRowSelectionOnClick
              onRowClick={(p) => {
                const id = (p.row as TripwireEntry).node_id;
                if (typeof id === "number") onSelectNode(id);
              }}
              initialState={{ pagination: { paginationModel: { pageSize: 25 } } }}
              pageSizeOptions={[25, 50, 100]}
              aria-label="Tripwire audit trail"
              sx={gridSx}
            />
          </Box>
        ) : (
          <NotAvailable>No tripwire events on {page.date}.</NotAvailable>
        )}
      </Box>
    </Box>
  );
}
