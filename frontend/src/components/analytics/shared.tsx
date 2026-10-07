"use client";

import type { ReactNode } from "react";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Chip from "@mui/material/Chip";
import Typography from "@mui/material/Typography";
import type { GridColDef, GridValidRowModel } from "@mui/x-data-grid";
import type { NodeRole } from "@/types/payload";
import { fmt, fmtPct } from "@/lib/v5";

export const ACCENT = "#60a5fa";
export const HOLDOUT = "#a78bfa";

export const gridSx = {
  border: "1px solid rgba(255,255,255,0.06)",
  bgcolor: "rgba(255,255,255,0.01)",
  color: "#cbd5e1",
  fontSize: "0.76rem",
  "& .MuiDataGrid-columnHeaders": {
    bgcolor: "rgba(255,255,255,0.03)",
    borderBottom: "1px solid rgba(255,255,255,0.06)",
  },
  "& .MuiDataGrid-columnHeader, & .MuiDataGrid-filler, & .MuiDataGrid-scrollbarFiller": {
    bgcolor: "#111214",
  },
  "& .MuiDataGrid-columnHeaderTitle": { color: "#64748b", fontWeight: 600, fontSize: "0.72rem" },
  "& .MuiDataGrid-cell": { borderBottom: "1px solid rgba(255,255,255,0.03)" },
  "& .MuiDataGrid-row": { cursor: "pointer" },
  "& .MuiDataGrid-row:hover": { bgcolor: "rgba(96,165,250,0.05)" },
  "& .MuiDataGrid-row.Mui-selected": { bgcolor: "rgba(96,165,250,0.12)" },
  "& .MuiDataGrid-row.Mui-selected:hover": { bgcolor: "rgba(96,165,250,0.16)" },
  "& .MuiDataGrid-footerContainer": { borderTop: "1px solid rgba(255,255,255,0.06)", color: "#475569" },
  "& .MuiTablePagination-root": { color: "#475569", fontSize: "0.72rem" },
  "& .MuiDataGrid-virtualScroller": { bgcolor: "transparent" },
  "& .MuiDataGrid-overlay": { bgcolor: "transparent", color: "#475569" },
} as const;

export const tooltipStyle = {
  background: "rgba(10,10,10,0.96)",
  border: "1px solid rgba(255,255,255,0.08)",
  color: "#e2e8f0",
  fontSize: "0.72rem",
  borderRadius: 6,
};

export const axisProps = {
  stroke: "#1e293b",
  tick: { fill: "#64748b", fontSize: 10 },
};

export const gridStroke = "rgba(255,255,255,0.04)";

/** Recharts tooltip formatter: ranges as [lo, hi], nulls as "—" (never 0). */
export function tooltipValue(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map((v) => fmt(v as number, 1)).join(", ")}]`;
  return typeof value === "number" ? fmt(value, 3) : "—";
}

/** Numeric DataGrid column; null renders as "—". */
export function numCol<R extends GridValidRowModel>(
  field: string,
  headerName: string,
  opts: { width?: number; digits?: number; pct?: boolean; description?: string } = {},
): GridColDef<R> {
  const { width = 96, digits = 2, pct = false, description } = opts;
  return {
    field,
    headerName,
    description: description ?? headerName,
    width,
    type: "number",
    valueFormatter: (value: number | null | undefined) => (pct ? fmtPct(value) : fmt(value, digits)),
  };
}

export function RoleChip({ role }: { role: NodeRole }) {
  return role === "holdout" ? (
    <Chip
      label="holdout"
      size="small"
      variant="outlined"
      sx={{ height: 18, fontSize: "0.65rem", color: HOLDOUT, borderColor: "rgba(167,139,250,0.5)" }}
    />
  ) : (
    <Chip
      label="observed"
      size="small"
      variant="outlined"
      sx={{ height: 18, fontSize: "0.65rem", color: "#6ee7b7", borderColor: "rgba(52,211,153,0.4)" }}
    />
  );
}

/** No observation anywhere in the analysis window: hollow, like the map ring. */
export function GhostChip({ title }: { title: string }) {
  return (
    <Chip
      label="ghost"
      size="small"
      variant="outlined"
      title={title}
      sx={{ height: 18, fontSize: "0.65rem", color: "#94a3b8", borderColor: "rgba(148,163,184,0.45)", borderStyle: "dashed" }}
    />
  );
}

export function NotAvailable({ children }: { children?: ReactNode }) {
  return (
    <Alert
      severity="info"
      sx={{
        bgcolor: "rgba(96,165,250,0.06)",
        color: "#93c5fd",
        border: "1px solid rgba(96,165,250,0.15)",
        "& .MuiAlert-icon": { color: ACCENT },
        fontSize: "0.8rem",
      }}
    >
      {children ?? "Not available for this model."}
    </Alert>
  );
}

export function SectionTitle({ children, hint }: { children: ReactNode; hint?: ReactNode }) {
  return (
    <Box sx={{ display: "flex", alignItems: "baseline", gap: 1, mb: 0.75 }}>
      <Typography variant="subtitle2" sx={{ color: "#94a3b8", fontSize: "0.8rem" }}>
        {children}
      </Typography>
      {hint && (
        <Typography variant="caption" sx={{ color: "#475569", fontSize: "0.7rem" }}>
          {hint}
        </Typography>
      )}
    </Box>
  );
}

export function StatCard({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <Box
      sx={{
        minWidth: 120,
        flex: "1 1 120px",
        p: 1.25,
        borderRadius: 1.5,
        border: "1px solid rgba(255,255,255,0.06)",
        bgcolor: "rgba(255,255,255,0.02)",
      }}
    >
      <Typography sx={{ color: "#64748b", fontSize: "0.68rem", textTransform: "uppercase", letterSpacing: "0.05em" }}>
        {label}
      </Typography>
      <Typography sx={{ color: "#e2e8f0", fontSize: "1.05rem", fontFamily: "monospace" }}>{value}</Typography>
      {hint && <Typography sx={{ color: "#475569", fontSize: "0.66rem" }}>{hint}</Typography>}
    </Box>
  );
}
