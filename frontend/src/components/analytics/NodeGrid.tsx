"use client";

import { useMemo, useState } from "react";
import Box from "@mui/material/Box";
import TextField from "@mui/material/TextField";
import { DataGrid } from "@mui/x-data-grid";
import type { GridColDef } from "@mui/x-data-grid";
import type { NodeRole, TemporalPage } from "@/types/payload";
import type { SpatialIndex } from "@/types/registry";
import { isGhost, lastReportedText, observedFraction } from "@/lib/v5";
import { GhostChip, gridSx, numCol, RoleChip, SectionTitle } from "./shared";

export interface NodeRow {
  id: number;
  location_name: string;
  node_role: NodeRole;
  /** No observation anywhere in the analysis window (tooltip text when true). */
  ghost: string | null;
  n_observed: number | null;
  rmse: number | null;
  mae: number | null;
  bias: number | null;
  interval80_coverage: number | null;
  mean_point: number | null;
  peak_point: number | null;
  mean_truth: number | null;
  observed_frac: number | null;
  semantic_tags: string;
}

export function nodeRows(spatial: SpatialIndex, page: TemporalPage): NodeRow[] {
  return spatial.nodes.map((n) => {
    const m = page.node_metrics[String(n.id)];
    const s = page.node_series[String(n.id)];
    return {
      id: n.id,
      location_name: n.location_name,
      node_role: n.node_role,
      ghost: isGhost(n) ? `Ghost: no data in the analysis window (${lastReportedText(n)})` : null,
      n_observed: m?.n_observed ?? null,
      rmse: m?.rmse ?? null,
      mae: m?.mae ?? null,
      bias: m?.bias ?? null,
      interval80_coverage: m?.interval80_coverage ?? null,
      mean_point: m?.mean_point ?? null,
      peak_point: m?.peak_point ?? null,
      mean_truth: m?.mean_truth ?? null,
      observed_frac: observedFraction(s?.truth),
      semantic_tags: n.semantic_tags.join(", "),
    };
  });
}

const columns: GridColDef<NodeRow>[] = [
  { field: "id", headerName: "ID", width: 60, type: "number" },
  { field: "location_name", headerName: "Location", flex: 1, minWidth: 160 },
  {
    field: "node_role",
    headerName: "Role",
    width: 150,
    renderCell: ({ value, row }) => (
      <Box sx={{ display: "flex", gap: 0.5, alignItems: "center", height: "100%" }}>
        <RoleChip role={value as NodeRole} />
        {row.ghost && <GhostChip title={row.ghost} />}
      </Box>
    ),
  },
  numCol("rmse", "RMSE"),
  numCol("mae", "MAE"),
  numCol("bias", "Bias", { description: "mean(point − truth)" }),
  numCol("interval80_coverage", "80% cov", { pct: true, description: "80% interval coverage" }),
  numCol("mean_point", "Mean point", { digits: 1 }),
  numCol("peak_point", "Peak point", { digits: 1 }),
  numCol("mean_truth", "Mean truth", { digits: 1 }),
  numCol("observed_frac", "Observed", { pct: true, description: "Fraction of slots with ground truth" }),
  { field: "semantic_tags", headerName: "POI tags", width: 160, sortable: false },
];

interface Props {
  spatial: SpatialIndex;
  page: TemporalPage;
  selectedNodeId: number | null;
  onSelect: (id: number) => void;
}

export default function NodeGrid({ spatial, page, selectedNodeId, onSelect }: Props) {
  const [filterText, setFilterText] = useState("");
  const rows = useMemo(() => nodeRows(spatial, page), [spatial, page]);

  const filtered = useMemo(() => {
    const q = filterText.trim().toLowerCase();
    if (!q) return rows;
    return rows.filter(
      (r) =>
        r.location_name.toLowerCase().includes(q) ||
        String(r.id) === q ||
        r.node_role === q ||
        (q === "ghost" && r.ghost != null) ||
        r.semantic_tags.toLowerCase().includes(q),
    );
  }, [rows, filterText]);

  return (
    <Box>
      <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 0.75 }}>
        <SectionTitle
          hint={`${spatial.metadata.num_nodes} nodes · ${spatial.metadata.num_holdout_nodes} holdout · ${spatial.metadata.num_ghost_nodes} ghost`}
        >
          Nodes · {page.date}
        </SectionTitle>
        <TextField
          size="small"
          placeholder="Filter by location, ID, role, ghost or POI tag…"
          value={filterText}
          onChange={(e) => setFilterText(e.target.value)}
          slotProps={{ htmlInput: { "aria-label": "Filter nodes" } }}
          sx={{
            ml: "auto",
            width: 340,
            "& .MuiOutlinedInput-root": {
              bgcolor: "rgba(255,255,255,0.03)",
              "& fieldset": { borderColor: "rgba(255,255,255,0.09)" },
              "&:hover fieldset": { borderColor: "rgba(96,165,250,0.35)" },
              "&.Mui-focused fieldset": { borderColor: "#60a5fa" },
            },
            "& input": { color: "#e2e8f0", fontSize: "0.8rem", py: 0.6 },
            "& input::placeholder": { color: "#475569", opacity: 1 },
          }}
        />
      </Box>
      <Box sx={{ height: 320 }}>
        <DataGrid
          rows={filtered}
          columns={columns}
          density="compact"
          disableRowSelectionOnClick
          rowSelectionModel={selectedNodeId != null ? [selectedNodeId] : []}
          onRowClick={(p) => onSelect(Number(p.id))}
          initialState={{
            sorting: { sortModel: [{ field: "rmse", sort: "desc" }] },
            pagination: { paginationModel: { pageSize: 25 } },
          }}
          pageSizeOptions={[25, 50, 100]}
          aria-label="Node metrics"
          getRowClassName={(p) => (p.row.ghost ? "node-ghost" : "")}
          sx={{ ...gridSx, "& .node-ghost": { color: "#64748b" } }}
        />
      </Box>
    </Box>
  );
}
