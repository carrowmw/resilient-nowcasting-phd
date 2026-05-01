"use client";

import { useEffect, useState, useMemo } from "react";
import Link from "next/link";
import AppBar from "@mui/material/AppBar";
import Toolbar from "@mui/material/Toolbar";
import Typography from "@mui/material/Typography";
import IconButton from "@mui/material/IconButton";
import TextField from "@mui/material/TextField";
import Tabs from "@mui/material/Tabs";
import Tab from "@mui/material/Tab";
import Box from "@mui/material/Box";
import Alert from "@mui/material/Alert";
import CircularProgress from "@mui/material/CircularProgress";
import { DataGrid } from "@mui/x-data-grid";
import type { GridColDef, GridRowSelectionModel } from "@mui/x-data-grid";
import {
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import type { MasterIndex, NodeEntry } from "@/types/payload";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface NodeRow {
  id: number;
  location_name: string;
  is_simulated: boolean;
  avg_mu: number;
  max_mu: number;
  avg_sigma: number;
  semantic_tags: string;
}

// Each chart point has numeric keys for mu and ci bounds per selected node.
type ChartPoint = Record<string, number>;

// ---------------------------------------------------------------------------
// Colour palette for multi-node overlay
// ---------------------------------------------------------------------------

const PALETTE = [
  "#60a5fa", // blue-400
  "#34d399", // emerald-400
  "#f59e0b", // amber-400
  "#f87171", // red-400
  "#a78bfa", // violet-400
  "#fb7185", // rose-400
  "#22d3ee", // cyan-400
  "#4ade80", // green-400
];

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function AnalyticsPage() {
  const [masterIndex, setMasterIndex] = useState<MasterIndex | null>(null);
  const [loading, setLoading] = useState(true);
  const [filterText, setFilterText] = useState("");
  const [selectedIds, setSelectedIds] = useState<number[]>([]);
  const [activeTab, setActiveTab] = useState(0);

  useEffect(() => {
    fetch("/data/master_index.json")
      .then((r) => (r.ok ? r.json() : null))
      .then((d: MasterIndex | null) => {
        setMasterIndex(d);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  // ---- DataGrid rows (computed aggregates) --------------------------------

  const rows: NodeRow[] = useMemo(() => {
    if (!masterIndex) return [];
    return masterIndex.nodes.map((node: NodeEntry) => {
      const preds = node.timeseries.map((e) => e.prediction);
      const sigmas = node.timeseries.map((e) => e.uncertainty);
      const n = preds.length || 1;
      const avg_mu = preds.reduce((a, b) => a + b, 0) / n;
      const max_mu = preds.length ? Math.max(...preds) : 0;
      const avg_sigma = sigmas.reduce((a, b) => a + b, 0) / n;
      return {
        id: node.id,
        location_name: node.location_name,
        is_simulated: node.is_simulated,
        avg_mu,
        max_mu,
        avg_sigma,
        semantic_tags: node.semantic_tags?.join(", ") || "—",
      };
    });
  }, [masterIndex]);

  // ---- Omnibar filter -----------------------------------------------------

  const filteredRows = useMemo(() => {
    const q = filterText.trim().toLowerCase();
    if (!q) return rows;
    return rows.filter(
      (r) =>
        r.location_name.toLowerCase().includes(q) ||
        String(r.id) === filterText.trim() ||
        r.semantic_tags.toLowerCase().includes(q)
    );
  }, [rows, filterText]);

  // ---- Chart data (Amendment 1: separate ci_lower / ci_upper fields) -------
  // Each point: { t, mu_{id}, ci_lower_{id}, ci_upper_{id} }
  // Rendered as: <Area dataKey={['ci_lower_{id}', 'ci_upper_{id}']} /> —
  // Recharts' native range support; no background-colour clipping needed.

  const chartData: ChartPoint[] = useMemo(() => {
    if (!masterIndex || selectedIds.length === 0) return [];
    const nodeMap = new Map(masterIndex.nodes.map((n) => [n.id, n]));
    const T = masterIndex.metadata.timesteps;
    return Array.from({ length: T }, (_, t) => {
      const point: ChartPoint = { t };
      for (const id of selectedIds) {
        const node = nodeMap.get(id);
        if (!node) continue;
        const ts = node.timeseries[t];
        if (!ts) continue;
        const halfCI = 1.96 * ts.uncertainty;
        point[`mu_${id}`] = ts.prediction;
        point[`ci_lower_${id}`] = Math.max(0, ts.prediction - halfCI);
        point[`ci_upper_${id}`] = ts.prediction + halfCI;
      }
      return point;
    });
  }, [masterIndex, selectedIds]);

  // ---- DataGrid column definitions ----------------------------------------

  const columns: GridColDef<NodeRow>[] = [
    { field: "id", headerName: "ID", width: 64, type: "number" },
    { field: "location_name", headerName: "Location", flex: 1, minWidth: 150 },
    { field: "is_simulated", headerName: "Sim", width: 56, type: "boolean" },
    {
      field: "avg_mu",
      headerName: "Avg μ",
      width: 95,
      type: "number",
      valueFormatter: (value: number) => value.toFixed(3),
    },
    {
      field: "max_mu",
      headerName: "Peak μ",
      width: 95,
      type: "number",
      valueFormatter: (value: number) => value.toFixed(3),
    },
    {
      field: "avg_sigma",
      headerName: "Avg σ",
      width: 95,
      type: "number",
      valueFormatter: (value: number) => value.toFixed(3),
    },
    {
      field: "semantic_tags",
      headerName: "POI Tags",
      flex: 1,
      minWidth: 120,
      sortable: false,
    },
  ];

  // ---- Loading / error states ---------------------------------------------

  if (loading) {
    return (
      <Box
        sx={{
          display: "flex",
          height: "100vh",
          alignItems: "center",
          justifyContent: "center",
          bgcolor: "#0a0a0a",
        }}
      >
        <CircularProgress sx={{ color: "#60a5fa" }} />
      </Box>
    );
  }

  if (!masterIndex) {
    return (
      <Box
        sx={{
          display: "flex",
          height: "100vh",
          alignItems: "center",
          justifyContent: "center",
          bgcolor: "#0a0a0a",
        }}
      >
        <Alert severity="error">
          Failed to load master_index.json — check that{" "}
          <code>export_ui_payload.py</code> has been run.
        </Alert>
      </Box>
    );
  }

  // ---- Full layout --------------------------------------------------------

  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        height: "100vh",
        bgcolor: "#0a0a0a",
        color: "#e2e8f0",
        overflow: "hidden",
      }}
    >
      {/* AppBar */}
      <AppBar
        position="static"
        elevation={0}
        sx={{
          bgcolor: "rgba(12,12,12,0.97)",
          borderBottom: "1px solid rgba(255,255,255,0.07)",
        }}
      >
        <Toolbar variant="dense" sx={{ minHeight: 48 }}>
          <Link href="/" style={{ color: "inherit", display: "flex" }}>
            <IconButton
              size="small"
              sx={{ color: "#60a5fa", mr: 1 }}
              aria-label="Back to map"
            >
              <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="currentColor"
              >
                <path d="M20 11H7.83l5.59-5.59L12 4l-8 8 8 8 1.41-1.41L7.83 13H20v-2z" />
              </svg>
            </IconButton>
          </Link>

          <Typography
            variant="subtitle1"
            sx={{ fontWeight: 700, letterSpacing: "0.05em", color: "#e2e8f0" }}
          >
            NSTAR Analytics
          </Typography>

          <Typography
            variant="caption"
            sx={{ ml: 2, color: "#334155", fontFamily: "monospace" }}
          >
            {masterIndex.metadata.num_nodes} nodes ·{" "}
            {masterIndex.metadata.timesteps}T ·{" "}
            {masterIndex.metadata.start_date}
          </Typography>
        </Toolbar>
      </AppBar>

      <Box
        sx={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          overflow: "hidden",
          p: 1.5,
          gap: 1.5,
        }}
      >
        {/* Omnibar */}
        <TextField
          fullWidth
          size="small"
          placeholder="Filter by location name, node ID, or POI tag…"
          value={filterText}
          onChange={(e) => setFilterText(e.target.value)}
          inputProps={{ "aria-label": "Filter nodes" }}
          sx={{
            "& .MuiOutlinedInput-root": {
              bgcolor: "rgba(255,255,255,0.03)",
              "& fieldset": { borderColor: "rgba(255,255,255,0.09)" },
              "&:hover fieldset": { borderColor: "rgba(96,165,250,0.35)" },
              "&.Mui-focused fieldset": { borderColor: "#60a5fa" },
            },
            "& input": { color: "#e2e8f0", fontSize: "0.85rem" },
            "& input::placeholder": { color: "#334155", opacity: 1 },
          }}
        />

        {/* DataGrid */}
        <Box sx={{ height: "36vh", minHeight: 160 }}>
          <DataGrid
            rows={filteredRows}
            columns={columns}
            checkboxSelection
            disableRowSelectionOnClick
            rowSelectionModel={selectedIds as GridRowSelectionModel}
            onRowSelectionModelChange={(ids: GridRowSelectionModel) =>
              setSelectedIds(ids as number[])
            }
            initialState={{
              sorting: { sortModel: [{ field: "avg_mu", sort: "desc" }] },
              pagination: { paginationModel: { pageSize: 25 } },
            }}
            pageSizeOptions={[25, 50, 100]}
            density="compact"
            sx={{
              border: "1px solid rgba(255,255,255,0.06)",
              bgcolor: "rgba(255,255,255,0.01)",
              color: "#cbd5e1",
              fontSize: "0.78rem",
              "& .MuiDataGrid-columnHeaders": {
                bgcolor: "rgba(255,255,255,0.03)",
                borderBottom: "1px solid rgba(255,255,255,0.06)",
              },
              "& .MuiDataGrid-columnHeaderTitle": {
                color: "#64748b",
                fontWeight: 600,
                fontSize: "0.75rem",
              },
              "& .MuiDataGrid-cell": {
                borderBottom: "1px solid rgba(255,255,255,0.03)",
              },
              "& .MuiDataGrid-row:hover": {
                bgcolor: "rgba(96,165,250,0.05)",
              },
              "& .MuiDataGrid-row.Mui-selected": {
                bgcolor: "rgba(96,165,250,0.09)",
              },
              "& .MuiDataGrid-row.Mui-selected:hover": {
                bgcolor: "rgba(96,165,250,0.13)",
              },
              "& .MuiCheckbox-root": { color: "#334155" },
              "& .MuiCheckbox-root.Mui-checked": { color: "#60a5fa" },
              "& .MuiDataGrid-footerContainer": {
                borderTop: "1px solid rgba(255,255,255,0.06)",
                color: "#334155",
              },
              "& .MuiTablePagination-root": {
                color: "#334155",
                fontSize: "0.75rem",
              },
              "& .MuiDataGrid-virtualScroller": { bgcolor: "transparent" },
            }}
          />
        </Box>

        {/* Tabs */}
        <Box sx={{ borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
          <Tabs
            value={activeTab}
            onChange={(_, v: number) => setActiveTab(v)}
            sx={{
              minHeight: 34,
              "& .MuiTab-root": {
                color: "#334155",
                textTransform: "none",
                fontSize: "0.78rem",
                minHeight: 34,
                py: 0,
                px: 2,
              },
              "& .Mui-selected": { color: "#60a5fa !important" },
              "& .Mui-disabled": { color: "#1e293b !important", opacity: 1 },
              "& .MuiTabs-indicator": { bgcolor: "#60a5fa", height: 2 },
            }}
          >
            <Tab label="Forecast Variance" value={0} />
            <Tab label="Network Influence" value={1} disabled />
            <Tab label="Self-Healing Log" value={2} disabled />
          </Tabs>
        </Box>

        {/* Tab 0 — Forecast Variance */}
        {activeTab === 0 && (
          <Box sx={{ flex: 1, overflow: "hidden", minHeight: 0 }}>
            {selectedIds.length === 0 ? (
              <Alert
                severity="info"
                sx={{
                  bgcolor: "rgba(96,165,250,0.07)",
                  color: "#93c5fd",
                  border: "1px solid rgba(96,165,250,0.15)",
                  "& .MuiAlert-icon": { color: "#60a5fa" },
                  fontSize: "0.82rem",
                }}
              >
                Select one or more nodes from the table above to plot their
                ZINB forecast variance (μ ± 1.96σ confidence interval).
              </Alert>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart
                  data={chartData}
                  margin={{ top: 4, right: 20, left: -12, bottom: 24 }}
                >
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="rgba(255,255,255,0.04)"
                  />
                  <XAxis
                    dataKey="t"
                    stroke="#1e293b"
                    tick={{ fill: "#334155", fontSize: 10 }}
                    label={{
                      value: "Timestep (15-min intervals)",
                      position: "insideBottom",
                      offset: -12,
                      fill: "#334155",
                      fontSize: 10,
                    }}
                  />
                  <YAxis
                    stroke="#1e293b"
                    tick={{ fill: "#334155", fontSize: 10 }}
                    width={48}
                  />
                  <Tooltip
                    contentStyle={{
                      background: "rgba(10,10,10,0.96)",
                      border: "1px solid rgba(255,255,255,0.08)",
                      color: "#e2e8f0",
                      fontSize: "0.72rem",
                      borderRadius: 6,
                    }}
                    formatter={(value: unknown, name: string) => [
                      Array.isArray(value)
                        ? `[${(value as number[]).map((v) => v.toFixed(2)).join(", ")}]`
                        : Number(value).toFixed(3),
                      name,
                    ]}
                  />
                  <Legend
                    wrapperStyle={{ color: "#475569", fontSize: "0.72rem" }}
                  />

                  {selectedIds.map((id, i) => {
                    const color = PALETTE[i % PALETTE.length];
                    const node = masterIndex.nodes.find((n) => n.id === id);
                    const label = node
                      ? node.location_name.replace(/^Ncl/, "")
                      : `Node ${id}`;

                    return [
                      /*
                       * Amendment 1 — native Recharts range Area.
                       * dataKey receives an array of two field names:
                       * Recharts reads [ci_lower_{id}, ci_upper_{id}] from
                       * each data point and fills the band between them.
                       * No background-colour clipping; the CartesianGrid
                       * remains fully visible through the transparent fill.
                       */
                      <Area
                        key={`ci_${id}`}
                        type="monotone"
                        dataKey={
                          [
                            `ci_lower_${id}`,
                            `ci_upper_${id}`,
                          ] as unknown as string
                        }
                        fill={color}
                        stroke="none"
                        fillOpacity={0.13}
                        name={`${label} 95% CI`}
                        legendType="square"
                        activeDot={false}
                        isAnimationActive={false}
                      />,
                      <Line
                        key={`mu_${id}`}
                        type="monotone"
                        dataKey={`mu_${id}`}
                        stroke={color}
                        strokeWidth={2}
                        dot={false}
                        name={`${label} μ`}
                        isAnimationActive={false}
                      />,
                    ];
                  })}
                </ComposedChart>
              </ResponsiveContainer>
            )}
          </Box>
        )}
      </Box>
    </Box>
  );
}
