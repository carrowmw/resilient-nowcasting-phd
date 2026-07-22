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
  BarChart,
  Bar,
  ReferenceLine,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import type {
  MasterIndex,
  NodeEntry,
  NodeRole,
  PITBin,
  UIPayload,
  AlphaNodeEntry,
  TripwireEntry,
} from "@/types/payload";
import type { SpatialIndex, TemporalPage, RunRegistry } from "@/types/registry";
import useTimeStore from "@/store/useTimeStore";
import { registryUrl, spatialUrl, pageUrl } from "@/lib/payloadSource";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface NodeRow {
  id: number;
  location_name: string;
  node_role: NodeRole;
  is_simulated: boolean;
  avg_mu: number;
  max_mu: number;
  avg_sigma: number | null; // null for models without a ZINB uncertainty head
  semantic_tags: string;
  mae_drop_nan: number | null;
  rmse_drop_nan: number | null;
  mae_fill_nan: number | null;
  rmse_fill_nan: number | null;
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
// Payload merge (mirrors MapVisualizer.tsx — kept in sync manually)
// ---------------------------------------------------------------------------

function mergePayload(spatial: SpatialIndex, page: TemporalPage): UIPayload {
  const m = spatial.metadata;
  const nodes: NodeEntry[] = spatial.nodes.map((n) => ({
    id: n.id,
    location_name: n.location_name,
    lat: n.lat,
    lon: n.lon,
    node_role: n.node_role,
    is_simulated: n.is_simulated,
    timeseries: page.node_timeseries[String(n.id)] ?? [],
    semantic_tags: n.semantic_tags,
  }));
  return {
    metadata: {
      num_nodes: m.num_nodes,
      num_real_nodes: m.num_real_nodes,
      num_simulated_nodes: m.num_simulated_nodes,
      num_holdout_nodes: m.num_holdout_nodes,
      holdout_manifest_sha: m.holdout_manifest_sha,
      timesteps: page.timesteps,
      seq_len: page.seq_len,
      veh_class: m.veh_class,
      dir: m.dir,
      start_date: page.date,
      end_date: page.date,
      wandb_run_id: m.wandb_run_id,
      git_commit_hash: m.git_commit_hash,
      hyperparameters: m.hyperparameters,
    },
    nodes,
    edges: page.edge_attention,
    pit_histogram: page.pit_histogram,
    zero_inflation_reliability: page.zero_inflation_reliability,
    global_imputation_burden: page.global_imputation_burden,
    global_healing_quality: page.global_healing_quality,
    attention_entropy_series: page.attention_entropy_series,
    tripwire_audit_trail: page.tripwire_audit_trail,
    alpha_node_leaderboard: page.alpha_node_leaderboard,
  };
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function AnalyticsPage() {
  const [masterIndex, setMasterIndex] = useState<MasterIndex | null>(null);
  const [loading, setLoading] = useState(true);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [filterText, setFilterText] = useState("");
  const [selectedIds, setSelectedIds] = useState<number[]>([]);
  const [activeTab, setActiveTab] = useState(0);

  const activeRunId = useTimeStore((s) => s.activeRunId);
  const activeDate = useTimeStore((s) => s.activeDate);
  const runRegistry = useTimeStore((s) => s.runRegistry);
  const setRunRegistry = useTimeStore((s) => s.setRunRegistry);
  const setActiveRunId = useTimeStore((s) => s.setActiveRunId);
  const setAvailableDates = useTimeStore((s) => s.setAvailableDates);
  const setActiveDate = useTimeStore((s) => s.setActiveDate);

  const [spatialIndex, setSpatialIndex] = useState<SpatialIndex | null>(null);

  // Stage 1 — registry (no-op if already loaded from map page)
  useEffect(() => {
    if (runRegistry.length > 0) return;
    const controller = new AbortController();
    fetch(registryUrl(), { signal: controller.signal })
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json() as Promise<RunRegistry>;
      })
      .then((registry) => {
        setFetchError(null);
        setRunRegistry(registry.runs);
        const def = registry.default_run_id ?? registry.runs[0]?.run_id ?? null;
        if (def) setActiveRunId(def);
      })
      .catch((err) => {
        if (err.name === "AbortError") return;
        setFetchError(`Failed to load run registry: ${err.message}`);
        setLoading(false);
      });
    return () => controller.abort();
  }, [runRegistry.length, setRunRegistry, setActiveRunId]);

  // Stage 2 — spatial index (re-fetches on run change)
  // Uses setSpatialIndex (state, not a ref) so Stage 3 re-triggers when spatial
  // arrives even if activeDate hasn't changed (e.g. navigating from the map page).
  useEffect(() => {
    if (!activeRunId) return;
    const controller = new AbortController();
    fetch(spatialUrl(activeRunId), { signal: controller.signal })
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json() as Promise<SpatialIndex>;
      })
      .then((spatial) => {
        setFetchError(null);
        setSpatialIndex(spatial);
        const dates = spatial.metadata.available_dates ?? [];
        setAvailableDates(dates);
        if (dates.length > 0) setActiveDate(dates[0]);
      })
      .catch((err) => {
        if (err.name === "AbortError") return;
        setFetchError(`Failed to load spatial index for ${activeRunId}: ${err.message}`);
        setLoading(false);
      });
    return () => controller.abort();
  }, [activeRunId, setAvailableDates, setActiveDate]);

  // Stage 3 — page fetch + merge (re-fetches on run, date, or spatial change)
  useEffect(() => {
    if (!activeRunId || !activeDate || !spatialIndex) return;
    // Guard against run-switch race: spatialIndex may still be for the previous run.
    if (spatialIndex.metadata.run_id !== activeRunId) return;
    const controller = new AbortController();
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setLoading(true);
    fetch(pageUrl(activeRunId, activeDate), { signal: controller.signal })
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json() as Promise<TemporalPage>;
      })
      .then((page) => {
        setFetchError(null);
        setMasterIndex(mergePayload(spatialIndex, page));
        setLoading(false);
      })
      .catch((err) => {
        if (err.name === "AbortError") return;
        setFetchError(`Failed to load page for ${activeRunId}/${activeDate}: ${err.message}`);
        setLoading(false);
      });
    return () => controller.abort();
  }, [activeRunId, activeDate, spatialIndex]);

  // ---- DataGrid rows (computed aggregates) --------------------------------

  const rows: NodeRow[] = useMemo(() => {
    if (!masterIndex) return [];
    return masterIndex.nodes.map((node: NodeEntry) => {
      const preds = node.timeseries.map((e) => e.prediction);
      // A baseline reports uncertainty=null; keep avg_sigma null rather than
      // averaging in fabricated zeros.
      const sigmas = node.timeseries
        .map((e) => e.uncertainty)
        .filter((v): v is number => v != null);
      const n = preds.length || 1;
      const avg_mu = preds.reduce((a, b) => a + b, 0) / n;
      const max_mu = preds.length ? Math.max(...preds) : 0;
      const avg_sigma =
        sigmas.length > 0 ? sigmas.reduce((a, b) => a + b, 0) / sigmas.length : null;
      const te = node.two_extremes;
      return {
        id: node.id,
        location_name: node.location_name,
        node_role:
          node.node_role ?? (node.is_simulated ? "simulated" : "observed"),
        is_simulated: node.is_simulated,
        avg_mu,
        max_mu,
        avg_sigma,
        semantic_tags: node.semantic_tags?.join(", ") || "—",
        mae_drop_nan: te?.mae_drop_nan ?? null,
        rmse_drop_nan: te?.rmse_drop_nan ?? null,
        mae_fill_nan: te?.mae_fill_nan ?? null,
        rmse_fill_nan: te?.rmse_fill_nan ?? null,
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
        // No CI band for models without uncertainty (baselines): collapse to the
        // point forecast rather than drawing a fabricated interval.
        const halfCI = ts.uncertainty != null ? 1.96 * ts.uncertainty : 0;
        point[`mu_${id}`] = ts.prediction;
        point[`ci_lower_${id}`] = Math.max(0, ts.prediction - halfCI);
        point[`ci_upper_${id}`] = ts.prediction + halfCI;
      }
      return point;
    });
  }, [masterIndex, selectedIds]);

  // ---- Reliability chart data (scatter dots) ------------------------------
  // The ComposedChart binds this data at the top level; the y = x reference
  // line is drawn presentationally via <ReferenceLine segment={...} />.

  const reliabilityData = useMemo(
    () => masterIndex?.zero_inflation_reliability ?? [],
    [masterIndex]
  );

  // ---- DataGrid column definitions ----------------------------------------

  const nullableFmt = (value: unknown): string => {
    if (value === null || value === undefined) return "—";
    return (value as number).toFixed(3);
  };

  const columns: GridColDef<NodeRow>[] = [
    { field: "id", headerName: "ID", width: 64, type: "number" },
    { field: "location_name", headerName: "Location", flex: 1, minWidth: 150 },
    { field: "node_role", headerName: "Role", width: 90, type: "string" },
    {
      field: "avg_mu",
      headerName: "Avg μ",
      width: 95,
      type: "number",
      valueFormatter: (value: number | null | undefined) =>
        value == null ? "—" : value.toFixed(3),
    },
    {
      field: "max_mu",
      headerName: "Peak μ",
      width: 95,
      type: "number",
      valueFormatter: (value: number | null | undefined) =>
        value == null ? "—" : value.toFixed(3),
    },
    {
      field: "avg_sigma",
      headerName: "Avg σ",
      width: 95,
      type: "number",
      // null for baselines (no uncertainty head) — show N/A, not a fake 0.
      valueFormatter: (value: number | null) =>
        value == null ? "N/A" : value.toFixed(3),
    },
    {
      field: "mae_drop_nan",
      headerName: "MAE (drop NaN)",
      width: 130,
      type: "number",
      valueFormatter: nullableFmt,
    },
    {
      field: "rmse_drop_nan",
      headerName: "RMSE (drop NaN)",
      width: 140,
      type: "number",
      valueFormatter: nullableFmt,
    },
    {
      field: "mae_fill_nan",
      headerName: "MAE (fill 0)",
      width: 110,
      type: "number",
      valueFormatter: nullableFmt,
    },
    {
      field: "rmse_fill_nan",
      headerName: "RMSE (fill 0)",
      width: 115,
      type: "number",
      valueFormatter: nullableFmt,
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
          {fetchError ?? "Failed to load payload — check the run registry and R2 bucket."}
        </Alert>
      </Box>
    );
  }

  const hasPIT =
    Array.isArray(masterIndex.pit_histogram) &&
    masterIndex.pit_histogram.length > 0;
  const hasReliability = reliabilityData.length > 0;

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

          {runRegistry.length > 0 && (
            <Box sx={{ ml: "auto", display: "flex", alignItems: "center", gap: 1 }}>
              <Typography
                variant="caption"
                sx={{ color: "#64748b", fontFamily: "monospace", fontSize: 11 }}
              >
                MODEL
              </Typography>
              <select
                value={activeRunId ?? ""}
                onChange={(e) => setActiveRunId(e.target.value)}
                aria-label="Select model run"
                style={{
                  background: "rgba(0,0,0,0.4)",
                  color: "#e2e8f0",
                  border: "1px solid rgba(148,163,184,0.3)",
                  borderRadius: 8,
                  padding: "4px 8px",
                  fontSize: 13,
                  fontFamily: "monospace",
                  cursor: "pointer",
                  outline: "none",
                }}
              >
                {runRegistry.map((r) => (
                  <option key={r.run_id} value={r.run_id} style={{ color: "#000" }}>
                    {r.run_id}
                  </option>
                ))}
              </select>
            </Box>
          )}
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
            <Tab label="Network Influence" value={1} />
            <Tab label="Self-Healing Log" value={2} />
            <Tab label="ZINB Calibration" value={3} />
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

        {/* Tab 1 — Network Influence */}
        {activeTab === 1 && (
          <Box sx={{ flex: 1, display: "flex", gap: 2, overflow: "hidden", flexDirection: { xs: "column", md: "row" } }}>
            <Box sx={{ flex: 1, minWidth: 300, display: "flex", flexDirection: "column" }}>
              <Typography variant="subtitle2" sx={{ mb: 1, color: "#94a3b8" }}>
                Alpha-Node Leaderboard
              </Typography>
              <Box sx={{ flex: 1, minHeight: 0 }}>
                <DataGrid
                  rows={(masterIndex?.alpha_node_leaderboard ?? []).map((r: AlphaNodeEntry) => ({ id: r.node_id, ...r }))}
                  columns={[
                    { field: "node_id", headerName: "ID", width: 60 },
                    { field: "location_name", headerName: "Location", flex: 1 },
                    { field: "total_outgoing_attention", headerName: "Outgoing Attention", width: 150, valueFormatter: (value: number) => value.toFixed(3) }
                  ]}
                  density="compact"
                  hideFooter
                  sx={{
                    bgcolor: "rgba(255,255,255,0.02)", color: "#cbd5e1",
                    "& .MuiDataGrid-columnHeaders": { bgcolor: "rgba(255,255,255,0.03)", borderBottom: "1px solid rgba(255,255,255,0.06)" },
                    "& .MuiDataGrid-cell": { borderBottom: "1px solid rgba(255,255,255,0.03)" },
                  }}
                />
              </Box>
            </Box>
            <Box sx={{ flex: 1, minWidth: 300 }}>
              <Typography variant="subtitle2" sx={{ mb: 1, color: "#94a3b8" }}>
                Spatial Over-Smoothing Index (Attention Entropy)
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <ComposedChart data={(masterIndex?.attention_entropy_series || []).map((v, i) => ({ t: i, entropy: v }))}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                  <XAxis dataKey="t" stroke="#1e293b" tick={{ fill: "#334155" }} />
                  <YAxis stroke="#1e293b" tick={{ fill: "#334155" }} />
                  <Tooltip contentStyle={{ background: "rgba(10,10,10,0.9)", borderColor: "#334155" }} />
                  <Line type="monotone" dataKey="entropy" stroke="#a78bfa" dot={false} strokeWidth={2} name="Shannon Entropy" />
                </ComposedChart>
              </ResponsiveContainer>
            </Box>
          </Box>
        )}

        {/* Tab 2 — Self-Healing Log */}
        {activeTab === 2 && (
          <Box sx={{ flex: 1, display: "flex", gap: 2, overflow: "hidden", flexDirection: { xs: "column", md: "row" } }}>
            <Box sx={{ flex: 1, minWidth: 300 }}>
              <Typography variant="subtitle2" sx={{ mb: 1, color: "#94a3b8" }}>
                Global Imputation Burden vs. Healing Quality
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                {(() => {
                  const burden = masterIndex?.global_imputation_burden ?? [];
                  const healingQ = masterIndex?.global_healing_quality;
                  const hasHealingQ = Array.isArray(healingQ) && healingQ.length > 0;
                  const chartData = burden.map((b, i) => ({
                    t: i,
                    burden: b * 100,
                    ...(hasHealingQ ? { mse: healingQ[i] } : {}),
                  }));
                  return (
                    <ComposedChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                      <XAxis dataKey="t" stroke="#1e293b" tick={{ fill: "#334155" }} />
                      <YAxis yAxisId="left" stroke="#f87171" tick={{ fill: "#f87171" }} orientation="left" />
                      <YAxis yAxisId="right" stroke="#60a5fa" tick={{ fill: "#60a5fa" }} orientation="right" />
                      <Tooltip contentStyle={{ background: "rgba(10,10,10,0.9)", borderColor: "#334155" }} />
                      <Legend />
                      <Area yAxisId="left" type="step" dataKey="burden" fill="#f87171" stroke="#ef4444" fillOpacity={0.2} name="% Sensors Offline" />
                      {hasHealingQ && (
                        <Line yAxisId="right" type="monotone" dataKey="mse" stroke="#60a5fa" dot={false} strokeWidth={2} name="Global Imputation MSE" />
                      )}
                    </ComposedChart>
                  );
                })()}
              </ResponsiveContainer>
            </Box>
            <Box sx={{ flex: 1, minWidth: 300, display: "flex", flexDirection: "column" }}>
              <Typography variant="subtitle2" sx={{ mb: 1, color: "#94a3b8" }}>
                Tripwire Audit Trail
              </Typography>
              <Box sx={{ flex: 1, minHeight: 0 }}>
                <DataGrid
                  rows={(masterIndex?.tripwire_audit_trail ?? []).map((r: TripwireEntry, i) => ({ id: i, ...r }))}
                  columns={[
                    { field: "timestep", headerName: "Time", width: 60 },
                    { field: "node_id", headerName: "Node ID", width: 80 },
                    { field: "event_type", headerName: "Safeguard Triggered", flex: 1 }
                  ]}
                  density="compact"
                  sx={{
                    bgcolor: "rgba(255,255,255,0.02)", color: "#cbd5e1",
                    "& .MuiDataGrid-columnHeaders": { bgcolor: "rgba(255,255,255,0.03)", borderBottom: "1px solid rgba(255,255,255,0.06)" },
                    "& .MuiDataGrid-cell": { borderBottom: "1px solid rgba(255,255,255,0.03)" },
                  }}
                />
              </Box>
            </Box>
          </Box>
        )}

        {/* Tab 3 — ZINB Calibration */}
        {activeTab === 3 && (
          <Box
            sx={{
              flex: 1,
              overflow: "auto",
              minHeight: 0,
              display: "flex",
              flexDirection: "column",
              gap: 2,
            }}
          >
            {!hasPIT && !hasReliability ? (
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
                Calibration data not present in this payload. Re-run{" "}
                <code>export_ui_payload.py</code> to generate the PIT histogram
                and zero-inflation reliability arrays.
              </Alert>
            ) : (
              <Box
                sx={{
                  display: "flex",
                  gap: 2,
                  flex: 1,
                  minHeight: 300,
                  flexWrap: "wrap",
                }}
              >
                {/* ---- Left: PIT Histogram ---- */}
                <Box
                  sx={{
                    flex: 1,
                    minWidth: 300,
                    display: "flex",
                    flexDirection: "column",
                    gap: 0.5,
                  }}
                >
                  <Typography
                    variant="caption"
                    sx={{ color: "#64748b", fontSize: "0.75rem" }}
                  >
                    PIT Histogram — uniform bars indicate perfect calibration
                  </Typography>
                  {hasPIT ? (
                    <ResponsiveContainer width="100%" height={280}>
                      <BarChart
                        data={masterIndex.pit_histogram as PITBin[]}
                        margin={{ top: 8, right: 16, left: -4, bottom: 28 }}
                      >
                        <CartesianGrid
                          strokeDasharray="3 3"
                          stroke="rgba(255,255,255,0.04)"
                        />
                        <XAxis
                          dataKey="bin_center"
                          stroke="#1e293b"
                          tick={{ fill: "#334155", fontSize: 10 }}
                          tickFormatter={(v: number) => v.toFixed(2)}
                          label={{
                            value: "PIT value  u = F_ZINB(y_obs)",
                            position: "insideBottom",
                            offset: -14,
                            fill: "#334155",
                            fontSize: 10,
                          }}
                        />
                        <YAxis
                          stroke="#1e293b"
                          tick={{ fill: "#334155", fontSize: 10 }}
                          width={48}
                          label={{
                            value: "Density",
                            angle: -90,
                            position: "insideLeft",
                            offset: 14,
                            fill: "#334155",
                            fontSize: 10,
                          }}
                        />
                        <Tooltip
                          contentStyle={{
                            background: "rgba(10,10,10,0.96)",
                            border: "1px solid rgba(255,255,255,0.08)",
                            color: "#e2e8f0",
                            fontSize: "0.72rem",
                            borderRadius: 6,
                          }}
                          formatter={(v: unknown) => [
                            Number(v).toFixed(4),
                            "density",
                          ]}
                        />
                        {/* Uniform reference: 1/10 = 0.10 */}
                        <ReferenceLine
                          y={0.1}
                          stroke="#f59e0b"
                          strokeDasharray="4 4"
                          label={{
                            value: "Uniform (0.10)",
                            position: "insideTopRight",
                            fill: "#f59e0b",
                            fontSize: 9,
                          }}
                        />
                        <Bar
                          dataKey="density"
                          fill="#60a5fa"
                          fillOpacity={0.8}
                          isAnimationActive={false}
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  ) : (
                    <Alert severity="warning" sx={{ fontSize: "0.78rem" }}>
                      PIT histogram not available in this payload.
                    </Alert>
                  )}
                </Box>

                {/* ---- Right: Zero-Inflation Reliability ---- */}
                <Box
                  sx={{
                    flex: 1,
                    minWidth: 300,
                    display: "flex",
                    flexDirection: "column",
                    gap: 0.5,
                  }}
                >
                  <Typography
                    variant="caption"
                    sx={{ color: "#64748b", fontSize: "0.75rem" }}
                  >
                    Zero-Inflation Reliability — points on the dashed y = x
                    line indicate perfect calibration
                  </Typography>
                  {hasReliability ? (
                    <ResponsiveContainer width="100%" height={280}>
                      {/*
                       * ComposedChart bound to a single, unified data array
                       * (reliabilityData). The scatter dots are the only
                       * interactive series; the dashed y = x perfect-calibration
                       * reference is drawn with a presentational <ReferenceLine>
                       * so it does not register as hoverable data points.
                       */}
                      <ComposedChart
                        data={reliabilityData}
                        margin={{ top: 8, right: 16, left: -4, bottom: 28 }}
                      >
                        <CartesianGrid
                          strokeDasharray="3 3"
                          stroke="rgba(255,255,255,0.04)"
                        />
                        <XAxis
                          dataKey="pi_center"
                          type="number"
                          domain={[0, 1]}
                          stroke="#1e293b"
                          tick={{ fill: "#334155", fontSize: 10 }}
                          tickFormatter={(v: number) => v.toFixed(1)}
                          label={{
                            value: "Predicted π (zero-inflation probability)",
                            position: "insideBottom",
                            offset: -14,
                            fill: "#334155",
                            fontSize: 10,
                          }}
                        />
                        <YAxis
                          dataKey="observed_zero_fraction"
                          type="number"
                          domain={[0, 1]}
                          stroke="#1e293b"
                          tick={{ fill: "#334155", fontSize: 10 }}
                          width={48}
                          label={{
                            value: "Observed zero fraction",
                            angle: -90,
                            position: "insideLeft",
                            offset: 14,
                            fill: "#334155",
                            fontSize: 10,
                          }}
                        />
                        <Tooltip
                          contentStyle={{
                            background: "rgba(10,10,10,0.96)",
                            border: "1px solid rgba(255,255,255,0.08)",
                            color: "#e2e8f0",
                            fontSize: "0.72rem",
                            borderRadius: 6,
                          }}
                          formatter={(v: unknown, name: string) => [
                            Number(v).toFixed(4),
                            name,
                          ]}
                        />
                        <Legend
                          wrapperStyle={{
                            color: "#475569",
                            fontSize: "0.72rem",
                          }}
                        />
                        {/*
                         * y = x perfect-calibration reference. Purely
                         * presentational — drawn between two axis coordinates
                         * so it never registers as interactive data points,
                         * which avoids tooltip misalignment and phantom
                         * highlights at x = 0 / x = 1.
                         */}
                        <ReferenceLine
                          segment={[
                            { x: 0, y: 0 },
                            { x: 1, y: 1 },
                          ]}
                          stroke="#f59e0b"
                          strokeDasharray="6 3"
                          strokeWidth={1.5}
                          label={{
                            value: "Perfect calibration (y = x)",
                            position: "insideTopLeft",
                            fill: "#f59e0b",
                            fontSize: 9,
                          }}
                        />
                        {/* Observed scatter — dots only, no connecting line */}
                        <Line
                          type="monotone"
                          dataKey="observed_zero_fraction"
                          stroke="none"
                          dot={{ r: 5, fill: "#34d399", stroke: "#34d399" }}
                          activeDot={{ r: 7, fill: "#34d399" }}
                          name="Observed zero fraction"
                          isAnimationActive={false}
                          legendType="circle"
                        />
                      </ComposedChart>
                    </ResponsiveContainer>
                  ) : (
                    <Alert severity="warning" sx={{ fontSize: "0.78rem" }}>
                      Reliability diagram not available in this payload.
                    </Alert>
                  )}
                </Box>
              </Box>
            )}
          </Box>
        )}
      </Box>
    </Box>
  );
}
