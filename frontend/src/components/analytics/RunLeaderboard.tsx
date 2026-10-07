"use client";

import { useMemo } from "react";
import Chip from "@mui/material/Chip";
import Box from "@mui/material/Box";
import { DataGrid } from "@mui/x-data-grid";
import type { GridColDef } from "@mui/x-data-grid";
import type { RunRegistryEntry, TestMetrics } from "@/types/registry";
import type { Likelihood } from "@/types/payload";
import { gridSx, numCol, SectionTitle } from "./shared";

export interface LeaderboardRow extends TestMetrics {
  id: string;
  rank: number;
  is_reference: boolean;
  experiment: string;
  description: string;
  likelihood: Likelihood;
  seq_len: number;
  w_obs_all_rmse: number | null;
  w_obs_ind_rmse: number | null;
  w_interval80_coverage: number | null;
  w_real_gap_mass_ratio: number | null;
}

export function leaderboardRows(runs: RunRegistryEntry[]): LeaderboardRow[] {
  return [...runs]
    .sort((a, b) => a.registry.rank - b.registry.rank)
    .map((r) => ({
      ...r.test_metrics,
      id: r.run_id,
      rank: r.registry.rank,
      is_reference: r.registry.is_reference,
      experiment: r.experiment,
      description: r.description,
      likelihood: r.forecast_likelihood,
      seq_len: r.seq_len,
      w_obs_all_rmse: r.window_metrics.obs_all_rmse,
      w_obs_ind_rmse: r.window_metrics.obs_ind_rmse,
      w_interval80_coverage: r.window_metrics.interval80_coverage,
      w_real_gap_mass_ratio: r.window_metrics.real_gap_mass_ratio,
    }));
}

const columns: GridColDef<LeaderboardRow>[] = [
  { field: "rank", headerName: "#", width: 52, type: "number" },
  {
    field: "is_reference",
    headerName: "Ref",
    width: 64,
    sortable: false,
    renderCell: ({ value }) =>
      value ? (
        <Chip
          label="REF"
          size="small"
          sx={{ height: 18, fontSize: "0.62rem", color: "#fbbf24", bgcolor: "rgba(251,191,36,0.12)" }}
        />
      ) : null,
  },
  { field: "experiment", headerName: "Experiment", width: 150 },
  { field: "description", headerName: "Description", flex: 1, minWidth: 220 },
  { field: "likelihood", headerName: "Likelihood", width: 110 },
  { field: "seq_len", headerName: "seq_len", width: 72, type: "number" },
  numCol("val_obs_all_rmse", "val RMSE", { description: "Validation obs_all RMSE (rank key)" }),
  numCol("test_obs_all_rmse", "test RMSE", { description: "Test RMSE, all observed targets" }),
  numCol("test_obs_ind_rmse", "test ind RMSE", { width: 110, description: "Test RMSE, holdout (inductive) nodes" }),
  numCol("test_so_12_forecast_rmse", "so_12 RMSE", { width: 100, description: "Test RMSE under a 12-step (3 h) synthetic outage" }),
  numCol("test_crps", "CRPS", { width: 80 }),
  numCol("test_collapse_rate", "collapse", { pct: true, width: 90, description: "Test collapse rate" }),
  numCol("test_real_gap_mass_ratio", "mass ratio", { width: 96, description: "Test real-gap mass ratio (1 = no collapse)" }),
  numCol("test_q90_coverage", "q90 cov", { pct: true, width: 86, description: "Test q90 coverage (target 90%)" }),
  numCol("w_obs_all_rmse", "win RMSE", { description: "Analysis-window RMSE, all observed" }),
  numCol("w_obs_ind_rmse", "win ind RMSE", { width: 110, description: "Analysis-window RMSE, holdout nodes" }),
  numCol("w_interval80_coverage", "win 80% cov", { pct: true, width: 104, description: "Analysis-window 80% interval coverage" }),
  numCol("w_real_gap_mass_ratio", "win mass ratio", { width: 116, description: "Analysis-window real-gap mass ratio" }),
];

interface Props {
  runs: RunRegistryEntry[];
  activeRunId: string | null;
  selectionRule: string | null;
  onSelect: (runId: string) => void;
}

export default function RunLeaderboard({ runs, activeRunId, selectionRule, onSelect }: Props) {
  const rows = useMemo(() => leaderboardRows(runs), [runs]);
  return (
    <Box>
      <SectionTitle hint={selectionRule ?? undefined}>Run leaderboard</SectionTitle>
      <Box sx={{ height: 280 }}>
        <DataGrid
          rows={rows}
          columns={columns}
          density="compact"
          disableRowSelectionOnClick
          rowSelectionModel={activeRunId ? [activeRunId] : []}
          onRowClick={(p) => onSelect(String(p.id))}
          initialState={{ sorting: { sortModel: [{ field: "rank", sort: "asc" }] } }}
          hideFooter={rows.length <= 100}
          aria-label="Run leaderboard"
          sx={gridSx}
        />
      </Box>
    </Box>
  );
}
