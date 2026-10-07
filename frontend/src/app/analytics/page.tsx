"use client";

import { useState, type ReactNode } from "react";
import Link from "next/link";
import AppBar from "@mui/material/AppBar";
import Toolbar from "@mui/material/Toolbar";
import Typography from "@mui/material/Typography";
import IconButton from "@mui/material/IconButton";
import Tabs from "@mui/material/Tabs";
import Tab from "@mui/material/Tab";
import Box from "@mui/material/Box";
import Alert from "@mui/material/Alert";
import CircularProgress from "@mui/material/CircularProgress";
import useTimeStore from "@/store/useTimeStore";
import usePayloadLoader from "@/lib/usePayloadLoader";
import DatePaginator from "@/components/DatePaginator";
import RunLeaderboard from "@/components/analytics/RunLeaderboard";
import NodeGrid from "@/components/analytics/NodeGrid";
import ForecastTab from "@/components/analytics/ForecastTab";
import OutageTab from "@/components/analytics/OutageTab";
import CalibrationTab from "@/components/analytics/CalibrationTab";
import NetworkTab from "@/components/analytics/NetworkTab";
import { ACCENT } from "@/components/analytics/shared";

const TABS = ["Forecast", "Outage resilience", "Calibration", "Network influence"] as const;

function Centered({ children }: { children: ReactNode }) {
  return (
    <Box sx={{ display: "flex", py: 6, alignItems: "center", justifyContent: "center" }}>{children}</Box>
  );
}

export default function AnalyticsPage() {
  const [activeTab, setActiveTab] = useState(0);

  // Registry → spatial → page (shared with the map; reuses loaded payloads).
  const fetchError = usePayloadLoader();

  const runRegistry = useTimeStore((s) => s.runRegistry);
  const selectionRule = useTimeStore((s) => s.selectionRule);
  const activeRunId = useTimeStore((s) => s.activeRunId);
  const activeDate = useTimeStore((s) => s.activeDate);
  const setActiveRunId = useTimeStore((s) => s.setActiveRunId);
  const selectedNodeId = useTimeStore((s) => s.selectedNodeId);
  const setSelectedNodeId = useTimeStore((s) => s.setSelectedNodeId);
  const dayPayload = useTimeStore((s) => s.dayPayload);
  const isLoadingPage = useTimeStore((s) => s.isLoadingPage);

  // Only show the payload once it matches the selection (no stale run/day).
  const loaded =
    dayPayload && dayPayload.runId === activeRunId && dayPayload.date === activeDate ? dayPayload : null;
  const entry = runRegistry.find((r) => r.run_id === activeRunId) ?? null;
  const caps = loaded?.spatial.metadata.capabilities ?? null;
  const node = loaded?.spatial.nodes.find((n) => n.id === selectedNodeId) ?? null;

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
      <AppBar
        position="static"
        elevation={0}
        sx={{ bgcolor: "rgba(12,12,12,0.97)", borderBottom: "1px solid rgba(255,255,255,0.07)" }}
      >
        <Toolbar variant="dense" sx={{ minHeight: 48, gap: 2 }}>
          <Link href="/" style={{ color: "inherit", display: "flex" }}>
            <IconButton size="small" sx={{ color: ACCENT }} aria-label="Back to map">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                <path d="M20 11H7.83l5.59-5.59L12 4l-8 8 8 8 1.41-1.41L7.83 13H20v-2z" />
              </svg>
            </IconButton>
          </Link>
          <Typography variant="subtitle1" sx={{ fontWeight: 700, letterSpacing: "0.05em", color: "#e2e8f0" }}>
            NSTAR Analytics
          </Typography>
          {entry && (
            <Typography variant="caption" sx={{ color: "#64748b", fontFamily: "monospace" }} noWrap>
              #{entry.registry.rank} {entry.run_id} · {entry.forecast_likelihood} · seq_len {entry.seq_len}
              {loaded ? ` · ${loaded.spatial.metadata.num_nodes} nodes · ${loaded.page.timesteps}×${loaded.page.step_minutes} min` : ""}
            </Typography>
          )}
          <Box sx={{ ml: "auto", display: "flex", alignItems: "center", gap: 1 }} className="font-mono text-xs text-slate-300">
            {isLoadingPage && <CircularProgress size={14} sx={{ color: ACCENT }} />}
            <DatePaginator />
          </Box>
        </Toolbar>
      </AppBar>

      <Box sx={{ flex: 1, overflow: "auto", p: 1.5, display: "flex", flexDirection: "column", gap: 2 }}>
        {fetchError && (
          <Alert severity="error" variant="outlined" sx={{ color: "#fca5a5" }}>
            {fetchError}
          </Alert>
        )}

        {runRegistry.length > 0 ? (
          <RunLeaderboard
            runs={runRegistry}
            activeRunId={activeRunId}
            selectionRule={selectionRule}
            onSelect={setActiveRunId}
          />
        ) : (
          !fetchError && (
            <Centered>
              <CircularProgress sx={{ color: ACCENT }} />
            </Centered>
          )
        )}

        {loaded && caps ? (
          <>
            <NodeGrid
              spatial={loaded.spatial}
              page={loaded.page}
              selectedNodeId={selectedNodeId}
              onSelect={setSelectedNodeId}
            />

            <Box sx={{ borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
              <Tabs
                value={activeTab}
                onChange={(_, v: number) => setActiveTab(v)}
                sx={{
                  minHeight: 34,
                  "& .MuiTab-root": { color: "#475569", textTransform: "none", fontSize: "0.8rem", minHeight: 34, py: 0, px: 2 },
                  "& .Mui-selected": { color: `${ACCENT} !important` },
                  "& .MuiTabs-indicator": { bgcolor: ACCENT, height: 2 },
                }}
              >
                {TABS.map((label, i) => (
                  <Tab key={label} label={label} value={i} />
                ))}
              </Tabs>
            </Box>

            <Box sx={{ pb: 4 }}>
              {activeTab === 0 && <ForecastTab page={loaded.page} node={node} capabilities={caps} />}
              {activeTab === 1 && <OutageTab page={loaded.page} node={node} />}
              {activeTab === 2 && (
                <CalibrationTab
                  calibration={loaded.page.calibration}
                  dayMetrics={loaded.page.day_metrics}
                  windowMetrics={loaded.spatial.metadata.window_metrics}
                  capabilities={caps}
                />
              )}
              {activeTab === 3 && (
                <NetworkTab page={loaded.page} capabilities={caps} onSelectNode={setSelectedNodeId} />
              )}
            </Box>
          </>
        ) : (
          activeRunId &&
          !fetchError && (
            <Centered>
              <CircularProgress sx={{ color: ACCENT }} />
            </Centered>
          )
        )}
      </Box>
    </Box>
  );
}
