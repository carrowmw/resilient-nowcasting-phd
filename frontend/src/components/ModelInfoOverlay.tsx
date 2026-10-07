"use client";

import { useState } from "react";
import type { ReactNode } from "react";
import useTimeStore from "@/store/useTimeStore";
import type { Capabilities } from "@/types/payload";
import type { RunRegistryEntry } from "@/types/registry";
import { fmt, fmtPct } from "@/lib/v5";

function Row({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", gap: 16, padding: "2px 0" }}>
      <span style={{ color: "#94a3b8" }}>{label}</span>
      <span style={{ color: "#e2e8f0", fontVariantNumeric: "tabular-nums", textAlign: "right" }}>{value}</span>
    </div>
  );
}

function Section({ title }: { title: string }) {
  return (
    <div style={{ color: "#60a5fa", fontSize: 10, letterSpacing: 1, marginTop: 12, marginBottom: 2 }}>
      {title}
    </div>
  );
}

function section(hp: Record<string, unknown>, key: string): Record<string, unknown> {
  const v = hp[key];
  return v != null && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : {};
}

function str(v: unknown): string {
  return v == null ? "—" : String(v);
}

const CAPABILITY_LABELS: [keyof Capabilities, string][] = [
  ["uncertainty", "interval"],
  ["zero_inflation", "π"],
  ["imputation", "impute"],
  ["kriging", "kriging"],
  ["graph", "graph"],
  ["outage", "outage"],
  ["quantiles", "quantiles"],
];

function CapabilityChips({ caps }: { caps: Capabilities }) {
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginTop: 4 }}>
      {CAPABILITY_LABELS.map(([key, label]) => (
        <span
          key={key}
          title={key}
          style={{
            fontSize: 10,
            padding: "1px 6px",
            borderRadius: 6,
            border: `1px solid ${caps[key] ? "rgba(52,211,153,0.5)" : "rgba(100,116,139,0.35)"}`,
            color: caps[key] ? "#6ee7b7" : "#475569",
            textDecoration: caps[key] ? "none" : "line-through",
          }}
        >
          {label}
        </span>
      ))}
    </div>
  );
}

/**
 * Glassmorphic card for the active run: registry placement, capabilities,
 * architecture, held-out test metrics, exporter window metrics and provenance.
 * Data comes entirely from the run registry entry (no extra fetch).
 */
export default function ModelInfoOverlay() {
  const [open, setOpen] = useState(false);
  const runRegistry = useTimeStore((s) => s.runRegistry);
  const activeRunId = useTimeStore((s) => s.activeRunId);

  const entry: RunRegistryEntry | undefined = runRegistry.find((r) => r.run_id === activeRunId);
  if (!entry) return null;

  const model = section(entry.hyperparameters, "model");
  const training = section(entry.hyperparameters, "training");
  const tm = entry.test_metrics;
  const wm = entry.window_metrics;

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-label="Toggle model info"
        style={{
          position: "absolute",
          top: 24,
          left: 320,
          zIndex: 20,
          width: 36,
          height: 36,
          borderRadius: "50%",
          background: "rgba(15,15,15,0.88)",
          border: "1px solid rgba(96,165,250,0.35)",
          backdropFilter: "blur(8px)",
          color: "#60a5fa",
          cursor: "pointer",
          fontStyle: "italic",
          fontWeight: 700,
        }}
      >
        i
      </button>

      {open && (
        <div
          role="dialog"
          aria-label="Model info"
          style={{
            position: "absolute",
            top: 70,
            left: 88,
            zIndex: 25,
            width: 320,
            maxHeight: "calc(100vh - 220px)",
            overflowY: "auto",
            background: "rgba(10,12,16,0.92)",
            border: "1px solid rgba(96,165,250,0.35)",
            backdropFilter: "blur(10px)",
            borderRadius: 12,
            padding: 16,
            color: "#e2e8f0",
            fontSize: 12,
            fontFamily: "monospace",
            boxShadow: "0 8px 40px rgba(0,0,0,0.6)",
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8 }}>
            <strong style={{ fontSize: 14 }}>
              #{entry.registry.rank} {entry.run_id}
            </strong>
            {entry.registry.is_reference && (
              <span style={{ fontSize: 10, color: "#fbbf24", border: "1px solid rgba(251,191,36,0.5)", borderRadius: 6, padding: "0 6px" }}>
                REFERENCE
              </span>
            )}
            <button
              type="button"
              onClick={() => setOpen(false)}
              aria-label="Close model info"
              style={{ marginLeft: "auto", background: "transparent", border: "none", color: "#94a3b8", cursor: "pointer", fontSize: 16 }}
            >
              ✕
            </button>
          </div>
          <div style={{ color: "#94a3b8", fontSize: 11, marginTop: 4 }}>{entry.description}</div>

          <Section title="MODEL" />
          <Row label="Experiment" value={entry.experiment} />
          <Row label="Group" value={entry.group} />
          <Row label="Likelihood" value={entry.forecast_likelihood} />
          <Row label="Look-back" value={`${entry.seq_len} slots`} />
          {entry.quantile_levels.length > 0 && (
            <Row label="Quantiles" value={entry.quantile_levels.join(", ")} />
          )}
          <CapabilityChips caps={entry.capabilities} />

          {Object.keys(model).length > 0 && (
            <>
              <Section title="ARCHITECTURE" />
              <Row label="Hidden" value={str(model.hidden_dim)} />
              <Row label="Layers" value={str(model.num_layers)} />
              <Row label="Heads" value={str(model.num_heads)} />
              <Row label="Embed" value={str(model.embedding_dim)} />
              <Row label="Dropout" value={str(model.dropout)} />
            </>
          )}

          {Object.keys(training).length > 0 && (
            <>
              <Section title="TRAINING" />
              {training.epochs != null && <Row label="Epochs" value={str(training.epochs)} />}
              {training.batch_size != null && <Row label="Batch" value={str(training.batch_size)} />}
              {training.learning_rate != null && <Row label="LR" value={str(training.learning_rate)} />}
            </>
          )}

          <Section title="HELD-OUT TEST" />
          <Row label="Val RMSE (rank key)" value={fmt(tm.val_obs_all_rmse)} />
          <Row label="RMSE all / ind" value={`${fmt(tm.test_obs_all_rmse)} / ${fmt(tm.test_obs_ind_rmse)}`} />
          <Row label="3 h outage RMSE" value={fmt(tm.test_so_12_forecast_rmse)} />
          <Row label="CRPS" value={fmt(tm.test_crps)} />
          <Row label="Collapse rate" value={fmtPct(tm.test_collapse_rate)} />
          <Row label="Gap mass ratio" value={fmt(tm.test_real_gap_mass_ratio)} />
          <Row label="q90 coverage" value={fmtPct(tm.test_q90_coverage)} />

          <Section title="ANALYSIS WINDOW" />
          <Row label="Period" value={`${entry.analysis_period.start_date} → ${entry.analysis_period.end_date}`} />
          <Row label="Days" value={String(entry.available_dates.length)} />
          <Row label="Nodes" value={`${entry.num_nodes} (${entry.num_holdout_nodes} holdout, ${entry.num_ghost_nodes} ghost)`} />
          <Row label="Observed" value={fmtPct(wm.observed_frac)} />
          <Row label="RMSE all / ind" value={`${fmt(wm.obs_all_rmse)} / ${fmt(wm.obs_ind_rmse)}`} />
          <Row label="80% coverage" value={fmtPct(wm.interval80_coverage)} />
          <Row label="Gap mass ratio" value={fmt(wm.real_gap_mass_ratio)} />

          <Section title="PROVENANCE" />
          <Row label="W&B run" value={entry.wandb_run_id ?? "—"} />
          <Row label="Commit" value={entry.git_commit_hash ? entry.git_commit_hash.slice(0, 9) : "—"} />
          <Row label="Holdout sha" value={entry.holdout_manifest_sha ? entry.holdout_manifest_sha.slice(0, 9) : "—"} />
        </div>
      )}
    </>
  );
}
