"use client";

import { useState } from "react";
import type { ReactNode } from "react";
import useTimeStore from "@/store/useTimeStore";
import type { RunRegistryEntry } from "@/types/registry";

function Row({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", gap: 16, padding: "2px 0" }}>
      <span style={{ color: "#94a3b8" }}>{label}</span>
      <span style={{ color: "#e2e8f0", fontVariantNumeric: "tabular-nums" }}>{value}</span>
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

function num(v: unknown, digits = 3): string {
  return typeof v === "number" ? v.toFixed(digits) : "—";
}

/**
 * Glassmorphic card showing the active run's training config, metrics and
 * analysis period. Self-contained: an "i" button toggles the card. Data comes
 * entirely from the run registry entry (no extra fetch).
 */
export default function ModelInfoOverlay() {
  const [open, setOpen] = useState(false);
  const runRegistry = useTimeStore((s) => s.runRegistry);
  const activeRunId = useTimeStore((s) => s.activeRunId);

  const entry: RunRegistryEntry | undefined = runRegistry.find(
    (r) => r.run_id === activeRunId,
  );
  if (!entry) return null;

  const model = (entry.train_config?.model ?? {}) as Record<string, unknown>;
  const training = (entry.train_config?.training ?? {}) as Record<string, unknown>;
  const m = entry.metrics_summary ?? {};

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
          style={{
            position: "absolute",
            top: 70,
            left: 88,
            zIndex: 25,
            width: 300,
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
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <strong style={{ fontSize: 14 }}>{entry.run_id}</strong>
            <button
              type="button"
              onClick={() => setOpen(false)}
              aria-label="Close model info"
              style={{ background: "transparent", border: "none", color: "#94a3b8", cursor: "pointer", fontSize: 16 }}
            >
              ✕
            </button>
          </div>

          <Section title="ARCHITECTURE" />
          <Row label="Hidden" value={String(model.hidden_dim ?? "—")} />
          <Row label="Layers" value={String(model.num_layers ?? "—")} />
          <Row label="Heads" value={String(model.num_heads ?? "—")} />
          <Row label="Embed" value={String(model.embedding_dim ?? "—")} />
          <Row label="Dropout" value={String(model.dropout ?? "—")} />

          {Object.keys(training).length > 0 && (
            <>
              <Section title="TRAINING" />
              {training.epochs != null && <Row label="Epochs" value={String(training.epochs)} />}
              {training.batch_size != null && <Row label="Batch" value={String(training.batch_size)} />}
              {training.learning_rate != null && <Row label="LR" value={String(training.learning_rate)} />}
            </>
          )}

          <Section title="VALIDATION" />
          <Row label="RMSE" value={num(m.val_rmse)} />
          <Row label="MAE" value={num(m.val_mae)} />
          <Row label="Zero Acc" value={num(m.val_zero_acc)} />

          <Section title="ANALYSIS" />
          <Row
            label="Period"
            value={`${entry.analysis_period.start_date ?? "—"} → ${entry.analysis_period.end_date ?? "—"}`}
          />
          <Row label="Days" value={String(entry.available_dates.length)} />
          <Row
            label="Nodes"
            value={`${entry.num_nodes} (${entry.num_nodes - entry.num_simulated_nodes} real + ${entry.num_simulated_nodes} sim)`}
          />

          <Section title="PROVENANCE" />
          <Row label="W&B run" value={entry.wandb_run_id} />
          <Row label="Commit" value={String(entry.git_commit).slice(0, 9)} />
        </div>
      )}
    </>
  );
}
