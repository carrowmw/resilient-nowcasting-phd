"use client";

import useTimeStore from "@/store/useTimeStore";

/**
 * Top-left dropdown to switch between W&B model runs. Selection updates
 * activeRunId; MapVisualizer's stage-2/3 effects cascade the spatial + page
 * reloads. Positioned to the right of the analytics button.
 */
export default function ModelSelector() {
  const runRegistry = useTimeStore((s) => s.runRegistry);
  const activeRunId = useTimeStore((s) => s.activeRunId);
  const setActiveRunId = useTimeStore((s) => s.setActiveRunId);

  if (runRegistry.length === 0) return null;

  return (
    <div
      style={{
        position: "absolute",
        top: 24,
        left: 88,
        zIndex: 20,
        display: "flex",
        alignItems: "center",
        gap: 8,
        background: "rgba(15,15,15,0.88)",
        border: "1px solid rgba(96,165,250,0.35)",
        backdropFilter: "blur(8px)",
        borderRadius: 12,
        padding: "8px 12px",
        boxShadow: "0 4px 24px rgba(0,0,0,0.5)",
      }}
    >
      <label
        htmlFor="model-selector"
        style={{ color: "#94a3b8", fontSize: 11, fontFamily: "monospace" }}
      >
        MODEL
      </label>
      <select
        id="model-selector"
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
    </div>
  );
}
