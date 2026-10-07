"use client";

import useTimeStore from "@/store/useTimeStore";

/**
 * Top-left dropdown listing registry runs by rank. Selection updates
 * activeRunId; the payload loader cascades the spatial + page reloads.
 * Positioned to the right of the analytics button.
 */
export default function ModelSelector() {
  const runRegistry = useTimeStore((s) => s.runRegistry);
  const activeRunId = useTimeStore((s) => s.activeRunId);
  const setActiveRunId = useTimeStore((s) => s.setActiveRunId);

  if (runRegistry.length === 0) return null;

  const active = runRegistry.find((r) => r.run_id === activeRunId);

  return (
    <div
      style={{
        position: "absolute",
        top: 24,
        left: 88,
        zIndex: 20,
        display: "flex",
        flexDirection: "column",
        gap: 4,
        maxWidth: 220,
        background: "rgba(15,15,15,0.88)",
        border: "1px solid rgba(96,165,250,0.35)",
        backdropFilter: "blur(8px)",
        borderRadius: 12,
        padding: "8px 12px",
        boxShadow: "0 4px 24px rgba(0,0,0,0.5)",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <label htmlFor="model-selector" style={{ color: "#94a3b8", fontSize: 11, fontFamily: "monospace" }}>
          MODEL
        </label>
        <select
          id="model-selector"
          value={activeRunId ?? ""}
          onChange={(e) => setActiveRunId(e.target.value)}
          aria-label="Select model run"
          title={active?.description}
          style={{
            minWidth: 0,
            flex: 1,
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
            <option key={r.run_id} value={r.run_id} title={r.description} style={{ color: "#000" }}>
              #{r.registry.rank} {r.run_id}
              {r.registry.is_reference ? " ★" : ""} — {r.description}
            </option>
          ))}
        </select>
      </div>
      {active && (
        <div
          style={{
            color: "#64748b",
            fontSize: 10,
            fontFamily: "monospace",
            whiteSpace: "nowrap",
            overflow: "hidden",
            textOverflow: "ellipsis",
          }}
          title={active.description}
        >
          {active.description}
        </div>
      )}
    </div>
  );
}
