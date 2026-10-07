"use client";

import useTimeStore from "@/store/useTimeStore";

export default function MapLegend() {
  const selectedNodeId = useTimeStore((s) => s.selectedNodeId);

  return (
    <div
      style={{
        position: "absolute",
        bottom: 24,
        left: 24,
        zIndex: 10,
        width: 250,
        background: "rgba(10, 12, 16, 0.85)",
        border: "1px solid rgba(255, 255, 255, 0.08)",
        borderRadius: "12px",
        padding: "14px",
        color: "#e2e8f0",
        fontSize: "11px",
        boxShadow: "0 8px 32px rgba(0, 0, 0, 0.5)",
        backdropFilter: "blur(8px)",
        pointerEvents: "auto",
        transition: "opacity 0.2s ease-in-out",
      }}
      aria-label="Map legend"
    >
      <div style={{ fontWeight: 600, fontSize: "12px", marginBottom: "10px", color: "#f8fafc", letterSpacing: "0.5px" }}>
        MAP LEGEND
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
        {/* Observed Nodes */}
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <svg width="20" height="20" viewBox="0 0 20 20" style={{ overflow: "visible" }}>
            <circle cx="10" cy="10" r="6" fill="#10b981" />
          </svg>
          <div>
            <div style={{ fontWeight: 500, color: "#f1f5f9" }}>Observed Sensor</div>
            <div style={{ color: "#94a3b8", fontSize: "10px" }}>Model sees its history when online.</div>
          </div>
        </div>

        {/* Holdout Nodes */}
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <svg width="20" height="20" viewBox="0 0 20 20" style={{ overflow: "visible" }}>
            <circle cx="10" cy="10" r="5" fill="#f59e0b" />
            <circle cx="10" cy="10" r="5" fill="none" stroke="#a78bfa" strokeWidth="1.5" />
            <circle cx="10" cy="10" r="8" fill="none" stroke="rgba(167, 139, 250, 0.6)" strokeWidth="1" strokeDasharray="2 2" />
          </svg>
          <div>
            <div style={{ fontWeight: 500, color: "#a78bfa" }}>Holdout (Zero-Shot)</div>
            <div style={{ color: "#94a3b8", fontSize: "10px" }}>Never trained on; blanked at inference.</div>
          </div>
        </div>

        {/* No interval */}
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <svg width="20" height="20" viewBox="0 0 20 20" style={{ overflow: "visible" }}>
            <circle cx="10" cy="10" r="6" fill="rgb(100,116,139)" />
          </svg>
          <div>
            <div style={{ fontWeight: 500, color: "#cbd5e1" }}>No Interval</div>
            <div style={{ color: "#94a3b8", fontSize: "10px" }}>Model has no predictive distribution.</div>
          </div>
        </div>

        {/* Ghost nodes */}
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <svg width="20" height="20" viewBox="0 0 20 20" style={{ overflow: "visible" }}>
            <circle cx="6" cy="10" r="4" fill="none" stroke="rgba(148,163,184,0.6)" strokeWidth="1.5" />
            <circle cx="15" cy="10" r="4" fill="none" stroke="rgba(167,139,250,0.5)" strokeWidth="1.5" />
          </svg>
          <div>
            <div style={{ fontWeight: 500, color: "#94a3b8" }}>Ghost (No Data)</div>
            <div style={{ color: "#94a3b8", fontSize: "10px" }}>No readings this window; forecast can&apos;t be checked.</div>
          </div>
        </div>

        {/* Graph Attention Edge */}
        {selectedNodeId === null && (
          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            <svg width="20" height="20" viewBox="0 0 20 20">
              <path d="M 2,15 Q 10,2 18,15" fill="none" stroke="#f97316" strokeWidth="2" />
            </svg>
            <div>
              <div style={{ fontWeight: 500, color: "#f97316" }}>Attention Connection</div>
              <div style={{ color: "#94a3b8", fontSize: "10px" }}>Graph attention weight (width & opacity).</div>
            </div>
          </div>
        )}

        {/* Ego-graph edges explanation */}
        {selectedNodeId !== null && (
          <>
            <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
              <svg width="20" height="20" viewBox="0 0 20 20">
                <path d="M 2,15 Q 10,2 18,15" fill="none" stroke="#60a5fa" strokeWidth="2" />
              </svg>
              <div>
                <div style={{ fontWeight: 500, color: "#60a5fa" }}>Incoming Attention</div>
                <div style={{ color: "#94a3b8", fontSize: "10px" }}>Neighbors influencing this node.</div>
              </div>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
              <svg width="20" height="20" viewBox="0 0 20 20">
                <path d="M 2,15 Q 10,2 18,15" fill="none" stroke="#fb7185" strokeWidth="2" />
              </svg>
              <div>
                <div style={{ fontWeight: 500, color: "#fb7185" }}>Outgoing Attention</div>
                <div style={{ color: "#94a3b8", fontSize: "10px" }}>This node influencing its neighbors.</div>
              </div>
            </div>
          </>
        )}
      </div>

      <div style={{ borderTop: "1px solid rgba(255, 255, 255, 0.08)", marginTop: "10px", paddingTop: "8px" }}>
        <div style={{ height: 6, borderRadius: 3, background: "linear-gradient(90deg, rgb(0,200,0), rgb(255,200,0), rgb(255,0,0))" }} />
        <div style={{ display: "flex", justifyContent: "space-between", color: "#94a3b8", fontSize: "9px", marginTop: "2px" }}>
          <span>narrow</span>
          <span>≥ 2× wide</span>
        </div>
        <div style={{ color: "#94a3b8", fontSize: "9px", marginTop: "4px" }}>
          Node Color = relative 80% interval width (hi−lo)/(point+1)
        </div>
        <div style={{ color: "#94a3b8", fontSize: "9px", marginTop: "2px" }}>
          Radius = point forecast
        </div>
      </div>
    </div>
  );
}
