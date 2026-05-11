'use client';

import useTimeStore from '@/store/useTimeStore';

/**
 * Floating global attention-threshold control.
 *
 * Positioned top-left below the Analytics Fab. Controls the minimum edge
 * weight required for a global topology arc to be rendered. Hidden when a
 * node is selected because the global topology layer is hidden in that state
 * (the ego-graph layer takes over and ignores the threshold).
 */
export default function AttentionControls() {
  const selectedNodeId = useTimeStore((s) => s.selectedNodeId);
  const globalAttentionThreshold = useTimeStore((s) => s.globalAttentionThreshold);
  const setGlobalAttentionThreshold = useTimeStore((s) => s.setGlobalAttentionThreshold);

  if (selectedNodeId !== null) return null;

  return (
    <div
      style={{
        position: 'absolute',
        top: 88,
        left: 16,
        zIndex: 20,
        minWidth: 180,
        background: 'rgba(15,15,15,0.88)',
        border: '1px solid rgba(96,165,250,0.25)',
        borderRadius: 10,
        backdropFilter: 'blur(8px)',
        padding: '10px 14px',
        boxShadow: '0 4px 24px rgba(0,0,0,0.5)',
      }}
      aria-label="Global attention threshold controls"
    >
      <label
        htmlFor="global-attention-threshold"
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          fontSize: 11,
          color: 'rgb(148,163,184)',
          marginBottom: 6,
          userSelect: 'none',
        }}
      >
        <span style={{ color: 'rgb(96,165,250)', fontWeight: 600 }}>
          Attention threshold
        </span>
        <span style={{ fontFamily: 'monospace', color: 'rgb(203,213,225)' }}>
          {globalAttentionThreshold.toFixed(2)}
        </span>
      </label>
      <input
        id="global-attention-threshold"
        type="range"
        min={0}
        max={1}
        step={0.01}
        value={globalAttentionThreshold}
        onChange={(e) => setGlobalAttentionThreshold(Number(e.target.value))}
        style={{ width: '100%', accentColor: 'rgb(96,165,250)' }}
        aria-label="Global attention edge weight threshold"
      />
    </div>
  );
}
