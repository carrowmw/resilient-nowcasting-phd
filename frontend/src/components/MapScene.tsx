'use client';

import dynamic from 'next/dynamic';
import Link from 'next/link';

const MapVisualizer = dynamic(() => import('./MapVisualizer'), {
  ssr: false,
  loading: () => (
    <div className="flex h-screen w-screen items-center justify-center bg-neutral-950 text-slate-400 text-sm">
      Loading map…
    </div>
  ),
});

const TimeSlider = dynamic(() => import('./TimeSlider'), { ssr: false });
const NodeInspector = dynamic(() => import('./NodeInspector'), { ssr: false });

export default function MapScene() {
  return (
    <main className="relative h-screen w-screen overflow-hidden">
      <MapVisualizer />
      <NodeInspector />
      <TimeSlider />

      {/* Floating button: navigate to the Analytics dashboard.
          Positioned top-left to avoid conflicting with the NodeInspector (top-right)
          and the TimeSlider (bottom). */}
      <Link
        href="/analytics"
        aria-label="Open analytics dashboard"
        style={{
          position: 'absolute',
          top: 24,
          left: 24,
          zIndex: 20,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          width: 48,
          height: 48,
          borderRadius: '50%',
          background: 'rgba(15,15,15,0.88)',
          border: '1px solid rgba(96,165,250,0.35)',
          backdropFilter: 'blur(8px)',
          color: '#60a5fa',
          textDecoration: 'none',
          boxShadow: '0 4px 24px rgba(0,0,0,0.5)',
          transition: 'border-color 0.15s, box-shadow 0.15s',
        }}
      >
        {/* Bar-chart icon (inline SVG — no MUI dep needed in this component) */}
        <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
          <path d="M3 13h2v8H3v-8zm4-5h2v13H7V8zm4-4h2v17h-2V4zm4 7h2v10h-2V11zm4-4h2v14h-2V7z" />
        </svg>
      </Link>
    </main>
  );
}
