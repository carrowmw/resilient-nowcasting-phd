'use client';

import dynamic from 'next/dynamic';

// Both components use WebGL / browser APIs — load them client-side only.
// `ssr: false` is required here and is only valid inside a Client Component
// (see Next.js 16 docs: https://nextjs.org/docs/app/guides/lazy-loading).
const MapVisualizer = dynamic(() => import('./MapVisualizer'), {
  ssr: false,
  loading: () => (
    <div className="flex h-screen w-screen items-center justify-center bg-neutral-950 text-slate-400 text-sm">
      Loading map…
    </div>
  ),
});

const TimeSlider = dynamic(() => import('./TimeSlider'), {
  ssr: false,
});

export default function MapScene() {
  return (
    <main className="relative h-screen w-screen overflow-hidden">
      <MapVisualizer />
      <TimeSlider />
    </main>
  );
}
