'use client';

import { useEffect, useRef, useCallback } from 'react';
import useTimeStore from '@/store/useTimeStore';
import { timestepLabel } from '@/lib/v5';
import DatePaginator from './DatePaginator';

export default function TimeSlider() {
  const currentTimestep = useTimeStore((s) => s.currentTimestep);
  const maxTimesteps = useTimeStore((s) => s.maxTimesteps);
  const isPlaying = useTimeStore((s) => s.isPlaying);
  const setTimestep = useTimeStore((s) => s.setTimestep);
  const togglePlayback = useTimeStore((s) => s.togglePlayback);
  const incrementTimestep = useTimeStore((s) => s.incrementTimestep);
  // Labels come from page.timestamps when a page is loaded; else t × 15 min.
  const timestamps = useTimeStore((s) => s.dayPayload?.page.timestamps);
  const stepMinutes = useTimeStore((s) => s.dayPayload?.page.step_minutes) ?? 15;

  // Playback animation loop. animateRef stores the latest callback so the RAF
  // loop recurses through the ref rather than closing over `animate`.
  const rafRef = useRef<number | null>(null);
  const lastTickRef = useRef<number>(0);
  const animateRef = useRef<(now: number) => void>(() => {});
  const TICK_INTERVAL_MS = 300; // one step every 300 ms

  const animate = useCallback(
    (now: number) => {
      if (now - lastTickRef.current >= TICK_INTERVAL_MS) {
        incrementTimestep();
        lastTickRef.current = now;
      }
      rafRef.current = requestAnimationFrame(animateRef.current);
    },
    [incrementTimestep]
  );

  useEffect(() => {
    animateRef.current = animate;
  }, [animate]);

  useEffect(() => {
    if (isPlaying) {
      lastTickRef.current = performance.now();
      rafRef.current = requestAnimationFrame(animateRef.current);
    } else {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    }
    return () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
      }
    };
  }, [isPlaying, animate]);

  // Seven evenly spaced tick labels across the day.
  const ticks = Array.from({ length: 7 }, (_, i) => Math.round((i * maxTimesteps) / 6));

  return (
    <div
      className="
        absolute bottom-6 left-1/2 -translate-x-1/2
        flex flex-col items-center gap-2
        bg-black/70 backdrop-blur-sm text-white
        rounded-xl px-6 py-4 shadow-2xl
        w-[min(560px,90vw)]
        z-10
      "
      aria-label="Time navigation controls"
    >
      {/* Header row */}
      <div className="flex w-full items-center justify-between text-xs font-mono text-slate-300">
        <DatePaginator />
        <span className="text-base font-semibold text-white tabular-nums">
          {timestepLabel(currentTimestep, timestamps, stepMinutes)}
        </span>
        <span>
          t = {currentTimestep} / {maxTimesteps}
        </span>
      </div>

      {/* Slider */}
      <input
        type="range"
        min={0}
        max={maxTimesteps}
        value={currentTimestep}
        onChange={(e) => setTimestep(Number(e.target.value))}
        className="w-full h-2 appearance-none rounded-full cursor-pointer accent-orange-400"
        aria-label="Timestep selector"
      />

      <div className="flex w-full justify-between px-0.5">
        {ticks.map((t, i) => (
          <span key={i} className="text-[10px] text-slate-400 font-mono">
            {timestepLabel(t, timestamps, stepMinutes).slice(0, 2)}h
          </span>
        ))}
      </div>

      {/* Play / Pause */}
      <button
        onClick={togglePlayback}
        className="
          mt-1 px-5 py-1.5 rounded-full text-sm font-medium
          bg-orange-500 hover:bg-orange-400 active:bg-orange-600
          transition-colors focus:outline-none focus:ring-2 focus:ring-orange-300
        "
        aria-label={isPlaying ? 'Pause playback' : 'Play time animation'}
      >
        {isPlaying ? '⏸ Pause' : '▶ Play'}
      </button>
    </div>
  );
}
