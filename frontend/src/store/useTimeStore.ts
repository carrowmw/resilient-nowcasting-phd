import { create } from 'zustand';
import type { AttentionEdge } from '@/types/payload';
import type { RunRegistryEntry } from '@/types/registry';

interface TimeState {
  currentTimestep: number;
  maxTimesteps: number;
  isPlaying: boolean;
  selectedNodeId: number | null;
  globalAttentionThreshold: number;
  egoGraphEdges: AttentionEdge[] | null;
  /** ISO date string (YYYY-MM-DD) for the currently displayed day. */
  startDate: string;

  // --- Multi-run + pagination state (Phase 1) ---
  activeRunId: string | null;
  activeDate: string | null;            // "2026-05-01"
  availableDates: string[];             // sorted, from spatial.json metadata
  runRegistry: RunRegistryEntry[];      // from run_registry.json
  isLoadingPage: boolean;               // true during a page.json fetch

  setTimestep: (step: number) => void;
  setMaxTimesteps: (max: number) => void;
  togglePlayback: () => void;
  incrementTimestep: () => void;
  setSelectedNodeId: (id: number | null) => void;
  setGlobalAttentionThreshold: (v: number) => void;
  setEgoGraphEdges: (edges: AttentionEdge[] | null) => void;
  setStartDate: (date: string) => void;

  // --- Multi-run + pagination actions ---
  setActiveRunId: (runId: string) => void;
  setActiveDate: (date: string) => void;
  setAvailableDates: (dates: string[]) => void;
  setRunRegistry: (runs: RunRegistryEntry[]) => void;
  setIsLoadingPage: (loading: boolean) => void;
  navigateDay: (direction: 'prev' | 'next') => void;
}

const useTimeStore = create<TimeState>((set, get) => ({
  currentTimestep: 0,
  maxTimesteps: 95,
  isPlaying: false,
  selectedNodeId: null,
  globalAttentionThreshold: 0.5,
  egoGraphEdges: null,
  startDate: '',

  activeRunId: null,
  activeDate: null,
  availableDates: [],
  runRegistry: [],
  isLoadingPage: false,

  setTimestep: (step: number) =>
    set({ currentTimestep: Math.max(0, Math.min(step, get().maxTimesteps)) }),

  setMaxTimesteps: (max: number) => set({ maxTimesteps: max }),

  togglePlayback: () => set((state) => ({ isPlaying: !state.isPlaying })),

  incrementTimestep: () =>
    set((state) => ({
      currentTimestep:
        state.currentTimestep >= state.maxTimesteps
          ? 0
          : state.currentTimestep + 1,
    })),

  setSelectedNodeId: (id: number | null) => set({ selectedNodeId: id }),

  // Clamp threshold to [0, 1] — attention weights live in [0, 1].
  setGlobalAttentionThreshold: (v: number) =>
    set({ globalAttentionThreshold: Math.max(0, Math.min(1, v)) }),

  setEgoGraphEdges: (edges: AttentionEdge[] | null) => set({ egoGraphEdges: edges }),

  setStartDate: (date: string) => set({ startDate: date }),

  setActiveRunId: (runId: string) => set({ activeRunId: runId }),

  // Changing the day resets the playhead and the current selection's stale detail.
  setActiveDate: (date: string) => set({ activeDate: date, startDate: date }),

  setAvailableDates: (dates: string[]) => set({ availableDates: dates }),

  setRunRegistry: (runs: RunRegistryEntry[]) => set({ runRegistry: runs }),

  setIsLoadingPage: (loading: boolean) => set({ isLoadingPage: loading }),

  navigateDay: (direction: 'prev' | 'next') => {
    const { availableDates, activeDate } = get();
    if (!activeDate || availableDates.length === 0) return;
    const idx = availableDates.indexOf(activeDate);
    if (idx === -1) return;
    const nextIdx = direction === 'next' ? idx + 1 : idx - 1;
    if (nextIdx < 0 || nextIdx >= availableDates.length) return; // boundary
    set({ activeDate: availableDates[nextIdx], startDate: availableDates[nextIdx] });
  },
}));

export default useTimeStore;
